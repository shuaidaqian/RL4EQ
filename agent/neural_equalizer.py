# -*- coding: utf-8 -*-
"""长上下文 Pilot 条件神经均衡器。"""

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EqualizerConfig:
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 3
    dim_feedforward: int = 192
    adapter_rank: int = 8
    max_len: int = 512
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    region_dim: int = 8
    dropout: float = 0.0


class ResidualAdapter(nn.Module):
    """在线阶段开放的瓶颈残差 Adapter。"""

    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.down = nn.Linear(d_model, rank)
        self.up = nn.Linear(rank, d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.up(F.gelu(self.down(value)))


class DilatedResidualBlock(nn.Module):
    """保持序列长度不变的扩张卷积残差块。"""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.project = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        x = self.norm(value).transpose(1, 2)
        x = self.project(F.gelu(self.conv(x))).transpose(1, 2)
        return residual + x


class PilotConditioner(nn.Module):
    """只聚合 Adapt Pilot，输出帧级信道条件向量。"""

    def __init__(self, d_model: int):
        super().__init__()
        hidden = max(d_model // 2, 32)
        self.feature = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.GELU(),
        )
        self.summary = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(
        self,
        rx_symbols: torch.Tensor,
        adapt_pilot_symbols: torch.Tensor,
        adapt_pilot_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = adapt_pilot_mask.bool()
        safe_symbols = torch.where(mask, adapt_pilot_symbols, torch.zeros_like(adapt_pilot_symbols))
        features = torch.cat([rx_symbols, safe_symbols.unsqueeze(-1)], dim=-1)
        embedded = self.feature(features) * mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(embedded.dtype)
        mean = embedded.sum(dim=1) / denom
        centered = (embedded - mean.unsqueeze(1)) * mask.unsqueeze(-1)
        variance = centered.square().sum(dim=1) / denom
        return self.summary(torch.cat([mean, torch.sqrt(variance + 1e-6)], dim=-1))


class AdapterTransformerBlock(nn.Module):
    """Transformer 编码块及其参数高效 Adapter。"""

    def __init__(self, config: EqualizerConfig):
        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.adapter = ResidualAdapter(config.d_model, config.adapter_rank)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.encoder(value))


class ExtremeDelayEqualizer(nn.Module):
    """扩张卷积、Pilot 条件和 Transformer 组成的 BPSK 均衡器。"""

    def __init__(self, config: EqualizerConfig):
        super().__init__()
        self.config = config
        self.region_embedding = nn.Embedding(3, config.region_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(2 + config.region_dim, config.d_model),
            nn.LayerNorm(config.d_model),
        )
        self.position_embedding = nn.Embedding(config.max_len, config.d_model)
        self.pilot_conditioner = PilotConditioner(config.d_model)
        self.condition_film = nn.Linear(config.d_model, config.d_model * 2)
        nn.init.zeros_(self.condition_film.weight)
        nn.init.zeros_(self.condition_film.bias)
        self.conv_blocks = nn.ModuleList(
            DilatedResidualBlock(config.d_model, dilation) for dilation in config.dilations
        )
        self.transformer_blocks = nn.ModuleList(
            AdapterTransformerBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, 1)

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * sum(int(value) for value in self.config.dilations)

    def forward(
        self,
        rx_symbols: torch.Tensor,
        region_ids: torch.Tensor,
        adapt_pilot_symbols: torch.Tensor,
        adapt_pilot_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, steps, _ = rx_symbols.shape
        if steps > self.config.max_len:
            raise ValueError(f"序列长度 {steps} 超过 max_len={self.config.max_len}。")
        region_features = self.region_embedding(region_ids.long())
        x = self.input_projection(torch.cat([rx_symbols.float(), region_features], dim=-1))
        positions = torch.arange(steps, device=rx_symbols.device)
        x = x + self.position_embedding(positions).unsqueeze(0).expand(batch_size, -1, -1)

        condition = self.pilot_conditioner(
            rx_symbols.float(), adapt_pilot_symbols.float(), adapt_pilot_mask
        )
        gamma, beta = self.condition_film(condition).chunk(2, dim=-1)
        x = x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        for block in self.conv_blocks:
            x = block(x)
        for block in self.transformer_blocks:
            x = block(x)
        logits = self.output_head(self.final_norm(x)).squeeze(-1)
        return logits, torch.sigmoid(logits)

    def freeze_all(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False

    def set_trainable_targets(self, train_adapters: bool, train_output: bool) -> None:
        self.freeze_all()
        if train_adapters:
            for block in self.transformer_blocks:
                for parameter in block.adapter.parameters():
                    parameter.requires_grad = True
        if train_output:
            for parameter in self.output_head.parameters():
                parameter.requires_grad = True

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def capture_peft_state(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.state_dict().items()
            if "adapter" in name or name.startswith("output_head")
        }

    def restore_peft_state(self, state: dict[str, torch.Tensor]) -> None:
        current = self.state_dict()
        for name, value in state.items():
            if name not in current:
                raise KeyError(f"未知 PEFT 参数: {name}")
            current[name].copy_(value.to(current[name].device))

