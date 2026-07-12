# -*- coding: utf-8 -*-
"""参数高效神经均衡器。

本模块从第二阶段开始服务在线自适应：主干 Transformer 默认冻结，
在线阶段只更新 Adapter 和输出头，控制更新规模与时延。
"""

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn


@dataclass
class EqualizerConfig:
    state_dim: int = 45
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dim_feedforward: int = 128
    adapter_rank: int = 8
    dropout: float = 0.0
    max_len: int = 512
    use_channel_encoder: bool = False
    channel_dim: int = 32
    use_sync_head: bool = False
    sync_dim: int = 32
    sync_delay_bins: int = 9


class ResidualAdapter(nn.Module):
    """瓶颈 Adapter，用少量参数吸收新信道偏移。"""

    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.down = nn.Linear(d_model, rank)
        self.act = nn.ReLU()
        self.up = nn.Linear(rank, d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


class NeuralChannelEncoder(nn.Module):
    """从训练序列和导频中提取帧级神经信道表征。"""

    def __init__(self, state_dim: int, channel_dim: int):
        super().__init__()
        hidden_dim = max(channel_dim * 2, 32)
        rx_dim = state_dim - 3
        feature_dim = state_dim + rx_dim
        self.known_embed = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, channel_dim),
            nn.ReLU(),
        )
        self.summary = nn.Sequential(
            nn.Linear(channel_dim * 2, channel_dim),
            nn.LayerNorm(channel_dim),
            nn.ReLU(),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        known_mask = states[..., -2:-1].clamp(0.0, 1.0)
        known_symbol = states[..., -3:-2]
        rx_window = states[..., :-3]
        corr_window = rx_window * known_symbol
        encoder_input = torch.cat([states, corr_window], dim=-1)
        embedded = self.known_embed(encoder_input) * known_mask
        denom = known_mask.sum(dim=1).clamp_min(1.0)
        mean = embedded.sum(dim=1) / denom
        var = (((embedded - mean.unsqueeze(1)) * known_mask) ** 2).sum(dim=1) / denom
        return self.summary(torch.cat([mean, torch.sqrt(var + 1e-6)], dim=-1))


class SyncPhaseDelayHead(nn.Module):
    """纯神经同步/相位/时延 latent head，只使用已知符号构造相关特征。"""

    def __init__(self, state_dim: int, sync_dim: int, delay_bins: int):
        super().__init__()
        rx_dim = state_dim - 3
        hidden_dim = max(sync_dim * 2, 32)
        feature_dim = rx_dim * 2 + 3
        self.feature = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sync_dim),
            nn.ReLU(),
        )
        self.summary = nn.Sequential(
            nn.Linear(sync_dim * 2, sync_dim),
            nn.LayerNorm(sync_dim),
            nn.ReLU(),
        )
        self.delay_head = nn.Linear(sync_dim, delay_bins)
        self.phase_head = nn.Linear(sync_dim, 2)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        known_mask = states[..., -2:-1].clamp(0.0, 1.0)
        known_symbol = states[..., -3:-2]
        rx_window = states[..., :-3]
        corr_window = rx_window * known_symbol
        feature_input = torch.cat([rx_window, corr_window, states[..., -3:]], dim=-1)
        embedded = self.feature(feature_input) * known_mask
        denom = known_mask.sum(dim=1).clamp_min(1.0)
        mean = embedded.sum(dim=1) / denom
        var = (((embedded - mean.unsqueeze(1)) * known_mask) ** 2).sum(dim=1) / denom
        latent = self.summary(torch.cat([mean, torch.sqrt(var + 1e-6)], dim=-1))
        delay_logits = self.delay_head(latent)
        phase = torch.tanh(self.phase_head(latent))
        return torch.cat([latent, delay_logits, phase], dim=-1)


class AdapterEqualizer(nn.Module):
    """Transformer + Adapter 的二分类均衡器。

    forward 输入形状为 ``(B, T, state_dim)``，输出 bit logits/probabilities。
    """

    def __init__(self, config: EqualizerConfig):
        super().__init__()
        self.config = config
        input_dim = config.state_dim
        self.channel_encoder = None
        if config.use_channel_encoder:
            self.channel_encoder = NeuralChannelEncoder(config.state_dim, config.channel_dim)
            input_dim += config.channel_dim
            self.channel_film = nn.Linear(config.channel_dim, config.d_model * 2)
            nn.init.zeros_(self.channel_film.weight)
            nn.init.zeros_(self.channel_film.bias)
        self.sync_head = None
        if config.use_sync_head:
            self.sync_head = SyncPhaseDelayHead(config.state_dim, config.sync_dim, config.sync_delay_bins)
            sync_out_dim = config.sync_dim + config.sync_delay_bins + 2
            self.sync_film = nn.Linear(sync_out_dim, config.d_model * 2)
            nn.init.zeros_(self.sync_film.weight)
            nn.init.zeros_(self.sync_film.bias)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, config.d_model),
            nn.LayerNorm(config.d_model),
        )
        self.pos_encoding = nn.Embedding(config.max_len, config.d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
        )
        self.backbone = nn.TransformerEncoder(enc_layer, num_layers=config.n_layers)
        self.adapter = ResidualAdapter(config.d_model, config.adapter_rank)
        self.output_head = nn.Linear(config.d_model, 1)

    def forward(self, states: torch.Tensor):
        bsz, steps, _ = states.shape
        pos = torch.arange(steps, device=states.device).unsqueeze(0).expand(bsz, -1)
        model_input = states
        channel_state = None
        if self.channel_encoder is not None:
            channel_state = self.channel_encoder(states).unsqueeze(1).expand(-1, steps, -1)
            model_input = torch.cat([states, channel_state], dim=-1)
        x = self.input_proj(model_input) + self.pos_encoding(pos)
        if channel_state is not None:
            gamma, beta = self.channel_film(channel_state[:, 0, :]).chunk(2, dim=-1)
            x = x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        if self.sync_head is not None:
            sync_state = self.sync_head(states)
            gamma, beta = self.sync_film(sync_state).chunk(2, dim=-1)
            x = x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        x = self.backbone(x)
        x = self.adapter(x)
        logits = self.output_head(x).squeeze(-1)
        return logits, torch.sigmoid(logits)

    def freeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def enable_parameter_efficient_tuning(
        self,
        train_adapter: bool = True,
        train_output: bool = True,
        train_sync: bool = False,
    ) -> None:
        """冻结主干，只开放 Adapter/输出头。"""
        self.freeze_all()
        if train_adapter:
            for param in self.adapter.parameters():
                param.requires_grad = True
        if train_output:
            for param in self.output_head.parameters():
                param.requires_grad = True
        if train_sync and self.sync_head is not None:
            for param in self.sync_head.parameters():
                param.requires_grad = True
            for param in self.sync_film.parameters():
                param.requires_grad = True

    def set_trainable_targets(self, train_adapter: bool, train_output: bool, train_sync: bool = False) -> None:
        self.enable_parameter_efficient_tuning(
            train_adapter=train_adapter,
            train_output=train_output,
            train_sync=train_sync,
        )

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (param for param in self.parameters() if param.requires_grad)

    def trainable_parameter_count(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)
