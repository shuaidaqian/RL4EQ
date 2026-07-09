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


class AdapterEqualizer(nn.Module):
    """Transformer + Adapter 的二分类均衡器。

    forward 输入形状为 ``(B, T, state_dim)``，输出 bit logits/probabilities。
    """

    def __init__(self, config: EqualizerConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Sequential(
            nn.Linear(config.state_dim, config.d_model),
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
        x = self.input_proj(states) + self.pos_encoding(pos)
        x = self.backbone(x)
        x = self.adapter(x)
        logits = self.output_head(x).squeeze(-1)
        return logits, torch.sigmoid(logits)

    def freeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def enable_parameter_efficient_tuning(self, train_adapter: bool = True, train_output: bool = True) -> None:
        """冻结主干，只开放 Adapter/输出头。"""
        self.freeze_all()
        if train_adapter:
            for param in self.adapter.parameters():
                param.requires_grad = True
        if train_output:
            for param in self.output_head.parameters():
                param.requires_grad = True

    def set_trainable_targets(self, train_adapter: bool, train_output: bool) -> None:
        self.enable_parameter_efficient_tuning(train_adapter=train_adapter, train_output=train_output)

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (param for param in self.parameters() if param.requires_grad)

    def trainable_parameter_count(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)
