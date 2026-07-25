# -*- coding: utf-8 -*-
"""物理引导展开式整帧神经均衡器。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from agent.cir_estimator import CIRCondition
from agent.peft import PEFTRegistry, mark_peft_group
from env.linear_operator import LinearChannelOperator


@dataclass(frozen=True)
class UnfoldedConfig:
    frame_len: int = 512
    max_delay: int = 40
    iterations: int = 4
    d_model: int = 96
    num_heads: int = 4
    adapter_rank: int = 8
    lora_rank: int = 8

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "UnfoldedConfig":
        return cls(**data)


class BottleneckAdapter(nn.Module):
    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.down = nn.Linear(d_model, rank)
        self.up = nn.Linear(rank, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(torch.relu(self.down(x)))


class LoRAResidual(nn.Module):
    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class DenoiserBlock(nn.Module):
    def __init__(self, config: UnfoldedConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(config.d_model, config.num_heads, batch_first=True)
        self.attn_lora = LoRAResidual(config.d_model, config.lora_rank)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model),
        )
        self.ffn_lora = LoRAResidual(config.d_model, config.lora_rank)
        self.adapter = BottleneckAdapter(config.d_model, config.adapter_rank)
        self.norm2 = nn.LayerNorm(config.d_model)
        mark_peft_group(self.attn_lora, "attention_lora")
        mark_peft_group(self.ffn_lora, "ffn_lora")
        mark_peft_group(self.adapter, "adapter")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out + self.attn_lora(x))
        x = self.norm2(x + self.ffn(x) + self.ffn_lora(x) + self.adapter(x))
        return x


class UnfoldedEqualizer(nn.Module):
    """显式 H/H^H 迭代与非因果 Transformer denoiser 结合的块均衡器。"""

    def __init__(self, config: UnfoldedConfig | None = None):
        super().__init__()
        self.config = config or UnfoldedConfig()
        self.feature_proj = nn.Linear(4, self.config.d_model)
        self.region_embedding = nn.Embedding(2, self.config.d_model)
        self.conditioner = nn.Sequential(
            nn.Linear(96, self.config.d_model * 2),
            nn.Tanh(),
            nn.Linear(self.config.d_model * 2, self.config.d_model * 2),
        )
        self.blocks = nn.ModuleList([DenoiserBlock(self.config) for _ in range(3)])
        self.head = nn.Linear(self.config.d_model, 1)
        self.alpha = nn.Parameter(torch.full((self.config.iterations,), 0.2))
        self.damping = nn.Parameter(torch.full((self.config.iterations,), 0.2))
        mark_peft_group(self.conditioner, "conditioner_film")
        mark_peft_group(self.head, "head")
        self.peft = PEFTRegistry(self)
        self.set_trainable_groups(set())

    def forward(
        self,
        rx_iq: torch.Tensor,
        condition: CIRCondition,
        region_ids: torch.Tensor,
        soft_tail: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rx_complex = torch.complex(rx_iq[..., 0], rx_iq[..., 1]).to(torch.complex64)
        operator = LinearChannelOperator(frame_len=rx_complex.shape[1], max_delay=condition.complex_cir.shape[1] - 1)
        soft_symbols = torch.zeros_like(rx_complex)
        logits = torch.zeros_like(rx_complex.real)
        for layer in range(self.config.iterations):
            residual = rx_complex - operator.forward(soft_symbols, condition.complex_cir, soft_tail)
            gradient = operator.adjoint(residual, condition.complex_cir)
            proposal = soft_symbols + self.alpha[layer].to(gradient.dtype) * gradient
            features = torch.stack((proposal.real, proposal.imag, residual.real, residual.imag), dim=-1)
            hidden = self.feature_proj(features) + self.region_embedding(region_ids.clamp_min(0).clamp_max(1))
            hidden = self._apply_film(hidden, condition)
            for block in self.blocks:
                hidden = block(hidden)
            logits = self.head(hidden).squeeze(-1)
            damping = torch.sigmoid(self.damping[layer])
            soft_update = torch.complex(torch.tanh(logits), torch.zeros_like(logits))
            soft_symbols = (1.0 - damping) * soft_update + damping * soft_symbols
        return logits, torch.sigmoid(logits)

    def _apply_film(self, hidden: torch.Tensor, condition: CIRCondition) -> torch.Tensor:
        latent = condition.latent_residual
        if latent.shape[1] < 96:
            latent = torch.nn.functional.pad(latent, (0, 96 - latent.shape[1]))
        latent = latent[:, :96].to(hidden.dtype)
        gamma_beta = self.conditioner(latent)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return hidden * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def set_trainable_groups(self, groups: set[str]) -> None:
        resolved = self.peft.resolve(groups)
        for parameter in self.parameters():
            parameter.requires_grad_(getattr(parameter, "_peft_group", None) in resolved)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.trainable_parameters())
