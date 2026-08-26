# -*- coding: utf-8 -*-
"""低维连续调制状态。

PPO 不直接输出整帧 bit，也不直接生成高维参数增量；它只输出有界低维调制
向量，用来调制神经块均衡器的 Adapter、FiLM、LoRA 和输出头。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModulationConfig:
    num_adapter_gates: int = 3
    num_lora_scales: int = 3

    @property
    def action_dim(self) -> int:
        return self.num_adapter_gates + 1 + self.num_lora_scales + 3


@dataclass(frozen=True)
class ModulationState:
    adapter_gates: torch.Tensor
    film_residual_scale: torch.Tensor
    lora_scales: torch.Tensor
    head_temperature: torch.Tensor
    head_bias: torch.Tensor
    confidence_threshold: torch.Tensor

    @staticmethod
    def identity(config: ModulationConfig, device=None) -> "ModulationState":
        return ModulationState(
            adapter_gates=torch.ones(config.num_adapter_gates, device=device, dtype=torch.float32),
            film_residual_scale=torch.tensor(0.0, device=device, dtype=torch.float32),
            lora_scales=torch.ones(config.num_lora_scales, device=device, dtype=torch.float32),
            head_temperature=torch.tensor(1.0, device=device, dtype=torch.float32),
            head_bias=torch.tensor(0.0, device=device, dtype=torch.float32),
            confidence_threshold=torch.tensor(0.0, device=device, dtype=torch.float32),
        )

    @staticmethod
    def from_raw(raw: torch.Tensor, config: ModulationConfig) -> "ModulationState":
        raw = raw.reshape(-1).to(dtype=torch.float32)
        _validate_action_dim(raw, config)
        adapter_end = config.num_adapter_gates
        film_idx = adapter_end
        lora_start = film_idx + 1
        lora_end = lora_start + config.num_lora_scales
        temp_idx = lora_end
        bias_idx = temp_idx + 1
        confidence_idx = bias_idx + 1
        return ModulationState(
            adapter_gates=2.0 * torch.sigmoid(raw[:adapter_end]),
            film_residual_scale=torch.sigmoid(raw[film_idx]) - 0.5,
            lora_scales=2.0 * torch.sigmoid(raw[lora_start:lora_end]),
            head_temperature=0.5 + 1.5 * torch.sigmoid(raw[temp_idx]),
            head_bias=2.0 * torch.sigmoid(raw[bias_idx]) - 1.0,
            confidence_threshold=torch.sigmoid(raw[confidence_idx]),
        )

    @staticmethod
    def from_vector(vector: torch.Tensor, config: ModulationConfig) -> "ModulationState":
        vector = vector.reshape(-1).to(dtype=torch.float32)
        _validate_action_dim(vector, config)
        adapter_end = config.num_adapter_gates
        film_idx = adapter_end
        lora_start = film_idx + 1
        lora_end = lora_start + config.num_lora_scales
        temp_idx = lora_end
        bias_idx = temp_idx + 1
        confidence_idx = bias_idx + 1
        return ModulationState(
            adapter_gates=vector[:adapter_end].clamp(0.0, 2.0),
            film_residual_scale=vector[film_idx].clamp(-0.5, 0.5),
            lora_scales=vector[lora_start:lora_end].clamp(0.0, 2.0),
            head_temperature=vector[temp_idx].clamp(0.5, 2.0),
            head_bias=vector[bias_idx].clamp(-1.0, 1.0),
            confidence_threshold=vector[confidence_idx].clamp(0.0, 1.0),
        )

    def to_vector(self) -> torch.Tensor:
        return torch.cat(
            [
                self.adapter_gates.reshape(-1),
                self.film_residual_scale.reshape(1),
                self.lora_scales.reshape(-1),
                self.head_temperature.reshape(1),
                self.head_bias.reshape(1),
                self.confidence_threshold.reshape(1),
            ]
        ).to(dtype=torch.float32)


def _validate_action_dim(vector: torch.Tensor, config: ModulationConfig) -> None:
    if vector.numel() != config.action_dim:
        raise ValueError(f"调制向量维度必须为 {config.action_dim}，实际为 {vector.numel()}。")
