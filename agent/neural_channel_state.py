# -*- coding: utf-8 -*-
"""把合法 Pilot 的状态残差映射为有界神经 CIR 调制。"""

from __future__ import annotations

import math
import torch
from torch import nn

from agent.peft import mark_peft_group


def known_pilot_equations(rx, symbols, known_mask, score_mask, memory):
    """未知历史与未知发送符号不参与任何拟合行。"""
    positions = torch.arange(memory, rx.numel(), device=rx.device)
    indices = positions[:, None] - torch.arange(memory + 1, device=rx.device)
    valid = score_mask[positions] & known_mask[indices].all(dim=1)
    indices, positions = indices[valid], positions[valid]
    if not positions.numel():
        raise ValueError("Pilot 信道状态任务没有完全已知的卷积行。")
    return symbols[indices], rx[positions]


class NeuralChannelStateAdapter(nn.Module):
    """离线学习状态调制，在线选择 gate head 或有界复数记忆 head。"""

    def __init__(self, *, acquisition_rows, rho_frame, max_direction_norm=.5, memory_taps=0, memory_output_scale=.25):
        super().__init__()
        if acquisition_rows < 1 or not 0 < rho_frame <= 1 or max_direction_norm <= 0:
            raise ValueError("信道状态先验需要合法 acquisition 行数、相关系数和方向上限。")
        self.acquisition_rows = int(acquisition_rows)
        self.rho_frame = float(rho_frame)
        self.max_direction_norm = float(max_direction_norm)
        self.features = nn.Sequential(nn.Linear(9, 16), nn.GELU(), nn.Linear(16, 16), nn.GELU())
        self.head = nn.Linear(16, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        mark_peft_group(self.features, "channel_state_features")
        mark_peft_group(self.head, "channel_state_head")
        if type(memory_taps) is not int or memory_taps < 0 or not math.isfinite(memory_output_scale) or memory_output_scale <= 0:
            raise ValueError("复数状态记忆需要非负整数 tap 数和正输出尺度。")
        self.memory_output_scale = float(memory_output_scale)
        self.memory = nn.Embedding(memory_taps, 2) if memory_taps else None
        self.memory_scale = nn.Linear(16, 2) if memory_taps else None
        if self.memory is not None:
            nn.init.zeros_(self.memory.weight)
            nn.init.zeros_(self.memory_scale.weight)
            nn.init.zeros_(self.memory_scale.bias)
            mark_peft_group(self.memory, "channel_state_memory")
            mark_peft_group(self.memory_scale, "channel_state_memory_scale")

    def prior_variance(self, prior, noise, frame_index):
        """方向构造与 MAP 目标共享同一个对角 acquisition/过程先验。"""
        acquisition_variance = noise / self.acquisition_rows
        drift = 1.0 - self.rho_frame ** (2 * (int(frame_index) + 1))
        return acquisition_variance + drift * (prior.abs().square() + acquisition_variance)

    def forward(self, rx, cir, noise, adapt_symbols, adapt_mask, frame_index):
        if int(frame_index) < 0:
            raise ValueError("神经状态分支只支持 acquisition 之后的数据帧。")
        directions, features = [], []
        for index in range(rx.shape[0]):
            # 合法状态特征不经旧网络反传；本阶段只学习新状态表示与输出 head。
            with torch.no_grad():
                prior = cir[index].detach()
                variance_noise = noise[index].reshape(()).detach().clamp_min(1e-6)
                design, target = known_pilot_equations(rx[index].detach(), adapt_symbols[index],
                    adapt_mask[index], adapt_mask[index], prior.numel() - 1)
                variance = self.prior_variance(prior, variance_noise, frame_index)
                precision = variance_noise / variance.clamp_min(1e-10)
                gram = design.conj().T @ design
                direction = torch.linalg.solve(gram + torch.diag(precision.to(gram.dtype)),
                    design.conj().T @ (target - design @ prior))
                direction = direction * (self.max_direction_norm / direction.norm().clamp_min(1e-12)).clamp(max=1.)
                std = variance.sqrt().clamp_min(1e-6)
                scale = prior.abs().square().mean().sqrt().clamp_min(1e-6)
                strength = gram.diag().real / (gram.diag().real + precision).clamp_min(1e-6)
                feature = torch.stack((prior.real / scale, prior.imag / scale,
                    direction.real / std, direction.imag / std, std.log() / 5.,
                    variance_noise.log().expand_as(std) / 5.,
                    prior.abs().square() / prior.abs().square().sum().clamp_min(1e-6),
                    torch.linspace(0, 1, prior.numel(), device=rx.device), strength), dim=-1)
                directions.append(direction)
                features.append(torch.tanh(feature))
        hidden = self.features(torch.stack(features))
        gates = 2. * torch.tanh(self.head(hidden).squeeze(-1))
        updated = cir + gates.to(cir.dtype) * torch.stack(directions)
        if self.memory is not None:
            if self.memory.num_embeddings != cir.shape[-1]:
                raise ValueError("复数状态记忆的 tap 数与 CIR 不一致。")
            # 记忆为零时精确保留原输出；离线只学尺度，在线仅更新受限记忆 head。
            scale = torch.exp(torch.tanh(self.memory_scale(hidden)))
            residual = self.memory_output_scale * torch.tanh(self.memory.weight).unsqueeze(0) * scale
            updated = updated + torch.complex(residual[..., 0], residual[..., 1]).to(cir.dtype)
        return updated
