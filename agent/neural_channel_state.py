# -*- coding: utf-8 -*-
"""把合法 Pilot 的状态残差映射为有界神经 CIR 调制。"""

from __future__ import annotations

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
    """离线学习逐 tap 的状态调制表示，在线只更新 17 参数 head。"""

    def __init__(self, *, acquisition_rows, rho_frame, max_direction_norm=.5):
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
        gates = 2. * torch.tanh(self.head(self.features(torch.stack(features))).squeeze(-1))
        return cir + gates.to(cir.dtype) * torch.stack(directions)
