# -*- coding: utf-8 -*-
"""残余 CFO 与慢相位扰动。

该模块只描述接收端同步残差造成的相位旋转，不包含硬件非线性、编码或高阶调制。
真实 impairment 参数不进入 receiver_view，传统 baseline 必须从 Adapt Pilot 自行估计。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class PhaseImpairmentSettings:
    cfo_cycles_per_symbol: float = 0.0
    phase_noise_std: float = 0.0


def settings_from_profile(
    profile: str,
    rng: np.random.Generator,
    *,
    cfo_cycles_per_symbol: float = 0.0,
    phase_noise_std: float = 0.0,
) -> PhaseImpairmentSettings:
    """根据 profile 采样 episode 级同步残差。

    显式传入的 cfo_cycles_per_symbol / phase_noise_std 优先级最高，便于单元测试和
    精准校准实验。
    """

    if cfo_cycles_per_symbol != 0.0 or phase_noise_std != 0.0:
        return PhaseImpairmentSettings(float(cfo_cycles_per_symbol), float(phase_noise_std))
    presets = {
        "clean": ((0.0, 0.0), (0.0, 0.0)),
        "cfo_tiny": ((0.0002, 0.0008), (0.0, 0.0)),
        "phase_tiny": ((0.0, 0.0), (0.0002, 0.0008)),
        "cfo_phase_tiny": ((0.0002, 0.0008), (0.0002, 0.0008)),
        "cfo_light": ((0.001, 0.003), (0.0, 0.0)),
        "phase_light": ((0.0, 0.0), (0.001, 0.003)),
        "cfo_phase_light": ((0.001, 0.003), (0.001, 0.003)),
        "cfo_phase_mid": ((0.003, 0.008), (0.003, 0.008)),
        "eme_slow_drift_v1": ((0.0008, 0.0008), (0.003, 0.003)),
    }
    if profile not in presets:
        raise ValueError(f"未知 impairment_profile：{profile}")
    cfo_range, phase_range = presets[profile]
    cfo = _sample_signed_abs(rng, cfo_range)
    phase_std = float(rng.uniform(*phase_range)) if phase_range[1] > 0 else 0.0
    return PhaseImpairmentSettings(cfo, phase_std)


def apply_phase_impairment(
    rx: torch.Tensor,
    *,
    global_start: int,
    settings: PhaseImpairmentSettings,
    phase_state: float,
    torch_rng: torch.Generator,
) -> tuple[torch.Tensor, float]:
    """对一帧接收符号施加连续 CFO 与慢相位随机游走。"""

    if settings.cfo_cycles_per_symbol == 0.0 and settings.phase_noise_std == 0.0 and phase_state == 0.0:
        return rx, phase_state
    frame_len = rx.numel()
    indices = torch.arange(global_start, global_start + frame_len, dtype=torch.float32, device=rx.device)
    phase = 2.0 * torch.pi * float(settings.cfo_cycles_per_symbol) * indices
    if settings.phase_noise_std > 0.0:
        increments = torch.randn(frame_len, generator=torch_rng, dtype=torch.float32) * float(settings.phase_noise_std)
        walk = torch.cumsum(increments, dim=0) + float(phase_state)
        phase = phase + walk.to(device=rx.device)
        next_phase_state = float(walk[-1].item())
    else:
        next_phase_state = float(phase_state)
    rotated = rx * torch.exp(1j * phase.to(torch.float32)).to(torch.complex64)
    return rotated.to(torch.complex64), next_phase_state


def _sample_signed_abs(rng: np.random.Generator, value_range: tuple[float, float]) -> float:
    low, high = value_range
    if high <= 0.0:
        return 0.0
    magnitude = float(rng.uniform(low, high))
    sign = -1.0 if float(rng.uniform()) < 0.5 else 1.0
    return sign * magnitude
