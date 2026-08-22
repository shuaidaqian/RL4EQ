# -*- coding: utf-8 -*-
"""传统 baseline 可用的 Pilot 相位/CFO 补偿。

补偿器只读取接收端可见的 Adapt Pilot：rx_symbols、adapt_symbols 和 adapt_mask。
Reward Pilot 与 Data 标签不参与估计。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PilotPhaseEstimate:
    phase0: float
    cfo_cycles_per_symbol: float
    pilot_count: int


def estimate_pilot_phase_line(
    receiver_view,
    reference_symbols: torch.Tensor | None = None,
    reference_mask: torch.Tensor | None = None,
    max_abs_cfo_cycles_per_symbol: float | None = None,
) -> PilotPhaseEstimate:
    """用 Adapt Pilot 相位误差拟合 phase(t)=phase0+2π*cfo*t。"""

    rx = receiver_view.rx_symbols.flatten().to(torch.complex64)
    tx = receiver_view.adapt_symbols.flatten().to(torch.complex64)
    mask = receiver_view.adapt_mask.flatten().bool()
    if reference_symbols is not None:
        tx = reference_symbols.flatten().to(torch.complex64)
    if reference_mask is not None:
        mask = mask & reference_mask.flatten().bool()
    mask = mask & (tx.abs() > 1e-6)
    pilot_count = int(mask.sum().item())
    if pilot_count < 2:
        return PilotPhaseEstimate(0.0, 0.0, pilot_count)
    indices = torch.nonzero(mask, as_tuple=False).flatten().to(torch.float32)
    phase_error = unwrap_phase(torch.angle(rx[mask] * torch.conj(tx[mask])))
    centered_t = indices - indices.mean()
    centered_phase = phase_error - phase_error.mean()
    denom = centered_t.square().sum().clamp_min(1e-8)
    slope = (centered_t * centered_phase).sum() / denom
    intercept = phase_error.mean() - slope * indices.mean()
    cfo = slope / (2.0 * torch.pi)
    if max_abs_cfo_cycles_per_symbol is not None:
        cfo = torch.clamp(
            cfo,
            min=-float(max_abs_cfo_cycles_per_symbol),
            max=float(max_abs_cfo_cycles_per_symbol),
        )
    return PilotPhaseEstimate(float(intercept.item()), float(cfo.item()), pilot_count)


def apply_phase_correction(rx_symbols: torch.Tensor, phase0: float, cfo_cycles_per_symbol: float) -> torch.Tensor:
    """按估计相位直线反向旋转整帧接收符号。"""

    rx = rx_symbols.flatten().to(torch.complex64)
    indices = torch.arange(rx.numel(), dtype=torch.float32, device=rx.device)
    phase = float(phase0) + 2.0 * torch.pi * float(cfo_cycles_per_symbol) * indices
    return (rx * torch.exp(-1j * phase).to(torch.complex64)).reshape_as(rx_symbols)


def unwrap_phase(phase: torch.Tensor) -> torch.Tensor:
    """兼容当前 PyTorch 版本的 1D 相位展开。"""

    flat = phase.flatten().to(torch.float32)
    if flat.numel() <= 1:
        return flat.reshape_as(phase)
    delta = flat[1:] - flat[:-1]
    two_pi = 2.0 * torch.pi
    correction = torch.where(delta > torch.pi, -two_pi, torch.where(delta < -torch.pi, two_pi, torch.zeros_like(delta)))
    offsets = torch.cat((torch.zeros(1, dtype=flat.dtype, device=flat.device), torch.cumsum(correction, dim=0)))
    return (flat + offsets).reshape_as(phase)
