# -*- coding: utf-8 -*-
"""贝叶斯递归 CIR 估计：acquisition 先验 + Pilot/Data 后验更新。

信息边界：
- acquisition 帧的发送符号在接收端已知，可构造先验协方差；
- Adapt Pilot 符号已知，可用于 Pilot 后验；
- Data 位置只用模型 soft 判决 ``tanh(logit/2)``，不读 Data 标签；
- 真实 CIR 只用于离线 oracle/NMSE 诊断，不进入估计。

Data 行的观测噪声包含符号方差经 CIR 能量传播后的项：
``R_t = sigma^2 + sum_l |h_l|^2 Var_{t-l}``，
因此低 SNR 下错误符号会自动降权，而不是按真实 SNR 调度。
"""

from __future__ import annotations

import torch


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.conj().transpose(-2, -1))


def _rows(symbols: torch.Tensor, positions: torch.Tensor, max_delay: int) -> torch.Tensor:
    return torch.stack([symbols[positions - delay] for delay in range(max_delay + 1)], dim=1)


def _phase_correct(receiver: torch.Tensor, positions: torch.Tensor, phase0: float, cfo: float) -> torch.Tensor:
    phase = torch.exp(-1j * (float(phase0) + 2.0 * torch.pi * float(cfo) * positions.to(torch.float32)))
    return receiver[positions] * phase.to(torch.complex64)


def acquisition_prior(
    acquisition,
    cfo: float,
    max_delay: int,
    noise_variance: float,
    *,
    ridge: float = 1e-2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从 acquisition 帧构造归一化 CIR 先验均值与协方差。"""

    device = acquisition.tx_symbols.device
    tx = acquisition.tx_symbols.to(torch.complex64)
    rx = acquisition.rx_symbols.to(torch.complex64)
    positions = torch.arange(max_delay, tx.numel(), device=device)
    corrected = _phase_correct(rx, positions, 0.0, cfo)
    design = _rows(tx, positions, max_delay)
    identity = torch.eye(max_delay + 1, device=device, dtype=torch.complex64)
    gram = design.conj().transpose(0, 1) @ design + float(ridge) * identity
    estimate = torch.linalg.solve(gram, design.conj().transpose(0, 1) @ corrected).to(torch.complex64)
    norm = estimate.norm().clamp_min(1e-12)
    covariance = float(noise_variance) * torch.linalg.inv(_symmetrize(gram)) / norm.square()
    return (estimate / norm).detach(), _symmetrize(covariance).detach()


def posterior_update(
    prior_mean: torch.Tensor,
    prior_covariance: torch.Tensor,
    design: torch.Tensor,
    observation: torch.Tensor,
    row_noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """加权高斯后验；``row_noise`` 为每个观测行的总噪声方差。"""

    device = prior_mean.device
    precision = torch.linalg.inv(_symmetrize(prior_covariance))
    weights = row_noise.reshape(-1).clamp_min(1e-8).reciprocal()
    information = design.conj().transpose(0, 1) @ (weights.unsqueeze(1) * design)
    lambda_post = _symmetrize(precision + information)
    rhs = precision @ prior_mean + design.conj().transpose(0, 1) @ (weights * observation)
    estimate = torch.linalg.solve(lambda_post, rhs).to(torch.complex64)
    covariance = torch.linalg.inv(lambda_post)
    norm = estimate.norm().clamp_min(1e-12)
    return (estimate / norm).detach(), _symmetrize(covariance / norm.square()).detach()


def pilot_posterior(
    frame,
    prior_mean: torch.Tensor,
    prior_covariance: torch.Tensor,
    max_delay: int,
    noise_variance: float,
    phase0: float,
    cfo: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapt Pilot 后验；只使用已知 Pilot 符号。"""

    device = prior_mean.device
    mask = frame.adapt_mask.to(device=device).bool()
    positions = torch.nonzero(mask, as_tuple=False).flatten()
    positions = positions[positions >= max_delay]
    if positions.numel() < 2:
        return prior_mean.detach(), prior_covariance.detach()
    symbols = frame.tx_symbols.to(device=device).to(torch.complex64)
    corrected = _phase_correct(frame.rx_symbols.to(device=device).to(torch.complex64), positions, phase0, cfo)
    design = _rows(symbols, positions, max_delay)
    row_noise = torch.full((positions.numel(),), float(noise_variance), device=device)
    return posterior_update(prior_mean, prior_covariance, design, corrected, row_noise)


def data_posterior(
    frame,
    logits: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_covariance: torch.Tensor,
    max_delay: int,
    noise_variance: float,
    phase0: float,
    cfo: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Data soft 判决后验；符号方差通过 CIR 能量传播到行噪声。"""

    device = prior_mean.device
    soft = torch.tanh(logits / 2.0)
    variance = (1.0 - soft.square()).clamp_min(0.0)
    mask = frame.data_mask.to(device=device).bool()
    positions = torch.nonzero(mask, as_tuple=False).flatten()
    positions = positions[positions >= max_delay]
    if positions.numel() < 2:
        return prior_mean.detach(), prior_covariance.detach()
    corrected = _phase_correct(frame.rx_symbols.to(device=device).to(torch.complex64), positions, phase0, cfo)
    real_design = _rows(soft, positions, max_delay)
    design = torch.complex(real_design, torch.zeros_like(real_design))
    cir_energy = prior_mean.abs().square()
    # sum_l |h_l|^2 Var_{t-l}：对每行按延迟展开符号方差。
    propagated = torch.zeros_like(variance[positions])
    for delay in range(max_delay + 1):
        propagated = propagated + cir_energy[delay].to(propagated.dtype) * variance[positions - delay]
    row_noise = float(noise_variance) + propagated
    return posterior_update(prior_mean, prior_covariance, design, corrected, row_noise)


def add_process_noise(covariance: torch.Tensor, process_noise: float) -> torch.Tensor:
    if process_noise <= 0.0:
        return covariance
    identity = torch.eye(covariance.shape[0], device=covariance.device, dtype=covariance.dtype)
    return _symmetrize(covariance + float(process_noise) * identity)
