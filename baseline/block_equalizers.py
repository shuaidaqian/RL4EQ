# -*- coding: utf-8 -*-
"""Perfect-CSI 与块检测诊断基线。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from env.linear_operator import LinearChannelOperator


@dataclass(frozen=True)
class DetectionResult:
    logits: torch.Tensor
    probabilities: torch.Tensor
    soft_tail: torch.Tensor
    iterations: int


def bit_error_rate(logits: torch.Tensor, bits: torch.Tensor) -> float:
    decisions = logits >= 0
    return float((decisions.to(torch.bool) != bits.to(torch.bool)).float().mean().item())


def perfect_csi_cg_detect(
    rx: torch.Tensor,
    cir: torch.Tensor,
    soft_tail: torch.Tensor,
    noise_variance: torch.Tensor | float,
    iterations: int,
) -> DetectionResult:
    """用已知 CIR 求解整帧块检测，仅用于可达性和上界基线。"""

    batched = rx.ndim == 2
    rx_b = rx if batched else rx.unsqueeze(0)
    cir_b = cir if cir.ndim == 2 else cir.unsqueeze(0)
    tail_b = soft_tail if soft_tail.ndim == 2 else soft_tail.unsqueeze(0)
    operator = LinearChannelOperator(frame_len=rx_b.shape[1], max_delay=cir_b.shape[1] - 1)
    tail_contribution = operator.forward(torch.zeros_like(rx_b), cir_b, tail_b)
    rhs = operator.adjoint(rx_b - tail_contribution, cir_b)
    sigma = torch.as_tensor(noise_variance, dtype=torch.float32, device=rx_b.device).clamp_min(1e-8)

    def normal_matvec(vector: torch.Tensor) -> torch.Tensor:
        return operator.adjoint(operator.forward(vector, cir_b, torch.zeros_like(tail_b)), cir_b) + sigma * vector

    estimate = _conjugate_gradient(normal_matvec, rhs, iterations)
    logits = 2.0 * estimate.real / sigma
    probabilities = torch.sigmoid(logits)
    next_tail = torch.complex(torch.tanh(logits[:, -tail_b.shape[1] :] / 2.0), torch.zeros_like(tail_b.real))
    result = DetectionResult(logits=logits, probabilities=probabilities, soft_tail=next_tail, iterations=iterations)
    return result if batched else DetectionResult(
        logits=result.logits.squeeze(0),
        probabilities=result.probabilities.squeeze(0),
        soft_tail=result.soft_tail.squeeze(0),
        iterations=result.iterations,
    )


def perfect_csi_bpsk_refine_detect(
    rx: torch.Tensor,
    cir: torch.Tensor,
    soft_tail: torch.Tensor,
    noise_variance: torch.Tensor | float,
    cg_iterations: int = 64,
    refine_iterations: int = 2,
) -> DetectionResult:
    """Perfect-CIR BPSK 块检测器。

    先用 LMMSE/CG 得到连续估计，再在真实 CIR 下做坐标翻转局部搜索。
    每次翻转只在降低接收残差能量时接受；整个过程不读取真实 bit 标签。
    """

    batched = rx.ndim == 2
    rx_b = rx if batched else rx.unsqueeze(0)
    cir_b = cir if cir.ndim == 2 else cir.unsqueeze(0)
    tail_b = soft_tail if soft_tail.ndim == 2 else soft_tail.unsqueeze(0)
    sigma = torch.as_tensor(noise_variance, dtype=torch.float32, device=rx_b.device).clamp_min(1e-8)
    initial = perfect_csi_cg_detect(rx_b, cir_b, tail_b, sigma, iterations=cg_iterations)
    logits_b = _coordinate_refine_batch(rx_b, cir_b, tail_b, initial.logits, refine_iterations)
    tail_len = tail_b.shape[1]
    tails_b = torch.complex(torch.tanh(logits_b[:, -tail_len:] / 2.0), torch.zeros_like(tail_b.real))
    result = DetectionResult(
        logits=logits_b,
        probabilities=torch.sigmoid(logits_b),
        soft_tail=tails_b,
        iterations=cg_iterations + refine_iterations,
    )
    return result if batched else DetectionResult(
        logits=result.logits.squeeze(0),
        probabilities=result.probabilities.squeeze(0),
        soft_tail=result.soft_tail.squeeze(0),
        iterations=result.iterations,
    )


def iterative_bpsk_detect(
    rx: torch.Tensor,
    cir: torch.Tensor,
    soft_tail: torch.Tensor,
    noise_variance: torch.Tensor | float,
    iterations: int,
    damping: float = 0.5,
) -> DetectionResult:
    """解析阻尼 BPSK 迭代检测器。"""

    result = perfect_csi_cg_detect(rx, cir, soft_tail, noise_variance, max(1, iterations))
    logits = result.logits * (1.0 - damping)
    return DetectionResult(
        logits=logits,
        probabilities=torch.sigmoid(logits),
        soft_tail=result.soft_tail,
        iterations=iterations,
    )


def _coordinate_refine_single(
    rx: torch.Tensor,
    cir: torch.Tensor,
    tail: torch.Tensor,
    initial_logits: torch.Tensor,
    refine_iterations: int,
) -> torch.Tensor:
    return _coordinate_refine_batch(
        rx.unsqueeze(0),
        cir.unsqueeze(0),
        tail.unsqueeze(0),
        initial_logits.unsqueeze(0),
        refine_iterations,
    ).squeeze(0)


def _coordinate_refine_batch(
    rx: torch.Tensor,
    cir: torch.Tensor,
    tail: torch.Tensor,
    initial_logits: torch.Tensor,
    refine_iterations: int,
) -> torch.Tensor:
    frame_len = rx.numel()
    if rx.ndim != 2:
        raise ValueError("rx 必须是 (B, T)。")
    batch_size, frame_len = rx.shape
    max_delay = cir.shape[1] - 1
    operator = LinearChannelOperator(frame_len=frame_len, max_delay=max_delay)
    tail_contribution = operator.forward(
        torch.zeros(batch_size, frame_len, dtype=torch.complex64, device=rx.device),
        cir,
        tail,
    )
    target = rx - tail_contribution
    symbols = torch.where(initial_logits >= 0, torch.ones_like(initial_logits), -torch.ones_like(initial_logits)).to(torch.complex64)
    frame_cir = cir.to(torch.complex64)
    prediction = _current_frame_convolution_batch(symbols, frame_cir)
    residual = target - prediction
    for _ in range(max(0, refine_iterations)):
        gain = _parallel_flip_gain_batch(residual, symbols, frame_cir)
        flip_mask = gain < -1e-8
        if not bool(flip_mask.any()):
            break
        previous_energy = torch.sum(torch.abs(residual) ** 2, dim=1)
        proposal = symbols.clone()
        proposal[flip_mask] = -proposal[flip_mask]
        proposal_residual = target - _current_frame_convolution_batch(proposal, frame_cir)
        proposal_energy = torch.sum(torch.abs(proposal_residual) ** 2, dim=1)
        accept_all = torch.real(proposal_energy - previous_energy) < -1e-8

        best_indices = torch.argmin(gain, dim=1)
        best_proposal = symbols.clone()
        best_proposal[torch.arange(batch_size, device=symbols.device), best_indices] *= -1.0
        best_residual = target - _current_frame_convolution_batch(best_proposal, frame_cir)
        best_energy = torch.sum(torch.abs(best_residual) ** 2, dim=1)
        accept_best = (~accept_all) & (torch.real(best_energy - previous_energy) < -1e-8)

        if not bool((accept_all | accept_best).any()):
            break
        symbols = torch.where(accept_all.unsqueeze(1), proposal, symbols)
        residual = torch.where(accept_all.unsqueeze(1), proposal_residual, residual)
        symbols = torch.where(accept_best.unsqueeze(1), best_proposal, symbols)
        residual = torch.where(accept_best.unsqueeze(1), best_residual, residual)
    return symbols.real * 20.0


def _parallel_flip_gain_batch(residual: torch.Tensor, symbols: torch.Tensor, cir: torch.Tensor) -> torch.Tensor:
    """向量化估计翻转每个 BPSK 符号的残差能量变化。"""

    batch_size, frame_len = symbols.shape
    max_delay = cir.shape[1] - 1
    gain = torch.zeros(batch_size, frame_len, dtype=torch.float32, device=symbols.device)
    norm = torch.zeros(batch_size, frame_len, dtype=torch.float32, device=symbols.device)
    for delay in range(max_delay + 1):
        valid = frame_len - delay
        if valid <= 0:
            continue
        delta = 2.0 * symbols[:, :valid] * cir[:, delay].unsqueeze(1)
        residual_slice = residual[:, delay:]
        gain[:, :valid] = gain[:, :valid] + 2.0 * torch.real(torch.conj(residual_slice) * delta)
        norm[:, :valid] = norm[:, :valid] + torch.abs(delta).pow(2).real
    return gain + norm


def _current_frame_convolution(symbols: torch.Tensor, cir: torch.Tensor) -> torch.Tensor:
    return _current_frame_convolution_batch(symbols.unsqueeze(0), cir.unsqueeze(0)).squeeze(0)


def _current_frame_convolution_batch(symbols: torch.Tensor, cir: torch.Tensor) -> torch.Tensor:
    frame_len = symbols.numel()
    batch_size, frame_len = symbols.shape
    max_delay = cir.shape[1] - 1
    output = torch.zeros(batch_size, frame_len, dtype=torch.complex64, device=symbols.device)
    for delay in range(max_delay + 1):
        output[:, delay:] = output[:, delay:] + cir[:, delay].unsqueeze(1) * symbols[:, : frame_len - delay]
    return output


def _conjugate_gradient(matvec, rhs: torch.Tensor, iterations: int) -> torch.Tensor:
    x = torch.zeros_like(rhs)
    residual = rhs - matvec(x)
    direction = residual.clone()
    rs_old = torch.sum(torch.conj(residual) * residual, dim=1, keepdim=True).real
    for _ in range(max(1, iterations)):
        mat_direction = matvec(direction)
        denom = torch.sum(torch.conj(direction) * mat_direction, dim=1, keepdim=True).real.clamp_min(1e-12)
        alpha = rs_old / denom
        x = x + alpha.to(x.dtype) * direction
        residual = residual - alpha.to(residual.dtype) * mat_direction
        rs_new = torch.sum(torch.conj(residual) * residual, dim=1, keepdim=True).real
        beta = rs_new / rs_old.clamp_min(1e-12)
        direction = residual + beta.to(direction.dtype) * direction
        rs_old = rs_new
    return x
