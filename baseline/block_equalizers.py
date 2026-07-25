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
