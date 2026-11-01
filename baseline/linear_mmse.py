# -*- coding: utf-8 -*-
"""无 CP 线性信道的传统线性 MMSE 校准参考；不含判决反馈、神经网络或 RL。"""

from __future__ import annotations

import math

import torch

from env.linear_operator import LinearChannelOperator


def linear_block_lmmse(rx, cir, tail, noise_variance, *, known_symbols=None,
                      known_mask=None, real_bpsk=False, iterations=128, rtol=1e-6):
    """用 CG 求线性高斯估计，处理真实块边界与已知 Adapt Pilot。

    复数模式最小化 ||y-Hx||² + sigma²||x||²；实 BPSK 线性先验使用
    Re(H^H H) 与 sigma²/2。只改变二阶统计，不做硬判决反馈或非线性搜索。
    调用方只能传 acquisition/Adapt 估计 CIR、可知历史和 Adapt 已知符号。
    """
    if rx.ndim != 1 or cir.ndim != 1 or tail.ndim != 1:
        raise ValueError("校准参考仅接受单帧一维输入。")
    if cir.numel() < 1 or tail.numel() != cir.numel() - 1:
        raise ValueError("历史尾长度必须等于 CIR 记忆长度。")
    variance = float(noise_variance)
    if not math.isfinite(variance) or variance <= 0 or iterations < 1 or not 0 < rtol < 1:
        raise ValueError("噪声方差、迭代次数或收敛容差无效。")
    rx = rx.to(torch.complex64)
    cir = cir.to(device=rx.device, dtype=torch.complex64)
    tail = tail.to(device=rx.device, dtype=torch.complex64)
    if not all(bool(torch.isfinite(value).all()) for value in (rx, cir, tail)):
        raise ValueError("接收信号、CIR 与历史尾必须有限。")
    if (known_mask is None) != (known_symbols is None):
        raise ValueError("已知符号和掩码必须同时提供。")
    fixed = torch.zeros_like(rx)
    mask = torch.zeros(rx.numel(), dtype=torch.bool, device=rx.device)
    if known_mask is not None:
        if known_mask.shape != rx.shape or known_symbols.shape != rx.shape:
            raise ValueError("已知符号和掩码必须与当前帧同形。")
        mask = known_mask.to(device=rx.device, dtype=torch.bool)
        # 掩码外的发送符号即使被污染也不会进入估计。
        fixed[mask] = known_symbols.to(device=rx.device, dtype=torch.complex64)[mask]
        if not bool(torch.isfinite(fixed).all()):
            raise ValueError("已知 Pilot 必须有限。")
        if real_bpsk and bool((fixed.imag != 0).any()):
            raise ValueError("实 BPSK 先验要求已知符号没有虚部。")
    unknown = ~mask
    operator = LinearChannelOperator(rx.numel(), cir.numel() - 1)
    target = rx - operator.forward(fixed, cir, tail)
    rhs = operator.adjoint(target, cir)
    ridge = variance / 2.0 if real_bpsk else variance
    if real_bpsk:
        rhs = rhs.real
    rhs = torch.where(unknown, rhs, torch.zeros_like(rhs))
    zero_tail = torch.zeros_like(tail)

    def normal(vector):
        projected = torch.where(unknown, vector, torch.zeros_like(vector))
        response = operator.adjoint(operator.forward(projected.to(torch.complex64), cir, zero_tail), cir)
        response = response.real if real_bpsk else response
        return torch.where(unknown, response + ridge * projected, vector)

    estimate = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    energy = torch.vdot(residual, residual).real
    initial_energy = float(energy)
    threshold = max(initial_energy * rtol * rtol, 1e-24)
    completed = 0
    for step in range(int(iterations)):
        if float(energy) <= threshold:
            break
        response = normal(direction)
        denominator = torch.vdot(direction, response).real
        if not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
            raise FloatingPointError("MMSE 正定系统的 CG 分母异常。")
        alpha = energy / denominator
        estimate = estimate + alpha * direction
        residual = residual - alpha * response
        next_energy = torch.vdot(residual, residual).real
        direction = residual + (next_energy / energy) * direction
        energy = next_energy
        completed = step + 1
    # 直接重算正规方程残差，避免只报告递推残差产生的虚假收敛。
    actual_energy = float(torch.vdot(rhs - normal(estimate), rhs - normal(estimate)).real)
    relative_residual = math.sqrt(actual_energy / initial_energy) if initial_energy > 0 else 0.0
    output = fixed + estimate.to(torch.complex64)
    return output, {
        "iterations": completed, "relative_normal_residual": relative_residual,
        "converged": relative_residual <= max(rtol * 3, 1e-7),
        "real_bpsk_linear_prior": bool(real_bpsk), "known_pilot_count": int(mask.sum()),
        "uses_neural_network": False, "uses_rl": False,
        "uses_decision_feedback": False, "linear_convolution_boundary": True,
    }
