# -*- coding: utf-8 -*-
"""已知 acquisition CFO 时的圆周相关相位估计，仅使用 Adapt 观测。"""

from __future__ import annotations

import torch

from baseline.traditional_equalizers import _pilot_reference_from_cir
from env.linear_operator import LinearChannelOperator


def _vectorized_reference(frame, cir, tail, *, clean_rows):
    """一次矩阵乘法生成合法参考，避免逐 tap 判断引起设备同步。"""
    memory = cir.numel()-1
    positions = torch.nonzero(frame.adapt_mask, as_tuple=False).flatten()
    if memory:
        padded_tail = torch.nn.functional.pad(tail.flatten(), (max(0, memory-tail.numel()), 0))[-memory:]
        history = torch.cat((padded_tail.to(cir.device, cir.dtype), frame.tx_symbols))
        known_history = torch.cat((torch.ones(memory, device=cir.device, dtype=torch.bool), frame.adapt_mask))
    else:
        history, known_history = frame.tx_symbols, frame.adapt_mask
    windows = history.unfold(0, memory+1, 1).flip(-1)[positions]
    known = known_history.unfold(0, memory+1, 1).flip(-1)[positions]
    active = torch.ones_like(cir, dtype=torch.bool) if clean_rows else cir.abs()>1e-8
    reference = windows@torch.where(active, cir, 0)
    valid = (known | ~active[None]).all(1)
    if clean_rows:
        valid &= positions>=memory
    else:
        valid &= reference.abs()>1e-6
    return positions[valid], reference[valid]


def circular_phase_features(frame, cir, tail, cfo, *, clean_rows=False, reference_backend="vectorized"):
    """在复数域累加相关量，避免低 SNR 解缠绕跳周污染公共相位。

    CIR 与 CFO 视作给定时，公共相位为复 AWGN 似然的闭式最大值。
    实际 CIR/tail/CFO 有误差，因此不宣称联合估计无偏或已校准。
    ``clean_rows`` 排除任何涉及未知前帧符号的行，用作独立消融。
    """

    if reference_backend == "vectorized":
        positions, prediction = _vectorized_reference(frame, cir, tail, clean_rows=clean_rows)
    elif reference_backend != "legacy":
        raise ValueError("未知参考波形计算实现。")
    elif clean_rows:
        memory = cir.numel() - 1
        positions = torch.arange(memory, frame.rx_symbols.numel(), device=cir.device)
        known = torch.stack([frame.adapt_mask[positions - k] for k in range(memory + 1)], 1).all(1)
        positions = positions[known]
        operator = LinearChannelOperator(frame_len=frame.rx_symbols.numel(), max_delay=memory)
        reference = operator.forward(frame.tx_symbols[None], cir[None], tail[None]).flatten()
        prediction = reference[positions]
    else:
        from types import SimpleNamespace
        view = SimpleNamespace(adapt_symbols=frame.tx_symbols, adapt_mask=frame.adapt_mask,
                               rx_symbols=frame.rx_symbols)
        reference, mask = _pilot_reference_from_cir(view, cir, tail)
        reference = reference.flatten()
        positions = torch.nonzero(frame.adapt_mask & mask.flatten(), as_tuple=False).flatten()
        prediction = reference[positions]
    if positions.numel() < 2:
        raise ValueError("圆周相位估计至少需要两个合法 Adapt 观测。")
    target = frame.rx_symbols[positions] * torch.exp(-2j * torch.pi * float(cfo) * positions.float())
    correlation = torch.vdot(prediction, target)
    phase0 = torch.angle(correlation)
    output = torch.zeros(16, dtype=torch.float32, device=cir.device)
    for block, indices in enumerate(torch.tensor_split(torch.arange(positions.numel(), device=cir.device), 4)):
        if not indices.numel():
            continue
        value, expected = target[indices], prediction[indices]
        residual_angle = torch.angle(value * expected.conj() * torch.exp(-1j * phase0))
        residual = value - expected * torch.exp(1j * phase0)
        output[4 * block:4 * block + 4] = torch.stack((
            phase0, torch.as_tensor(cfo, dtype=torch.float32, device=cir.device),
            residual_angle.var(unbiased=False),
            residual.abs().square().mean() / expected.abs().square().mean().clamp_min(1e-8),
        ))
    return output
