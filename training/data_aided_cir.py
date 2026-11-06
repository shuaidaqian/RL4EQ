# -*- coding: utf-8 -*-
"""数据辅助 CIR 精化：只用接收波形与模型 soft 判决更新物理状态。

与 ``agent.cir_estimator.decision_directed_cir_update`` 的 Pilot-only 历史
协议不同，本模块允许把已接收 Data 位置的模型 soft 判决作为 LS 行，但：

- 不读取 Data 发送符号、Data 比特或真实 CIR；
- 只用当前帧已经完整接收的波形；
- 更新后的 CIR 只用于下一帧，属于因果的跨帧物理状态；
- 置信度低于阈值的 Data 位置被排除，降低错误传播。

它是物理状态估计器，不是神经参数更新。若在线分支要把它算作研究增益，
必须与纯 Pilot CIR 对照，并在四臂中单独归因。
"""

from __future__ import annotations

import torch


def data_aided_cir_update(
    frame,
    logits: torch.Tensor,
    max_delay: int,
    previous_cir: torch.Tensor,
    *,
    alpha: float = 0.5,
    confidence_threshold: float = 0.5,
    ridge: float = 1e-3,
    phase0: float = 0.0,
    cfo: float = 0.0,
) -> torch.Tensor:
    """用 Data 位置 soft 判决做加权 LS，返回归一化 CIR。

    ``logits`` 必须是当前帧完整接收波形的前向输出。函数只使用
    ``frame.rx_symbols``、``frame.data_mask`` 与 ``logits``；不读取
    ``frame.bits`` 或 Data 位置的 ``tx_symbols``。

    ``phase0``/``cfo`` 来自 Adapt Pilot 的相位线估计；提供时先把接收波形
    去旋转到与 acquisition CIR 相同的相位坐标，避免残余 CFO 被吸进 tap 增益。
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha 必须位于 [0, 1]。")
    if ridge < 0.0:
        raise ValueError("ridge 必须非负。")
    device = logits.device
    previous = previous_cir.to(device=device, dtype=torch.complex64).reshape(-1)
    if previous.numel() != max_delay + 1:
        raise ValueError("previous_cir 长度必须为 max_delay + 1。")
    soft = torch.tanh(logits / 2.0)
    rx = frame.rx_symbols.to(device=device, dtype=torch.complex64)
    if phase0 != 0.0 or cfo != 0.0:
        positions = torch.arange(rx.numel(), device=device, dtype=torch.float32)
        correction = torch.exp(
            -1j * (float(phase0) + 2.0 * torch.pi * float(cfo) * positions)
        ).to(torch.complex64)
        rx = rx * correction
    data_mask = frame.data_mask.to(device=device).bool()
    positions = torch.arange(soft.numel(), device=device)
    valid = data_mask & (positions >= max_delay) & (soft.abs() >= float(confidence_threshold))
    if int(valid.sum()) < max_delay + 1:
        return previous / previous.norm().clamp_min(1e-12)

    index = positions[valid]
    rows = torch.stack([soft[index - delay] for delay in range(max_delay + 1)], dim=1)
    targets = rx[index]
    weights = soft[index].abs().clamp_min(1e-3).sqrt().unsqueeze(1)
    design = torch.complex(rows, torch.zeros_like(rows)) * weights
    observation = targets * weights.squeeze(1)
    identity = torch.eye(max_delay + 1, device=device, dtype=torch.complex64)
    # 正规方程等价于增广 LS，但只需求解 (L x L) 系统；复数 lstsq 在
    # Windows/CUDA 上会回退到很慢的实现。
    gram = design.conj().transpose(0, 1) @ design + ridge * identity
    rhs = design.conj().transpose(0, 1) @ observation + ridge * previous
    estimate = torch.linalg.solve(gram, rhs).to(torch.complex64)
    estimate = estimate / estimate.norm().clamp_min(1e-12)
    blended = (1.0 - alpha) * previous + alpha * estimate
    return blended / blended.norm().clamp_min(1e-12)


def pilot_ridge_cir_update(
    frame,
    max_delay: int,
    previous_cir: torch.Tensor,
    *,
    alpha: float = 0.5,
    ridge: float = 1e-2,
    phase0: float = 0.0,
    cfo: float = 0.0,
) -> torch.Tensor:
    """Pilot-only 全 tap ridge LS：已知 Adapt Pilot + 相位补偿 + acquisition 先验。

    适配 prefix 布局：只使用帧首 Adapt Pilot 中卷积行完全落在 Pilot 内的位置。
    与 ``pilot_sparse_cir_update`` 的 top-K 稀疏支持不同，这里对全部 tap 做
    ridge 正则，避免支持选择不稳定；仍不读取 Data 或 Reward 标签。
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha 必须位于 [0, 1]。")
    if ridge <= 0.0:
        raise ValueError("Pilot ridge 必须为正数，保证欠定系统可解。")
    device = previous_cir.device
    previous = previous_cir.to(device=device, dtype=torch.complex64).reshape(-1)
    if previous.numel() != max_delay + 1:
        raise ValueError("previous_cir 长度必须为 max_delay + 1。")
    rx = frame.rx_symbols.to(device=device, dtype=torch.complex64)
    if phase0 != 0.0 or cfo != 0.0:
        positions_all = torch.arange(rx.numel(), device=device, dtype=torch.float32)
        correction = torch.exp(
            -1j * (float(phase0) + 2.0 * torch.pi * float(cfo) * positions_all)
        ).to(torch.complex64)
        rx = rx * correction
    pilot_mask = frame.adapt_mask.to(device=device).bool()
    pilot_positions = torch.nonzero(pilot_mask, as_tuple=False).flatten()
    pilot_positions = pilot_positions[pilot_positions >= max_delay]
    if pilot_positions.numel() < 2:
        return previous / previous.norm().clamp_min(1e-12)
    tx = frame.tx_symbols.to(device=device, dtype=torch.complex64)
    rows = torch.stack([tx[pilot_positions - delay] for delay in range(max_delay + 1)], dim=1)
    targets = rx[pilot_positions]
    identity = torch.eye(max_delay + 1, device=device, dtype=torch.complex64)
    gram = rows.conj().transpose(0, 1) @ rows + ridge * identity
    rhs = rows.conj().transpose(0, 1) @ targets + ridge * previous
    estimate = torch.linalg.solve(gram, rhs).to(torch.complex64)
    estimate = estimate / estimate.norm().clamp_min(1e-12)
    blended = (1.0 - alpha) * previous + alpha * estimate
    return blended / blended.norm().clamp_min(1e-12)
