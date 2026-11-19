# -*- coding: utf-8 -*-
"""从离线 CIR 轨迹估计一阶复信道动态统计。"""

from __future__ import annotations

import math

import torch


def estimate_ar1_statistics(trajectories: list[torch.Tensor]) -> dict[str, torch.Tensor | float | int]:
    """估计共同 AR(1) 系数和逐 tap 平稳功率。

    每条轨迹形状为 [时间, tap]。实现假定复 CIR 的条件均值为零、创新跨 tap 不相关，
    与 EME 生成器的复高斯 tap 演化相符。估计器仅用于离线训练集或开发集。
    """

    if not trajectories:
        raise ValueError("至少需要一条 CIR 轨迹。")
    length = trajectories[0].shape[-1]
    if length < 1:
        raise ValueError("CIR 轨迹必须至少包含一个 tap。")
    cross = 0j
    previous_energy = 0.0
    power = torch.zeros(length, dtype=torch.float64)
    state_count = 0
    transition_count = 0
    per_trajectory_rho = []
    for trajectory in trajectories:
        if trajectory.ndim != 2 or trajectory.shape[1] != length or trajectory.shape[0] < 2:
            raise ValueError("每条 CIR 轨迹必须是相同 tap 数且至少两帧的二维张量。")
        values = trajectory.to(dtype=torch.complex128, device="cpu")
        previous, current = values[:-1], values[1:]
        cross += torch.sum(previous.conj() * current).item()
        previous_energy += torch.sum(previous.abs().square()).item()
        trajectory_energy = torch.sum(previous.abs().square()).item()
        if trajectory_energy > 0.0:
            per_trajectory_rho.append(float((
                torch.sum(previous.conj() * current).item() / trajectory_energy
            ).real))
        power += current.abs().square().sum(dim=0).to(torch.float64)
        state_count += int(current.shape[0])
        transition_count += int(previous.shape[0])
    if previous_energy <= 0.0:
        raise ValueError("CIR 轨迹能量为零，无法估计 AR(1) 系数。")
    rho_pooled = float((cross / previous_energy).real)
    rho = float(sum(per_trajectory_rho) / len(per_trajectory_rho))
    if not math.isfinite(rho) or not 0.0 <= rho < 1.0:
        raise ValueError(f"估计得到非法 AR(1) 系数 rho={rho}。")
    stationary_power = power / state_count
    return {
        "rho": rho,
        "rho_energy_weighted": rho_pooled,
        "per_trajectory_rho": per_trajectory_rho,
        "stationary_power": stationary_power,
        "transition_count": transition_count,
        "state_count": state_count,
    }
