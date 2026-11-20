# -*- coding: utf-8 -*-
"""固定有限帧似然下的BPSK EP数学原型，尚未接入在线接收实验。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class BpskEPResult:
    logits: torch.Tensor
    gaussian_mean: torch.Tensor
    gaussian_variance: torch.Tensor
    site_precision: torch.Tensor
    site_linear: torch.Tensor
    rejected_site_updates: tuple[int, ...]


def bpsk_ep(precision, linear, *, steps=5, damping=.5, variance_floor=1e-8):
    """对 exp(-xᵀAx/2+bᵀx) 与均匀BPSK先验做阻尼EP。

    输入为实数似然精度A和线性项b，可来自正确边界的有限卷积，
    不要求循环信道。各维初始高斯site为N(0,1)。每轮先算cavity，
    BPSK矩匹配后做自然参数阻尼；非正精度的新site保持原值。
    本函数是固定点估计信道下的数学原型，不包括CIR或同步不确定性。
    """

    if (precision.ndim != 2 or precision.shape[0] != precision.shape[1]
            or linear.ndim != 1 or precision.shape[0] != linear.numel() or linear.numel() == 0):
        raise ValueError('EP需要非空方阵精度与等维线性项。')
    if (precision.dtype not in {torch.float32, torch.float64} or linear.dtype != precision.dtype
            or precision.device != linear.device):
        raise ValueError('EP输入须在同一设备上使用相同实浮点类型。')
    if type(steps) is not int or steps < 0:
        raise ValueError('EP更新次数须为非负整数。')
    if (not isinstance(damping, (int, float)) or not math.isfinite(damping) or not 0 < damping <= 1
            or not isinstance(variance_floor, (int, float)) or not 0 < variance_floor < 1):
        raise ValueError('阻尼须在(0,1]，方差下限须在(0,1)。')
    if not bool(torch.isfinite(precision).all() and torch.isfinite(linear).all()):
        raise ValueError('EP输入不能包含非有限数。')
    tolerance = 100 * torch.finfo(precision.dtype).eps
    if not torch.allclose(precision, precision.T, rtol=tolerance, atol=tolerance):
        raise ValueError('EP似然精度必须对称。')
    # 拒绝不定似然；不能把错误正规方程靠site正则项掩盖。
    eigen_min = torch.linalg.eigvalsh(precision).min()
    scale = precision.norm().clamp_min(1.)
    if bool(eigen_min < -tolerance * scale):
        raise ValueError('EP似然精度必须半正定。')

    site_precision = torch.ones_like(linear)
    site_linear = torch.zeros_like(linear)
    rejected = []

    def gaussian_and_cavity():
        chol = torch.linalg.cholesky(precision + torch.diag(site_precision))
        mean = torch.cholesky_solve((linear + site_linear)[:, None], chol)[:, 0]
        variance = torch.cholesky_inverse(chol).diagonal()
        cavity_precision = variance.reciprocal() - site_precision
        cavity_linear = mean / variance - site_linear
        return mean, variance, cavity_precision, cavity_linear

    for _ in range(steps):
        mean, variance, cavity_precision, cavity_linear = gaussian_and_cavity()
        tilted_mean = torch.tanh(cavity_linear)
        tilted_variance = (1 - tilted_mean.square()).clamp_min(variance_floor)
        proposal_precision = tilted_variance.reciprocal() - cavity_precision
        proposal_linear = tilted_mean / tilted_variance - cavity_linear
        valid = (proposal_precision > 0) & torch.isfinite(proposal_precision) & torch.isfinite(proposal_linear)
        rejected.append(int((~valid).sum()))
        site_precision = torch.where(valid,
            (1 - damping) * site_precision + damping * proposal_precision, site_precision)
        site_linear = torch.where(valid,
            (1 - damping) * site_linear + damping * proposal_linear, site_linear)
    mean, variance, _, cavity_linear = gaussian_and_cavity()
    # 在±1处，高斯cavity的二次项相同；LLR=2*cavity_linear。
    # 这是EP近似边缘，耦合信道中不称精确BPSK后验。
    return BpskEPResult(2 * cavity_linear, mean, variance, site_precision, site_linear, tuple(rejected))
