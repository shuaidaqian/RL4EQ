# -*- coding: utf-8 -*-
"""固定物理接收机上的神经检测子空间，不修改 CIR 或同步状态。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class PilotObservation:
    """在线边界：仅含已接收特征、Adapt 标签及独立验证标签。"""

    base_logits: torch.Tensor
    features: torch.Tensor
    logit_scale: torch.Tensor
    adapt_mask: torch.Tensor
    adapt_labels: torch.Tensor
    validation_mask: torch.Tensor
    validation_labels: torch.Tensor


class DetectionSubspace(nn.Module):
    """离线学习非线性残差基，在线仅更新实数系数；不存储物理 tap。"""

    def __init__(self, dimension: int, architecture: str = "pointwise_mlp_v1",
                 learned_gradient_transport: bool = False):
        super().__init__()
        if dimension not in {2, 4, 8}:
            raise ValueError("检测子空间只允许预先指定的2/4/8维。")
        if architecture not in {"pointwise_mlp_v1", "temporal_conv_v1"}:
            raise ValueError("检测子空间基网络结构未登记。")
        self.dimension = dimension
        self.architecture = architecture
        self.learned_gradient_transport = bool(learned_gradient_transport)
        if architecture == "temporal_conv_v1":
            self.basis_network = nn.Sequential(
                nn.Conv1d(7, 16, kernel_size=5, padding=2), nn.GELU(),
                nn.Conv1d(16, dimension, kernel_size=1), nn.Tanh())
        else:
            self.basis_network = nn.Sequential(nn.Linear(7, 16), nn.Tanh(), nn.Linear(16, dimension), nn.Tanh())
        # 零系数精确保留来源网络；普通监督首步先学习系数，随后表示也可获得梯度。
        self.coefficients = nn.Parameter(torch.zeros(dimension))
        if self.learned_gradient_transport:
            # 单位阵初始化时与原始增量完全一致；下三角因子保证映射对称正定。
            self.gradient_transport_raw = nn.Parameter(torch.zeros(dimension, dimension))

    def gradient_transport_matrix(self):
        """返回有界对称正定的低维梯度传输矩阵；关闭时返回None。"""
        if not self.learned_gradient_transport:
            return None
        raw = self.gradient_transport_raw
        strict_lower = torch.tril(raw, diagonal=-1)
        strict_lower = 0.25 * torch.tanh(strict_lower)
        diagonal = torch.exp(0.5 * torch.tanh(torch.diagonal(raw)))
        factor = strict_lower + torch.diag(diagonal)
        return factor @ factor.transpose(-1, -2)

    def basis(self, observation: PilotObservation):
        if self.architecture == "temporal_conv_v1":
            features = observation.features.transpose(0, 1).unsqueeze(0)
            basis = self.basis_network(features).squeeze(0).transpose(0, 1)
        else:
            basis = self.basis_network(observation.features)
        return basis * observation.logit_scale / math.sqrt(self.dimension)

    def logits(self, observation: PilotObservation, coefficients=None):
        coefficients = self.coefficients if coefficients is None else coefficients
        return observation.base_logits + self.basis(observation) @ coefficients


@torch.no_grad()
def capture_observation(model, context):
    """从已有物理残差支路取特征；钩子局限于本次前向，异常时也移除。"""
    from training.pilot_transfer import context_logits

    if model.symbol_residual is None:
        raise ValueError("来源接收机必须已有符号残差支路。")
    captured = []
    hook = model.symbol_residual.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    try:
        base = context_logits(model, context, visible="full_rx").detach()
    finally:
        hook.remove()
    if len(captured) != 1 or captured[0].shape != (1, base.numel(), 7):
        raise ValueError("物理检测特征的来源或尺寸不符合固定协议。")
    view = context.frame.receiver_view()
    # 将有界符号残差换成 logit 尺度；仅依赖接收机已知噪声设定。
    scale = (.5 / context.condition.noise_variance.reshape(()).clamp_min(1e-6)).reshape(1, 1)
    return PilotObservation(base, captured[0][0].clone(), scale,
        view.adapt_mask, (view.adapt_symbols.real > 0).float(),
        context.frame.reward_mask, (view.reward_symbols.real > 0).float())
