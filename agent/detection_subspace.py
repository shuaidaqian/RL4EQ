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


class _DilatedResidualBlock(nn.Module):
    """以残差旁路保留局部可辨方向，同时扩展时间感受野。"""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GELU(), nn.Conv1d(channels, channels, kernel_size=1))

    def forward(self, value):
        return value + 0.1 * self.conv(value)


class _TemporalDilatedResidualBasis(nn.Module):
    """输出127符号感受野的逐位置神经基。"""

    def __init__(self, dimension: int):
        super().__init__()
        self.input = nn.Conv1d(7, 16, kernel_size=1)
        self.blocks = nn.Sequential(*[
            _DilatedResidualBlock(16, dilation) for dilation in (1, 2, 4, 8, 16, 32)])
        self.output = nn.Sequential(nn.Conv1d(16, dimension, kernel_size=1), nn.Tanh())

    def forward(self, features):
        return self.output(self.blocks(self.input(features)))


class DetectionSubspace(nn.Module):
    """离线学习非线性残差基，在线仅更新实数系数；不存储物理 tap。"""

    def __init__(self, dimension: int, architecture: str = "pointwise_mlp_v1",
                 learned_gradient_transport: bool = False,
                 learned_context_update: bool = False):
        super().__init__()
        if dimension not in {2, 4, 8, 10, 12, 14, 16}:
            raise ValueError("检测子空间只允许预先指定的2/4/8/10/12/14/16维。")
        if architecture not in {"pointwise_mlp_v1", "temporal_conv_v1", "temporal_dilated_127_v1",
                                "temporal_dilated_residual_127_v1", "pilot_context_mlp_v1"}:
            raise ValueError("检测子空间基网络结构未登记。")
        self.dimension = dimension
        self.architecture = architecture
        self.learned_gradient_transport = bool(learned_gradient_transport)
        self.learned_context_update = bool(learned_context_update)
        self.use_context_update = self.learned_context_update
        if architecture == "pilot_context_mlp_v1":
            self.context_feature = nn.Sequential(nn.Linear(7, 32), nn.Tanh())
            self.context_projection = nn.Linear(dimension, 32, bias=False)
            self.context_output = nn.Linear(32, 1, bias=False)
        elif architecture == "temporal_conv_v1":
            self.basis_network = nn.Sequential(
                nn.Conv1d(7, 16, kernel_size=5, padding=2), nn.GELU(),
                nn.Conv1d(16, dimension, kernel_size=1), nn.Tanh())
        elif architecture == "temporal_dilated_127_v1":
            layers = []
            channels = 7
            for dilation in (1, 2, 4, 8, 16, 32):
                layers.extend((nn.Conv1d(channels, 16, kernel_size=3,
                    padding=dilation, dilation=dilation), nn.GELU()))
                channels = 16
            layers.extend((nn.Conv1d(16, dimension, kernel_size=1), nn.Tanh()))
            self.basis_network = nn.Sequential(*layers)
        elif architecture == "temporal_dilated_residual_127_v1":
            self.basis_network = _TemporalDilatedResidualBasis(dimension)
        else:
            self.basis_network = nn.Sequential(nn.Linear(7, 16), nn.Tanh(), nn.Linear(16, dimension), nn.Tanh())
        # 零系数精确保留来源网络；普通监督首步先学习系数，随后表示也可获得梯度。
        self.coefficients = nn.Parameter(torch.zeros(dimension))
        if self.learned_gradient_transport:
            # 单位阵初始化时与原始增量完全一致；下三角因子保证映射对称正定。
            self.gradient_transport_raw = nn.Parameter(torch.zeros(dimension, dimension))
        if self.learned_context_update:
            # 输入Adapt梯度方向、历史分歧和梯度尺度；零输出初始化保留原Adam首步。
            self.context_update_network = nn.Sequential(
                nn.Linear(2 * dimension + 1, 16), nn.Tanh(), nn.Linear(16, dimension))
            nn.init.zeros_(self.context_update_network[-1].weight)
            nn.init.zeros_(self.context_update_network[-1].bias)

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

    def context_update_delta(self, history_gradients, learning_rate):
        """仅由Adapt梯度及历史统计产生有界低维增量，不读取验证或Data标签。"""
        if not self.learned_context_update or not self.use_context_update:
            return None
        gradients = torch.stack(history_gradients)
        mean = gradients.mean(dim=0)
        scale = mean.norm().clamp_min(1e-12)
        disagreement = (gradients - mean).square().mean(dim=0).clamp_min(1e-12).sqrt() / scale
        log_scale = torch.log10(scale).clamp(-12., 2.) / 12.
        features = torch.cat((mean / scale, disagreement, log_scale.reshape(1)))
        return 0.5 * learning_rate * torch.tanh(self.context_update_network(features))

    def basis(self, observation: PilotObservation, coefficients=None):
        if self.architecture == "pilot_context_mlp_v1":
            hidden = self.context_feature(observation.features)
            context = self.coefficients if coefficients is None else coefficients
            local_slope = 1. - torch.tanh(hidden + self.context_projection(context)).square()
            basis = (local_slope * self.context_output.weight) @ self.context_projection.weight
            return basis * observation.logit_scale / math.sqrt(self.dimension)
        if self.architecture in {"temporal_conv_v1", "temporal_dilated_127_v1",
                                 "temporal_dilated_residual_127_v1"}:
            features = observation.features.transpose(0, 1).unsqueeze(0)
            basis = self.basis_network(features).squeeze(0).transpose(0, 1)
        else:
            basis = self.basis_network(observation.features)
        return basis * observation.logit_scale / math.sqrt(self.dimension)

    def logits(self, observation: PilotObservation, coefficients=None):
        coefficients = self.coefficients if coefficients is None else coefficients
        if self.architecture == "pilot_context_mlp_v1":
            hidden = self.context_feature(observation.features)
            modulated = torch.tanh(hidden + self.context_projection(coefficients)) - torch.tanh(hidden)
            residual = (modulated @ self.context_output.weight.transpose(0, 1)).squeeze(-1)
            residual = residual * observation.logit_scale.reshape(-1) / math.sqrt(self.dimension)
            return observation.base_logits + residual
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
