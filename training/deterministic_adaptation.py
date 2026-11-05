# -*- coding: utf-8 -*-
"""确定性神经记忆更新；不构造策略、动作或 PPO 状态。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from training.pilot_transfer import ChannelStateSignalObjective, channel_state_reward_loss, context_logits


@dataclass(frozen=True)
class DeterministicSettings:
    """开发前固定的优化与保护参数；不由 Data 指标调整。"""

    learning_rate: float = .02
    steps: int = 3
    step_radius: float = .15
    total_radius: float = .8
    proximal_weight: float = .1
    min_validation_improvement: float = 1e-5
    reset_to_origin_on_validation: bool = False
    origin_retention_decay: float = 1.0
    adapt_logit_temperature: float = 1.0
    surrogate_gate_gradient: bool = False
    gradient_transport_regularization: float = 0.0
    inner_optimizer: str = "adam"
    gn_damping: float = 1e-2
    gn_max_backtracks: int = 8

    def __post_init__(self):
        values = (self.learning_rate, self.step_radius, self.total_radius)
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("学习率和参数半径必须为正有限数。")
        if type(self.steps) is not int or self.steps < 1:
            raise ValueError("更新步数必须为正整数。")
        if any(not math.isfinite(value) or value < 0 for value in
               (self.proximal_weight, self.min_validation_improvement)):
            raise ValueError("近端权重与验证阈值必须非负且有限。")
        if type(self.reset_to_origin_on_validation) is not bool:
            raise ValueError("是否允许验证驱动的初始系数回退必须是布尔值。")
        if not math.isfinite(self.origin_retention_decay) or not 0 < self.origin_retention_decay <= 1:
            raise ValueError("离线锚点收缩系数必须在(0,1]内。")
        if not math.isfinite(self.adapt_logit_temperature) or self.adapt_logit_temperature <= 0:
            raise ValueError("Adapt Pilot logit温度必须为正有限数。")
        if type(self.surrogate_gate_gradient) is not bool:
            raise ValueError("直通门控梯度开关必须为布尔值。")
        if not math.isfinite(self.gradient_transport_regularization) or self.gradient_transport_regularization < 0:
            raise ValueError("梯度传输映射正则必须是非负有限数。")
        if self.inner_optimizer not in {"adam", "damped_gauss_newton"}:
            raise ValueError("子空间内循环优化器只允许Adam或阻尼Gauss–Newton。")
        if not math.isfinite(self.gn_damping) or self.gn_damping <= 0:
            raise ValueError("Gauss–Newton阻尼必须为正有限数。")
        if type(self.gn_max_backtracks) is not int or not 0 <= self.gn_max_backtracks <= 32:
            raise ValueError("Gauss–Newton回溯次数必须在0到32之间。")


def validation_metrics(model, context, logits):
    """验证符号只在候选完成后读取，不参与 Adapt 反传。"""
    mask = context.frame.reward_mask
    if not bool(mask.any()):
        raise ValueError("确定性微调需要独立验证 Pilot。")
    labels = context.frame.receiver_view().reward_symbols.real > 0
    return float(channel_state_reward_loss(model, context)), float(
        ((logits[mask] >= 0) != labels[mask]).float().mean())


class DeterministicMemoryAdapter:
    """单帧 Adapt 信号目标，双重投影，当前验证守门，接受后持久保留。"""

    def __init__(self, model, settings: DeterministicSettings):
        self.model, self.settings = model, settings
        if model.channel_state is None or model.channel_state.memory is None:
            raise ValueError("确定性记忆更新需要已启用神经记忆支路。")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.parameter = model.channel_state.memory.weight
        self.parameter.requires_grad_(True)
        self.origin = self.parameter.detach().clone()
        self.parameter_name = "channel_state.memory.weight"

    def step(self, context):
        """返回守门前后完整输出；输入上下文必须已去除未知标签与真值。"""
        settings, parameter = self.settings, self.parameter
        initial = parameter.detach().clone()
        with torch.no_grad():
            before = context_logits(self.model, context, visible="full_rx").detach()
            loss_before, ber_before = validation_metrics(self.model, context, before)
        # 每帧重置优化器，避免拒绝候选后遗留动量；这是明确的开发协议。
        optimizer = torch.optim.AdamW([parameter], lr=settings.learning_rate, weight_decay=0.)
        objective = ChannelStateSignalObjective(self.model, role="adapt")
        losses, gradient_norm = [], 0.
        for _ in range(settings.steps):
            optimizer.zero_grad(set_to_none=True)
            loss = objective(context) + settings.proximal_weight * (parameter - initial).square().sum()
            if not bool(torch.isfinite(loss)):
                with torch.no_grad():
                    parameter.copy_(initial)
                raise FloatingPointError("Adapt 目标非有限，停止该开发运行。")
            loss.backward()
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                with torch.no_grad():
                    parameter.copy_(initial)
                raise FloatingPointError("神经记忆梯度缺失或非有限。")
            gradient_norm = float(torch.nn.utils.clip_grad_norm_([parameter], 1.))
            optimizer.step()
            losses.append(float(loss.detach()))
            with torch.no_grad():
                delta = parameter - initial
                parameter.copy_(initial + delta * (settings.step_radius / delta.norm().clamp_min(1e-12)).clamp(max=1.))
                delta = parameter - self.origin
                parameter.copy_(self.origin + delta * (settings.total_radius / delta.norm().clamp_min(1e-12)).clamp(max=1.))
        with torch.no_grad():
            candidate = context_logits(self.model, context, visible="full_rx").detach()
            loss_after, ber_after = validation_metrics(self.model, context, candidate)
            delta_norm = float((parameter - initial).norm())
            accepted = bool(delta_norm > 0 and math.isfinite(loss_after)
                and loss_after < loss_before - settings.min_validation_improvement
                and ber_after <= ber_before)
            candidate_memory = parameter.detach().clone()
            if not accepted:
                parameter.copy_(initial)
            retained = candidate if accepted else before
            metrics = dict(accepted=accepted, adapt_losses=losses, gradient_norm=gradient_norm,
                delta_norm=delta_norm, retained_total_norm=float((parameter - self.origin).norm()),
                validation_loss_before=loss_before, validation_loss_candidate=loss_after,
                validation_ber_before=ber_before, validation_ber_candidate=ber_after,
                parameter_name=self.parameter_name, parameter_count=parameter.numel(),
                optimizer_reset_each_frame=True, uses_rl=False, data_labels_used_for_control=False)
        optimizer.zero_grad(set_to_none=True)
        return metrics, before, candidate, retained, candidate_memory
