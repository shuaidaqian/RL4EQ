# -*- coding: utf-8 -*-
"""训练与在线共用的确定性系数更新，包括投影、近端和验证守门。"""

from __future__ import annotations

import math
import torch
from torch.nn import functional as F

from training.deterministic_adaptation import DeterministicSettings


def project(vector, center, radius):
    delta = vector - center
    return center + delta * (radius / delta.norm().clamp_min(1e-12)).clamp(max=1.)


def pilot_loss(logits, observation, role):
    if role == "adapt":
        mask, labels = observation.adapt_mask, observation.adapt_labels
    elif role == "validation":
        mask, labels = observation.validation_mask, observation.validation_labels
    else:
        raise ValueError("更新角色只允许Adapt或验证。")
    if not bool(mask.any()):
        raise ValueError("缺少合法Pilot。")
    return F.binary_cross_entropy_with_logits(logits[mask], labels[mask])


def coefficient_update(base, basis, observation, initial, origin, settings: DeterministicSettings, *, create_graph=False):
    """每帧重置Adam状态；返回候选、保留值及审计指标，不接收Data标签。"""
    if not create_graph:
        base, basis, origin = base.detach(), basis.detach(), origin.detach()
        initial = initial.detach().clone().requires_grad_(True)
    current = initial
    first, second = torch.zeros_like(current), torch.zeros_like(current)
    losses = []
    for step in range(1, settings.steps + 1):
        loss = pilot_loss(base + basis @ current, observation, "adapt")
        loss = loss + settings.proximal_weight * (current - initial).square().sum()
        gradient, = torch.autograd.grad(loss, current, create_graph=create_graph)
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("子空间目标或梯度非有限，停止开发运行。")
        gradient_norm = gradient.norm()
        # 与torch的clip_grad_norm_(max_norm=1)采用同一分母和稳定项。
        gradient = gradient * (1. / (gradient_norm + 1e-6)).clamp(max=1.)
        first = .9 * first + .1 * gradient
        second = .999 * second + .001 * gradient.square()
        current = current - settings.learning_rate * (first / (1 - .9**step)) / (
            (second / (1 - .999**step)).clamp_min(1e-24).sqrt() + 1e-8)
        current = project(project(current, initial, settings.step_radius), origin, settings.total_radius)
        losses.append(float(loss.detach()))
        if not create_graph:
            current = current.detach().requires_grad_(True)
    before_logits, candidate_logits = base + basis @ initial, base + basis @ current
    mask, labels = observation.validation_mask, observation.validation_labels.bool()
    before_loss = float(pilot_loss(before_logits, observation, "validation").detach())
    candidate_loss = float(pilot_loss(candidate_logits, observation, "validation").detach())
    before_ber = float(((before_logits.detach()[mask] >= 0) != labels[mask]).float().mean())
    candidate_ber = float(((candidate_logits.detach()[mask] >= 0) != labels[mask]).float().mean())
    accepted = bool((current.detach() - initial.detach()).norm() > 0
        and math.isfinite(candidate_loss) and candidate_loss < before_loss - settings.min_validation_improvement
        and candidate_ber <= before_ber)
    retained = current if accepted else initial
    metrics = dict(accepted=accepted, adapt_losses=losses, gradient_norm=float(gradient_norm.detach()),
        validation_loss_before=before_loss, validation_loss_candidate=candidate_loss,
        validation_ber_before=before_ber, validation_ber_candidate=candidate_ber,
        delta_norm=float((current - initial).detach().norm()),
        retained_total_norm=float((retained - origin).detach().norm()),
        parameter_count=current.numel(), data_labels_used_for_control=False,
        optimizer_reset_each_frame=True, uses_rl=False)
    if not create_graph:
        current, retained = current.detach(), retained.detach()
    return current, retained, metrics


def update_subspace(model, observation, initial, origin, settings, *, create_graph=False):
    basis = model.basis(observation)
    return coefficient_update(observation.base_logits, basis, observation, initial, origin,
                              settings, create_graph=create_graph)
