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


def pilot_loss(logits, observation, role, *, temperature=1.0):
    if role == "adapt":
        mask, labels = observation.adapt_mask, observation.adapt_labels
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Adapt Pilot logit温度必须为正有限数。")
        logits = logits / temperature
    elif role == "validation":
        mask, labels = observation.validation_mask, observation.validation_labels
    else:
        raise ValueError("更新角色只允许Adapt或验证。")
    if not bool(mask.any()):
        raise ValueError("缺少合法Pilot。")
    return F.binary_cross_entropy_with_logits(logits[mask], labels[mask])


def coefficient_update(base, basis, observation, initial, origin, settings: DeterministicSettings, *,
                       adapt_history=None, create_graph=False, gradient_transport=None,
                       context_update=None):
    """每帧重置Adam；内循环可联合最近多帧Adapt，验证只用当前帧。"""
    if not create_graph:
        base, basis, origin = base.detach(), basis.detach(), origin.detach()
        initial = initial.detach().clone().requires_grad_(True)
        gradient_transport = gradient_transport.detach() if gradient_transport is not None else None
    history = adapt_history or [(base, basis, observation)]
    if not history or history[-1][2] is not observation:
        raise ValueError("Adapt历史必须非空且以当前帧Pilot观测结束。")
    if not create_graph:
        history = [(item_base.detach(), item_basis.detach(), item_observation)
                   for item_base, item_basis, item_observation in history]
    current = initial
    first, second = torch.zeros_like(current), torch.zeros_like(current)
    losses = []
    for step in range(1, settings.steps + 1):
        frame_losses = [pilot_loss(item_base + item_basis @ current, item_observation, "adapt",
                          temperature=settings.adapt_logit_temperature)
                        for item_base, item_basis, item_observation in history]
        loss = torch.stack(frame_losses).mean()
        loss = loss + settings.proximal_weight * (current - initial).square().sum()
        if context_update is None:
            gradient, = torch.autograd.grad(loss, current, create_graph=create_graph)
            history_gradients = None
        else:
            history_gradients = [torch.autograd.grad(frame_loss, current,
                create_graph=create_graph)[0] for frame_loss in frame_losses]
            gradient = torch.stack(history_gradients).mean(dim=0)
            gradient = gradient + 2 * settings.proximal_weight * (current - initial)
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("子空间目标或梯度非有限，停止开发运行。")
        gradient_norm = gradient.norm()
        # 与torch的clip_grad_norm_(max_norm=1)采用同一分母和稳定项。
        gradient = gradient * (1. / (gradient_norm + 1e-6)).clamp(max=1.)
        first = .9 * first + .1 * gradient
        second = .999 * second + .001 * gradient.square()
        adam_delta = -settings.learning_rate * (first / (1 - .9**step)) / (
            (second / (1 - .999**step)).clamp_min(1e-24).sqrt() + 1e-8)
        if gradient_transport is not None:
            if gradient_transport.shape != (current.numel(), current.numel()):
                raise ValueError("梯度传输矩阵维数与在线系数不一致。")
            adam_delta = gradient_transport @ adam_delta
        if context_update is not None:
            adam_delta = adam_delta + context_update(history_gradients, settings.learning_rate)
        if not bool(torch.isfinite(adam_delta).all()):
            raise FloatingPointError("Pilot在线系数增量非有限，停止开发运行。")
        current = current + adam_delta
        current = project(project(current, initial, settings.step_radius), origin, settings.total_radius)
        losses.append(float(loss.detach()))
        if not create_graph:
            current = current.detach().requires_grad_(True)
    before_logits, candidate_logits, origin_logits = (
        base + basis @ initial, base + basis @ current, base + basis @ origin)
    mask, labels = observation.validation_mask, observation.validation_labels.bool()
    before_loss = float(pilot_loss(before_logits, observation, "validation").detach())
    candidate_loss = float(pilot_loss(candidate_logits, observation, "validation").detach())
    origin_loss = float(pilot_loss(origin_logits, observation, "validation").detach())
    before_ber = float(((before_logits.detach()[mask] >= 0) != labels[mask]).float().mean())
    candidate_ber = float(((candidate_logits.detach()[mask] >= 0) != labels[mask]).float().mean())
    origin_ber = float(((origin_logits.detach()[mask] >= 0) != labels[mask]).float().mean())
    accepted = bool((current.detach() - initial.detach()).norm() > 0
        and math.isfinite(candidate_loss) and candidate_loss < before_loss - settings.min_validation_improvement
        and candidate_ber <= before_ber)
    retained = current if accepted else initial
    if accepted and settings.origin_retention_decay < 1:
        retained = origin + settings.origin_retention_decay * (retained - origin)
    retained_logits = base + basis @ retained
    retained_loss = float(pilot_loss(retained_logits, observation, "validation").detach())
    retained_ber = float(((retained_logits.detach()[mask] >= 0) != labels[mask]).float().mean())
    accepted = bool(accepted and math.isfinite(retained_loss)
        and retained_loss < before_loss - settings.min_validation_improvement
        and retained_ber <= before_ber)
    if not accepted:
        # 外层训练保持线上拒绝时的精确前向值，但可选地将候选梯度直通到外层。
        retained = (current + (initial - current).detach()
                    if create_graph and settings.surrogate_gate_gradient else initial)
        retained_logits = before_logits
        retained_loss, retained_ber = before_loss, before_ber
    candidate_accepted = accepted
    reset_to_origin = False
    if settings.reset_to_origin_on_validation:
        retained_loss = retained_loss if accepted else before_loss
        retained_ber = retained_ber if accepted else before_ber
        reset_to_origin = bool(
            math.isfinite(origin_loss)
            and origin_loss < retained_loss - settings.min_validation_improvement
            and origin_ber <= retained_ber
            and (retained.detach() - origin.detach()).norm() > 0
        )
        if reset_to_origin:
            retained = (retained + (origin - retained).detach()
                        if create_graph and settings.surrogate_gate_gradient else origin)
            accepted = False
    metrics = dict(accepted=accepted, adapt_losses=losses, gradient_norm=float(gradient_norm.detach()),
        candidate_accepted=candidate_accepted, reset_to_origin=reset_to_origin,
        validation_loss_before=before_loss, validation_loss_candidate=candidate_loss,
        validation_ber_before=before_ber, validation_ber_candidate=candidate_ber,
        validation_loss_origin=origin_loss, validation_ber_origin=origin_ber,
        validation_loss_retained=retained_loss, validation_ber_retained=retained_ber,
        adapt_history_frames=len(history),
        delta_norm=float((current - initial).detach().norm()),
        retained_total_norm=float((retained - origin).detach().norm()),
        parameter_count=current.numel(), data_labels_used_for_control=False,
        gradient_transport_enabled=gradient_transport is not None,
        gradient_transport_condition=(float(torch.linalg.cond(gradient_transport.detach()))
                                      if gradient_transport is not None else 1.0),
        context_update_enabled=context_update is not None,
        optimizer_reset_each_frame=True, uses_rl=False)
    if not create_graph:
        current, retained = current.detach(), retained.detach()
    return current, retained, metrics


def update_subspace(model, observation, initial, origin, settings, *, adapt_history=None, create_graph=False):
    basis = model.basis(observation)
    history = None
    if adapt_history is not None:
        history = [(item.base_logits, model.basis(item), item) for item in adapt_history]
    return coefficient_update(observation.base_logits, basis, observation, initial, origin,
                              settings, adapt_history=history, create_graph=create_graph,
                              gradient_transport=model.gradient_transport_matrix(),
                              context_update=(model.context_update_delta
                                  if model.learned_context_update and model.use_context_update else None))
