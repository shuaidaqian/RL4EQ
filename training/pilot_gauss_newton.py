# -*- coding: utf-8 -*-
"""合法Adapt Pilot驱动的低维阻尼Gauss–Newton候选更新。"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from training.subspace_adaptation import project


def damped_gauss_newton_candidate(model, observation, current, origin, *, damping=1e-2,
                                  proximal_weight=0.1, temperature=1.0,
                                  step_radius=0.15, total_radius=0.8,
                                  max_backtracks=8, create_graph=False):
    """仅用当前Adapt Pilot构造Fisher曲率，返回有界候选和数值诊断。

    验证Pilot不参与此函数；调用方须在候选形成后独立执行验证门控。
    """
    values = (damping, proximal_weight, temperature, step_radius, total_radius)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Gauss–Newton参数必须是有限数。")
    if damping <= 0 or proximal_weight < 0 or temperature <= 0 or step_radius <= 0 or total_radius <= 0:
        raise ValueError("阻尼、温度和信赖域半径必须为正，近端系数不得为负。")
    if max_backtracks < 0 or max_backtracks > 32:
        raise ValueError("回溯次数必须在0到32之间。")

    mask = observation.adapt_mask.bool()
    if not bool(mask.any()):
        raise ValueError("缺少合法Adapt Pilot。")
    labels = observation.adapt_labels[mask].to(dtype=current.dtype)
    start = current.clone() if create_graph else current.detach().clone()
    anchor = origin.to(device=start.device, dtype=start.dtype)
    if not create_graph:
        anchor = anchor.detach()

    def adapt_objective(coefficients):
        scaled = model.logits(observation, coefficients)[mask] / temperature
        pilot_bce = F.binary_cross_entropy_with_logits(scaled, labels)
        proximal = proximal_weight * (coefficients - anchor).square().sum()
        return pilot_bce + proximal

    with torch.enable_grad():
        variable = start if create_graph else start.detach().requires_grad_(True)
        scaled_logits = model.logits(observation, variable)[mask] / temperature
        probabilities = torch.sigmoid(scaled_logits if create_graph else scaled_logits.detach())
        jacobian = torch.autograd.functional.jacobian(
            lambda coefficients: model.logits(observation, coefficients)[mask] / temperature,
            variable, create_graph=create_graph, vectorize=True)
        count = max(int(mask.sum()), 1)
        gradient = jacobian.transpose(0, 1) @ (probabilities - labels) / count
        gradient = gradient + 2 * proximal_weight * (variable - anchor)
        curvature_weights = (probabilities * (1 - probabilities)).clamp_min(1e-8)
        fisher = jacobian.transpose(0, 1) @ (curvature_weights.unsqueeze(1) * jacobian) / count
        identity = torch.eye(start.numel(), device=start.device, dtype=start.dtype)
        system = fisher + (damping + 2 * proximal_weight) * identity
        candidate_step = torch.linalg.solve(system, -gradient)

    if not (torch.isfinite(jacobian).all() and torch.isfinite(system).all()
            and torch.isfinite(candidate_step).all()):
        raise FloatingPointError("Pilot Jacobian或阻尼Gauss–Newton解出现非有限值。")

    candidate_step = candidate_step * (step_radius / candidate_step.norm().clamp_min(1e-12)).clamp(max=1.)
    candidate = project(project(start + candidate_step, start, step_radius), anchor, total_radius)
    before = float(adapt_objective(start).detach())
    accepted = False
    used_backtracks = 0
    for backtrack in range(max_backtracks + 1):
        trial = project(project(start + candidate_step, start, step_radius), anchor, total_radius)
        after = float(adapt_objective(trial).detach())
        if math.isfinite(after) and after <= before:
            candidate = trial if create_graph else trial.detach()
            accepted, used_backtracks = True, backtrack
            break
        candidate_step = candidate_step * 0.5
        used_backtracks = backtrack + 1
    if not accepted:
        candidate = start

    singular_values = torch.linalg.svdvals(jacobian.detach())
    tolerance = (singular_values.max() * max(jacobian.shape) * torch.finfo(singular_values.dtype).eps
                 if singular_values.numel() else torch.tensor(0., device=start.device))
    rank = int((singular_values > tolerance).sum()) if singular_values.numel() else 0
    raw_condition = (float(singular_values.max() / singular_values.min())
        if singular_values.numel() and float(singular_values.min()) > 0 else math.inf)
    return candidate, dict(
        accepted_by_adapt_backtracking=accepted,
        adapt_objective_before=before,
        adapt_objective_candidate=float(adapt_objective(candidate).detach()),
        jacobian_rank=rank,
        jacobian_rows=int(jacobian.shape[0]),
        parameter_dimension=int(jacobian.shape[1]),
        jacobian_condition=raw_condition if math.isfinite(raw_condition) else None,
        fisher_condition=float(torch.linalg.cond(system.detach())),
        damping=float(damping),
        backtracks=used_backtracks,
        proposed_step_norm=float(candidate_step.detach().norm()),
        applied_step_norm=float((candidate.detach() - start.detach()).norm()),
        total_origin_norm=float((candidate.detach() - anchor.detach()).norm()),
        adapt_only=True,
        validation_used_for_candidate=False,
        data_used_for_candidate=False,
    )


def update_subspace_gauss_newton(model, observation, initial, origin, settings, *,
                                 adapt_history=None, create_graph=False):
    """生成GN候选并套用独立当前帧验证Pilot门控，供训练与在线共用。"""
    history = adapt_history or [observation]
    if len(history) != 1 or history[0] is not observation:
        raise ValueError("Gauss–Newton首版只允许单帧Adapt Pilot历史。")
    candidate, geometry = damped_gauss_newton_candidate(model, observation, initial, origin,
        damping=getattr(settings, "gn_damping", 1e-2),
        proximal_weight=settings.proximal_weight,
        temperature=settings.adapt_logit_temperature,
        step_radius=settings.step_radius,
        total_radius=settings.total_radius,
        max_backtracks=getattr(settings, "gn_max_backtracks", 8),
        create_graph=create_graph)
    before_logits = model.logits(observation, initial)
    candidate_logits = model.logits(observation, candidate)
    origin_logits = model.logits(observation, origin)
    mask = observation.validation_mask
    labels = observation.validation_labels.bool()
    before_loss = float(F.binary_cross_entropy_with_logits(
        before_logits[mask], labels[mask].to(before_logits.dtype)).detach())
    candidate_loss = float(F.binary_cross_entropy_with_logits(
        candidate_logits[mask], labels[mask].to(candidate_logits.dtype)).detach())
    origin_loss = float(F.binary_cross_entropy_with_logits(
        origin_logits[mask], labels[mask].to(origin_logits.dtype)).detach())
    before_ber = float(((before_logits.detach()[mask] >= 0) != labels[mask]).float().mean())
    candidate_ber = float(((candidate_logits.detach()[mask] >= 0) != labels[mask]).float().mean())
    origin_ber = float(((origin_logits.detach()[mask] >= 0) != labels[mask]).float().mean())
    candidate_accepted = bool(geometry["accepted_by_adapt_backtracking"]
        and math.isfinite(candidate_loss)
        and candidate_loss < before_loss - settings.min_validation_improvement
        and candidate_ber <= before_ber)
    retained = candidate if candidate_accepted else initial
    if candidate_accepted and settings.origin_retention_decay < 1:
        retained = origin + settings.origin_retention_decay * (retained - origin)
    retained_logits = model.logits(observation, retained)
    retained_loss = float(F.binary_cross_entropy_with_logits(
        retained_logits[mask], labels[mask].to(retained_logits.dtype)).detach())
    retained_ber = float(((retained_logits.detach()[mask] >= 0) != labels[mask]).float().mean())
    accepted = bool(candidate_accepted and retained_loss < before_loss - settings.min_validation_improvement
        and retained_ber <= before_ber)
    if not accepted:
        retained = (candidate + (initial - candidate).detach()
                    if create_graph and settings.surrogate_gate_gradient else initial)
        retained_loss, retained_ber = before_loss, before_ber
    guarded_loss = retained_loss if accepted else before_loss
    guarded_ber = retained_ber if accepted else before_ber
    guarded = retained if accepted else initial
    reset_to_origin = bool(settings.reset_to_origin_on_validation
        and origin_loss < guarded_loss - settings.min_validation_improvement
        and origin_ber <= guarded_ber and float((guarded.detach() - origin.detach()).norm()) > 0)
    if reset_to_origin:
        retained = (retained + (origin - retained).detach()
                    if create_graph and settings.surrogate_gate_gradient else origin)
        accepted = False
        retained_loss, retained_ber = origin_loss, origin_ber
    metrics = dict(accepted=accepted, candidate_accepted=candidate_accepted,
        reset_to_origin=reset_to_origin, adapt_losses=[geometry["adapt_objective_before"]],
        gradient_norm=geometry["proposed_step_norm"], validation_loss_before=before_loss,
        validation_loss_candidate=candidate_loss, validation_loss_origin=origin_loss,
        validation_loss_retained=retained_loss, validation_ber_before=before_ber,
        validation_ber_candidate=candidate_ber, validation_ber_origin=origin_ber,
        validation_ber_retained=retained_ber, adapt_history_frames=1,
        delta_norm=float((candidate - initial).detach().norm()),
        retained_total_norm=float((retained.detach() - origin.detach()).norm()),
        parameter_count=int(initial.numel()), data_labels_used_for_control=False,
        optimizer_reset_each_frame=True, uses_rl=False, **geometry)
    if not create_graph:
        candidate, retained = candidate.detach(), retained.detach()
    return candidate, retained, metrics
