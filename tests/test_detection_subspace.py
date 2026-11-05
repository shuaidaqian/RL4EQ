# -*- coding: utf-8 -*-
"""固定信道子空间的数值一致性、参数作用与信息边界。"""

import copy
from dataclasses import fields, replace
import json

import pytest
import torch

from agent.detection_subspace import DetectionSubspace, PilotObservation, capture_observation
from evaluation.adaptability_audit import deployable_frame
from evaluation.subspace_development import (frozen_error_conditioned_loss, outer_data_loss,
    paired_online_gain_loss, read_manifest, sequence_loss)
from training.deterministic_adaptation import DeterministicSettings
from training.pilot_risk_gate import FEATURE_NAMES, PilotRiskRegressor, validation_risk_features
from training.subspace_adaptation import coefficient_update, pilot_loss, project, update_subspace
from scripts.diagnose_subspace_data_oracle import projected_data_oracle


def fixture_observation():
    torch.manual_seed(7)
    mask = torch.arange(40) < 20
    validation = (torch.arange(40) >= 20) & (torch.arange(40) < 28)
    return PilotObservation(torch.randn(40), torch.randn(40, 7), torch.tensor([[2.]]),
        mask, torch.randint(0, 2, (40,)).float(), validation, torch.randint(0, 2, (40,)).float())


@pytest.mark.parametrize("dimension", [2,4,8,10,12,14,16])
def test_zero_coefficients_preserve_source_and_jacobian_matches_finite_difference(dimension):
    observation = fixture_observation()
    model = DetectionSubspace(dimension)
    torch.testing.assert_close(model.logits(observation), observation.base_logits, atol=0, rtol=0)
    basis = model.basis(observation)
    for index in range(dimension):
        delta = torch.zeros(dimension)
        delta[index] = .01
        actual = (model.logits(observation, delta) - observation.base_logits) / .01
        torch.testing.assert_close(actual, basis[:, index], atol=2e-5, rtol=2e-4)
    assert basis[~(observation.adapt_mask | observation.validation_mask)].norm() > 0
    assert {field.name for field in fields(observation)} == {
        "base_logits","features","logit_scale","adapt_mask","adapt_labels","validation_mask","validation_labels"}


def test_read_only_data_oracle_improves_a_reachable_toy_boundary_within_radius():
    basis = torch.eye(2)
    base = torch.tensor([-0.2, 0.2])
    labels = torch.tensor([1., 0.])
    coeff, loss = projected_data_oracle(basis, base, labels, radius=0.5, steps=80)
    assert coeff.norm() <= 0.5 + 1e-6
    assert loss < torch.nn.functional.binary_cross_entropy_with_logits(base, labels).item()
    assert ((base + basis @ coeff > 0) == (labels > 0.5)).all()


@pytest.mark.parametrize("architecture", ["pointwise_mlp_v1", "temporal_conv_v1", "temporal_dilated_127_v1", "temporal_dilated_residual_127_v1", "pilot_context_mlp_v1"])
def test_subspace_architectures_preserve_shape_and_zero_coefficient_identity(architecture):
    observation = fixture_observation()
    model = DetectionSubspace(4, architecture=architecture)
    basis = model.basis(observation)
    assert basis.shape == (40, 4)
    torch.testing.assert_close(model.logits(observation), observation.base_logits, atol=0, rtol=0)
    assert torch.isfinite(basis).all()


@pytest.mark.parametrize("steps", [1,3])
def test_functional_adam_projection_matches_torch_and_unrolled_training(steps):
    observation = fixture_observation()
    model = DetectionSubspace(4)
    basis = model.basis(observation).detach()
    settings = DeterministicSettings(steps=steps, step_radius=.015, total_radius=.02)
    origin = torch.zeros(4)
    initial = torch.full((4,), .002)
    candidate, retained, control = coefficient_update(observation.base_logits, basis, observation,
        initial, origin, settings)
    parameter = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.AdamW([parameter], lr=settings.learning_rate, weight_decay=0.)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = pilot_loss(observation.base_logits+basis@parameter, observation, "adapt")
        loss = loss + settings.proximal_weight*(parameter-initial).square().sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([parameter], 1.)
        optimizer.step()
        with torch.no_grad():
            parameter.copy_(project(project(parameter, initial, settings.step_radius), origin, settings.total_radius))
    torch.testing.assert_close(candidate, parameter, atol=2e-7, rtol=2e-5)
    graph_initial = initial.clone().requires_grad_(True)
    graph_candidate, graph_retained, graph_control = coefficient_update(observation.base_logits, basis,
        observation, graph_initial, origin, settings, create_graph=True)
    torch.testing.assert_close(graph_candidate, candidate, atol=2e-7, rtol=2e-5)
    torch.testing.assert_close(graph_retained, retained, atol=2e-7, rtol=2e-5)
    assert graph_control["accepted"] == control["accepted"]
    assert control["retained_total_norm"] <= settings.total_radius + 1e-6


def test_validation_labels_do_not_enter_candidate_and_rejection_restores_initial():
    observation = fixture_observation()
    model = DetectionSubspace(4)
    initial = torch.zeros(4)
    settings = DeterministicSettings(min_validation_improvement=100.)
    first = update_subspace(model, observation, initial, initial, settings)
    changed = replace(observation, validation_labels=1-observation.validation_labels)
    second = update_subspace(model, changed, initial, initial, settings)
    torch.testing.assert_close(first[0], second[0], atol=0, rtol=0)
    torch.testing.assert_close(first[1], initial, atol=0, rtol=0)
    assert first[2]["accepted"] is False


def test_straight_through_gate_keeps_exact_rejection_forward_and_passes_outer_gradient():
    observation = fixture_observation()
    model = DetectionSubspace(4)
    initial = torch.zeros(4, requires_grad=True)
    origin = torch.zeros(4)
    settings = DeterministicSettings(min_validation_improvement=100., surrogate_gate_gradient=True)
    candidate, retained, metrics = update_subspace(model, observation, initial, origin, settings,
        create_graph=True)
    assert metrics["accepted"] is False
    assert torch.equal(retained.detach(), initial.detach())
    (retained * torch.arange(1., 5.)).sum().backward()
    assert any(parameter.grad is not None and parameter.grad.norm() > 0 for parameter in model.parameters())
    assert initial.grad is not None and torch.isfinite(initial.grad).all()


def test_adapt_history_changes_inner_update_but_validation_stays_on_current_frame():
    observation = fixture_observation()
    previous = replace(observation, adapt_labels=1-observation.adapt_labels)
    model = DetectionSubspace(4, architecture="temporal_conv_v1")
    origin = torch.zeros(4)
    initial = torch.zeros(4)
    settings = DeterministicSettings(proximal_weight=0.)
    current_only = update_subspace(model, observation, initial, origin, settings)
    with_history = update_subspace(model, observation, initial, origin, settings,
        adapt_history=[previous, observation])
    assert not torch.equal(current_only[0], with_history[0])
    assert with_history[2]["adapt_history_frames"] == 2
    poisoned_validation = replace(observation, validation_labels=1-observation.validation_labels)
    changed_validation = update_subspace(model, poisoned_validation, initial, origin, settings,
        adapt_history=[previous, poisoned_validation])
    torch.testing.assert_close(with_history[0], changed_validation[0], atol=0, rtol=0)
    assert changed_validation[2]["adapt_history_frames"] == 2


def test_validation_can_reset_stale_history_to_offline_origin_without_data_labels():
    # Adapt Pilot 推动当前记忆，验证 Pilot 显示离线初始点更可靠；Data不在接口中。
    observation = PilotObservation(
        base_logits=torch.zeros(4), features=torch.zeros(4, 7), logit_scale=torch.ones(1, 1),
        adapt_mask=torch.tensor([True, True, False, False]),
        adapt_labels=torch.tensor([1., 1., 0., 0.]),
        validation_mask=torch.tensor([False, False, True, True]),
        validation_labels=torch.tensor([0., 0., 0., 0.]),
    )
    basis = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [1., 0.]])
    settings = DeterministicSettings(steps=1, learning_rate=.02, step_radius=.15,
        total_radius=.8, proximal_weight=0., reset_to_origin_on_validation=True)
    initial, origin = torch.tensor([.5, 0.]), torch.zeros(2)
    candidate, retained, metrics = coefficient_update(
        observation.base_logits, basis, observation, initial, origin, settings)
    assert metrics["candidate_accepted"] is False
    assert metrics["reset_to_origin"] is True
    assert metrics["accepted"] is False
    torch.testing.assert_close(retained, origin, atol=0, rtol=0)
    assert metrics["validation_loss_origin"] < metrics["validation_loss_before"]
    graph_candidate, graph_retained, graph_metrics = coefficient_update(
        observation.base_logits, basis, observation,
        initial.clone().requires_grad_(True), origin.clone().requires_grad_(True), settings,
        create_graph=True)
    torch.testing.assert_close(graph_candidate.detach(), candidate, atol=2e-7, rtol=2e-5)
    torch.testing.assert_close(graph_retained.detach(), retained, atol=2e-7, rtol=2e-5)
    assert graph_metrics["candidate_accepted"] is metrics["candidate_accepted"]
    assert graph_metrics["reset_to_origin"] is metrics["reset_to_origin"]


def test_origin_retention_decay_is_applied_before_final_validation_gate():
    observation = fixture_observation()
    model = DetectionSubspace(4)
    settings = DeterministicSettings(origin_retention_decay=.5)
    origin = torch.zeros(4)
    candidate, retained, metrics = coefficient_update(
        observation.base_logits, model.basis(observation).detach(), observation,
        torch.zeros(4), origin, settings)
    if metrics["accepted"]:
        torch.testing.assert_close(retained, origin + .5 * (candidate - origin), atol=0, rtol=0)
        assert metrics["validation_loss_retained"] <= metrics["validation_loss_before"]


def test_adapt_logit_temperature_changes_only_inner_objective():
    observation = fixture_observation()
    logits = torch.randn(40)
    base = pilot_loss(logits, observation, "adapt")
    same = pilot_loss(logits, observation, "adapt", temperature=1.)
    tempered = pilot_loss(logits, observation, "adapt", temperature=.5)
    validation = pilot_loss(logits, observation, "validation")
    torch.testing.assert_close(base, same, atol=0, rtol=0)
    assert not torch.isclose(base, tempered)
    torch.testing.assert_close(validation, pilot_loss(logits, observation, "validation", temperature=.5), atol=0, rtol=0)


def test_capture_does_not_read_data_truth_and_removes_temporary_hook():
    from tests.test_pilot_transfer import _symbol_fixture
    _, model, context = _symbol_fixture()
    original = capture_observation(model, replace(context, frame=deployable_frame(context.frame)))
    symbols, bits = context.frame.tx_symbols.clone(), context.frame.bits.clone()
    symbols[context.frame.data_mask] = complex(float("nan"), float("nan"))
    bits[context.frame.data_mask] = ~bits[context.frame.data_mask]
    poisoned = replace(context.frame, tx_symbols=symbols, bits=bits,
        true_cir=torch.full((5,), float("nan")), tail_symbols=torch.full((4,), float("nan")))
    alternate = capture_observation(model, replace(context, frame=deployable_frame(poisoned)))
    for field in fields(original):
        torch.testing.assert_close(getattr(original,field.name), getattr(alternate,field.name), atol=0, rtol=0)
    assert not model.symbol_residual._forward_pre_hooks


def test_outer_future_labels_change_only_outer_loss_not_current_candidate():
    observation = fixture_observation()
    model = DetectionSubspace(4)
    data_mask = ~(observation.adapt_mask | observation.validation_mask)
    row = dict(observation=observation, offline=dict(data_mask=data_mask,bits=torch.zeros(40,dtype=torch.bool)))
    session = dict(frames=[row,copy.deepcopy(row)])
    settings = DeterministicSettings()
    first_candidate = update_subspace(model,observation,model.coefficients,model.coefficients,settings)[0]
    first_loss = sequence_loss(model,session,settings,1,adapt=True,create_graph=True)
    session["frames"][1]["offline"]["bits"] = torch.ones(40,dtype=torch.bool)
    second_loss = sequence_loss(model,session,settings,1,adapt=True,create_graph=True)
    second_candidate = update_subspace(model,observation,model.coefficients,model.coefficients,settings)[0]
    assert not torch.isclose(first_loss,second_loss)
    torch.testing.assert_close(first_candidate,second_candidate,atol=0,rtol=0)
    second_loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_development_manifest_keeps_original_channel_and_disjoint_roles():
    spec = read_manifest("configs/audits/detection_subspace_long_v1.json")
    assert spec["channel_changes"] is False
    assert spec["config"] == "configs/eme_long_memory_v2_shared_pilot_memory_rl.json"


def test_temperature2_manifest_changes_only_the_adapt_loss_scale():
    baseline = read_manifest("configs/audits/detection_subspace_long_v6_adapt_history8_v1.json")
    candidate = read_manifest("configs/audits/detection_subspace_long_v7_adapt_temperature2_v1.json")
    assert candidate["update"]["adapt_logit_temperature"] == 2.0
    assert baseline["update"].get("adapt_logit_temperature", 1.0) == 1.0
    assert candidate["config_sha256"] == baseline["config_sha256"]
    assert candidate["checkpoint_sha256"] == baseline["checkpoint_sha256"]
    assert candidate["data_roles"] == baseline["data_roles"]
    assert candidate["channel_changes"] is False


def test_v8_manifest_enables_surrogate_gradient_only_for_offline_outer_training():
    baseline = read_manifest("configs/audits/detection_subspace_long_v7_adapt_temperature2_v1.json")
    candidate = read_manifest("configs/audits/detection_subspace_long_v8_surrogate_gate_v1.json")
    assert candidate["update"]["surrogate_gate_gradient"] is True
    assert baseline["update"].get("surrogate_gate_gradient", False) is False
    assert candidate["config_sha256"] == baseline["config_sha256"]
    assert candidate["data_roles"] == baseline["data_roles"]
    assert candidate["heldout_confirmation"] is False


def test_v9_transport_is_spd_bounded_and_identity_initialized_update_is_unchanged():
    observation = fixture_observation()
    settings = DeterministicSettings(proximal_weight=0., surrogate_gate_gradient=True)
    origin = torch.zeros(4)
    initial = torch.tensor([.01, -.01, .005, 0.])
    baseline = DetectionSubspace(4)
    transported = DetectionSubspace(4, learned_gradient_transport=True)
    transported.load_state_dict(baseline.state_dict(), strict=False)
    torch.testing.assert_close(transported.gradient_transport_matrix(), torch.eye(4), atol=0, rtol=0)
    no_transport = update_subspace(baseline, observation, initial, origin, settings)
    identity_transport = update_subspace(transported, observation, initial, origin, settings)
    torch.testing.assert_close(identity_transport[0], no_transport[0], atol=0, rtol=0)
    torch.testing.assert_close(identity_transport[1], no_transport[1], atol=0, rtol=0)

    with torch.no_grad():
        transported.gradient_transport_raw[1, 0] = 1.2
        transported.gradient_transport_raw[2, 0] = -1.0
        transported.gradient_transport_raw.diagonal().copy_(torch.tensor([.7, -.4, .2, -.8]))
    matrix = transported.gradient_transport_matrix()
    eigenvalues = torch.linalg.eigvalsh(matrix)
    assert torch.isfinite(matrix).all()
    assert torch.all(eigenvalues > 0)
    assert float(matrix.detach().abs().max()) < 2.
    assert float(torch.linalg.cond(matrix).detach()) < 20.
    altered = update_subspace(transported, observation, initial, origin, settings)
    assert not torch.equal(altered[0], no_transport[0])


def test_v9_outer_adaptation_loss_has_finite_nonzero_transport_gradient():
    observation = fixture_observation()
    model = DetectionSubspace(4, learned_gradient_transport=True)
    data_mask = ~(observation.adapt_mask | observation.validation_mask)
    row = dict(observation=observation,
        offline=dict(data_mask=data_mask, bits=torch.randint(0, 2, (40,), dtype=torch.bool)))
    session = dict(frames=[row, copy.deepcopy(row)])
    settings = replace(DeterministicSettings(surrogate_gate_gradient=True),
                       gradient_transport_regularization=0.)
    loss = sequence_loss(model, session, settings, 1, adapt=True, create_graph=True)
    gradient, = torch.autograd.grad(loss, model.gradient_transport_raw, allow_unused=False)
    assert torch.isfinite(gradient).all()
    assert float(gradient.norm()) > 0.


def test_v9_manifest_keeps_fixed_channel_and_enables_only_learned_transport():
    spec = read_manifest("configs/audits/detection_subspace_long_v9_gradient_transport_v1.json")
    assert spec["learned_gradient_transport"] is True
    assert spec["channel_changes"] is False
    assert spec["heldout_confirmation"] is False
    assert spec["config"] == "configs/eme_long_memory_v2_shared_pilot_memory_rl.json"
    assert spec["snrs_db"] == [0, 5, 10, 15]


def test_context_update_zero_initialization_matches_adam_and_uses_only_adapt_history():
    observation = fixture_observation()
    previous = replace(observation, adapt_labels=1-observation.adapt_labels)
    baseline = DetectionSubspace(4)
    contextual = DetectionSubspace(4, learned_context_update=True)
    contextual.load_state_dict(baseline.state_dict(), strict=False)
    contextual.use_context_update = True
    settings = DeterministicSettings(proximal_weight=0., surrogate_gate_gradient=True)
    origin = torch.zeros(4)
    history = [previous, observation]
    ordinary = update_subspace(baseline, observation, origin, origin, settings,
                               adapt_history=history)
    zero_initialized = update_subspace(contextual, observation, origin, origin, settings,
                                       adapt_history=history)
    torch.testing.assert_close(zero_initialized[0], ordinary[0], atol=0, rtol=0)
    assert zero_initialized[2]["context_update_enabled"] is True

    changed_validation = replace(observation,
        validation_labels=1-observation.validation_labels)
    alternate = update_subspace(contextual, changed_validation, origin, origin, settings,
        adapt_history=[previous, changed_validation])
    torch.testing.assert_close(alternate[0], zero_initialized[0], atol=0, rtol=0)
    with torch.no_grad():
        contextual.context_update_network[-1].weight[0, 0] = 1.
    changed_adapt = update_subspace(contextual, observation, origin, origin, settings,
        adapt_history=history)
    assert not torch.equal(changed_adapt[0], zero_initialized[0])


def test_context_update_outer_data_loss_trains_nonlinear_rule():
    observation = fixture_observation()
    model = DetectionSubspace(4, learned_context_update=True)
    data_mask = ~(observation.adapt_mask | observation.validation_mask)
    row = dict(observation=observation,
        offline=dict(data_mask=data_mask, bits=torch.randint(0, 2, (40,), dtype=torch.bool)))
    session = dict(frames=[row, copy.deepcopy(row)], inner_history_frames=1)
    settings = replace(DeterministicSettings(surrogate_gate_gradient=True),
                       gradient_transport_regularization=0.)
    loss = sequence_loss(model, session, settings, 1, adapt=True, create_graph=True)
    gradient, = torch.autograd.grad(loss, model.context_update_network[-1].weight,
                                   allow_unused=False)
    assert torch.isfinite(gradient).all()
    assert float(gradient.norm()) > 0.


def test_v10_manifest_keeps_protocol_and_enables_context_update_only():
    spec = read_manifest("configs/audits/detection_subspace_long_v10_context_update_v1.json")
    assert spec["learned_context_update"] is True
    assert spec["learned_gradient_transport"] is False
    assert spec["channel_changes"] is False
    assert spec["heldout_confirmation"] is False
    assert spec["config"] == "configs/eme_long_memory_v2_shared_pilot_memory_rl.json"
    assert spec["snrs_db"] == [0, 5, 10, 15]


def test_v14_dimension_audit_manifest_adds_only_sixteen_dimension_candidate():
    spec = read_manifest("configs/audits/detection_subspace_long_v14_dimension16_audit_v1.json")
    assert spec["dimensions"] == [2, 4, 8, 16]
    assert spec["channel_changes"] is False
    assert spec["heldout_confirmation"] is False


def test_dimension_sweep_manifest_preserves_fixed_channel_and_candidate_order():
    spec = read_manifest("configs/audits/detection_subspace_dimension_sweep_v1.json")
    assert spec["dimensions"] == [2, 4, 8, 10, 12, 14, 16]
    assert spec["channel_changes"] is False
    assert spec["heldout_confirmation"] is False


def test_v16_manifest_keeps_dimension_screen_tied_to_error_conditioned_training():
    spec = read_manifest("configs/audits/detection_subspace_long_v16_dim12_frozen_error_v1.json")
    assert spec["dimensions"] == [2, 4, 8, 10, 12, 14, 16]
    assert spec["training"]["outer_objective"] == "frozen_error_conditioned"
    assert spec["channel_changes"] is False


def test_v17_manifest_only_adds_learned_gradient_transport_to_v16_update():
    spec = read_manifest("configs/audits/detection_subspace_long_v17_dim12_gradient_transport_v1.json")
    assert spec["learned_gradient_transport"] is True
    assert spec["learned_context_update"] is True
    assert spec["training"]["outer_objective"] == "frozen_error_conditioned"
    assert spec["channel_changes"] is False
    assert spec["heldout_confirmation"] is False


def test_v18_manifest_changes_only_temporal_receptive_field():
    spec = read_manifest("configs/audits/detection_subspace_long_v18_dilated127_v1.json")
    assert spec["architecture"] == "temporal_dilated_127_v1"
    assert spec["learned_gradient_transport"] is False
    assert spec["learned_context_update"] is True
    assert spec["training"]["outer_objective"] == "frozen_error_conditioned"
    assert spec["channel_changes"] is False
    assert spec["heldout_confirmation"] is False


def test_v19_manifest_uses_residual_dilations_and_keeps_fixed_channel():
    spec = read_manifest("configs/audits/detection_subspace_long_v19_dilated127_residual_v1.json")
    assert spec["architecture"] == "temporal_dilated_residual_127_v1"
    assert spec["training"]["outer_objective"] == "frozen_error_conditioned"
    assert spec["channel_changes"] is False
    assert spec["heldout_confirmation"] is False


def test_v20_manifest_is_context_conditioned_and_preserves_label_boundary():
    spec = read_manifest("configs/audits/detection_subspace_long_v20_pilot_context_adapter_v1.json")
    assert spec["architecture"] == "pilot_context_mlp_v1"
    assert spec["learned_context_update"] is True
    assert spec["learned_gradient_transport"] is False
    assert spec["channel_changes"] is False
    assert spec["heldout_confirmation"] is False


def test_pilot_context_adapter_is_zero_at_origin_and_jacobian_matches_finite_difference():
    observation = fixture_observation()
    model = DetectionSubspace(4, architecture="pilot_context_mlp_v1")
    origin = torch.zeros(4)
    torch.testing.assert_close(model.logits(observation, origin), observation.base_logits, atol=0, rtol=0)
    direction = torch.tensor([0.4, -0.2, 0.3, -0.1])
    epsilon = 1e-3
    finite_difference = (model.logits(observation, epsilon * direction)
        - model.logits(observation, -epsilon * direction)) / (2 * epsilon)
    jacobian_product = model.basis(observation) @ direction
    torch.testing.assert_close(finite_difference, jacobian_product, atol=2e-4, rtol=2e-3)
    context = torch.tensor([0.03, -0.02, 0.01, 0.04])
    finite_difference = (model.logits(observation, context + epsilon * direction)
        - model.logits(observation, context - epsilon * direction)) / (2 * epsilon)
    jacobian_product = model.basis(observation, context) @ direction
    torch.testing.assert_close(finite_difference, jacobian_product, atol=2e-4, rtol=2e-3)


def test_smooth_ber_outer_objective_is_differentiable_and_tracks_signed_margin():
    bits = torch.tensor([True, False])
    row = dict(offline=dict(data_mask=torch.tensor([True, True]), bits=bits))
    logits = torch.tensor([0.2, -0.2], requires_grad=True)
    correct = outer_data_loss(logits, row, "smooth_ber", 1.0)
    wrong = outer_data_loss(-logits, row, "smooth_ber", 1.0)
    assert correct < wrong
    gradient, = torch.autograd.grad(correct, logits)
    assert torch.isfinite(gradient).all()
    assert gradient[0] < 0 and gradient[1] > 0
    with pytest.raises(ValueError):
        outer_data_loss(logits, row, "smooth_ber", 0.)


def test_paired_online_gain_objective_keeps_frozen_quality_and_pushes_online_below_it():
    frozen = torch.tensor(.2, requires_grad=True)
    online = torch.tensor(.2, requires_grad=True)
    loss = paired_online_gain_loss(frozen, online)
    grad_frozen, grad_online = torch.autograd.grad(loss, (frozen, online))
    torch.testing.assert_close(grad_frozen, torch.tensor(.5))
    torch.testing.assert_close(grad_online, torch.tensor(1.5))

    spec = read_manifest("configs/audits/detection_subspace_long_v12_paired_online_gain_v1.json")
    assert spec["training"]["outer_objective"] == "online_gain"
    assert spec["training"]["smooth_ber_temperature"] == 1.0


def test_frozen_error_conditioned_objective_rewards_correcting_frozen_errors():
    bits = torch.tensor([1., 1., 0., 1.])
    row = dict(offline=dict(data_mask=torch.ones(4, dtype=torch.bool), bits=bits))
    frozen = torch.tensor([-.5, 2., -2., 1.])
    unchanged = frozen.clone().requires_grad_(True)
    corrected = torch.tensor([.5, 2., -2., 1.], requires_grad=True)
    unchanged_loss = frozen_error_conditioned_loss(frozen, unchanged, row)
    corrected_loss = frozen_error_conditioned_loss(frozen, corrected, row)
    assert corrected_loss < unchanged_loss
    gradient, = torch.autograd.grad(unchanged_loss, unchanged)
    assert gradient[0] < 0
    assert gradient[1] < 0
    assert gradient[2] > 0

    spec = read_manifest("configs/audits/detection_subspace_long_v13_frozen_error_outer_v1.json")
    assert spec["training"]["outer_objective"] == "frozen_error_conditioned"


def test_v11_manifest_keeps_protocol_and_fixes_outer_surrogate_temperature():
    spec = read_manifest("configs/audits/detection_subspace_long_v11_smooth_ber_outer_v1.json")
    assert spec["training"]["outer_objective"] == "smooth_ber"
    assert spec["training"]["smooth_ber_temperature"] == 1.0
    assert spec["learned_context_update"] is True
    assert spec["channel_changes"] is False
    assert spec["heldout_confirmation"] is False
    assert spec["config"] == "configs/eme_long_memory_v2_shared_pilot_memory_rl.json"
    assert spec["snrs_db"] == [0, 5, 10, 15]


def test_pilot_risk_features_and_regressor_use_fixed_pilot_only_schema():
    observation = PilotObservation(
        base_logits=torch.zeros(4), features=torch.zeros(4, 7), logit_scale=torch.ones(1, 1),
        adapt_mask=torch.tensor([True, True, False, False]),
        adapt_labels=torch.tensor([1., 1., 0., 0.]),
        validation_mask=torch.tensor([False, False, True, True]),
        validation_labels=torch.tensor([0., 0., 1., 0.]))
    before, after = torch.tensor([0., 0., -.2, .1]), torch.tensor([0., 0., .2, .2])
    features = validation_risk_features(before, after, observation, .02, .1)
    assert features.shape == (len(FEATURE_NAMES),)
    assert torch.isfinite(features).all()
    x = torch.stack([features, features + .1, features - .1, features + .2])
    y = torch.tensor([.1, .2, 0., .3])
    model = PilotRiskRegressor.fit(x, y, ridge=.1)
    restored = PilotRiskRegressor.from_dict(model.to_dict())
    torch.testing.assert_close(model.predict(x), restored.predict(x))
    with pytest.raises(ValueError):
        PilotRiskRegressor.fit(x[:, :-1], y)


def test_pilot_risk_gate_only_adds_a_rejection_after_existing_hard_gate():
    observation = PilotObservation(
        base_logits=torch.zeros(4), features=torch.zeros(4, 7), logit_scale=torch.ones(1, 1),
        adapt_mask=torch.tensor([True, True, False, False]),
        adapt_labels=torch.ones(4),
        validation_mask=torch.tensor([False, False, True, True]),
        validation_labels=torch.ones(4))
    basis = torch.ones(4, 1)
    settings = DeterministicSettings(steps=1, learning_rate=.02, proximal_weight=0.,
        reset_to_origin_on_validation=False, origin_retention_decay=1.)
    origin = torch.zeros(1)
    candidate, hard_retained, hard_metrics = coefficient_update(
        observation.base_logits, basis, observation, origin, origin, settings)
    assert hard_metrics["accepted"] is True
    gated_candidate, gated_retained, gated_metrics = coefficient_update(
        observation.base_logits, basis, observation, origin, origin, settings,
        validation_risk_gate=lambda features: features.new_tensor(-1.))
    torch.testing.assert_close(gated_candidate, candidate)
    assert gated_metrics["candidate_accepted"] is True
    assert gated_metrics["validation_risk_gate_rejected"] is True
    assert gated_metrics["accepted"] is False
    torch.testing.assert_close(gated_retained, origin)
    torch.testing.assert_close(hard_retained, candidate)


def test_pilot_risk_replacement_can_accept_a_hard_rejected_candidate_from_pilot_score():
    observation = PilotObservation(
        base_logits=torch.zeros(4), features=torch.zeros(4, 7), logit_scale=torch.ones(1, 1),
        adapt_mask=torch.tensor([True, True, False, False]),
        adapt_labels=torch.ones(4),
        validation_mask=torch.tensor([False, False, True, True]),
        validation_labels=torch.zeros(4))
    basis = torch.ones(4, 1)
    settings = DeterministicSettings(steps=1, learning_rate=.02, proximal_weight=0.,
        reset_to_origin_on_validation=False, origin_retention_decay=1.)
    origin = torch.zeros(1)
    candidate, hard_retained, hard_metrics = coefficient_update(
        observation.base_logits, basis, observation, origin, origin, settings)
    assert hard_metrics["accepted"] is False
    risk_candidate, risk_retained, risk_metrics = coefficient_update(
        observation.base_logits, basis, observation, origin, origin, settings,
        validation_risk_gate=lambda features: features.new_tensor(.1),
        validation_risk_gate_mode="replace")
    torch.testing.assert_close(risk_candidate, candidate)
    assert risk_metrics["candidate_accepted"] is False
    assert risk_metrics["accepted"] is True
    assert risk_metrics["validation_risk_gate_rejected"] is False
    torch.testing.assert_close(risk_retained, candidate)
    torch.testing.assert_close(hard_retained, origin)


def test_pilot_risk_manifest_fixes_base_checkpoint_split_and_threshold():
    import json
    spec = json.loads(open("configs/audits/pilot_validation_risk_gate_v1.json", encoding="utf-8").read())
    assert spec["channel_changes"] is False
    assert spec["heldout_confirmation"] is False
    assert spec["risk_gate"]["threshold"] == 0.
    assert spec["risk_gate"]["ridge"] == .1
    assert spec["risk_gate"]["feature_names"] == list(FEATURE_NAMES)


def test_dense_memory_origin_reset_equals_physical_branch():
    from tests.test_adaptability_audit import _fixture
    from training.deterministic_adaptation import DeterministicMemoryAdapter
    from training.pilot_transfer import context_logits
    model, context = _fixture()
    neural = copy.deepcopy(model)
    adapter = DeterministicMemoryAdapter(neural,DeterministicSettings())
    adapter.step(context)
    retained = neural.channel_state.memory.weight.detach().clone()
    with torch.no_grad():
        neural.channel_state.memory.weight.copy_(adapter.origin)
        torch.testing.assert_close(context_logits(neural,context),context_logits(model,context),atol=0,rtol=0)
        neural.channel_state.memory.weight.copy_(retained)
