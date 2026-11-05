# -*- coding: utf-8 -*-
"""验证 rank-1 Adapt Pilot 阻尼 Gauss–Newton 候选。"""

import pytest
import torch

from training.rank1_online import (
    damped_rank1_gauss_newton_candidate,
    differentiable_damped_rank1_gauss_newton_candidate,
)


def _linear_adapt_logits(features):
    return lambda parameters: features @ parameters


def test_rank1_candidate_matches_damped_logistic_fisher_step():
    parameters = torch.tensor([0.2, -0.1, 0.05], dtype=torch.float64)
    basis = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float64)
    features = torch.tensor([
        [1.0, 0.5, -0.2],
        [-0.3, 0.8, 0.7],
        [0.2, -1.0, 0.4],
        [0.9, 0.3, 0.1],
    ], dtype=torch.float64)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    damping = 0.7

    candidate, diagnostics = damped_rank1_gauss_newton_candidate(
        parameters=parameters,
        basis=basis,
        adapt_logits_fn=_linear_adapt_logits(features),
        adapt_labels=labels,
        damping=damping,
        max_step_norm=1.0,
    )

    direction = basis / basis.norm()
    logits = features @ parameters
    jacobian = features @ direction
    probability = torch.sigmoid(logits)
    gradient = ((probability - labels) * jacobian).mean()
    fisher = (probability * (1.0 - probability) * jacobian.square()).mean()
    expected_step = -gradient / (fisher + damping)
    expected = parameters + direction * expected_step

    torch.testing.assert_close(candidate, expected)
    assert diagnostics["no_op"] is False
    assert diagnostics["adapt_count"] == labels.numel()
    assert diagnostics["gradient"] == pytest.approx(float(gradient))
    assert diagnostics["fisher"] == pytest.approx(float(fisher))
    assert candidate.requires_grad is False


def test_differentiable_rank1_candidate_matches_logistic_step_and_outer_gradients():
    parameters = torch.tensor([0.25, -0.18, 0.12], dtype=torch.float64, requires_grad=True)
    basis = torch.tensor([0.7, -0.3, 0.5], dtype=torch.float64, requires_grad=True)
    features = torch.tensor([
        [1.0, 0.2, -0.4],
        [-0.5, 0.7, 0.3],
        [0.4, -0.8, 1.0],
        [0.6, 0.1, 0.2],
    ], dtype=torch.float64)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    damping = 0.8

    candidate, diagnostics = differentiable_damped_rank1_gauss_newton_candidate(
        parameters=parameters,
        basis=basis,
        adapt_logits_fn=_linear_adapt_logits(features),
        adapt_labels=labels,
        damping=damping,
        max_step_norm=1.0,
    )

    direction = basis / basis.norm()
    logits = features @ parameters
    jacobian = features @ direction
    probabilities = torch.sigmoid(logits)
    gradient = ((probabilities - labels) * jacobian).mean()
    fisher = (probabilities * (1.0 - probabilities) * jacobian.square()).mean()
    expected = parameters + direction * (-gradient / (fisher + damping))
    torch.testing.assert_close(candidate, expected)

    target = torch.tensor([-0.1, 0.25, 0.4], dtype=torch.float64)
    outer_loss = (candidate - target).square().sum()
    parameter_gradient, basis_gradient = torch.autograd.grad(outer_loss, (parameters, basis))
    assert torch.isfinite(parameter_gradient).all()
    assert torch.isfinite(basis_gradient).all()
    assert torch.linalg.vector_norm(parameter_gradient) > 0
    assert torch.linalg.vector_norm(basis_gradient) > 0
    assert candidate.requires_grad
    assert diagnostics["gradient"].requires_grad
    assert diagnostics["fisher"].requires_grad


def test_rank1_candidate_respects_single_step_trust_region():
    parameters = torch.zeros(3)
    basis = torch.tensor([0.0, 3.0, 4.0])
    features = torch.tensor([[10.0, -3.0, 2.0], [-5.0, 4.0, 8.0]])
    labels = torch.tensor([1.0, 0.0])
    radius = 0.015

    candidate, diagnostics = damped_rank1_gauss_newton_candidate(
        parameters=parameters,
        basis=basis,
        adapt_logits_fn=_linear_adapt_logits(features),
        adapt_labels=labels,
        damping=1e-5,
        max_step_norm=radius,
    )

    delta = candidate - parameters
    assert float(torch.linalg.vector_norm(delta)) <= radius + 1e-7
    assert diagnostics["step_norm"] <= radius + 1e-7
    torch.testing.assert_close(delta, basis / basis.norm() * diagnostics["step_norm"] * torch.sign(
        torch.dot(delta, basis)))


def test_constant_adapt_logits_produce_finite_no_op_even_under_no_grad():
    parameters = torch.tensor([0.3, -0.2])
    basis = torch.tensor([1.0, 1.0])
    labels = torch.tensor([1.0, 0.0, 1.0])

    with torch.no_grad():
        candidate, diagnostics = damped_rank1_gauss_newton_candidate(
            parameters=parameters,
            basis=basis,
            adapt_logits_fn=lambda _parameters: torch.zeros_like(labels),
            adapt_labels=labels,
            damping=0.1,
            max_step_norm=0.05,
        )

    torch.testing.assert_close(candidate, parameters)
    assert diagnostics["no_op"] is True
    assert diagnostics["no_op_reason"] == "zero_adapt_gradient"
    assert diagnostics["step_norm"] == 0.0
    assert torch.isfinite(candidate).all()


def test_empty_adapt_mask_is_finite_no_op():
    parameters = torch.tensor([0.1, 0.2])
    labels = torch.tensor([1.0, 0.0])
    candidate, diagnostics = damped_rank1_gauss_newton_candidate(
        parameters=parameters,
        basis=torch.tensor([1.0, 0.0]),
        adapt_logits_fn=lambda value: torch.stack((value[0], value[1])),
        adapt_labels=labels,
        adapt_mask=torch.tensor([False, False]),
        damping=0.01,
        max_step_norm=0.1,
    )
    torch.testing.assert_close(candidate, parameters)
    assert diagnostics["no_op"] is True
    assert diagnostics["no_op_reason"] == "empty_adapt_mask"
    assert diagnostics["adapt_count"] == 0


@pytest.mark.parametrize("damping", [0.0, -0.1, float("inf"), float("nan")])
def test_damping_must_be_finite_and_positive(damping):
    with pytest.raises(ValueError, match="damping"):
        damped_rank1_gauss_newton_candidate(
            parameters=torch.zeros(2),
            basis=torch.tensor([1.0, 0.0]),
            adapt_logits_fn=lambda value: value[:1],
            adapt_labels=torch.tensor([1.0]),
            damping=damping,
            max_step_norm=0.1,
        )


def test_non_finite_adapt_logits_fail_closed():
    with pytest.raises(FloatingPointError, match="logits"):
        damped_rank1_gauss_newton_candidate(
            parameters=torch.zeros(2),
            basis=torch.tensor([1.0, 0.0]),
            adapt_logits_fn=lambda _value: torch.tensor([float("nan")]),
            adapt_labels=torch.tensor([1.0]),
            damping=0.1,
            max_step_norm=0.1,
        )
