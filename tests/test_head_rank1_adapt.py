# -*- coding: utf-8 -*-
"""rank-1共享head更新必须留在Pilot Fisher子空间和累计信赖域内。"""

import torch
from types import SimpleNamespace

from scripts.prototype_head_rank1_adapt import (
    _adapt_update_contexts, _aggregate_reward_metrics, _cumulative_delta_radius,
    _posthoc_data_metrics, _project_candidate,
)
from training.rank1_online import _scalar_output_jacobian, scalar_gaussian_pilot_update


def test_projected_head_candidate_stays_in_rank_one_basis_and_radius():
    names = ["head.weight", "head.bias"]
    origin = {"head.weight": torch.zeros(1, 4), "head.bias": torch.zeros(1)}
    before = {"head.weight": torch.tensor([[0.01, 0.0, 0.0, 0.0]]),
              "head.bias": torch.tensor([0.0])}
    candidate = {"head.weight": torch.tensor([[0.03, 0.04, -0.02, 0.01]]),
                 "head.bias": torch.tensor([0.03])}
    basis = torch.zeros(5)
    basis[0] = 1.0
    projected, raw_norm, projected_norm = _project_candidate(
        candidate, before, origin, names, basis, radius=0.02)
    flat = torch.cat((projected["head.weight"].reshape(-1), projected["head.bias"].reshape(-1)))
    assert flat[0] == torch.tensor(0.02)
    assert torch.count_nonzero(flat[1:]) == 0
    assert torch.linalg.vector_norm(flat) <= 0.0200001
    assert raw_norm > projected_norm > 0


def test_data_metrics_are_scored_after_online_action_and_report_continuous_bce():
    """离线事后指标同时报告BER/BCE和正向改善，不提供控制输入。"""
    frame = SimpleNamespace(data_mask=torch.tensor([True, True, False]),
                            bits=torch.tensor([1.0, 0.0, 1.0]))
    frozen = torch.tensor([0.2, 0.1, 4.0])
    online = torch.tensor([1.0, -1.0, -4.0])

    metrics = _posthoc_data_metrics(frozen, online, frame)

    assert metrics["ber_frozen"] == 0.5
    assert metrics["ber_online"] == 0.0
    assert metrics["data_bce_gain"] > 0
    assert set(metrics) == {"ber_frozen", "ber_online", "data_bce_frozen",
                            "data_bce_online", "data_bce_gain"}


def test_reward_window_metrics_pool_pilot_counts_without_data_labels():
    """跨帧门控按独立Reward Pilot数量加权，不受帧错误率分母差异影响。"""
    ber, bce, count = _aggregate_reward_metrics([
        (0.5, 0.4, 4),
        (0.0, 0.2, 12),
    ])

    assert count == 16
    assert ber == 0.125
    assert abs(bce - 0.25) < 1e-7


def test_adapt_history_uses_only_recent_contexts_and_includes_current():
    """历史更新只使用最近窗口，并且当前Adapt帧始终纳入。"""
    history = list(range(6))

    assert _adapt_update_contexts(history, 6, 4) == [3, 4, 5, 6]
    assert _adapt_update_contexts(history, 6, 1) == [6]
    assert _adapt_update_contexts([], 6, 4) == [6]


def test_radius2_branch_only_doubles_cumulative_trust_region():
    """半径消融只改变累计边界，保留逐帧更新配置。"""
    settings = {"max_delta_norm": 0.05}

    assert _cumulative_delta_radius(settings, "full_head_adam") == 0.05
    assert _cumulative_delta_radius(settings, "full_head_adam_radius2") == 0.10


def test_scalar_gaussian_pilot_update_preserves_spd_and_accumulates_adapt_information():
    """Adapt score更新均值，Pilot Fisher收缩方差且保持正定下限。"""
    mean = torch.tensor(0.0, dtype=torch.float64)
    variance = torch.tensor(1.0, dtype=torch.float64)
    candidate, posterior, diagnostics = scalar_gaussian_pilot_update(
        coefficient=mean, variance=variance, gradient=-0.2, fisher=0.4,
        process_noise=0.1, information_scale=1.0, damping=0.01,
        max_step_norm=0.05, variance_floor=1e-4)

    assert candidate > mean
    assert torch.linalg.vector_norm(candidate - mean) <= 0.0500001
    assert 0 < posterior < variance + 0.1
    assert diagnostics["fisher"] == 0.4


def test_scalar_gaussian_pilot_update_zero_score_keeps_mean_but_updates_uncertainty():
    """无均值更新时仍吸收合法Adapt信息，且方差不得跌破下限。"""
    mean = torch.tensor(-0.2)
    variance = torch.tensor(0.3)
    candidate, posterior, _ = scalar_gaussian_pilot_update(
        coefficient=mean, variance=variance, gradient=0.0, fisher=2.0,
        process_noise=0.01, information_scale=1.0, damping=0.0,
        max_step_norm=0.05, variance_floor=0.02)

    assert torch.equal(candidate, mean)
    assert posterior >= 0.02
    assert posterior < variance + 0.01


def test_scalar_output_jacobian_chunking_matches_closed_form():
    """分块VJP与已知输出导数逐项一致。"""
    coefficient = torch.tensor(0.4, requires_grad=True)
    weights = torch.arange(1, 10, dtype=torch.float32)
    logits = weights * coefficient + coefficient.square()
    jacobian = _scalar_output_jacobian(logits, coefficient, chunk_size=3)
    expected = weights + 2.0 * coefficient.detach()

    assert torch.allclose(jacobian, expected)
