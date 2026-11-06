# -*- coding: utf-8 -*-
"""rank-1共享head更新必须留在Pilot Fisher子空间和累计信赖域内。"""

import torch
from types import SimpleNamespace

from scripts.prototype_head_rank1_adapt import (
    _adapt_update_contexts, _aggregate_reward_metrics, _cumulative_delta_radius,
    _posthoc_data_metrics, _project_candidate,
)


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
