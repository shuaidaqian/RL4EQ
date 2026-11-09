# -*- coding: utf-8 -*-
"""自适应 Q 只读实现的纯函数契约。"""

import torch

from scripts.diagnose_adaptive_q_online import _select_q


def test_select_q_prefers_zero_on_stationary_clean_data():
    torch.manual_seed(11)
    rows, dim = 60, 10
    truth = torch.randn(dim, dtype=torch.complex64)
    design = torch.randn(rows, dim, dtype=torch.complex64)
    observation = design @ truth
    prior_mean = truth.clone()
    prior_cov = torch.eye(dim, dtype=torch.complex64) * 0.02
    row_noise = torch.full((rows,), 0.01)
    selected = _select_q(
        prior_mean, prior_cov, design, observation, row_noise,
        candidates=[0.0, 1e-4, 1e-2],
    )
    assert selected == 0.0


def test_select_q_returns_a_candidate():
    torch.manual_seed(13)
    rows, dim = 40, 8
    design = torch.randn(rows, dim, dtype=torch.complex64)
    observation = torch.randn(rows, dtype=torch.complex64)
    prior_mean = torch.zeros(dim, dtype=torch.complex64)
    prior_cov = torch.eye(dim, dtype=torch.complex64)
    row_noise = torch.ones(rows)
    candidates = [0.0, 1e-5, 1e-3]
    selected = _select_q(prior_mean, prior_cov, design, observation, row_noise, candidates)
    assert selected in candidates
