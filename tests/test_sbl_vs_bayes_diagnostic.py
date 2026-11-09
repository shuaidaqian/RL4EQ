# -*- coding: utf-8 -*-
"""SBL 对照诊断的纯函数契约。"""

import pytest
import torch

from scripts.diagnose_sbl_vs_bayes import (
    _gaussian_posterior,
    _ls_update,
    _sbl_update,
)


def _nmse(estimate, truth):
    est = estimate.reshape(-1).to(torch.complex128)
    tru = truth.reshape(-1).to(torch.complex128)
    scale = torch.vdot(est, tru) / torch.vdot(est, est).real.clamp_min(1e-12)
    residual = tru - scale * est
    return float((residual.abs().square().sum() / tru.abs().square().sum().clamp_min(1e-12)).real)


def test_gaussian_posterior_matches_closed_form():
    torch.manual_seed(3)
    rows, dim = 40, 9
    design = torch.randn(rows, dim, dtype=torch.complex64)
    observation = torch.randn(rows, dtype=torch.complex64)
    prior_mean = torch.randn(dim, dtype=torch.complex64)
    prior_cov = torch.eye(dim, dtype=torch.complex64) * 2.0
    row_noise = torch.linspace(0.5, 1.5, rows)
    mean, covariance = _gaussian_posterior(
        prior_mean, prior_cov, design, observation, row_noise,
    )
    precision = torch.linalg.inv(prior_cov)
    weights = row_noise.reciprocal()
    information = design.conj().transpose(0, 1) @ (weights.unsqueeze(1) * design)
    expected = torch.linalg.solve(
        precision + information,
        precision @ prior_mean + design.conj().transpose(0, 1) @ (weights * observation),
    )
    torch.testing.assert_close(mean, expected)
    assert covariance.shape == (dim, dim)


def test_ls_update_matches_least_squares_with_small_ridge():
    torch.manual_seed(5)
    rows, dim = 60, 12
    design = torch.randn(rows, dim, dtype=torch.complex64)
    truth = torch.randn(dim, dtype=torch.complex64)
    observation = design @ truth
    estimate = _ls_update(design, observation, ridge=1e-10)
    expected = torch.linalg.lstsq(design, observation).solution
    torch.testing.assert_close(estimate, expected, atol=1e-4, rtol=1e-3)


def test_sbl_beats_ls_on_sparse_channel():
    torch.manual_seed(7)
    rows, dim, support = 80, 24, 3
    truth = torch.zeros(dim, dtype=torch.complex64)
    truth[:support] = torch.tensor([1.0, 0.6, -0.4], dtype=torch.complex64)
    design = torch.randn(rows, dim, dtype=torch.complex64)
    observation = design @ truth + 0.05 * torch.randn(rows, dtype=torch.complex64)
    least_squares = _ls_update(design, observation, ridge=1e-6)
    sparse, _, _ = _sbl_update(
        design, observation, torch.full((rows,), 0.0025),
        torch.zeros(dim, dtype=torch.complex64),
        torch.ones(dim, dtype=torch.float32), iterations=30,
    )
    assert _nmse(sparse, truth) < _nmse(least_squares, truth)
    assert _nmse(sparse, truth) < 0.05
