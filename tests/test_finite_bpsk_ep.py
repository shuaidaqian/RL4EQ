# -*- coding: utf-8 -*-
"""用闭式和穷举BPSK边缘检验EP自然参数，保持有限帧边界。"""

import itertools

import pytest
import torch

from evaluation.finite_bpsk_ep import bpsk_ep


def _exact_logits(precision, linear):
    symbols = torch.tensor(list(itertools.product([-1., 1.], repeat=linear.numel())), dtype=linear.dtype)
    energy = -.5 * torch.einsum('bi,ij,bj->b', symbols, precision, symbols) + symbols @ linear
    return torch.stack([torch.logsumexp(energy[symbols[:, index] == 1], dim=0)
        - torch.logsumexp(energy[symbols[:, index] == -1], dim=0) for index in range(linear.numel())])


@pytest.mark.parametrize('steps', [0, 1, 5])
def test_orthogonal_channel_matches_exact_bpsk_llr_for_any_ep_round(steps):
    precision = torch.diag(torch.tensor([.2, 1., 20., 100.], dtype=torch.float64))
    linear = torch.tensor([-.3, 0., 1.5, 20.], dtype=torch.float64)
    result = bpsk_ep(precision, linear, steps=steps)
    torch.testing.assert_close(result.logits, 2 * linear, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(result.logits, _exact_logits(precision, linear), rtol=1e-7, atol=1e-7)
    assert bool((result.gaussian_variance > 0).all())


def test_zero_information_does_not_create_bit_bias_or_overconfidence():
    result = bpsk_ep(torch.zeros(3, 3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64))
    torch.testing.assert_close(result.logits, torch.zeros(3, dtype=torch.float64), atol=0, rtol=0)
    torch.testing.assert_close(result.gaussian_variance, torch.ones(3, dtype=torch.float64), atol=0, rtol=0)


def test_finite_convolution_and_sign_symmetry_without_cir_or_label_access():
    # 末行不补循环响应；显式有限矩阵的正规项与未知符号等维。
    h = torch.tensor([[1., 0., 0.], [.6, 1., 0.], [0., .6, 1.]], dtype=torch.float64)
    precision = 2 * h.T @ h / .4
    linear = 2 * h.T @ torch.tensor([.8, -.2, .3], dtype=torch.float64) / .4
    positive = bpsk_ep(precision, linear, steps=5)
    negative = bpsk_ep(precision, -linear, steps=5)
    torch.testing.assert_close(positive.logits, -negative.logits, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(positive.site_precision, negative.site_precision, atol=1e-9, rtol=1e-9)
    assert bool(torch.isfinite(positive.logits).all())
    assert len(positive.rejected_site_updates) == 5
    exact = _exact_logits(precision, linear)
    assert bool(torch.isfinite(exact).all())
    # EP是近似算法；此处不要求它等于或逐次接近穷举后验。


@pytest.mark.parametrize('precision', [torch.tensor([[1., 2.], [0., 1.]], dtype=torch.float64),
    torch.tensor([[1., 2.], [2., 1.]], dtype=torch.float64)])
def test_invalid_likelihood_is_not_hidden_by_gaussian_site_prior(precision):
    with pytest.raises(ValueError):
        bpsk_ep(precision, torch.zeros(2, dtype=torch.float64))
