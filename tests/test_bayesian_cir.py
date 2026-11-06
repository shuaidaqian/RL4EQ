# -*- coding: utf-8 -*-
"""验证贝叶斯 CIR 先验/后验的数值、协方差收缩与 Data 标签隔离。"""

from dataclasses import replace

import torch

from env.frame_structure import FrameConfig, FrameGenerator
from env.linear_operator import LinearChannelOperator
from training.bayesian_cir import (
    acquisition_prior, add_process_noise, data_posterior, pilot_posterior,
)


def _fixture():
    frame = FrameGenerator(FrameConfig(frame_len=128, total_pilot=64, max_delay=4), seed=7).generate(0)
    cir = torch.tensor([0.8, 0.0, 0.3, 0.0, 0.6], dtype=torch.complex64)
    tail = torch.zeros(4, dtype=torch.complex64)
    operator = LinearChannelOperator(frame_len=128, max_delay=4)
    rx = operator.forward(
        frame.tx_symbols.to(torch.complex64).unsqueeze(0), cir.unsqueeze(0), tail.unsqueeze(0),
    ).squeeze(0)
    acquisition = replace(frame, rx_symbols=rx)
    logits = frame.tx_symbols.real.float() * 10.0
    return acquisition, cir, logits


def test_acquisition_prior_recovers_known_channel_and_covariance():
    acquisition, cir, _ = _fixture()
    mean, covariance = acquisition_prior(acquisition, 0.0, 4, 0.1, ridge=1e-6)
    torch.testing.assert_close(mean, cir / cir.norm(), rtol=1e-3, atol=1e-4)
    assert covariance.shape == (5, 5)
    assert torch.isfinite(covariance).all()
    torch.testing.assert_close(covariance, covariance.conj().transpose(0, 1), rtol=1e-6, atol=1e-8)
    assert torch.diagonal(covariance).real.min() > 0.0


def test_pilot_posterior_shrinks_covariance():
    acquisition, _, _ = _fixture()
    mean, covariance = acquisition_prior(acquisition, 0.0, 4, 0.1, ridge=1e-3)
    post_mean, post_cov = pilot_posterior(acquisition, mean, covariance, 4, 0.1, 0.0, 0.0)
    assert torch.isfinite(post_mean).all() and torch.isfinite(post_cov).all()
    assert torch.trace(post_cov).real < torch.trace(covariance).real


def test_data_posterior_is_label_free_and_shrinks_covariance():
    acquisition, _, logits = _fixture()
    mean, covariance = acquisition_prior(acquisition, 0.0, 4, 0.1, ridge=1e-3)
    post_mean, post_cov = data_posterior(
        acquisition, logits, mean, covariance, 4, 0.1, 0.0, 0.0,
    )
    assert torch.trace(post_cov).real < torch.trace(covariance).real
    bits = acquisition.bits.clone()
    bits[acquisition.data_mask] = ~bits[acquisition.data_mask]
    poisoned = replace(acquisition, bits=bits)
    poisoned_mean, poisoned_cov = data_posterior(
        poisoned, logits, mean, covariance, 4, 0.1, 0.0, 0.0,
    )
    torch.testing.assert_close(post_mean, poisoned_mean, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(post_cov, poisoned_cov, rtol=1e-6, atol=1e-6)


def test_process_noise_increases_covariance_diagonal():
    covariance = torch.eye(5, dtype=torch.complex64)
    updated = add_process_noise(covariance, 1e-2)
    torch.testing.assert_close(torch.diagonal(updated), torch.full((5,), 1.01, dtype=torch.complex64))
