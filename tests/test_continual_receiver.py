# -*- coding: utf-8 -*-
"""持续接收机的噪声功率、时序和隐藏标签隔离契约。"""

from dataclasses import replace

import pytest
import torch

from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from env.frame_structure import FrameConfig, FrameGenerator
from training.continual_receiver import (
    ARMS, ContinualReceiver, estimate_acquisition, gaussian_update, receiver_frame,
)


def acquisition_fixture():
    rng = torch.Generator().manual_seed(42)
    tx = (2 * torch.randint(0, 2, (1024,), generator=rng) - 1).to(torch.complex64)
    cir = torch.tensor([0.9 + .2j, .25 - .1j, -.1j])
    rx = sum(cir[k] * torch.roll(tx, k) for k in range(3))
    noise = .1 * torch.complex(torch.randn(1024, generator=rng), torch.randn(1024, generator=rng))
    return tx, rx + noise, cir


def test_acquisition_noise_and_gain_are_identifiable_without_snr():
    tx, rx, cir = acquisition_fixture()
    result = estimate_acquisition(tx, rx, 2, cfo_limit=0)
    assert result.observations == 1022 and result.rank == 3
    assert result.noise_source == "acquisition_residual"
    assert .017 < result.noise_variance < .023
    torch.testing.assert_close(result.mean, cir, atol=.012, rtol=0)
    scaled = estimate_acquisition(tx, rx * 3, 2, cfo_limit=0)
    assert scaled.noise_variance == pytest.approx(9 * result.noise_variance, rel=1e-5)
    torch.testing.assert_close(scaled.mean, result.mean * 3, atol=1e-6, rtol=1e-5)


def test_gaussian_pilot_update_matches_batch_without_double_counting():
    rng = torch.Generator().manual_seed(11)
    design = torch.randn((40, 3), generator=rng).to(torch.complex64)
    target = torch.randn(40, generator=rng).to(torch.complex64)
    mean = torch.zeros(3, dtype=torch.complex64)
    cov = torch.eye(3, dtype=torch.complex64)
    noise = torch.full((40,), .2)
    sequential = gaussian_update(mean, cov, design[:20], target[:20], noise[:20])
    sequential = gaussian_update(*sequential, design[20:], target[20:], noise[20:])
    batch = gaussian_update(mean, cov, design, target, noise)
    torch.testing.assert_close(sequential[0], batch[0], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(sequential[1], batch[1], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("arm", ARMS)
def test_hidden_labels_reward_and_truth_do_not_change_receiver(arm):
    tx, rx, _ = acquisition_fixture()
    acq = estimate_acquisition(tx, rx, 2, cfo_limit=0)
    frame = FrameGenerator(FrameConfig(frame_len=128, total_pilot=64, max_delay=2), seed=1).generate(0)
    frame = replace(frame, rx_symbols=torch.randn(128, dtype=torch.complex64),
                    true_cir=torch.ones(3, dtype=torch.complex64), tail_symbols=torch.ones(2))
    poison_tx = frame.tx_symbols.clone()
    poison_tx[~frame.adapt_mask] *= -1
    poison = replace(frame, bits=~frame.bits, tx_symbols=poison_tx,
                     true_cir=frame.true_cir * 10, tail_symbols=frame.tail_symbols * 10)
    model = UnfoldedEqualizer(UnfoldedConfig(frame_len=128, max_delay=2, iterations=1,
        d_model=8, num_heads=2, physics_warm_start_iterations=1))
    model.eval()
    first = ContinualReceiver(acq, torch.zeros(2, dtype=torch.complex64), arm=arm)
    second = ContinualReceiver(acq, torch.zeros(2, dtype=torch.complex64), arm=arm)
    a, info = first.step(model, receiver_frame(frame))
    b, _ = second.step(model, receiver_frame(poison))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(first.mean, second.mean, rtol=0, atol=0)
    assert info["neural_forwards"] == 1
    assert info["pilot_updates"] == int(arm != "acquisition")
    if arm in {"recursive_pilot", "recursive_soft"}:
        again, next_info = first.step(model, receiver_frame(frame))
        assert torch.isfinite(again).all()
        assert next_info["neural_forwards"] == 1
    with pytest.raises(ValueError, match="隐藏"):
        first.step(model, frame)


def test_rank_deficient_acquisition_cannot_estimate_noise():
    with pytest.raises(ValueError, match="满列秩"):
        estimate_acquisition(torch.ones(128), torch.ones(128), 3, cfo_limit=0)
