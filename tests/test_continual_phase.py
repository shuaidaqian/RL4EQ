# -*- coding: utf-8 -*-
"""圆周相位估计的同步恢复与未知符号隔离。"""

from dataclasses import replace

import pytest
import torch

from env.frame_structure import FrameConfig, FrameGenerator
from env.linear_operator import LinearChannelOperator
from training.continual_phase import circular_phase_features
from training.continual_receiver import receiver_frame


@pytest.mark.parametrize('phase', [3.13, -3.13, .7])
@pytest.mark.parametrize('clean_rows', [False, True])
def test_circular_phase_recovers_boundary_phase_and_known_cfo(phase, clean_rows):
    frame = FrameGenerator(FrameConfig(frame_len=128, total_pilot=64, max_delay=2), seed=7).generate(0)
    cir = torch.tensor([.9+.1j, .2j, .1], dtype=torch.complex64)
    tail = torch.tensor([1., -1.], dtype=torch.complex64)
    cfo = .001
    reference = LinearChannelOperator(frame_len=128, max_delay=2).forward(frame.tx_symbols[None], cir[None], tail[None]).flatten()
    rx = reference*torch.exp(1j*(phase+2*torch.pi*cfo*torch.arange(128)))
    visible = receiver_frame(replace(frame, rx_symbols=rx))
    features = circular_phase_features(visible, cir, tail, cfo, clean_rows=clean_rows)
    circular_error = torch.angle(torch.exp(1j*(features[0]-phase)))
    assert abs(float(circular_error)) < 1e-5
    poison = replace(visible, rx_symbols=visible.rx_symbols.clone())
    poison.rx_symbols[~visible.adapt_mask] = 1e6+1e6j
    other = circular_phase_features(poison, cir, tail, cfo, clean_rows=clean_rows)
    torch.testing.assert_close(features, other, rtol=0, atol=0)


@pytest.mark.parametrize('clean_rows', [False, True])
def test_vectorized_reference_matches_legacy_with_sparse_taps_and_long_memory(clean_rows):
    frame = FrameGenerator(FrameConfig(frame_len=1024, total_pilot=256, max_delay=116), seed=8).generate(0)
    rng = torch.Generator().manual_seed(423)
    cir = torch.randn(117, dtype=torch.complex64, generator=rng)/117**.5
    cir[5:10] = 0
    tail = torch.tanh(torch.randn(116, generator=rng)).to(torch.complex64)
    rx = LinearChannelOperator(frame_len=1024, max_delay=116).forward(frame.tx_symbols[None], cir[None], tail[None]).flatten()
    rx = rx*torch.exp(1j*(.7+2*torch.pi*.0004*torch.arange(1024)))
    rx += .1*torch.randn(1024, dtype=torch.complex64, generator=rng)
    visible = receiver_frame(replace(frame, rx_symbols=rx))
    original = circular_phase_features(visible, cir, tail, .0004, clean_rows=clean_rows, reference_backend='legacy')
    vectorized = circular_phase_features(visible, cir, tail, .0004, clean_rows=clean_rows)
    torch.testing.assert_close(vectorized, original, atol=2e-6, rtol=2e-6)
