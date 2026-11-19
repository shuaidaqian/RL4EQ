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
