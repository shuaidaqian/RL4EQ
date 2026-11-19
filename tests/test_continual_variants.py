# -*- coding: utf-8 -*-
"""验证神经分支消融与真实调用次数，不把同步学习模块误称非神经。"""

from dataclasses import replace

import pytest
import torch

from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from env.frame_structure import FrameConfig, FrameGenerator
from evaluation.continual_variants import ForwardCounter, PhysicsBranch, validate_variants
from training.continual_receiver import ContinualReceiver, estimate_acquisition, receiver_frame
from tests.test_continual_receiver import acquisition_fixture


def test_physics_branch_matches_zero_transformer_residual_and_call_counter():
    tx, rx, _ = acquisition_fixture()
    acq = estimate_acquisition(tx, rx, 2, cfo_limit=0)
    config = UnfoldedConfig(frame_len=128, max_delay=2, iterations=1, d_model=8, num_heads=2,
        enable_phase_correction_branch=True, phase_correction_initial_scale=1.,
        physics_warm_start_iterations=2, neural_residual_scale=0.)
    model = UnfoldedEqualizer(config).eval()
    frame = FrameGenerator(FrameConfig(frame_len=128, total_pilot=64, max_delay=2), seed=12).generate(0)
    frame = receiver_frame(replace(frame, rx_symbols=torch.randn(128, dtype=torch.complex64)))
    full = ForwardCounter(model)
    physical = ForwardCounter(PhysicsBranch(model))
    receivers = [ContinualReceiver(acq, torch.zeros(2, dtype=torch.complex64),
        arm='recursive_pilot', phase_estimator='circular_clean') for _ in range(2)]
    a, _ = receivers[0].step(full, frame)
    b, _ = receivers[1].step(physical, frame)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert full.calls == physical.calls == 1


def test_variant_manifest_rejects_unknown_or_ambiguous_comparisons():
    variant = dict(name='c', arm='recursive_pilot', checkpoint='example.pt')
    assert validate_variants(dict(variants=[variant])) == [variant]
    with pytest.raises(ValueError, match='互异'):
        validate_variants(dict(variants=[variant, variant]))
    with pytest.raises(ValueError, match='配对'):
        validate_variants(dict(variants=[variant], pairs=[['c', 'missing']]))
    physical = dict(name='physical', arm='recursive_pilot', checkpoint='example.pt',
                    decoder='physics_branch', context_source='c')
    assert len(validate_variants(dict(variants=[variant, physical]))) == 2
    with pytest.raises(ValueError, match='前面'):
        validate_variants(dict(variants=[physical, variant]))
    with pytest.raises(ValueError, match='同一 checkpoint'):
        validate_variants(dict(variants=[variant, dict(physical, checkpoint='other.pt')]))
