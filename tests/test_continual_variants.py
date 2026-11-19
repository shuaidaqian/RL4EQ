# -*- coding: utf-8 -*-
"""验证神经分支消融与真实调用次数，不把同步学习模块误称非神经。"""

from dataclasses import replace

import pytest
import torch

from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from agent.cir_estimator import CIRCondition
from env.frame_structure import FrameConfig, FrameGenerator
from env.linear_operator import LinearChannelOperator
from evaluation.continual_variants import (
    ForwardCounter,
    KnownAdaptRealPhysicsBranch,
    PhysicsBranch,
    validate_variants,
)
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


def test_real_bpsk_known_adapt_solver_matches_reduced_real_normal_equations():
    torch.manual_seed(31)
    frame_len, max_delay = 9, 2
    cir = torch.tensor([0.8 + 0.1j, 0.3 - 0.2j, -0.15 + 0.05j], dtype=torch.complex64)
    tail = torch.tensor([1.0, -1.0], dtype=torch.complex64)
    symbols = torch.randint(0, 2, (frame_len,)).float().mul(2).sub(1)
    adapt_mask = torch.tensor([1, 1, 0, 0, 1, 0, 0, 0, 0], dtype=torch.bool)
    noise_variance = torch.tensor([0.08])
    operator = LinearChannelOperator(frame_len, max_delay)
    rx = operator.forward(symbols.to(torch.complex64), cir, tail)
    rx = rx + 0.01 * (torch.randn(frame_len) + 1j * torch.randn(frame_len))
    condition = CIRCondition(
        complex_cir=cir.unsqueeze(0), support_probability=torch.ones(1, max_delay + 1),
        noise_variance=noise_variance, confidence=torch.ones(1),
        latent_residual=torch.zeros(1, 1),
    )
    model = UnfoldedEqualizer(UnfoldedConfig(
        frame_len=frame_len, max_delay=max_delay, iterations=1, d_model=8, num_heads=2,
        physics_warm_start_iterations=24,
    )).eval()
    known_symbols = torch.where(adapt_mask, symbols, torch.zeros_like(symbols))
    logits = model._real_bpsk_known_adapt_logits(
        rx, condition, tail, known_symbols, adapt_mask,
    )

    known = torch.where(adapt_mask, symbols, torch.zeros_like(symbols)).to(torch.complex64)
    residual_rx = rx - operator.forward(known, cir, tail)
    unknown_indices = torch.where(~adapt_mask)[0]
    columns = []
    for index in unknown_indices.tolist():
        basis = torch.zeros(frame_len, dtype=torch.complex64)
        basis[index] = 1.0
        columns.append(operator.forward(basis, cir, torch.zeros_like(tail)))
    design = torch.stack(columns, dim=1)
    gram = (design.conj().T @ design).real
    rhs = (design.conj().T @ residual_rx).real
    expected = torch.linalg.solve(gram + noise_variance.item() / 2 * torch.eye(len(columns)), rhs)
    torch.testing.assert_close(logits[0, unknown_indices], 2 * expected / noise_variance.item(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(logits[0, adapt_mask], 40 * symbols[adapt_mask], rtol=0, atol=0)

    changed_unobserved = known_symbols.clone()
    changed_unobserved[~adapt_mask] = -changed_unobserved[~adapt_mask]
    repeated = model._real_bpsk_known_adapt_logits(
        rx, condition, tail, changed_unobserved, adapt_mask,
    )
    torch.testing.assert_close(logits, repeated, rtol=0, atol=0)


def test_known_adapt_physics_branch_is_a_valid_matched_decoder():
    variant = dict(name='base', arm='recursive_pilot', checkpoint='model.pt')
    constrained = dict(
        name='constrained', arm='recursive_pilot', checkpoint='model.pt',
        decoder='known_adapt_real_physics', context_source='base',
    )
    assert validate_variants(dict(variants=[variant, constrained])) == [variant, constrained]
