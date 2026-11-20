# -*- coding: utf-8 -*-
"""独立矩阵参考、历史输出保持和缓存隔离，防止把求解误差当模型增益。"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from agent.cir_estimator import CIRCondition
from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from evaluation.continual_variants import KnownAdaptRealPhysicsBranch, validate_variants
from scripts.compare_continual_variants import (
    SOLVER_DIAGNOSTIC_FIELDS, compact_stream_row, validate_solver_diagnostics,
)


@pytest.mark.parametrize('backend', ['direct', 'fft'])
@pytest.mark.parametrize('iterations', [1, 8, 32])
def test_reported_residual_matches_independent_finite_frame_matrix(monkeypatch, backend, iterations):
    monkeypatch.setenv('RL4EQ_LINEAR_OPERATOR_BACKEND', backend)
    length, memory = 24, 3
    cir = torch.tensor([.6 + .1j, -.2 + .3j, .15j, -.1 + .05j], dtype=torch.complex64)
    delays = torch.arange(length)[:, None] - torch.arange(length)[None, :]
    matrix = torch.where((delays >= 0) & (delays <= memory), cir[delays.clamp(0, memory)], 0)
    known_mask = torch.arange(length) < 6
    known = torch.where(known_mask, (-1.) ** torch.arange(length), 0.).to(torch.complex64)
    unknown = torch.cos(torch.arange(length) * .73) * .1
    unknown[known_mask] = 0
    received = matrix @ (known + unknown)
    condition = CIRCondition(complex_cir=cir[None], support_probability=torch.ones(1, memory + 1),
        noise_variance=torch.tensor([.4]), confidence=torch.ones(1), latent_residual=torch.zeros(1, 96))
    holder = SimpleNamespace(config=SimpleNamespace(physics_warm_start_iterations=iterations))
    args = (holder, received[None], condition, torch.zeros(1, memory, dtype=torch.complex64),
            known[None], known_mask[None])
    legacy = UnfoldedEqualizer._real_bpsk_known_adapt_logits(*args)
    output, diagnostics = UnfoldedEqualizer._real_bpsk_known_adapt_logits(
        *args, return_solver_diagnostics=True)
    torch.testing.assert_close(output, legacy, atol=0, rtol=0)
    design = matrix[:, ~known_mask].to(torch.complex128)
    rhs = (design.mH @ (received - matrix @ known).to(torch.complex128)).real
    normal = (design.mH @ design).real + .2 * torch.eye(design.shape[1], dtype=torch.float64)
    recovered = output[0, ~known_mask].double() * .4 / 2
    reference_relative = (rhs - normal @ recovered).norm() / rhs.norm()
    assert diagnostics['iterations'] == iterations
    torch.testing.assert_close(diagnostics['rhs_norm'].double(), rhs.norm()[None], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(diagnostics['relative_residual'].double(), reference_relative[None],
        rtol=1e-3, atol=1e-6)
    if iterations == 32:
        exact = torch.linalg.solve(normal, rhs)
        torch.testing.assert_close(recovered, exact, rtol=2e-3, atol=2e-4)


def test_solver_override_does_not_mutate_cached_checkpoint_or_other_candidate():
    config = UnfoldedConfig(frame_len=16, max_delay=2, iterations=1, d_model=8, num_heads=2,
        physics_warm_start_iterations=8)
    model = UnfoldedEqualizer(config)
    legacy = KnownAdaptRealPhysicsBranch(model)
    accurate = KnownAdaptRealPhysicsBranch(model, solver_iterations=32, solver_diagnostics=True)
    assert model.config.physics_warm_start_iterations == legacy.config.physics_warm_start_iterations == 8
    assert accurate.config.physics_warm_start_iterations == 32
    assert accurate.model is not model
    for key, value in model.state_dict().items():
        torch.testing.assert_close(accurate.model.state_dict()[key], value, rtol=0, atol=0)
    accurate.model.config = replace(accurate.model.config, physics_warm_start_iterations=64)
    assert model.config.physics_warm_start_iterations == 8


@pytest.mark.parametrize('invalid', [0, -1, 8., True, '8'])
def test_manifest_rejects_invalid_solver_iterations(invalid):
    with pytest.raises(ValueError, match='正整数'):
        validate_variants(dict(variants=[dict(name='x', arm='recursive_pilot', checkpoint='x.pt',
            decoder='known_adapt_real_physics', physics_solver_iterations=invalid)]))


@pytest.mark.parametrize('decoder', ['neural', 'physics_branch', 'known_adapt_real_neural'])
def test_solver_override_cannot_silently_change_other_decoder(decoder):
    with pytest.raises(ValueError, match='只适用于'):
        validate_variants(dict(variants=[dict(name='x', arm='recursive_pilot', checkpoint='x.pt',
            decoder=decoder, physics_solver_iterations=32)]))


def test_stream_preserves_solver_metrics_and_refuses_incomplete_or_misregistered_values(tmp_path):
    from tests.test_long_integrity_prefix import _fixture, _write
    from scripts.check_long_integrity_prefix import inspect_prefix

    path, rows, manifest, config = _fixture(tmp_path)
    for variant in manifest['variants']:
        variant.update(physics_solver_iterations=32, physics_solver_diagnostics=True)
    for row in rows:
        row.update(physics_solver_iterations=32, physics_solver_relative_residual=.001,
            physics_solver_rhs_norm=1.)
        compact = compact_stream_row(row, schema_version=3)
        assert all(compact[key] == row[key] for key in SOLVER_DIAGNOSTIC_FIELDS)
        validate_solver_diagnostics(compact, manifest['variants'][0])
    _write(path, rows)
    assert inspect_prefix(path, manifest, config)['rows'] == len(rows)
    for key, invalid in [('physics_solver_iterations', 8), ('physics_solver_iterations', True),
                         ('physics_solver_relative_residual', float('nan')),
                         ('physics_solver_rhs_norm', -1.)]:
        corrupt = dict(rows[0], **{key: invalid})
        with pytest.raises(ValueError):
            validate_solver_diagnostics(corrupt, manifest['variants'][0])
    del rows[0]['physics_solver_rhs_norm']
    with pytest.raises(ValueError, match='完整'):
        compact_stream_row(rows[0], schema_version=3)
    with pytest.raises(ValueError, match='不一致'):
        validate_solver_diagnostics(rows[1], {})
