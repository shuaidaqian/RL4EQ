# -*- coding: utf-8 -*-
"""确定性微调、真值隔离、持续状态和诊断坐标的契约测试。"""

import copy
from dataclasses import replace
import json
import subprocess
import sys

import pytest
import torch

from evaluation.adaptability_audit import (
    AuditEnvironment, DiagnosticTruth, ExactStateDiagnostic, LegalStateTracker,
    corrected_block, deployable_frame, fixed_candidate_probe, information_metrics, memory_jacobian_metrics, run_audit,
)
from env.comm_env import CommEnvConfig
from env.impairments import PhaseImpairmentSettings, apply_phase_impairment
from training.deterministic_adaptation import DeterministicMemoryAdapter, DeterministicSettings
from training.pilot_transfer import context_logits


def _fixture():
    from tests.test_pilot_transfer import _channel_memory_fixture
    _, model, context = _channel_memory_fixture()
    return model, replace(context, frame=deployable_frame(context.frame))


@pytest.mark.parametrize("cfo,phase_std", [(0., 0.), (.0007, .003)])
def test_phase_trace_preserves_legacy_waveform_and_random_stream(cfo, phase_std):
    rx = torch.randn(32, dtype=torch.complex64)
    settings = PhaseImpairmentSettings(cfo, phase_std)
    first = torch.Generator().manual_seed(44)
    second = torch.Generator().manual_seed(44)
    legacy, legacy_state = apply_phase_impairment(rx, global_start=192, settings=settings,
        phase_state=0., phase_rng=first)
    traced, traced_state, phase = apply_phase_impairment(rx, global_start=192, settings=settings,
        phase_state=0., phase_rng=second, return_phase=True)
    torch.testing.assert_close(legacy, traced, atol=0, rtol=0)
    torch.testing.assert_close(first.get_state(), second.get_state(), atol=0, rtol=0)
    torch.testing.assert_close(traced * torch.exp(-1j*phase), rx, atol=3e-7, rtol=3e-7)
    assert legacy_state == traced_state


def test_truth_snapshot_follows_current_and_lookahead_frame_without_receiver_fields():
    env = AuditEnvironment(CommEnvConfig(level="B", max_delay=4, frame_len=128,
        total_pilot=64, seed=42, impairment_profile="cfo_phase_tiny"))
    env.reset_episode()
    current = env.next_frame(lookahead_len=4)
    truth = env.truth_for(current)
    future = env.next_frame(lookahead_len=4)
    torch.testing.assert_close(truth.cir, current.true_cir, atol=0, rtol=0)
    torch.testing.assert_close(truth.lookahead_cir, future.true_cir, atol=0, rtol=0)
    torch.testing.assert_close(truth.lookahead_phase, env.truth_for(future).phase[:4], atol=0, rtol=0)
    assert not hasattr(current.receiver_view(), "phase")
    assert not hasattr(current.receiver_view(), "true_cir")
    received = deployable_frame(current)
    assert received.true_cir is None and received.tail_symbols is None
    assert not bool(received.bits.any())
    assert bool((received.tx_symbols[received.data_mask] == 0).all())


def test_legal_tracker_matches_static_joint_solution_and_rejects_repeated_frame():
    from evaluation.joint_pilot_state import joint_posterior
    model, context = _fixture()
    frame = replace(context.frame, frame_index=0)
    prior = context.condition.complex_cir[0]
    tracker = LegalStateTracker(prior, .1, 124, 1.)
    old_mean, old_cov = tracker.mean.clone(), tracker.covariance.clone()
    _, block = tracker.context(model, frame, 0., context.tail, 10.)
    mean, covariance = joint_posterior(old_mean, old_cov, [block])
    torch.testing.assert_close(tracker.mean, mean)
    torch.testing.assert_close(tracker.covariance, covariance)
    with pytest.raises(ValueError, match="禁止重复"):
        tracker.context(model, frame, 0., context.tail, 10.)


@pytest.mark.parametrize("accept", [False, True])
def test_memory_update_projects_commits_or_restores_without_changing_other_weights(monkeypatch, accept):
    import training.deterministic_adaptation as module
    model, context = _fixture()
    calls = []
    def guard(*args):
        calls.append(True)
        return (2. if len(calls) == 1 else (1. if accept else 3.)), .25
    monkeypatch.setattr(module, "validation_metrics", guard)
    adapter = DeterministicMemoryAdapter(model, DeterministicSettings(step_radius=.02, total_radius=.025))
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    metrics, before, candidate, retained, memory = adapter.step(context)
    assert metrics["accepted"] is accept
    assert metrics["delta_norm"] > 0
    assert metrics["delta_norm"] <= .020001
    assert metrics["retained_total_norm"] <= .025001
    assert metrics["gradient_norm"] > 0
    assert metrics["uses_rl"] is False
    if accept:
        torch.testing.assert_close(model.channel_state.memory.weight, memory, rtol=0, atol=0)
        torch.testing.assert_close(retained, candidate, rtol=0, atol=0)
    else:
        torch.testing.assert_close(model.channel_state.memory.weight, initial[adapter.parameter_name], rtol=0, atol=0)
        torch.testing.assert_close(retained, before, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        if name != adapter.parameter_name:
            torch.testing.assert_close(value, initial[name], rtol=0, atol=0)


def test_data_labels_truth_and_tail_poison_do_not_change_two_frame_control_or_state():
    model, context = _fixture()
    twins = [copy.deepcopy(model), copy.deepcopy(model)]
    adapters = [DeterministicMemoryAdapter(branch, DeterministicSettings()) for branch in twins]
    trackers = [LegalStateTracker(context.condition.complex_cir[0], .1, 124, .999) for _ in twins]
    tail = context.tail
    for index in range(2):
        frame = replace(context.frame, frame_index=index)
        symbols, bits = frame.tx_symbols.clone(), frame.bits.clone()
        symbols[frame.data_mask] = complex(float("nan"), float("nan"))
        bits[frame.data_mask] = ~bits[frame.data_mask]
        poisoned = replace(frame, tx_symbols=symbols, bits=bits,
            true_cir=torch.full_like(context.condition.complex_cir[0], float("nan")),
            tail_symbols=torch.full_like(tail, float("nan")))
        outcomes = []
        for branch, adapter, tracker, received in zip(twins, adapters, trackers, (frame, poisoned)):
            received = deployable_frame(received)
            physical_context, block = tracker.context(branch, received, 0., tail, 10.)
            outcomes.append(adapter.step(physical_context))
        assert outcomes[0][0] == outcomes[1][0]
        for left, right in zip(outcomes[0][1:], outcomes[1][1:]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        torch.testing.assert_close(trackers[0].mean, trackers[1].mean, rtol=0, atol=0)
        torch.testing.assert_close(trackers[0].covariance, trackers[1].covariance, rtol=0, atol=0)
        with torch.no_grad():
            values = torch.tanh(outcomes[0][3][-tail.numel():]/2)
            tail = torch.complex(values, torch.zeros_like(values))


def test_exact_state_diagnostic_uses_explicit_truth_restores_scope_and_preserves_weights():
    model, context = _fixture()
    snapshot = {name: value.clone() for name, value in model.state_dict().items()}
    diagnostic = ExactStateDiagnostic.from_model(model)
    truth = DiagnosticTruth(context.condition.complex_cir[0], torch.zeros(context.frame.rx_symbols.numel()))
    original = diagnostic.diagnostic_logits(context, truth)
    poisoned = replace(context, frame=replace(context.frame, true_cir=torch.zeros_like(truth.cir),
        tail_symbols=torch.full_like(context.tail, 999j)))
    torch.testing.assert_close(original, diagnostic.diagnostic_logits(poisoned, truth), atol=0, rtol=0)
    assert diagnostic._diagnostic_truth is None
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, snapshot[name], rtol=0, atol=0)
        torch.testing.assert_close(diagnostic.state_dict()[name], value, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="隔离快照"):
        context_logits(diagnostic, context)


def test_information_geometry_reports_nullspace_and_real_parameter_rank():
    geometry = information_metrics(torch.ones(3, 5, dtype=torch.complex64))
    assert geometry["rank"] == 1 and geometry["full_column_condition_number"] is None
    model, context = _fixture()
    block = corrected_block(model, context)
    metrics = memory_jacobian_metrics(model, context, block.design)
    assert metrics["columns"] == 2 * context.condition.complex_cir.numel()
    assert metrics["rank"] == metrics["columns"]


def test_audit_rejects_partial_snr_before_loading_checkpoint(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(schema_version="adaptability-development-audit-v1",
        role="development", seeds=[1], snrs_db=[5])), encoding="utf-8")
    with pytest.raises(ValueError, match="完整四 SNR"):
        run_audit(manifest, tmp_path / "output")


def test_audit_cli_rejects_legacy_condition_overrides_before_loading_manifest():
    result = subprocess.run([sys.executable, "online_train.py", "--method", "audit",
        "--audit-manifest", "missing.json", "--snrs", "10"], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 2
    assert "审计条件只能由版本化清单指定" in result.stderr


def test_fixed_candidate_probe_is_read_only_and_future_labels_only_change_scoring():
    model, context = _fixture()
    models = {"raw": copy.deepcopy(model), "guarded": copy.deepcopy(model)}
    original = context.frame
    with torch.no_grad():
        physical = context_logits(model, context)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    first = fixed_candidate_probe(models, context, physical, original, seed=1, snr=10, accepted=False)
    changed = replace(original, bits=~original.bits)
    second = fixed_candidate_probe(models, context, physical, changed, seed=1, snr=10, accepted=False)
    assert abs(first['ber_raw'] + second['ber_raw'] - 1.) < 1e-6
    assert first['future_metrics_used_for_control'] is False
    for branch in models.values():
        for name, value in branch.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_smoke_audit_runs_all_snrs_records_hashes_and_refuses_overwrite(tmp_path, monkeypatch):
    import evaluation.adaptability_audit as module
    model, context = _fixture()
    checkpoint, config_path = tmp_path / 'model.pt', tmp_path / 'config.json'
    checkpoint.write_bytes(b'fixture-checkpoint')
    config = dict(frame_len=model.config.frame_len, max_delay=model.config.max_delay,
        pilot_total=64, reward_pilot_total=16, receiver_lookahead_len=model.config.max_delay,
        model=model.config.to_dict(), rho_frame=.999, rho=.999)
    config_path.write_text(json.dumps(config), encoding='utf-8')
    monkeypatch.setattr(module, '_load_branch', lambda *args: (model,
        {'metrics': {'history': [], 'receiver_lookahead_len': model.config.max_delay}}))
    spec = dict(schema_version='adaptability-development-audit-v1', role='development',
        seeds=[222], snrs_db=[0,5,10,15], state_split='offline_train', from_session_start=True,
        acquisition_count=1, config=str(config_path), config_sha256=module.sha256(config_path),
        checkpoint=str(checkpoint), checkpoint_sha256=module.sha256(checkpoint),
        frames=128, smoke_frames=2, deterministic_update={'steps': 1},
        fixed_candidate_offsets=[0,1,8,32,128], information_history_frames=2, jacobian_frame_indices=[0])
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(spec), encoding='utf-8')
    result = run_audit(manifest, tmp_path / 'output', smoke=True)
    assert result['records'] == 8 and result['fixed_candidate_records'] == 8
    assert result['fixed_candidate_offsets_observed'] == [0, 1]
    assert result['core_confirmed'] is False and result['uses_rl'] is False
    assert result['raw_sha256'] == module.sha256(tmp_path / 'output' / 'frame_metrics.jsonl')
    with pytest.raises(ValueError, match='禁止覆盖'):
        run_audit(manifest, tmp_path / 'output', smoke=True)


@pytest.mark.parametrize("settings", [dict(steps=0), dict(learning_rate=float("nan")), dict(total_radius=-1)])
def test_invalid_update_budget_is_rejected(settings):
    with pytest.raises(ValueError):
        DeterministicSettings(**settings)
