# -*- coding: utf-8 -*-
"""当前 Adapt Pilot 驱动物理相位/CFO 状态跟踪的契约测试。"""

from types import SimpleNamespace

import pytest
import torch


def _pilot_frame(cfo: float, phase0: float = 0.2, noise: float = 0.0):
    length = 96
    tx = torch.where(
        torch.arange(length) % 2 == 0,
        torch.ones(length, dtype=torch.complex64),
        -torch.ones(length, dtype=torch.complex64),
    )
    indices = torch.arange(length, dtype=torch.float32)
    phase = phase0 + 2.0 * torch.pi * float(cfo) * indices
    rx = tx * torch.exp(1j * phase).to(torch.complex64)
    if noise:
        generator = torch.Generator().manual_seed(123)
        rx = rx + noise * (
            torch.randn(length, generator=generator)
            + 1j * torch.randn(length, generator=generator)
        ).to(torch.complex64)
    return SimpleNamespace(
        rx_symbols=rx,
        tx_symbols=tx,
        adapt_mask=torch.ones(length, dtype=torch.bool),
        reward_mask=torch.zeros(length, dtype=torch.bool),
        data_mask=torch.zeros(length, dtype=torch.bool),
        model_region_ids=torch.ones(length, dtype=torch.long),
        receiver_view=lambda: SimpleNamespace(
            rx_symbols=rx,
            adapt_symbols=tx,
            adapt_mask=torch.ones(length, dtype=torch.bool),
            model_region_ids=torch.ones(length, dtype=torch.long),
        ),
    )


def test_adapt_pilot_recovers_current_residual_cfo_and_phase():
    from agent.cir_estimator import PilotPhysicalState, track_pilot_physical_state

    frame = _pilot_frame(cfo=0.00072, phase0=0.31)
    cir = torch.zeros(41, dtype=torch.complex64)
    cir[0] = 1.0 + 0.0j
    state = track_pilot_physical_state(
        frame,
        cir,
        torch.zeros(40, dtype=torch.complex64),
        previous=PilotPhysicalState(),
        cfo_limit=0.0012,
        smoothing=1.0,
    )

    assert state.confidence > 0.5
    assert state.cfo_cycles_per_symbol == pytest.approx(0.00072, abs=2e-5)
    assert state.phase0 == pytest.approx(0.31, abs=2e-3)


def test_low_confidence_pilot_keeps_previous_physical_state():
    from agent.cir_estimator import PilotPhysicalState, track_pilot_physical_state

    frame = _pilot_frame(cfo=0.00072, phase0=0.31, noise=2.0)
    cir = torch.zeros(41, dtype=torch.complex64)
    cir[0] = 1.0 + 0.0j
    previous = PilotPhysicalState(phase0=0.12, cfo_cycles_per_symbol=0.0004, confidence=0.8)
    state = track_pilot_physical_state(
        frame,
        cir,
        torch.zeros(40, dtype=torch.complex64),
        previous=previous,
        cfo_limit=0.0012,
        smoothing=1.0,
        min_confidence=0.95,
    )

    assert state.confidence < 0.95
    assert state.phase0 == pytest.approx(previous.phase0)
    assert state.cfo_cycles_per_symbol == pytest.approx(previous.cfo_cycles_per_symbol)


def test_pilot_online_state_records_joint_tracking_without_data_labels():
    import compare
    from agent.cir_estimator import PilotPhysicalState

    assert hasattr(compare, "PilotOnlineMethodState")
    state = PilotPhysicalState()
    assert state.data_labels_used_online is False


@pytest.mark.parametrize("mode", ["phase_only", "cir_only"])
def test_state_tracking_component_modes_isolate_physical_state(mode, monkeypatch):
    from dataclasses import replace

    import agent.state_tracking as tracking
    from agent.cir_estimator import PilotPhysicalState
    from agent.state_tracking import initial_state, update_from_adapt_pilot

    frame = _pilot_frame(cfo=0.0004)
    prior_cir = torch.zeros(41, dtype=torch.complex64)
    prior_cir[0] = 1.0
    prior = initial_state(prior_cir, torch.zeros(40, dtype=torch.complex64))
    prior = replace(
        prior,
        phase=PilotPhysicalState(phase0=0.1, cfo_cycles_per_symbol=0.0002, confidence=0.8),
    )
    candidate_cir = prior_cir.clone()
    candidate_cir[1] = 0.2
    observed_phase = PilotPhysicalState(phase0=0.3, cfo_cycles_per_symbol=0.0006, confidence=0.9)
    monkeypatch.setattr(tracking, "track_pilot_physical_state", lambda *args, **kwargs: observed_phase)
    monkeypatch.setattr(tracking, "pilot_sparse_cir_update", lambda *args, **kwargs: candidate_cir)

    update = update_from_adapt_pilot(
        frame,
        prior,
        snr_db=10.0,
        track_cir=mode != "phase_only",
        track_phase=mode != "cir_only",
    )

    assert update.committed
    if mode == "phase_only":
        assert torch.equal(update.candidate.cir, prior.cir)
        assert update.candidate.phase.cfo_cycles_per_symbol == pytest.approx(0.0006)
    else:
        assert torch.equal(update.candidate.cir, candidate_cir)
        assert update.candidate.phase.cfo_cycles_per_symbol == pytest.approx(0.0002)
    assert torch.equal(update.candidate.tail.mean, prior.tail.mean)


def test_uncertainty_aware_soft_tail_downweights_low_confidence_symbols():
    from agent.tail_state import SoftTailState, update_soft_tail_state

    previous = SoftTailState(
        mean=torch.ones(4, dtype=torch.complex64),
        variance=torch.zeros(4),
    )
    logits = torch.zeros(4)
    updated = update_soft_tail_state(logits, previous=previous, alpha=1.0)

    assert torch.all(updated.confidence < previous.confidence)
    assert torch.all(updated.mean.real > 0.5)
    assert torch.all(updated.mean.real < 1.0)


def test_uncertainty_aware_soft_tail_high_confidence_can_update():
    from agent.tail_state import SoftTailState, update_soft_tail_state

    previous = SoftTailState(
        mean=-torch.ones(4, dtype=torch.complex64),
        variance=torch.ones(4),
    )
    updated = update_soft_tail_state(torch.full((4,), 8.0), previous=previous, alpha=1.0)

    assert torch.all(updated.confidence > 0.95)
    assert torch.all(updated.mean.real > 0.9)


def test_soft_tail_update_uses_only_adapt_pilot_observations():
    from agent.state_tracking import TailBelief, initial_state, update_tail_from_adapt_pilot

    length = 96
    tx = torch.ones(length, dtype=torch.complex64)
    mask = torch.zeros(length, dtype=torch.bool)
    mask[:48] = True
    prior_tail = torch.zeros(40, dtype=torch.complex64)
    prior_tail[-1] = -1.0
    cir = torch.zeros(41, dtype=torch.complex64)
    cir[1] = 1.0
    rx = torch.zeros(length, dtype=torch.complex64)
    rx[0] = 1.0
    rx[1:48] = 1.0
    rx[48:] = 1.0

    def make_frame(observations):
        return type(
            "PilotFrame",
            (),
            {
                "adapt_mask": mask,
                "receiver_view": lambda self: type(
                    "ReceiverView",
                    (),
                    {"rx_symbols": observations, "adapt_symbols": tx, "adapt_mask": mask},
                )(),
            },
        )()

    posterior = initial_state(cir, prior_tail)
    posterior = type(posterior)(
        posterior.cir,
        posterior.cir_variance,
        posterior.phase,
        TailBelief(prior_tail, torch.ones(40), 0.0),
        posterior.innovation_variance,
        posterior.confidence,
    )
    changed_data = rx.clone()
    changed_data[48:] = torch.randn(48).to(torch.complex64) * 100.0
    estimate_a = update_tail_from_adapt_pilot(make_frame(rx), posterior, snr_db=30.0, minimum_rows=1)
    estimate_b = update_tail_from_adapt_pilot(
        make_frame(changed_data), posterior, snr_db=30.0, minimum_rows=1
    )

    assert torch.equal(estimate_a.mean, estimate_b.mean)
    assert torch.equal(estimate_a.variance, estimate_b.variance)
    assert estimate_a.mean[-1].real > 0.9


def test_rejected_joint_candidate_keeps_entire_prior_including_tail():
    from dataclasses import replace

    from agent.cir_estimator import PilotPhysicalState
    from agent.state_tracking import (
        PilotStateUpdate,
        commit_if_safe,
        initial_state,
        is_safe_to_commit,
    )

    cir = torch.zeros(41, dtype=torch.complex64)
    cir[0] = 1.0
    prior = initial_state(cir, torch.full((40,), -0.25, dtype=torch.complex64))
    candidate = replace(
        prior,
        cir=torch.zeros_like(prior.cir),
        phase=PilotPhysicalState(phase0=0.7, confidence=0.1),
        tail=replace(
            prior.tail,
            mean=torch.full_like(prior.tail.mean, 0.8),
        ),
    )
    update = PilotStateUpdate(prior, candidate, True, "low_confidence", 224)

    assert not is_safe_to_commit(update, minimum_confidence=0.15)
    committed = commit_if_safe(update, minimum_confidence=0.15)
    assert torch.equal(committed.cir, prior.cir)
    assert committed.phase.phase0 == pytest.approx(prior.phase.phase0)
    assert torch.equal(committed.tail.mean, prior.tail.mean)
