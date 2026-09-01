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
