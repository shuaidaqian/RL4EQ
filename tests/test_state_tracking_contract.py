# -*- coding: utf-8 -*-

from dataclasses import replace

import torch

from agent.pilot_risk import compute_pilot_risk
from agent.state_tracking import (
    TailBelief,
    initial_state,
    update_from_adapt_pilot,
    update_tail_from_adapt_pilot,
)
from env.frame_structure import Frame


def _frame() -> Frame:
    bits = torch.tensor([True, False, True, False, True, False, True, False])
    tx = torch.complex(bits.float() * 2.0 - 1.0, torch.zeros(8))
    adapt = torch.tensor([True, True, True, True, False, False, False, False])
    reward = torch.tensor([False, False, False, False, True, True, False, False])
    return Frame(
        frame_index=0,
        bits=bits,
        tx_symbols=tx,
        rx_symbols=tx.clone(),
        adapt_mask=adapt,
        reward_mask=reward,
        data_mask=~(adapt | reward),
        model_region_ids=torch.zeros(8, dtype=torch.long),
        adapt_block_lengths=(4,),
        tail_symbols=torch.zeros(2, dtype=torch.complex64),
        true_cir=torch.tensor([1 + 0j, 0.2 + 0j]),
    )


def test_state_update_is_invariant_to_data_labels_and_waveform():
    frame = _frame()
    prior = initial_state(torch.tensor([1 + 0j, 0.2 + 0j]), torch.zeros(2, dtype=torch.complex64))
    first = update_from_adapt_pilot(frame, prior, snr_db=10.0, min_confidence=0.0)
    changed = replace(
        frame,
        bits=frame.bits.clone().logical_not(),
        rx_symbols=torch.where(frame.data_mask, torch.tensor(3 + 2j), frame.rx_symbols),
    )
    second = update_from_adapt_pilot(changed, prior, snr_db=10.0, min_confidence=0.0)
    assert torch.allclose(first.candidate.cir, second.candidate.cir, atol=1e-6, rtol=1e-6)
    assert first.candidate.phase == second.candidate.phase


def test_tail_belief_clone_and_risk_do_not_read_data_labels():
    frame = _frame()
    posterior = initial_state(torch.tensor([1 + 0j, 0.2 + 0j]), torch.zeros(1, dtype=torch.complex64))
    risk_a = compute_pilot_risk(frame, posterior)
    changed = replace(frame, bits=frame.bits.logical_not())
    risk_b = compute_pilot_risk(changed, posterior)
    assert risk_a == risk_b
    assert risk_a.pilot_subset == "reward"
    assert risk_a.online_waveform_visibility == "reward_pilot_waveform_only"
    updated = update_tail_from_adapt_pilot(frame, posterior, snr_db=10.0)
    changed_frame = replace(
        frame,
        rx_symbols=torch.where(frame.data_mask, torch.tensor(99 + 77j), frame.rx_symbols),
        bits=frame.bits.logical_not(),
    )
    changed_updated = update_tail_from_adapt_pilot(changed_frame, posterior, snr_db=10.0)
    assert updated.mean.shape == posterior.tail.mean.shape
    assert torch.equal(updated.mean, changed_updated.mean)
    assert torch.equal(updated.variance, changed_updated.variance)


def test_reward_risk_is_invariant_to_adapt_pilot_but_adapt_risk_is_explicit():
    frame = _frame()
    posterior = initial_state(torch.tensor([1 + 0j, 0.2 + 0j]), torch.zeros(1, dtype=torch.complex64))
    changed = replace(
        frame,
        rx_symbols=torch.where(frame.adapt_mask, torch.tensor(8 - 3j), frame.rx_symbols),
    )
    reward_a = compute_pilot_risk(frame, posterior)
    reward_b = compute_pilot_risk(changed, posterior)
    assert reward_a == reward_b
    adapt = compute_pilot_risk(frame, posterior, pilot_subset="adapt")
    assert adapt.pilot_subset == "adapt"
    assert adapt.online_waveform_visibility == "reward_pilot_waveform_only"


def test_risk_rejects_unknown_pilot_subset():
    frame = _frame()
    posterior = initial_state(torch.tensor([1 + 0j, 0.2 + 0j]), torch.zeros(1, dtype=torch.complex64))
    try:
        compute_pilot_risk(frame, posterior, pilot_subset="data")
    except ValueError:
        pass
    else:
        raise AssertionError("代理风险不允许把 Data 作为观测子集")


def test_tail_belief_rejects_shape_mismatch():
    try:
        TailBelief(torch.zeros(2, dtype=torch.complex64), torch.zeros(3), 0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("形状不一致的 tail belief 必须被拒绝")
