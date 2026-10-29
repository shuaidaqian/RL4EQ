"""验证独立 Pilot 子窗口的遮罩、候选选择与 Data 波形隔离。"""

from dataclasses import replace

import pytest
import torch

from agent.cir_estimator import condition_from_cir
from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from env.frame_structure import FrameConfig, FrameGenerator
from scripts.replay_pilot_independent_subwindows import (
    _CANDIDATES,
    _make_candidate_models,
    _rank_candidates,
    _reward_gate,
    candidate_specs,
    split_adapt_subwindows,
)


def _fixture():
    torch.manual_seed(21)
    frame = FrameGenerator(
        FrameConfig(frame_len=128, total_pilot=64, layout="prefix", max_delay=4, reward_pilot_total=16),
        seed=9,
    ).generate(0)
    position = torch.arange(128, dtype=torch.float32)
    rx = frame.tx_symbols * torch.exp(1j * (0.2 + 2 * torch.pi * 0.001 * position))
    frame = replace(frame, rx_symbols=rx.to(torch.complex64))
    model = UnfoldedEqualizer(UnfoldedConfig(
        frame_len=128,
        max_delay=4,
        iterations=1,
        d_model=16,
        num_heads=4,
        enable_phase_correction_branch=True,
        phase_correction_initial_scale=1.0,
    ))
    model.attach_online_phase_trend_adapter()
    cir = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.complex64)
    condition = condition_from_cir(cir, 10)
    tail = torch.zeros(4, dtype=torch.complex64)
    return model.eval(), frame, condition, tail


def test_independent_pilot_views_expose_only_their_own_waveforms():
    _, frame, _, _ = _fixture()
    frame = replace(
        frame,
        true_cir=torch.ones(5, dtype=torch.complex64),
        tail_symbols=torch.ones(4, dtype=torch.complex64),
    )

    update, rank, reward, update_mask, rank_mask = split_adapt_subwindows(frame, 0.25)

    assert int(update_mask.sum()) == 36
    assert int(rank_mask.sum()) == 12
    assert not bool(torch.any(update_mask & rank_mask))
    assert torch.count_nonzero(update.rx_symbols[~update_mask]) == 0
    assert torch.count_nonzero(rank.rx_symbols[~rank_mask]) == 0
    assert torch.count_nonzero(reward.rx_symbols[~(frame.adapt_mask | frame.reward_mask)]) == 0
    assert torch.count_nonzero(update.bits[~update_mask]) == 0
    assert torch.count_nonzero(rank.bits[~rank_mask]) == 0
    assert torch.count_nonzero(reward.bits[frame.data_mask]) == 0
    assert torch.count_nonzero(update.tx_symbols[~update_mask]) == 0
    assert torch.count_nonzero(rank.tx_symbols[~rank_mask]) == 0
    assert torch.count_nonzero(reward.tx_symbols[frame.data_mask]) == 0
    assert update.true_cir is rank.true_cir is reward.true_cir is None
    assert update.tail_symbols is rank.tail_symbols is reward.tail_symbols is None
    assert update.reward_mask.sum() == rank.reward_mask.sum() == 0
    assert update.data_mask.sum() == rank.data_mask.sum() == reward.data_mask.sum() == 0


def test_subwindow_split_rejects_invalid_fraction():
    _, frame, _, _ = _fixture()
    for fraction in (0.0, 1.0, -0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="holdout_fraction"):
            split_adapt_subwindows(frame, fraction)


def test_independent_rank_selects_identity_when_all_candidates_tie_at_zero():
    rows = [
        {"candidate_name": name, "rank_pilot_bce_improvement": 0.0, "peft_delta_norm": 0.0 if name == "identity" else 0.1}
        for name in ("identity", "phase_trend_bce", "head_bce")
    ]
    assert _rank_candidates(rows)["candidate_name"] == "identity"


def test_candidate_ranking_can_use_independent_signal_reconstruction():
    rows = [
        {
            "candidate_name": "phase_bce",
            "rank_pilot_bce_improvement": 0.5,
            "rank_pilot_signal_improvement": -0.1,
            "peft_delta_norm": 0.01,
        },
        {
            "candidate_name": "phase_signal",
            "rank_pilot_bce_improvement": 0.1,
            "rank_pilot_signal_improvement": 0.2,
            "peft_delta_norm": 0.02,
        },
    ]

    assert _rank_candidates(rows, rank_score="bce")["candidate_name"] == "phase_bce"
    assert _rank_candidates(rows, rank_score="signal_reconstruction")["candidate_name"] == "phase_signal"


def test_multiscale_candidate_bank_matches_predeclared_safe_scales():
    candidates = candidate_specs("multiscale_signal", lr=0.005, steps=5, max_delta_norm=0.05)

    assert [row["name"] for row in candidates] == [
        "identity",
        "phase_signal_conservative",
        "phase_signal_standard",
        "phase_signal_moderate",
    ]
    assert [row["lr"] for row in candidates] == pytest.approx([0.0, 0.000625, 0.00125, 0.0025])
    assert [row["max_delta_norm"] for row in candidates] == pytest.approx([0.0, 0.0125, 0.05, 0.05])
    assert all(row["steps"] == 5 for row in candidates[1:])


def test_fixed_moderate_bank_has_one_update_candidate_and_identity():
    candidates = candidate_specs("fixed_moderate_signal", lr=0.005, steps=5, max_delta_norm=0.05)

    assert [row["name"] for row in candidates] == ["identity", "phase_signal_fixed_moderate"]
    assert candidates[1]["lr"] == pytest.approx(0.0025)
    assert candidates[1]["steps"] == 5
    assert candidates[1]["max_delta_norm"] == pytest.approx(0.05)


def test_reward_signal_guard_only_changes_reward_pilot_acceptance():
    common = {
        "snr": 10,
        "freeze_below_snr_db": 5,
        "reward_bce_improvement": 0.001,
        "min_reward_improvement": 0.0005,
        "reward_errors_before": 4,
        "reward_errors_after": 4,
        "reward_signal_improvement": -0.01,
    }

    assert _reward_gate(**common, require_signal_improvement=False)
    assert not _reward_gate(**common, require_signal_improvement=True)
    assert _reward_gate(
        **{**common, "reward_signal_improvement": 0.01},
        require_signal_improvement=True,
    )


def test_data_and_reward_waveforms_cannot_change_update_or_ranked_candidates():
    frozen, frame, condition, tail = _fixture()
    update, rank, _, _, _ = split_adapt_subwindows(frame, 0.25)
    candidates = candidate_specs("objective", lr=0.005, steps=2)
    first_rows, first_state = _make_candidate_models(
        frozen, update, rank, condition, tail, candidate_bank=candidates, lr=0.005, steps=2,
    )

    hidden = frame.reward_mask | frame.data_mask
    changed_rx = frame.rx_symbols.clone()
    changed_rx[hidden] = torch.complex(
        torch.full_like(changed_rx[hidden].real, 123.0),
        torch.full_like(changed_rx[hidden].imag, -87.0),
    )
    changed_bits = frame.bits.clone()
    changed_bits[hidden] = ~changed_bits[hidden]
    changed_tx = frame.tx_symbols.clone()
    changed_tx[hidden] = -changed_tx[hidden]
    poisoned = replace(frame, rx_symbols=changed_rx, bits=changed_bits, tx_symbols=changed_tx)
    update_poisoned, rank_poisoned, _, _, _ = split_adapt_subwindows(poisoned, 0.25)
    second_rows, second_state = _make_candidate_models(
        frozen, update_poisoned, rank_poisoned, condition, tail,
        candidate_bank=candidates, lr=0.005, steps=2,
    )

    assert [row["candidate_name"] for row in first_rows] == [item["name"] for item in _CANDIDATES]
    assert [row["rank_pilot_bce_improvement"] for row in first_rows] == pytest.approx(
        [row["rank_pilot_bce_improvement"] for row in second_rows], abs=1e-8,
    )
    assert first_state["selected"]["candidate_name"] == second_state["selected"]["candidate_name"]
    for name in first_state["models"]:
        for key, value in first_state["models"][name].state_dict().items():
            assert torch.equal(value, second_state["models"][name].state_dict()[key])
