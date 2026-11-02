# -*- coding: utf-8 -*-
"""窗口反馈复核必须重建真实成本，并与 Data 事后指标无关。"""

import copy

import pytest

from scripts.diagnose_window_reward_accounting import PROPOSED, summarize_reward_windows


def _rows():
    return [{
        "method": PROPOSED, "snr_db": 5, "seed": 1, "frame": frame,
        "data_labels_used_online": False, "policy_action_name": "peft_memory_map_small",
        "policy_window_closed": frame == 2, "window_size": 2,
        "reward_pilot_loss_before": 1., "reward_pilot_loss_after": .999,
        "reward_pilot_ber_before": 0., "reward_pilot_ber_after": 0.,
        "peft_delta_norm": .2, "action_delta_norm": 0., "reward": -.001 if frame == 2 else 0.,
        "peft_frame_accepted": True, "window_reward_rollback": frame == 2,
        "trajectory_peft_delta_norm_after_reward": 0. if frame == 2 else .2,
    } for frame in (1, 2)]


def test_reconstruction_distinguishes_attempts_frame_acceptance_and_terminal_memory():
    result = summarize_reward_windows(_rows())
    item = result["windows"][0]
    assert item["action_cost"] == pytest.approx(.002)
    assert item["pilot_utility_before_cost"] == pytest.approx(.001)
    assert item["positive_pilot_utility_cancelled_by_cost"]
    assert item["accepted_candidate_frames"] == 2
    assert item["terminal_retained_delta_norm"] == 0.
    assert result["partial_windows"] == []
    rows = copy.deepcopy(_rows())
    for row in rows:
        row.update(ber_data=-1000, data_bce_improvement_eval_only=float("nan"))
    assert summarize_reward_windows(rows) == result


def test_incomplete_windows_are_reported_without_fabricated_reward():
    result = summarize_reward_windows(_rows()[:1])
    assert result["windows"] == []
    assert result["partial_windows"][0]["unclosed_frames"] == 1


@pytest.mark.parametrize("problem", ["reward", "frames", "action", "labels"])
def test_invalid_accounting_or_information_declarations_are_rejected(problem):
    rows = _rows()
    if problem == "reward":
        rows[-1]["reward"] = 1.
    elif problem == "frames":
        rows[-1]["frame"] = 3
    elif problem == "action":
        rows[-1]["policy_action_name"] = "identity"
    else:
        rows[-1]["data_labels_used_online"] = True
    with pytest.raises(ValueError):
        summarize_reward_windows(rows)
