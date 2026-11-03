# -*- coding: utf-8 -*-
"""长会话不能用缺帧轨迹、即时动作前对照或单 seed 区间替代独立配对。"""

import pytest

from scripts.summarize_continuous_rl_compare import FROZEN, PROPOSED, summarize_continuous_rows


def _rows():
    return [{
        "method": method, "snr_db": 10, "seed": 1, "environment_seed": 50001,
        "frame": frame, "channel_frame_index": frame - 1, "pilot_layout": "prefix",
        "rx_sha256": "a" * 64, "data_labels_used_online": False,
        "reward_receiver_input": "full_rx", "data_waveform_used_online": True,
        "ber_data": .2 if method == FROZEN else .2 - frame * .001,
        "policy_action_name": "peft_memory_map_small", "policy_loss": .01 if frame == 6 else None,
        "policy_window_closed": frame % 2 == 0, "reward": .01,
        "window_reward_rollback": False, "trajectory_peft_delta_norm_after_reward": frame * .01,
    } for frame in range(1, 7) for method in (FROZEN, PROPOSED)]


def _summarize(rows):
    return summarize_continuous_rows(rows, expected_frames=6, segment_frames=2, frame_seconds=.1)


def test_continuous_segments_use_independent_frozen_and_do_not_invent_single_seed_ci():
    result = _summarize(_rows())
    item = result["per_snr"][0]
    assert result["duration_seconds"] == pytest.approx(.6)
    assert item["online_vs_frozen_gain"]["mean"] == pytest.approx(.0035)
    assert item["online_vs_frozen_gain"]["ci95"] is None
    assert item["late_minus_early_online_gain"]["mean"] == pytest.approx(.004)
    assert [segment["online_vs_frozen_gain"]["mean"] for segment in item["segments"]] == pytest.approx([.0015, .0035, .0055])
    assert item["ppo_update_frames"] == 1


@pytest.mark.parametrize("problem", ["missing", "duplicate", "physical_age", "rx", "labels", "method"])
def test_continuous_summary_rejects_incomplete_or_invalid_controls(problem):
    rows = _rows()
    if problem == "missing":
        rows = rows[2:]
    elif problem == "duplicate":
        rows.append(rows[0])
    elif problem == "physical_age":
        rows[-1]["channel_frame_index"] = 0
    elif problem == "rx":
        rows[-1]["rx_sha256"] = "b" * 64
    elif problem == "labels":
        rows[-1]["data_labels_used_online"] = True
    else:
        rows[-1]["method"] = "即时动作前对照"
    with pytest.raises(ValueError):
        _summarize(rows)


@pytest.mark.parametrize("problem", ["none", "length", "hash", "missing_hash", "age", "version"])
def test_continuous_summary_audits_joint_received_window(problem):
    rows = _rows()
    for row in rows:
        row.update(lookahead_len=4, lookahead_protocol_version="cross-frame-prefix-v1",
                   lookahead_rx_sha256="c" * 64, lookahead_frame_index=row["frame"])
    result = _summarize(rows)
    assert result["receiver_window_audit"]["shared_lookahead_rx_sha256_passed"]
    if problem == "none":
        rows[-1].update(lookahead_len=0, lookahead_protocol_version="none",
                        lookahead_rx_sha256=None, lookahead_frame_index=None)
    elif problem == "length":
        rows[-1]["lookahead_len"] = 3
    elif problem == "hash":
        rows[-1]["lookahead_rx_sha256"] = "d" * 64
    elif problem == "missing_hash":
        rows[-1].pop("lookahead_rx_sha256")
    elif problem == "age":
        rows[-1]["lookahead_frame_index"] = 100
    else:
        rows[-1]["lookahead_protocol_version"] = "none"
    with pytest.raises(ValueError):
        _summarize(rows)
