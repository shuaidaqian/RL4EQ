# -*- coding: utf-8 -*-
"""配对统计不得接受缺失轨迹、不同波形或在线信息隔离失败的记录。"""

import pytest

from scripts.summarize_calibrated_compare import METHODS, PROPOSED, summarize_rows


def _rows():
    rows = []
    for snr in [0, 5, 10, 15]:
        for seed in [1, 2]:
            for frame in [1, 2, 3]:
                for method in METHODS:
                    row = {
                        "snr_db": snr, "seed": seed, "frame": frame, "method": method,
                        "ber_data": 0.19 if method == PROPOSED else 0.20,
                        "environment_seed": 50_000 + seed, "rx_sha256": "a" * 64,
                        "traditional_calibration_sha256": "c" * 64,
                        "data_labels_used_online": False, "data_waveform_used_online": False,
                        "traditional_calibration_applied": True, "uses_reward_or_data_labels": False,
                        "uses_neural_network": False, "uses_rl": False,
                        "policy_action_name": "peft_head_fast", "peft_frame_accepted": True,
                        "data_logits_abs_delta_mean_eval_only": 0.02,
                    }
                    rows.append(row)
    return rows


def test_summary_uses_independent_seed_units_and_separates_late_gain():
    result = summarize_rows(_rows())
    assert result["paired_frames"] == 24
    assert result["main_aggregation_allowed"] is False
    for item in result["per_snr"]:
        assert item["online_vs_frozen_gain"]["mean"] == pytest.approx(0.01)
        assert item["online_vs_frozen_gain"]["seeds"] == 2
        assert item["late_minus_early_online_gain"]["mean"] == pytest.approx(0)


def test_summary_accepts_explicit_full_rx_but_rejects_mixed_or_unlabelled_visibility():
    rows = _rows()
    proposed = [row for row in rows if row["method"] == PROPOSED]
    for row in proposed:
        row.update(reward_receiver_input="full_rx", data_waveform_used_online=True)
    result = summarize_rows(rows)
    assert result["information_audit"]["data_waveform_used_online"] is True
    proposed[0].update(reward_receiver_input="pilot_only", data_waveform_used_online=False)
    with pytest.raises(ValueError, match="不同 Reward"):
        summarize_rows(rows)
    proposed[0]["data_waveform_used_online"] = True
    with pytest.raises(ValueError, match="隔离声明"):
        summarize_rows(rows)


def test_summary_separates_policy_updates_retained_memory_and_rejected_windows():
    rows = _rows()
    for row in rows:
        if row["method"] == PROPOSED:
            row.update(policy_loss=.1 if row["frame"] == 3 else None,
                       policy_window_closed=row["frame"] in {2, 3},
                       reward=.02 if row["frame"] == 2 else -.01,
                       window_reward_rollback=row["frame"] == 3,
                       trajectory_peft_delta_norm_after_reward=.4 if row["frame"] == 2 else 0.)
    for item in summarize_rows(rows)["per_snr"]:
        assert item["ppo_update_frames"] == 2
        assert item["closed_windows"] == 4
        assert item["positive_reward_windows"] == 2
        assert item["window_rollbacks"] == 2
        assert item["max_retained_trajectory_peft_delta_norm"] == pytest.approx(.4)
    # 历史结果缺失累计参数日志时，不能把未记录误报为零偏移。
    assert all(item["max_retained_trajectory_peft_delta_norm"] is None
               for item in summarize_rows(_rows())["per_snr"])


@pytest.mark.parametrize("failure", ["duplicate", "missing_method", "different_rx", "hidden_labels", "mixed_protocol"])
def test_summary_rejects_invalid_pairing_or_online_information(failure):
    rows = _rows()
    if failure == "duplicate":
        rows.append(dict(rows[0]))
    elif failure == "missing_method":
        rows.pop(0)
    elif failure == "different_rx":
        rows[0]["rx_sha256"] = "b" * 64
    elif failure == "hidden_labels":
        next(row for row in rows if row["method"] == PROPOSED)["data_labels_used_online"] = True
    else:
        rows[0]["traditional_calibration_sha256"] = "d" * 64
    with pytest.raises(ValueError):
        summarize_rows(rows)
