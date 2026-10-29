"""验证不同 Pilot Reward 评分的配对汇总与 Data 隔离。"""

import json

import pytest

from scripts.summarize_pilot_reward_score_replay import summarize


def _payload(score, accepted, ber, *, waveform=False):
    return {
        "metric": "cumulative_pilot_state_mapping_future_frame_replay",
        "checkpoint": "same-checkpoint.pt",
        "arguments": {
            "reward_score": score,
            "output": f"logs/{score}.json",
            "adapt_objective": "signal_reconstruction",
            "snrs": [10],
            "seeds": [2],
            "frames": 8,
            "train_frame_counts": [1],
            "validation_frames": 2,
            "update_scope": "state_mapping",
            "reward_aggregation": "aggregate",
            "max_delta_norm": 0.002,
            "lr": 0.0005,
            "steps": 1,
            "delay": 116,
            "pilot_total": 256,
            "reward_pilot_total": 32,
            "gap_seconds": 600,
            "condition_source": "pilot_cir_phase",
            "joint_bce_weight": 0.5,
        },
        "rows": [{
            "snr_db": 10,
            "seed": 2,
            "train_frame_count": 1,
            "test_frame_index": index,
            "initial_reward_gate_accepted": accepted,
            "candidate_data_ber_improvement": ber,
            "candidate_data_bce_improvement": 0.1,
            "data_ber_improvement": ber if accepted else 0.0,
            "data_bce_improvement": 0.1 if accepted else 0.0,
            "candidate_data_logits_abs_mean_delta": 0.03,
            "candidate_data_soft_abs_mean_delta": 0.01,
            "data_labels_used_online": False,
            "data_waveform_used_online": waveform,
        } for index in (5, 6, 7)],
    }


def _write_payload(tmp_path, name, payload):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_reward_score_summary_uses_matched_trials_and_future_data(tmp_path):
    paths = [
        _write_payload(tmp_path, "bce", _payload("bce_relative", True, 0.01)),
        _write_payload(tmp_path, "signal", _payload("signal_reconstruction", False, 0.01)),
    ]
    result = summarize(paths)
    assert result["candidate_data_used_for_online_decision"] is False
    assert result["online_data_waveform_isolation_passed"] is True
    assert result["score_summary"][0]["acceptance_fraction"] == 1.0
    assert result["score_summary"][0]["accepted_data_ber_positive_fraction"] == 1.0
    assert result["score_summary"][1]["acceptance_fraction"] == 0.0
    assert result["score_summary"][1]["rejected_but_data_ber_positive_fraction"] == 1.0
    assert len(result["paired_acceptance"]) == 2


def test_reward_score_summary_rejects_data_waveform_access(tmp_path):
    path = _write_payload(tmp_path, "leak", _payload("bce_relative", True, 0.01, waveform=True))
    with pytest.raises(ValueError, match="Data 波形隔离"):
        summarize([path])


def test_reward_score_summary_requires_matched_protocols(tmp_path):
    first = _payload("bce_relative", True, 0.01)
    second = _payload("signal_reconstruction", False, 0.01)
    second["arguments"]["pilot_total"] = 160
    paths = [
        _write_payload(tmp_path, "bce", first),
        _write_payload(tmp_path, "signal", second),
    ]
    with pytest.raises(ValueError, match="固定条件不一致"):
        summarize(paths)
