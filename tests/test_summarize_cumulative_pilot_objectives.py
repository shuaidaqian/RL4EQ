"""验证累计 Pilot objective 汇总的配对和隔离契约。"""

import json

import pytest

from scripts.summarize_cumulative_pilot_objectives import summarize


def _payload(objective, reward, ber, *, waveform=False):
    return {
        "metric": "cumulative_pilot_state_mapping_future_frame_replay",
        "checkpoint": "fixed.pt",
        "arguments": {
            "adapt_objective": objective,
            "snrs": [10],
            "seeds": [4],
            "frames": 8,
            "train_frame_counts": [1],
            "validation_frames": 2,
            "reward_score": "joint_relative",
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
            "seed": 4,
            "train_frame_count": 1,
            "test_frame_index": 7,
            "gate_reward_improvements": [reward, reward],
            "adapt_bce_before": 0.2,
            "adapt_bce_after": 0.19,
            "initial_reward_gate_accepted": True,
            "candidate_data_ber_improvement": ber,
            "candidate_data_bce_improvement": 0.01,
            "data_ber_improvement": ber,
            "data_bce_improvement": 0.01,
            "candidate_data_logits_abs_mean_delta": 0.02,
            "candidate_data_soft_abs_mean_delta": 0.01,
            "data_labels_used_online": False,
            "data_waveform_used_online": waveform,
            "diagnostic_uses_data_labels": True,
        }],
    }


def test_summary_pairs_pilot_reward_with_future_data_only_afterward(tmp_path):
    paths = []
    for index, payload in enumerate((
        _payload("bce", 0.2, 0.01),
        _payload("signal_reconstruction", 0.1, -0.01),
    )):
        path = tmp_path / f"{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)

    result = summarize(paths)
    assert result["data_metrics_used_online"] is False
    assert result["online_data_waveform_isolation_passed"] is True
    assert result["candidate_count"] == 2
    ranking = result["within_trial_ranking"][0]
    assert ranking["reward_vs_candidate_data_ber"]["agreement"] == 1.0
    assert ranking["best_by_pilot_reward"] == "bce"
    assert ranking["best_by_data_ber_eval_only"] == "bce"


def test_summary_rejects_any_data_waveform_online_flag(tmp_path):
    path = tmp_path / "leak.json"
    path.write_text(json.dumps(_payload("bce", 0.2, 0.01, waveform=True)), encoding="utf-8")
    with pytest.raises(ValueError, match="Data 波形隔离"):
        summarize([path])


def test_summary_rejects_mismatched_replay_conditions(tmp_path):
    first = _payload("bce", 0.2, 0.01)
    second = _payload("joint", 0.1, -0.01)
    second["arguments"]["pilot_total"] = 160
    paths = []
    for index, payload in enumerate((first, second)):
        path = tmp_path / f"{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)
    with pytest.raises(ValueError, match="固定条件不一致"):
        summarize(paths)
