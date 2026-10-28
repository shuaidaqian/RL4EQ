# -*- coding: utf-8 -*-

import pytest


def _frame(seed: int, snr: float, frame: int, *, data_waveform_used_online: bool = False):
    ranking = [
        {
            "name": "small",
            "score": 0.1,
            "data_bce_improvement_eval_only": 0.01,
            "data_ber_improvement_eval_only": 0.02,
            "data_logits_abs_delta_mean_eval_only": 0.2,
            "data_logits_direction_change_rate_eval_only": 0.02,
            "data_soft_output_delta_mean_eval_only": 0.03,
        },
        {
            "name": "large",
            "score": 0.2,
            "data_bce_improvement_eval_only": 0.03,
            "data_ber_improvement_eval_only": 0.04,
            "data_logits_abs_delta_mean_eval_only": 0.4,
            "data_logits_direction_change_rate_eval_only": 0.04,
            "data_soft_output_delta_mean_eval_only": 0.05,
        },
    ]
    return {
        "method": "Pilot-Driven Online Adaptation",
        "snr_db": snr,
        "seed": seed,
        "frame": frame,
        "online_update_candidate": "large",
        "data_waveform_used_online": data_waveform_used_online,
        "data_labels_used_online": False,
        "reward_pilot_candidate_ranking": ranking,
    }


def test_summary_reports_pilot_top1_and_data_isolation():
    from scripts.summarize_pilot_ranking_replay import summarize

    source_rows = [_frame(seed, 10.0, 1) for seed in (0, 1)]
    candidate_rows = []
    for source in source_rows:
        for candidate in source["reward_pilot_candidate_ranking"]:
            candidate_rows.append(
                {
                    "snr_db": source["snr_db"],
                    "seed": source["seed"],
                    "frame": source["frame"],
                    "candidate_name": candidate["name"],
                    "pilot_score": candidate["score"],
                    "data_bce_improvement": candidate["data_bce_improvement_eval_only"],
                    "data_ber_improvement": candidate["data_ber_improvement_eval_only"],
                    "data_logits_abs_delta": candidate["data_logits_abs_delta_mean_eval_only"],
                    "data_logits_direction_change_rate": candidate["data_logits_direction_change_rate_eval_only"],
                    "data_soft_output_delta": candidate["data_soft_output_delta_mean_eval_only"],
                }
            )
    result = summarize(candidate_rows, source_rows)

    assert result["trial_count"] == 2
    assert result["candidate_count"] == 4
    assert result["pilot_top1_matches_data_bce_best_fraction"] == pytest.approx(1.0)
    assert result["pilot_top1_matches_data_ber_best_fraction"] == pytest.approx(1.0)
    assert result["actual_selected"]["mean_data_ber_improvement"] == pytest.approx(0.04)
    assert result["online_data_isolation_passed"] is True
    assert result["data_metrics_used_for_online_decision"] is False


def test_summary_rejects_data_waveform_used_online(tmp_path):
    from scripts.summarize_pilot_ranking_replay import _load_rows

    path = tmp_path / "frame_metrics.jsonl"
    path.write_text(
        __import__("json").dumps(_frame(0, 10.0, 1, data_waveform_used_online=True)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Data 波形"):
        _load_rows([str(path)])


@pytest.mark.parametrize("flag,value", [
    ("data_labels_used_online", None),
    ("data_waveform_used_online", None),
    ("data_labels_used_online", "false"),
    ("data_waveform_used_online", 0),
    ("data_waveform_used_online", True),
])
def test_load_checks_isolation_even_without_candidates(tmp_path, flag, value):
    import json

    from scripts.summarize_pilot_ranking_replay import _load_rows

    row = _frame(0, 10.0, 1)
    row["reward_pilot_candidate_ranking"] = []
    if value is None:
        del row[flag]
    else:
        row[flag] = value
    path = tmp_path / "frame_metrics.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        _load_rows([str(path)])


def test_within_frame_ordering_exposes_misleading_pooled_correlation():
    from evaluation.metrics import spearman_reward_data
    from scripts.summarize_pilot_ranking_replay import _within_trial_ordering

    trials = {
        ("日志", 5.0, 1, 1): [
            {"pilot_score": 0.0, "data_ber_improvement": 1.0},
            {"pilot_score": 1.0, "data_ber_improvement": 0.0},
        ],
        ("日志", 5.0, 2, 1): [
            {"pilot_score": 10.0, "data_ber_improvement": 11.0},
            {"pilot_score": 11.0, "data_ber_improvement": 10.0},
        ],
    }
    rows = [row for candidates in trials.values() for row in candidates]
    pooled = spearman_reward_data(
        [row["pilot_score"] for row in rows],
        [row["data_ber_improvement"] for row in rows],
        threshold=0.6,
    )
    within = _within_trial_ordering(trials, "data_ber_improvement")

    assert pooled.correlation > 0.0
    assert within["comparable_trial_count"] == 2
    assert within["discordant_pair_count"] == 2
    assert within["frame_weighted_agreement"] == 0.0
    assert within["per_seed"]["1"]["frame_weighted_agreement"] == 0.0


def test_within_frame_ordering_does_not_count_ties_as_success():
    import math

    from scripts.summarize_pilot_ranking_replay import _within_trial_ordering

    trials = {
        ("日志", 10.0, 1, 1): [
            {"pilot_score": 0.0, "data_ber_improvement": 0.0},
            {"pilot_score": 1.0, "data_ber_improvement": 0.0},
        ]
    }
    result = _within_trial_ordering(trials, "data_ber_improvement")

    assert result["data_target_tie_pair_count"] == 1
    assert result["comparable_trial_count"] == 0
    assert math.isnan(result["frame_weighted_agreement"])
