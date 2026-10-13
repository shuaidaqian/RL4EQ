# -*- coding: utf-8 -*-

import pytest

from scripts.evaluate_pilot_reliability_oof import FEATURE_NAMES, evaluate_rows


def _rows():
    rows = []
    for seed in (0, 1, 2):
        for snr in (5.0, 10.0):
            for frame_index in (0, 1):
                common = {
                    "seed": seed,
                    "snr_db": snr,
                    "frame_index": frame_index,
                    "phase0": 0.01 * seed,
                    "cfo_cycles_per_symbol": 0.0001 * seed,
                    "phase_residual_variance": 0.1,
                    "phase_residual_energy": 0.2,
                    "reconstruction_error": 0.3,
                    "confidence": 0.8,
                    "data_labels_used_online": False,
                }
                rows.extend(
                    [
                        {
                            **common,
                            "candidate_name": "identity",
                            "objective": "bce",
                            "data_ber_improvement": 0.0,
                            "data_bce_improvement": 0.0,
                        },
                        {
                            **common,
                            "candidate_name": "phase_trend_bce",
                            "objective": "bce",
                            "adapt_bce_improvement": 0.01,
                            "reward_loss_improvement": 0.01,
                            "peft_delta_norm": 0.02,
                            "data_ber_improvement": -0.001,
                            "data_bce_improvement": -0.002,
                        },
                        {
                            **common,
                            "candidate_name": "phase_trend_signal_reconstruction",
                            "objective": "pilot_signal_reconstruction",
                            "pilot_reconstruction_improvement": 0.02,
                            "reward_loss_improvement": 0.02,
                            "peft_delta_norm": 0.03,
                            "data_ber_improvement": 0.002,
                            "data_bce_improvement": 0.003,
                        },
                    ]
                )
    return rows


def test_oof_calibrator_holds_out_whole_seeds_and_keeps_data_as_diagnostic_only():
    assert all(not name.startswith("data_") for name in FEATURE_NAMES)

    result = evaluate_rows(_rows(), alpha=5.0)

    assert result["fold_strategy"] == "leave_one_seed_out"
    assert result["heldout_seeds"] == [0, 1, 2]
    assert result["data_labels_used_online"] is False
    assert result["diagnostic_uses_data_labels"] is True
    assert result["offline_targets_include_data_labels"] is True
    assert result["trial_count"] == 12
    assert len(result["oof_predictions"]) == 36
    assert all(
        fold["heldout_seed"] not in fold["training_seeds"]
        for fold in result["fold_summaries"]
    )
    assert result["oof_policy_comparison"]["freeze"]["update_coverage"] == 0.0


def test_oof_calibrator_rejects_logs_that_used_data_labels_online():
    rows = _rows()
    rows[0]["data_labels_used_online"] = True

    with pytest.raises(ValueError, match="在线使用了 Data 标签"):
        evaluate_rows(rows)
