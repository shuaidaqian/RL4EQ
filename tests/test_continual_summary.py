# -*- coding: utf-8 -*-
"""防止将 Frozen 退化引起的相对收益误报为 Online 自身改善。"""

import pytest

from evaluation.continual_summary import interval, summarize_development


def result_fixture():
    result = {"status": "completed", "protocol": {"frames": 8, "snrs": [0], "seeds": [1, 2],
              "arms": ["acquisition", "current_pilot", "recursive_pilot"]},
              "rows": [], "acquisitions": [{"snr_db": 0, "noise_estimate": 1., "noise_truth_diagnostic": 1.}]}
    for seed in [1, 2]:
        for arm in result["protocol"]["arms"]:
            for frame in range(8):
                errors = 20 + frame if arm == "acquisition" else 10
                result["rows"].append({"seed": seed, "arm": arm, "snr_db": 0,
                    "frame_index": frame, "errors": errors, "bits": 100, "rx_sha256": str(frame),
                    "elapsed_ms": 1.})
    return result


def test_relative_growth_does_not_imply_online_improvement():
    summary = summarize_development(result_fixture())["per_snr"]["0"]
    assert summary["paired"]["acquisition_minus_recursive_pilot"]["late_minus_early_gain"]["mean"] > 0
    assert summary["arms"]["recursive_pilot"]["early_minus_late_ber"]["mean"] == 0


def test_incomplete_run_or_waveform_mismatch_cannot_be_summarized():
    result = result_fixture()
    result["status"] = "running"
    with pytest.raises(ValueError, match="未完成"):
        summarize_development(result)
    result["status"] = "completed"
    result["rows"][0]["rx_sha256"] = "different"
    with pytest.raises(ValueError, match="波形"):
        summarize_development(result)


def test_zero_observed_difference_cannot_prove_population_equivalence():
    result = interval([0., 0., 0.])
    assert result['mean'] == 0
    assert result['ci95'] is None
    assert result['interval_status'] == 'zero_empirical_variance_not_equivalence'


def test_long_sessions_get_power_of_two_windows_through_observation_horizon():
    result = result_fixture()
    result['protocol']['frames'] = 256
    result['protocol']['seeds'] = [1]
    result['rows'] = []
    for arm in result['protocol']['arms']:
        for frame in range(256):
            result['rows'].append({
                'seed':1,'arm':arm,'snr_db':0,'frame_index':frame,
                'errors':10,'bits':100,'rx_sha256':str(frame),'elapsed_ms':1.0,
            })
    windows = summarize_development(result)['per_snr']['0']['arms']['recursive_pilot']['windows']
    assert [(x['start'],x['stop']) for x in windows] == [
        (0,1),(1,4),(4,8),(8,16),(16,32),(32,64),(64,128),(128,256),
    ]
