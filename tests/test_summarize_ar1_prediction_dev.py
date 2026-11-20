import json

import pytest

from scripts.summarize_ar1_prediction_dev import (
    AUDITED_INITIAL_PREDICTION_OMISSION_SHA256, summarize,
)


def _fixture(tmp_path):
    manifest = {
        "seeds": [1, 2],
        "snrs": [0],
        "frames": 20,
        "variants": [{"name": "base"}, {"name": "candidate"}],
        "pairs": [["base", "candidate"]],
        "pre_registered_windows": [[1, 4], [4, 8], [8, 16], [16, 20]],
    }
    result = {
        "status": "completed",
        "row_count": 80,
        "rows_file": "result.jsonl",
        "protocol": {"experiment": "synthetic"},
    }
    manifest_path = tmp_path / "manifest.json"
    result_path = tmp_path / "result.json"
    rows_path = tmp_path / "result.jsonl"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with rows_path.open("w", encoding="utf-8") as stream:
        for seed in manifest["seeds"]:
            for frame in range(manifest["frames"]):
                for arm, errors in (("base", 2), ("candidate", 1)):
                    stream.write(json.dumps({
                        "snr_db": 0,
                        "seed": seed,
                        "frame_index": frame,
                        "arm": arm,
                        "errors": errors,
                        "bits": 100,
                        "elapsed_ms": 1.0,
                        "rx_sha256": "paired-waveform",
                    }) + "\n")
    return result_path, manifest_path, rows_path


def test_summarize_uses_session_pairs_and_fixed_windows(tmp_path):
    result_path, manifest_path, _ = _fixture(tmp_path)

    summary, report = summarize(result_path, manifest_path)

    assert summary["row_count"] == 80
    assert len(summary["snrs"]["0"]["windows"]) == 5
    assert summary["snrs"]["0"]["windows"][0]["pair_gain_mean_ci95_t"]["base|candidate"][0] == pytest.approx(0.01)
    assert "仅作开发描述" in summary["confidence_intervals"]
    assert "不能对协方差校准" in report


def test_summarize_rejects_unpaired_waveform_hashes(tmp_path):
    result_path, manifest_path, rows_path = _fixture(tmp_path)
    lines = rows_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[1])
    row["rx_sha256"] = "different-waveform"
    lines[1] = json.dumps(row)
    rows_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="波形哈希不一致"):
        summarize(result_path, manifest_path)


def test_summarize_reports_known_initial_timing_limitation_from_recorded_source(tmp_path):
    result_path, manifest_path, _ = _fixture(tmp_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["protocol"]["sources"] = {
        "training/continual_receiver.py": AUDITED_INITIAL_PREDICTION_OMISSION_SHA256,
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")

    summary, report = summarize(result_path, manifest_path)

    assert summary["initial_state_timing_audit"]["status"] == "known_initial_prediction_omission"
    assert "首个 Data Pilot 前没有状态预测" in report
    assert "不代表配对影响抵消" in report


def test_summarize_does_not_treat_unknown_source_as_timing_verified(tmp_path):
    result_path, manifest_path, _ = _fixture(tmp_path)
    summary, report = summarize(result_path, manifest_path)
    assert summary["initial_state_timing_audit"]["status"] == "not_classified"
    assert "首个 Data Pilot 前没有状态预测" not in report
