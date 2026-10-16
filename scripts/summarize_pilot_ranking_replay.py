# -*- coding: utf-8 -*-
"""汇总 Pilot-only 候选排序与 Data 事后收益的关系。

输入是 compare.py 生成的 ``frame_metrics.jsonl``。脚本只在实验结束后读取
候选的事后 Data 指标，用于评估 Pilot 排序是否有预测能力；它不会被在线入口
调用，也不会修改任何 checkpoint 或在线状态。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.metrics import spearman_reward_data


ONLINE_METHOD = "Pilot-Driven Online Adaptation"


def _finite(value: object) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"诊断指标必须是有限数：{value!r}")
    return number


def _input_files(inputs: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            path = path / "frame_metrics.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"找不到 frame_metrics.jsonl：{path}")
        files.append(path)
    return files


def _load_rows(inputs: Iterable[str]) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    source_rows: list[dict] = []
    for path in _input_files(inputs):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            source_row = dict(row)
            source_row["_source"] = str(path)
            source_rows.append(source_row)
            if row.get("method") != ONLINE_METHOD:
                continue
            ranking = row.get("reward_pilot_candidate_ranking")
            if not isinstance(ranking, list) or not ranking:
                continue
            if row.get("data_labels_used_online", False):
                raise ValueError(f"{path}:{line_number} 声称在线使用了 Data 标签。")
            if row.get("data_waveform_used_online", False):
                raise ValueError(f"{path}:{line_number} 声称在线使用了 Data 波形。")
            for candidate in ranking:
                if not isinstance(candidate, dict):
                    raise ValueError(f"{path}:{line_number} 候选记录不是对象。")
                required = (
                    "name",
                    "score",
                    "data_bce_improvement_eval_only",
                    "data_ber_improvement_eval_only",
                )
                missing = [name for name in required if name not in candidate]
                if missing:
                    raise ValueError(
                        f"{path}:{line_number} 候选 {candidate.get('name')} 缺少字段：{missing}"
                    )
                rows.append(
                    {
                        "source": str(path),
                        "snr_db": _finite(row["snr_db"]),
                        "seed": int(row["seed"]),
                        "frame": int(row.get("frame", row.get("frame_index", 0))),
                        "candidate_name": str(candidate["name"]),
                        "pilot_score": _finite(candidate["score"]),
                        "data_bce_improvement": _finite(
                            candidate["data_bce_improvement_eval_only"]
                        ),
                        "data_ber_improvement": _finite(
                            candidate["data_ber_improvement_eval_only"]
                        ),
                        "data_logits_abs_delta": _finite(
                            candidate.get("data_logits_abs_delta_mean_eval_only", 0.0)
                        ),
                        "data_logits_direction_change_rate": _finite(
                            candidate.get(
                                "data_logits_direction_change_rate_eval_only", 0.0
                            )
                        ),
                        "data_soft_output_delta": _finite(
                            candidate.get("data_soft_output_delta_mean_eval_only", 0.0)
                        ),
                        "window_guard_passed": bool(
                            candidate.get("window_guard_passed", False)
                        ),
                        "hard_ber_guard_passed": bool(
                            candidate.get("hard_ber_guard_passed", False)
                        ),
                        "signal_guard_passed": bool(
                            candidate.get("signal_guard_passed", False)
                        ),
                    }
                )
    return rows, source_rows


def _metric_summary(rows: list[dict], target: str) -> dict:
    result = spearman_reward_data(
        [row["pilot_score"] for row in rows],
        [row[target] for row in rows],
        threshold=0.6,
    )
    return {
        "spearman": float(result.correlation),
        "sample_count": int(result.n),
        "gate_pass": bool(result.passed),
    }


def _mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return float(sum(values) / len(values)) if values else math.nan


def _trial_groups(rows: list[dict]) -> dict[tuple[float, int, int], list[dict]]:
    grouped: dict[tuple[str, float, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("source", "")),
                row["snr_db"],
                row["seed"],
                row["frame"],
            )
        ].append(row)
    return grouped


def _summarize_group(rows: list[dict]) -> dict:
    return {
        "trial_count": len(rows),
        "mean_data_bce_improvement": _mean(rows, "data_bce_improvement"),
        "mean_data_ber_improvement": _mean(rows, "data_ber_improvement"),
        "positive_data_bce_fraction": float(
            sum(row["data_bce_improvement"] > 0.0 for row in rows) / len(rows)
        )
        if rows
        else math.nan,
        "positive_data_ber_fraction": float(
            sum(row["data_ber_improvement"] > 0.0 for row in rows) / len(rows)
        )
        if rows
        else math.nan,
    }


def summarize(rows: list[dict], source_rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("输入没有包含 Pilot-Driven Online Adaptation 的候选记录。")
    trials = _trial_groups(rows)
    pilot_top_matches_bce = 0
    pilot_top_matches_ber = 0
    actual_selected_rows: list[dict] = []
    actual_selected_names: list[str] = []
    top_rows: list[dict] = []
    for key, candidates in sorted(trials.items()):
        pilot_top = max(candidates, key=lambda row: row["pilot_score"])
        bce_best = max(
            [{"candidate_name": "identity", "data_bce_improvement": 0.0}, *candidates],
            key=lambda row: row["data_bce_improvement"],
        )
        ber_best = max(
            [{"candidate_name": "identity", "data_ber_improvement": 0.0}, *candidates],
            key=lambda row: row["data_ber_improvement"],
        )
        if pilot_top["candidate_name"] == bce_best["candidate_name"]:
            pilot_top_matches_bce += 1
        if pilot_top["candidate_name"] == ber_best["candidate_name"]:
            pilot_top_matches_ber += 1
        top_rows.append(pilot_top)
        source = next(
            source_row
            for source_row in source_rows
            if source_row.get("method") == ONLINE_METHOD
            and str(source_row.get("_source", "")) == key[0]
            and float(source_row.get("snr_db")) == key[1]
            and int(source_row.get("seed")) == key[2]
            and int(source_row.get("frame", source_row.get("frame_index", 0))) == key[3]
        )
        selected_name = str(source.get("online_update_candidate", "skip"))
        actual_selected_names.append(selected_name)
        selected = next(
            (row for row in candidates if row["candidate_name"] == selected_name),
            None,
        )
        if selected is None:
            selected = {
                "candidate_name": "identity",
                "data_bce_improvement": 0.0,
                "data_ber_improvement": 0.0,
            }
        actual_selected_rows.append(selected)

    by_snr: dict[str, dict] = {}
    trial_keys = sorted(trials)
    for snr in sorted({key[1] for key in trial_keys}):
        # Use trial-level selected rows to keep each frame equally weighted.
        indices = [index for index, key in enumerate(sorted(trials)) if key[1] == snr]
        selected = [actual_selected_rows[index] for index in indices]
        by_snr[str(snr)] = _summarize_group(selected)
    by_seed: dict[str, dict] = {}
    for seed in sorted({key[2] for key in trial_keys}):
        indices = [index for index, key in enumerate(sorted(trials)) if key[2] == seed]
        selected = [actual_selected_rows[index] for index in indices]
        by_seed[str(seed)] = _summarize_group(selected)

    isolation_flags = [
        bool(row.get("data_waveform_used_online", False))
        or bool(row.get("data_labels_used_online", False))
        for row in source_rows
        if row.get("method") == ONLINE_METHOD
    ]
    return {
        "schema_version": "pilot-ranking-replay-summary-v1",
        "method": ONLINE_METHOD,
        "trial_count": len(trials),
        "candidate_count": len(rows),
        "pilot_score_vs_data_bce": _metric_summary(rows, "data_bce_improvement"),
        "pilot_score_vs_data_ber": _metric_summary(rows, "data_ber_improvement"),
        "pilot_top1_matches_data_bce_best_fraction": float(
            pilot_top_matches_bce / len(trials)
        ),
        "pilot_top1_matches_data_ber_best_fraction": float(
            pilot_top_matches_ber / len(trials)
        ),
        "actual_selected": _summarize_group(actual_selected_rows),
        "actual_selected_update_coverage": float(
            sum(name not in {"skip", "identity"} for name in actual_selected_names)
            / len(actual_selected_names)
        ),
        "by_snr": by_snr,
        "by_seed": by_seed,
        "online_data_isolation_passed": not any(isolation_flags),
        "online_method_rows": len(isolation_flags),
        "diagnostic_uses_data_labels": True,
        "data_metrics_used_for_online_decision": False,
        "actual_selected_rows": [
            {
                "candidate_name": name,
                "data_bce_improvement": float(row["data_bce_improvement"]),
                "data_ber_improvement": float(row["data_ber_improvement"]),
            }
            for name, row in zip(actual_selected_names, actual_selected_rows)
        ],
        "pilot_top_rows": [
            {
                "snr_db": row["snr_db"],
                "seed": row["seed"],
                "frame": row["frame"],
                "candidate_name": row["candidate_name"],
                "pilot_score": row["pilot_score"],
                "data_bce_improvement": row["data_bce_improvement"],
                "data_ber_improvement": row["data_ber_improvement"],
            }
            for row in top_rows
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="一个或多个 compare 输出目录或 frame_metrics.jsonl。",
    )
    parser.add_argument("--output", required=True, help="汇总 JSON 输出路径。")
    args = parser.parse_args()
    rows, source_rows = _load_rows(args.input)
    result = summarize(rows, source_rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )
    print(f"saved {output}")


if __name__ == "__main__":
    main()
