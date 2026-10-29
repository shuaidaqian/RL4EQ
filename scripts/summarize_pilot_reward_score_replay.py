# -*- coding: utf-8 -*-
"""比较 Pilot Reward 评分对同一 PEFT 候选的放行可靠性。"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def _mean(values: list[float]) -> float:
    numbers = [float(value) for value in values if math.isfinite(float(value))]
    return sum(numbers) / len(numbers) if numbers else math.nan


def _trial_rows(paths: list[Path]) -> tuple[dict[str, list[dict]], dict]:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not payloads:
        raise ValueError("至少需要一个 Reward replay 文件。")
    reference = payloads[0]
    shared = dict(reference.get("arguments", {}))
    shared.pop("reward_score", None)
    shared.pop("output", None)
    grouped: dict[str, list[dict]] = defaultdict(list)
    seen_scores = set()
    for payload in payloads:
        if payload.get("metric") != "cumulative_pilot_state_mapping_future_frame_replay":
            raise ValueError("输入文件不是累计 Pilot-only replay 结果。")
        if payload.get("checkpoint") != reference.get("checkpoint"):
            raise ValueError("不同 Reward 分数必须使用同一 checkpoint。")
        args = dict(payload.get("arguments", {}))
        score = str(args.pop("reward_score", ""))
        args.pop("output", None)
        if not score:
            raise ValueError("回放缺少 Reward 评分名称。")
        if args != shared:
            changed = sorted(
                key for key in set(args) | set(shared) if args.get(key) != shared.get(key)
            )
            raise ValueError(f"不同 Reward 回放的固定条件不一致：{changed}")
        if score in seen_scores:
            raise ValueError(f"Reward 评分重复：{score}")
        seen_scores.add(score)
        trials: dict[tuple, list[dict]] = defaultdict(list)
        for row in payload.get("rows", []):
            if row.get("data_labels_used_online") is not False:
                raise ValueError("回放记录违反 Data 标签隔离。")
            if row.get("data_waveform_used_online") is not False:
                raise ValueError("回放记录违反 Data 波形隔离。")
            key = (
                float(row["snr_db"]), int(row["seed"]),
                int(row["train_frame_count"]),
            )
            trials[key].append(row)
        for key, rows in trials.items():
            grouped[score].append({
                "snr_db": key[0],
                "seed": key[1],
                "train_frame_count": key[2],
                "test_frames": len(rows),
                "accepted": bool(rows[0]["initial_reward_gate_accepted"]),
                "candidate_data_ber_improvement": _mean([
                    row["candidate_data_ber_improvement"] for row in rows
                ]),
                "candidate_data_bce_improvement": _mean([
                    row["candidate_data_bce_improvement"] for row in rows
                ]),
                "committed_data_ber_improvement": _mean([
                    row["data_ber_improvement"] for row in rows
                ]),
                "committed_data_bce_improvement": _mean([
                    row["data_bce_improvement"] for row in rows
                ]),
                "candidate_data_logits_abs_delta": _mean([
                    row["candidate_data_logits_abs_mean_delta"] for row in rows
                ]),
                "candidate_data_soft_output_delta": _mean([
                    row["candidate_data_soft_abs_mean_delta"] for row in rows
                ]),
                "future_data_ber_positive": _mean([
                    float(row["candidate_data_ber_improvement"] > 0.0) for row in rows
                ]),
            })
    score_names = set(grouped)
    if len(score_names) != len(payloads):
        raise ValueError("Reward 回放中存在重复或空的评分类别。")
    trial_keys = {
        score: {
            (row["snr_db"], row["seed"], row["train_frame_count"])
            for row in rows
        }
        for score, rows in grouped.items()
    }
    if any(keys != next(iter(trial_keys.values())) for keys in trial_keys.values()):
        raise ValueError("不同 Reward 分数必须覆盖完全相同的 seed/SNR/窗口配对。")
    return grouped, {
        "checkpoint": reference.get("checkpoint"),
        "shared_arguments": shared,
    }


def summarize(paths: list[Path]) -> dict:
    grouped, metadata = _trial_rows(paths)
    score_summary = []
    by_snr = []
    trial_rows = []
    for score, rows in sorted(grouped.items()):
        accepted = [row for row in rows if row["accepted"]]
        rejected = [row for row in rows if not row["accepted"]]
        score_summary.append({
            "reward_score": score,
            "candidate_trials": len(rows),
            "acceptance_fraction": len(accepted) / len(rows),
            "accepted_data_ber_positive_fraction": _mean([
                row["future_data_ber_positive"] for row in accepted
            ]),
            "accepted_data_ber_negative_fraction": _mean([
                float(row["candidate_data_ber_improvement"] < 0.0) for row in accepted
            ]),
            "rejected_but_data_ber_positive_fraction": _mean([
                row["future_data_ber_positive"] for row in rejected
            ]),
            "mean_candidate_data_ber_improvement": _mean([
                row["candidate_data_ber_improvement"] for row in rows
            ]),
            "mean_candidate_data_bce_improvement": _mean([
                row["candidate_data_bce_improvement"] for row in rows
            ]),
            "mean_committed_data_ber_improvement": _mean([
                row["committed_data_ber_improvement"] for row in rows
            ]),
            "positive_committed_data_ber_fraction": _mean([
                float(row["committed_data_ber_improvement"] > 0.0) for row in rows
            ]),
            "mean_committed_data_bce_improvement": _mean([
                row["committed_data_bce_improvement"] for row in rows
            ]),
            "mean_data_logits_abs_delta": _mean([
                row["candidate_data_logits_abs_delta"] for row in rows
            ]),
            "mean_data_soft_output_delta": _mean([
                row["candidate_data_soft_output_delta"] for row in rows
            ]),
        })
        for snr in sorted({row["snr_db"] for row in rows}):
            values = [row for row in rows if row["snr_db"] == snr]
            by_snr.append({
                "reward_score": score,
                "snr_db": snr,
                "candidate_trials": len(values),
                "acceptance_fraction": _mean([float(row["accepted"]) for row in values]),
                "mean_candidate_data_ber_improvement": _mean([
                    row["candidate_data_ber_improvement"] for row in values
                ]),
                "mean_committed_data_ber_improvement": _mean([
                    row["committed_data_ber_improvement"] for row in values
                ]),
                "positive_committed_data_ber_fraction": _mean([
                    float(row["committed_data_ber_improvement"] > 0.0) for row in values
                ]),
            })
        trial_rows.extend({"reward_score": score, **row} for row in rows)

    score_names = sorted(grouped)
    paired = []
    first = grouped[score_names[0]]
    key_fields = ("snr_db", "seed", "train_frame_count")
    reference = {
        tuple(row[field] for field in key_fields): row for row in first
    }
    for score in score_names:
        rows = {tuple(row[field] for field in key_fields): row for row in grouped[score]}
        for key, candidate in rows.items():
            baseline = reference[key]
            paired.append({
                "reward_score": score,
                "snr_db": key[0],
                "seed": key[1],
                "train_frame_count": key[2],
                "accepted": candidate["accepted"],
                "raw_candidate_data_ber_improvement": candidate["candidate_data_ber_improvement"],
                "committed_data_ber_improvement": candidate["committed_data_ber_improvement"],
                "raw_candidate_outcome_positive": candidate["candidate_data_ber_improvement"] > 0.0,
                "baseline_reward_score": score_names[0],
                "baseline_accepted": baseline["accepted"],
                "acceptance_matches_baseline": candidate["accepted"] == baseline["accepted"],
            })

    return {
        "schema_version": "pilot-reward-score-replay-summary-v1",
        **metadata,
        "candidate_data_used_for_online_decision": False,
        "online_data_waveform_isolation_passed": True,
        "online_data_label_isolation_passed": True,
        "diagnostic_uses_data_labels": True,
        "score_summary": score_summary,
        "by_snr": by_snr,
        "trial_rows": trial_rows,
        "paired_acceptance": paired,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = summarize([Path(value) for value in args.inputs])
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    print(f"已保存 Reward 评分对比：{target}；方法数={len(payload['score_summary'])}")


if __name__ == "__main__":
    main()
