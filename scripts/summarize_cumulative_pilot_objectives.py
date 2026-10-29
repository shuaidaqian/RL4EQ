# -*- coding: utf-8 -*-
"""汇总 Adapt Pilot 目标与累计窗口的 Pilot-only replay。"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def _finite(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else math.nan


def _pairwise_agreement(rows: list[dict], target: str) -> dict:
    concordant = discordant = tied = 0
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            score_delta = left["pilot_reward_score"] - right["pilot_reward_score"]
            target_delta = left[target] - right[target]
            if score_delta == 0 or target_delta == 0:
                tied += 1
            elif (score_delta > 0) == (target_delta > 0):
                concordant += 1
            else:
                discordant += 1
    comparable = concordant + discordant
    return {
        "comparable_pairs": comparable,
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "tied_pairs": tied,
        "agreement": concordant / comparable if comparable else math.nan,
    }


def _aggregate_pairwise_agreement(groups: dict[tuple, list[dict]], target: str) -> dict:
    concordant = discordant = tied = 0
    for rows in groups.values():
        result = _pairwise_agreement(rows, target)
        concordant += int(result["concordant_pairs"])
        discordant += int(result["discordant_pairs"])
        tied += int(result["tied_pairs"])
    comparable = concordant + discordant
    return {
        "comparable_pairs": comparable,
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "tied_pairs": tied,
        "agreement": concordant / comparable if comparable else math.nan,
        "interpretation": "只在相同 SNR、seed 和累计窗口的目标之间配对；并列不计入分母。",
    }


def summarize(paths: list[Path]) -> dict:
    if not paths:
        raise ValueError("至少需要一个累计 Pilot replay 文件。")

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    reference = payloads[0]
    reference_args = reference["arguments"]
    shared_fields = (
        "snrs", "seeds", "frames", "train_frame_counts", "validation_frames",
        "reward_score", "update_scope", "reward_aggregation", "max_delta_norm",
        "lr", "steps", "delay", "pilot_total", "reward_pilot_total", "gap_seconds",
        "condition_source", "joint_bce_weight",
    )
    for payload in payloads:
        if payload.get("metric") != "cumulative_pilot_state_mapping_future_frame_replay":
            raise ValueError("输入文件不是累计 Pilot-only replay 结果。")
        if payload.get("checkpoint") != reference.get("checkpoint"):
            raise ValueError("不同 replay 必须使用同一个离线 checkpoint。")
        for field in shared_fields:
            if payload["arguments"].get(field) != reference_args.get(field):
                raise ValueError(f"不同 replay 的固定条件不一致：{field}")

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for payload in payloads:
        objective = str(payload["arguments"]["adapt_objective"])
        for row in payload["rows"]:
            if row.get("data_labels_used_online") is not False:
                raise ValueError("回放记录违反 Data 标签隔离。")
            if row.get("data_waveform_used_online") is not False:
                raise ValueError("回放记录违反 Data 波形隔离。")
            if row.get("diagnostic_uses_data_labels") is not True:
                raise ValueError("回放缺少事后 Data 指标边界标记。")
            key = (
                float(row["snr_db"]), int(row["seed"]),
                int(row["train_frame_count"]), objective,
            )
            grouped[key].append(row)

    candidates = []
    for (snr, seed, train_count, objective), rows in sorted(grouped.items()):
        if len({int(row["test_frame_index"]) for row in rows}) != len(rows):
            raise ValueError("同一候选的事后测试帧重复。")
        reward_values = [value for row in rows for value in row["gate_reward_improvements"]]
        candidates.append({
            "snr_db": snr,
            "seed": seed,
            "train_frame_count": train_count,
            "adapt_objective": objective,
            "test_frames": len(rows),
            "pilot_reward_score": _finite(reward_values),
            "pilot_adapt_bce_improvement": _finite([
                row["adapt_bce_before"] - row["adapt_bce_after"] for row in rows
            ]),
            "reward_gate_accepted": bool(rows[0]["initial_reward_gate_accepted"]),
            "candidate_data_ber_improvement": _finite([
                row["candidate_data_ber_improvement"] for row in rows
            ]),
            "candidate_data_bce_improvement": _finite([
                row["candidate_data_bce_improvement"] for row in rows
            ]),
            "committed_data_ber_improvement": _finite([
                row["data_ber_improvement"] for row in rows
            ]),
            "committed_data_bce_improvement": _finite([
                row["data_bce_improvement"] for row in rows
            ]),
            "candidate_data_logits_abs_delta": _finite([
                row["candidate_data_logits_abs_mean_delta"] for row in rows
            ]),
            "candidate_data_soft_output_delta": _finite([
                row["candidate_data_soft_abs_mean_delta"] for row in rows
            ]),
            "test_data_ber_improved_fraction": _finite([
                float(row["candidate_data_ber_improvement"] > 0) for row in rows
            ]),
        })

    ranking_groups: dict[tuple, list[dict]] = defaultdict(list)
    by_objective: dict[tuple, list[dict]] = defaultdict(list)
    by_snr: dict[float, list[dict]] = defaultdict(list)
    for row in candidates:
        ranking_groups[(row["snr_db"], row["seed"], row["train_frame_count"])].append(row)
        by_objective[(row["adapt_objective"], row["train_frame_count"])].append(row)
        by_snr[row["snr_db"]].append(row)

    ranking = []
    for (snr, seed, train_count), rows in sorted(ranking_groups.items()):
        ranking.append({
            "snr_db": snr,
            "seed": seed,
            "train_frame_count": train_count,
            "candidate_count": len(rows),
            "reward_vs_candidate_data_bce": _pairwise_agreement(
                rows, "candidate_data_bce_improvement"
            ),
            "reward_vs_candidate_data_ber": _pairwise_agreement(
                rows, "candidate_data_ber_improvement"
            ),
            "best_by_pilot_reward": max(rows, key=lambda row: row["pilot_reward_score"])["adapt_objective"],
            "best_by_data_bce_eval_only": max(
                rows, key=lambda row: row["candidate_data_bce_improvement"]
            )["adapt_objective"],
            "best_by_data_ber_eval_only": max(
                rows, key=lambda row: row["candidate_data_ber_improvement"]
            )["adapt_objective"],
        })

    objective_summary = []
    for (objective, train_count), rows in sorted(by_objective.items()):
        objective_summary.append({
            "adapt_objective": objective,
            "train_frame_count": train_count,
            "seed_snr_trials": len(rows),
            "reward_acceptance_fraction": _finite([
                float(row["reward_gate_accepted"]) for row in rows
            ]),
            "mean_adapt_bce_improvement": _finite([
                row["pilot_adapt_bce_improvement"] for row in rows
            ]),
            "mean_candidate_data_ber_improvement": _finite([
                row["candidate_data_ber_improvement"] for row in rows
            ]),
            "mean_candidate_data_bce_improvement": _finite([
                row["candidate_data_bce_improvement"] for row in rows
            ]),
            "positive_data_ber_trial_fraction": _finite([
                float(row["candidate_data_ber_improvement"] > 0) for row in rows
            ]),
            "mean_committed_data_ber_improvement": _finite([
                row["committed_data_ber_improvement"] for row in rows
            ]),
            "positive_committed_ber_trial_fraction": _finite([
                float(row["committed_data_ber_improvement"] > 0) for row in rows
            ]),
        })

    snr_summary = []
    for snr, rows in sorted(by_snr.items()):
        snr_summary.append({
            "snr_db": snr,
            "candidate_trials": len(rows),
            "mean_candidate_data_ber_improvement": _finite([
                row["candidate_data_ber_improvement"] for row in rows
            ]),
            "mean_candidate_data_bce_improvement": _finite([
                row["candidate_data_bce_improvement"] for row in rows
            ]),
            "mean_committed_data_ber_improvement": _finite([
                row["committed_data_ber_improvement"] for row in rows
            ]),
        })

    return {
        "schema_version": "cumulative-pilot-objective-summary-v1",
        "checkpoint": reference["checkpoint"],
        "shared_arguments": reference_args,
        "candidate_count": len(candidates),
        "ranking_group_count": len(ranking),
        "data_metrics_used_online": False,
        "online_data_waveform_isolation_passed": True,
        "online_data_label_isolation_passed": True,
        "diagnostic_uses_data_labels": True,
        "ranking_interpretation": "同一 seed、SNR、累计窗口内比较 Pilot Reward 分数与未来 Data 指标；Data 仅事后评估。",
        "objective_summary": objective_summary,
        "snr_summary": snr_summary,
        "overall_reward_vs_candidate_data_bce": _aggregate_pairwise_agreement(
            ranking_groups, "candidate_data_bce_improvement"
        ),
        "overall_reward_vs_candidate_data_ber": _aggregate_pairwise_agreement(
            ranking_groups, "candidate_data_ber_improvement"
        ),
        "within_trial_ranking": ranking,
        "candidate_trials": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="不同 Adapt 目标的 JSON replay 文件。")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = summarize([Path(value) for value in args.inputs])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )
    print(f"已保存汇总：{output}；候选={result['candidate_count']}，配对组={result['ranking_group_count']}")


if __name__ == "__main__":
    main()
