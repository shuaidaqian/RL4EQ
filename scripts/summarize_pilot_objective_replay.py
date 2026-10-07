# -*- coding: utf-8 -*-
"""汇总 Pilot-only replay 中不同目标对 Data BER 的排序能力。

该脚本只读取诊断日志。Data 标签仅用于事后计算相关性，不参与任何在线更新。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    try:
        from scipy.stats import spearmanr

        value = spearmanr(x, y).statistic
        return None if value is None or not np.isfinite(value) else float(value)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    source = Path(args.log_dir) / "window_peft_rows.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("objective", "unknown"),
            row.get("level"),
            row.get("delay"),
            row.get("snr_db"),
            row.get("pilot_total"),
            row.get("pilot_layout"),
            row.get("seed"),
            row.get("window_index"),
        )
        grouped[key].append(row)

    summaries = []
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        candidates = [row for row in group if row.get("action_name") != "identity"]
        if not candidates:
            continue
        reward = [float(row.get("reward_loss_improvement", 0.0)) for row in candidates]
        data = [float(row.get("data_ber_improvement", 0.0)) for row in candidates]
        summaries.append(
            {
                "objective": key[0],
                "level": key[1],
                "delay": key[2],
                "snr_db": key[3],
                "pilot_total": key[4],
                "pilot_layout": key[5],
                "seed": key[6],
                "window_index": key[7],
                "candidate_count": len(candidates),
                "spearman_reward_loss_vs_data_ber": _spearman(reward, data),
                "reward_improved_fraction": float(sum(value > 0.0 for value in reward) / len(reward)),
                "data_improved_fraction": float(sum(value > 0.0 for value in data) / len(data)),
                "best_action_by_reward": candidates[int(np.argmax(reward))]["action_name"],
                "best_action_by_data": candidates[int(np.argmax(data))]["action_name"],
            }
        )

    by_objective: dict[str, list[dict]] = defaultdict(list)
    for row in summaries:
        by_objective[str(row["objective"])].append(row)
    objective_summary = []
    for objective, values in sorted(by_objective.items()):
        correlations = [row["spearman_reward_loss_vs_data_ber"] for row in values if row["spearman_reward_loss_vs_data_ber"] is not None]
        objective_summary.append(
            {
                "objective": objective,
                "windows": len(values),
                "mean_spearman": None if not correlations else float(np.mean(correlations)),
                "median_spearman": None if not correlations else float(np.median(correlations)),
                "fraction_spearman_ge_0_6": 0.0 if not correlations else float(sum(value >= 0.6 for value in correlations) / len(correlations)),
                "fraction_data_improved": float(np.mean([row["data_improved_fraction"] for row in values])),
            }
        )
    payload = {
        "metric": "pilot_objective_replay_ranking",
        "source": str(source),
        "diagnostic_uses_data_labels": True,
        "online_policy_uses_data_labels": False,
        "objective_summary": objective_summary,
        "window_summary": summaries,
    }
    output = Path(args.output) if args.output else Path(args.log_dir) / "pilot_objective_replay_summary.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
