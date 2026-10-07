# -*- coding: utf-8 -*-
"""比较 Pilot-only 代理目标对后续 Data BER 的排序能力。

该脚本只读取已有 compare 逐帧日志。Data 标签只用于事后相关性诊断，绝不参与
在线更新、候选选择或回滚。脚本不重新运行模型，适合先筛选窗口目标，再决定是否
进入新的在线实现。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.metrics import spearman_reward_data


def _read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else math.nan


def _trajectory_groups(rows: list[dict]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("method") != "Pilot-Driven Online Adaptation":
            continue
        key = (
            row.get("delay"),
            row.get("snr_db"),
            row.get("pilot_total"),
            row.get("reward_pilot_total"),
            row.get("pilot_layout"),
            row.get("seed"),
        )
        groups[key].append(row)
    for values in groups.values():
        values.sort(key=lambda item: int(item.get("frame", 0)))
    return groups


def _event_rows(rows: list[dict], window_size: int) -> list[dict]:
    events = []
    for trajectory, values in _trajectory_groups(rows).items():
        for index, row in enumerate(values):
            if not bool(row.get("online_update_scheduled", True)):
                continue
            window = values[index : index + int(window_size)]
            future = values[index + 1 : index + int(window_size)]
            if not future:
                continue
            reward_loss = float(row.get("reward_pilot_loss_before", 0.0)) - float(
                row.get("reward_pilot_loss_after", 0.0)
            )
            reward_windows = [float(value) for value in row.get("reward_pilot_window_gains", [])]
            cumulative_reward = _mean(reward_windows) if reward_windows else reward_loss
            # 状态距离是 Pilot-only 审计量；旧日志没有该字段时退化为零，且不伪造漂移。
            state_distance = float(row.get("pilot_state_drift_distance_to_frame_start", 0.0))
            state_gate_score = state_distance * float(row.get("pilot_phase_confidence", 0.0))
            future_data = [
                float(values[index].get("ber_data", 0.0))
                - float(values[index + offset].get("ber_data", 0.0))
                for offset in range(1, len(future) + 1)
            ]
            events.append(
                {
                    "trajectory": trajectory,
                    "event_frame": int(row.get("frame", 0)),
                    "window_size": int(window_size),
                    "future_data_ber_improvement": _mean(future_data),
                    "single_frame_bce": reward_loss,
                    "cumulative_reward_pilot": cumulative_reward,
                    "joint_reconstruction_proxy": cumulative_reward - 0.01 * state_gate_score,
                    "state_gate_score": state_gate_score,
                    "data_labels_used_online": False,
                    "diagnostic_uses_data_labels": True,
                }
            )
    return events


def _summarize(events: list[dict], score_name: str) -> dict:
    usable = [event for event in events if math.isfinite(event["future_data_ber_improvement"])]
    scores = [float(event[score_name]) for event in usable]
    targets = [float(event["future_data_ber_improvement"]) for event in usable]
    result = spearman_reward_data(scores, targets, threshold=0.6)
    return {
        "objective": score_name,
        "events": len(usable),
        "spearman": float(result.correlation),
        "spearman_gate_pass": bool(result.passed),
        "mean_future_data_ber_improvement": _mean(targets),
        "fraction_future_data_improved": _mean([float(value > 0.0) for value in targets]),
        "diagnostic_uses_data_labels": True,
        "online_policy_uses_data_labels": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-sizes", nargs="*", type=int, default=[2, 4, 8])
    args = parser.parse_args()
    rows = _read_rows(Path(args.log_dir) / "frame_metrics.jsonl")
    summaries = []
    events_payload = []
    objectives = ["single_frame_bce", "cumulative_reward_pilot", "joint_reconstruction_proxy"]
    for window_size in args.window_sizes:
        events = _event_rows(rows, int(window_size))
        events_payload.extend(events)
        for objective in objectives:
            summary = _summarize(events, objective)
            summary["window_size"] = int(window_size)
            summaries.append(summary)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "pilot-objective-ranking-v1",
                "source_log_dir": str(args.log_dir),
                "window_sizes": [int(value) for value in args.window_sizes],
                "summaries": summaries,
                "diagnostic_uses_data_labels": True,
                "online_policy_uses_data_labels": False,
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    with (output / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in events_payload:
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=True) + "\n")
    print(f"saved {output}; events={len(events_payload)}")


if __name__ == "__main__":
    main()
