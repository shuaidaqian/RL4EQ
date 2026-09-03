# -*- coding: utf-8 -*-
"""从 compare 逐帧日志生成 Pilot-only replay 诊断。"""

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
from evaluation.research_diagnostics import build_pilot_replay_events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-sizes", nargs="*", type=int, default=[1, 2, 4, 8])
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    rows = [
        json.loads(line)
        for line in (log_dir / "frame_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_events: list[dict] = []
    summaries: list[dict] = []
    for window_size in args.window_sizes:
        events = build_pilot_replay_events(rows, hold_frames=int(window_size))
        for event in events:
            all_events.append({**event, "window_size": int(window_size)})
        summaries.extend(_summarize_window(events, int(window_size)))
    payload = {
        "schema_version": "pilot-replay-diagnostic-v1",
        "source_log_dir": str(log_dir),
        "window_sizes": [int(value) for value in args.window_sizes],
        "diagnostic_uses_data_labels": True,
        "online_policy_uses_data_labels": False,
        "summaries": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )
    with (output_dir / "replay_events.jsonl").open("w", encoding="utf-8") as handle:
        for event in all_events:
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=True) + "\n")
    print(f"saved {output_dir}")
    print(f"events {len(all_events)}")


def _summarize_window(events: list[dict], window_size: int) -> list[dict]:
    groups: dict[str, list[dict]] = {"all_scheduled": events, "accepted_peft": [event for event in events if event["peft_update_applied"]]}
    summaries = []
    for group_name, group in groups.items():
        summaries.append(_summarize_group(group, window_size, group_name, None))
        by_snr: dict[float, list[dict]] = defaultdict(list)
        for event in group:
            by_snr[float(event["trajectory"][1])].append(event)
        for snr_db, snr_events in sorted(by_snr.items()):
            summaries.append(_summarize_group(snr_events, window_size, group_name, snr_db))
    return summaries


def _summarize_group(events: list[dict], window_size: int, group_name: str, snr_db: float | None) -> dict:
    usable = [
        event
        for event in events
        if math.isfinite(float(event["future_data_ber_improvement"]))
    ]
    rewards = [float(event["reward_loss_improvement"]) for event in usable]
    future_data = [float(event["future_data_ber_improvement"]) for event in usable]
    correlation = spearman_reward_data(rewards, future_data, threshold=0.6)
    return {
        "window_size": int(window_size),
        "group": str(group_name),
        "snr_db": snr_db,
        "event_count": len(events),
        "usable_event_count": len(usable),
        "spearman_reward_vs_future_data": float(correlation.correlation),
        "spearman_n": int(correlation.n),
        "spearman_gate_pass": bool(correlation.passed),
        "mean_future_data_ber_improvement": _mean(future_data),
        "fraction_future_data_improved": _fraction_positive(future_data),
        "diagnostic_uses_data_labels": True,
        "online_policy_uses_data_labels": False,
    }


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else math.nan


def _fraction_positive(values: list[float]) -> float:
    return float(sum(value > 0.0 for value in values) / len(values)) if values else math.nan


if __name__ == "__main__":
    main()
