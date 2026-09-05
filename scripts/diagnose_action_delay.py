# -*- coding: utf-8 -*-
"""诊断在线动作的即时与跨帧 Reward Pilot 反馈。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Iterable, Mapping


def summarize_action_delay(
    rows: Iterable[Mapping[str, object]],
    horizons: tuple[int, ...] = (0, 1, 2, 4, 8),
    delayed_margin: float = 1e-6,
) -> dict:
    """按相同配置和 seed 对齐未来帧，汇总每个动作的延迟 reward。

    该函数只读取 Reward Pilot 产生的 `bandit_reward`（或兼容的
    `reward_gain`），Data BER 即使存在也只被视为离线评估列，不能参与诊断
    的动作选择或控制器训练。
    """

    selected_horizons = tuple(int(value) for value in horizons)
    if not selected_horizons or any(value < 0 for value in selected_horizons):
        raise ValueError("horizons 必须包含至少一个非负整数。")
    if len(set(selected_horizons)) != len(selected_horizons):
        raise ValueError("horizons 不能包含重复值。")
    if float(delayed_margin) < 0.0 or not math.isfinite(float(delayed_margin)):
        raise ValueError("delayed_margin 必须是非负有限数。")

    records = [dict(row) for row in rows]
    records = [
        row
        for row in records
        if "action" in row
        and (row.get("bandit_reward") is not None or row.get("reward_gain") is not None)
    ]
    if not records:
        raise ValueError("动作延迟诊断至少需要一条记录。")
    if any(bool(row.get("data_labels_used_online", False)) for row in records):
        raise ValueError("动作延迟诊断拒绝在线使用 Data 标签的记录。")

    key_fields = ("seed", "delay", "snr_db", "pilot_total", "pilot_layout", "state_split")
    indexed: dict[tuple[object, ...], dict[int, dict]] = defaultdict(dict)
    for row in records:
        if "frame" not in row:
            raise ValueError("动作延迟诊断需要在线记录包含 frame 字段。")
        reward_value = row.get("bandit_reward", row.get("reward_gain"))
        if reward_value is None or not math.isfinite(float(reward_value)):
            raise ValueError("动作延迟诊断需要有限的 bandit_reward 或 reward_gain。")
        key = tuple(row.get(field) for field in key_fields)
        indexed[key][int(row["frame"])] = row

    accumulators: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    support: dict[str, dict[str, set[object]]] = defaultdict(
        lambda: {"seeds": set(), "snrs": set()}
    )
    for key, frames in indexed.items():
        for frame, anchor in frames.items():
            action = str(anchor["action"])
            if action == "skip":
                continue
            if anchor.get("update_applied") is False:
                continue
            if anchor.get("adaptation_accepted") is False:
                continue
            for horizon in selected_horizons:
                target = frames.get(frame + horizon)
                if target is None:
                    continue
                value = target.get("bandit_reward", target.get("reward_gain"))
                accumulators[action][horizon].append(float(value))
                support[action]["seeds"].add(key[0])
                support[action]["snrs"].add(key[2])

    by_action: dict[str, dict[str, dict[str, float]]] = {}
    for action, horizon_values in sorted(accumulators.items()):
        by_action[action] = {}
        for horizon in selected_horizons:
            values = horizon_values.get(horizon, [])
            by_action[action][str(horizon)] = {
                "count": float(len(values)),
                "mean_reward": float(sum(values) / max(1, len(values))),
            }

    delayed_effect_detected = False
    for action, summaries in by_action.items():
        if action == "skip" or "0" not in summaries:
            continue
        immediate = summaries["0"]
        if immediate["count"] == 0.0:
            continue
        delayed_values = [
            summary["mean_reward"]
            for horizon, summary in summaries.items()
            if horizon != "0" and summary["count"] > 0.0
        ]
        enough_support = (
            len(support[action]["seeds"]) >= 2
            and len(support[action]["snrs"]) >= 2
        )
        if (
            enough_support
            and delayed_values
            and max(delayed_values) > immediate["mean_reward"] + float(delayed_margin)
        ):
            delayed_effect_detected = True
            break

    return {
        "horizons": list(selected_horizons),
        "by_action": by_action,
        "support": {
            action: {
                "seed_count": len(values["seeds"]),
                "snr_count": len(values["snrs"]),
            }
            for action, values in support.items()
        },
        "delayed_effect_detected": delayed_effect_detected,
        "recommended_controller": (
            "investigate_safe_recurrent_double_dqn"
            if delayed_effect_detected
            else "contextual_bandit"
        ),
        "diagnostic_uses_data_labels": False,
    }


def _read_rows(path: Path) -> list[dict]:
    """读取 JSONL 或包含 rows 的 JSON。"""

    source = path / "frame_metrics.jsonl" if path.is_dir() else path
    if source.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(source.read_text(encoding="utf-8"))
    return payload["rows"] if isinstance(payload, dict) and "rows" in payload else payload


def main() -> None:
    parser = argparse.ArgumentParser(description="诊断在线动作的跨帧延迟效应")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize_action_delay(_read_rows(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
