# -*- coding: utf-8 -*-
"""汇总多帧轨迹四臂开发评估的配对窗口收益与趋势。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.pilot_transfer import paired_interval


def summarize(path: Path, window_frames: int):
    grouped = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            grouped[(row["branch"], float(row["snr_db"]), int(row["seed"]))].append(row)

    session_windows = {}
    for key, rows in grouped.items():
        rows.sort(key=lambda item: int(item["frame"]))
        if [int(row["frame"]) for row in rows] != list(range(len(rows))):
            raise ValueError(f"会话帧索引不连续：{key}")
        windows = []
        for start in range(0, len(rows), window_frames):
            block = rows[start:start + window_frames]
            if len(block) != window_frames:
                raise ValueError(f"会话帧数不能被窗口长度整除：{key}")
            gain = float(np.mean([item["ber_frozen"] - item["ber_online"] for item in block]))
            windows.append(gain)
        session_windows[key] = windows

    output = {"source": str(path.resolve()), "window_frames": window_frames,
              "gain_positive_means_better": True, "per_snr": []}
    snrs = sorted({key[1] for key in session_windows})
    for snr in snrs:
        branches = {}
        for branch in ("baseline", "post_adapt"):
            seeds = sorted(key[2] for key in session_windows if key[0] == branch and key[1] == snr)
            if not seeds:
                raise ValueError(f"缺少分支数据：{branch}, SNR={snr}")
            matrix = np.asarray([session_windows[(branch, snr, seed)] for seed in seeds], dtype=float)
            slopes = [float(np.polyfit(np.arange(matrix.shape[1]), row, 1)[0]) for row in matrix]
            branches[branch] = {
                "seeds": seeds,
                "window_gain": [paired_interval(matrix[:, index].tolist())
                                for index in range(matrix.shape[1])],
                "late_minus_early": paired_interval((matrix[:, -1] - matrix[:, 0]).tolist()),
                "session_slope_per_window": paired_interval(slopes),
            }
        seeds = sorted(key[2] for key in session_windows if key[0] == "baseline" and key[1] == snr)
        if any(("post_adapt", snr, seed) not in session_windows for seed in seeds):
            raise ValueError(f"A/B缺少配对会话：SNR={snr}")
        interactions = np.asarray([
            np.asarray(session_windows[("post_adapt", snr, seed)])
            - np.asarray(session_windows[("baseline", snr, seed)])
            for seed in seeds
        ])
        branches["trajectory_minus_supervised_gain_interaction_by_window"] = [
            paired_interval(interactions[:, index].tolist()) for index in range(interactions.shape[1])
        ]
        branches["trajectory_minus_supervised_late_early_interaction"] = paired_interval(
            ((interactions[:, -1] - interactions[:, 0]).tolist())
        )
        output["per_snr"].append({"snr_db": snr, **branches})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="四臂 frame_metrics.jsonl 路径")
    parser.add_argument("--window-frames", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.window_frames < 1:
        parser.error("--window-frames 必须为正整数")
    result = summarize(args.input, args.window_frames)
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
