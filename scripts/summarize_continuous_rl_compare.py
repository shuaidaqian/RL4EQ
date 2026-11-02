# -*- coding: utf-8 -*-
"""审计独立 Frozen 与窗口 RL 的完整连续会话，单 seed 不生成推断区间。"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.pilot_transfer import paired_interval

FROZEN = "Frozen Offline NN"
PROPOSED = "RL-Modulated Neural Block Equalizer"


def summarize_continuous_rows(rows, *, expected_frames, segment_frames, frame_seconds):
    if min(expected_frames, segment_frames, frame_seconds) <= 0 or expected_frames % segment_frames:
        raise ValueError("完整会话必须包含整数个预设分段，且时长为正。")
    pairs = defaultdict(dict)
    for row in rows:
        method = row["method"]
        if method not in {FROZEN, PROPOSED} or row.get("data_labels_used_online") is not False:
            raise ValueError("只接受标签隔离的独立 Frozen 与正式 RL 配对。")
        if row.get("pilot_layout") != "prefix" or row["environment_seed"] != 50_000 + row["seed"]:
            raise ValueError("实际 seed 或前缀 Pilot 协议不匹配。")
        if method == PROPOSED and (row.get("reward_receiver_input") != "full_rx"
                or row.get("data_waveform_used_online") is not True
                or row.get("channel_frame_index") != row["frame"] - 1):
            raise ValueError("RL 整帧接收或从首帧连续物理年龄不匹配。")
        key = (row["snr_db"], row["seed"], row["frame"])
        if method in pairs[key]:
            raise ValueError("同一配对帧存在重复方法。")
        pairs[key][method] = row
    trajectories = defaultdict(list)
    for (snr, seed, frame), pair in pairs.items():
        if set(pair) != {FROZEN, PROPOSED}:
            raise ValueError("缺少独立 Frozen 或正式 RL。")
        hashes = {row["rx_sha256"] for row in pair.values()}
        if len(hashes) != 1 or any(len(value) != 64 for value in hashes):
            raise ValueError("同帧接收波形不一致。")
        trajectories[(snr, seed)].append((frame, pair))
    if not trajectories:
        raise ValueError("没有完整连续会话。")
    for trajectory in trajectories.values():
        trajectory.sort(key=lambda item: item[0])
        if [frame for frame, _ in trajectory] != list(range(1, expected_frames + 1)):
            raise ValueError("帧数不足、重复或不连续，不能生成完整会话结论。")
    snrs = sorted({snr for snr, _ in trajectories})
    seeds = sorted({seed for _, seed in trajectories})
    if set(trajectories) != {(snr, seed) for snr in snrs for seed in seeds}:
        raise ValueError("各 SNR 的独立 seed 集合不同。")
    results = []
    for snr in snrs:
        def paired_stats(start, stop):
            values = [{method: float(np.mean([pair[method]["ber_data"]
                      for _, pair in trajectories[(snr, seed)][start:stop]]))
                      for method in (FROZEN, PROPOSED)} for seed in seeds]
            return {"ber_data": {method: paired_interval([value[method] for value in values])
                                  for method in (FROZEN, PROPOSED)},
                    "online_vs_frozen_gain": paired_interval([value[FROZEN] - value[PROPOSED] for value in values])}

        third = max(1, expected_frames // 3)
        late_minus_early = []
        for seed in seeds:
            gains = [pair[FROZEN]["ber_data"] - pair[PROPOSED]["ber_data"]
                     for _, pair in trajectories[(snr, seed)]]
            late_minus_early.append(float(np.mean(gains[-third:]) - np.mean(gains[:third])))
        proposed = [pair[PROPOSED] for seed in seeds for _, pair in trajectories[(snr, seed)]]
        results.append({"snr_db": snr, **paired_stats(0, expected_frames),
            "late_minus_early_online_gain": paired_interval(late_minus_early),
            "early_late_frames_each": third,
            "segments": [{"first_frame": start + 1, "last_frame": start + segment_frames,
                "start_seconds": start * frame_seconds, "end_seconds": (start + segment_frames) * frame_seconds,
                **paired_stats(start, start + segment_frames)}
                for start in range(0, expected_frames, segment_frames)],
            "policy_action_frames": dict(Counter(row["policy_action_name"] for row in proposed)),
            "ppo_update_frames": sum(row["policy_loss"] is not None for row in proposed),
            "positive_reward_windows": sum(row["policy_window_closed"] and row["reward"] > 0 for row in proposed),
            "window_rollbacks": sum(row["window_reward_rollback"] for row in proposed),
            "max_retained_trajectory_peft_delta_norm": max(row["trajectory_peft_delta_norm_after_reward"] for row in proposed)})
    return {"schema_version": "continuous-rl-paired-diagnostic-v1", "diagnostic_only": True,
            "main_aggregation_allowed": False, "seeds": seeds, "frames_per_seed": expected_frames,
            "duration_seconds": expected_frames * frame_seconds, "per_snr": results,
            "interpretation": "独立 Frozen 配对的连续机制探测；没有传统主比较，单 seed 不生成区间。"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=2048)
    parser.add_argument("--segment-frames", type=int, default=512)
    args = parser.parse_args()
    source, target = Path(args.input), Path(args.output)
    if target.exists():
        raise ValueError("输出已存在，请使用新的路径。")
    raw = (source / "frame_metrics.jsonl").read_bytes()
    manifest = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    channel = manifest["effective_channel"]
    result = summarize_continuous_rows([json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()],
        expected_frames=args.frames, segment_frames=args.segment_frames,
        frame_seconds=channel["frame_len"] / channel["symbol_rate_hz"])
    result["raw_file_sha256"] = hashlib.sha256(raw).hexdigest()
    result["comparison_manifest"] = manifest
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(target), "duration_seconds": result["duration_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
