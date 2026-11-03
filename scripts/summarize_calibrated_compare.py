# -*- coding: utf-8 -*-
"""审计统一校准比较的配对完整性，并按独立信道 seed 汇总。"""

from __future__ import annotations

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

from baseline.calibration import CALIBRATED_METHODS
from evaluation.pilot_transfer import paired_interval
from evaluation.receiver_protocol import audit_lookahead_rows


FROZEN = "Frozen Offline NN"
PROPOSED = "RL-Modulated Neural Block Equalizer"
METHODS = (*CALIBRATED_METHODS, FROZEN, PROPOSED)


def summarize_rows(rows):
    """先拒绝不完整配对或信息隔离失败，再计算 seed 级区间。"""
    if not rows or {row["method"] for row in rows} != set(METHODS):
        raise ValueError("此诊断需要三类校准传统方法、Frozen 和正式 RL 方法。")
    lookahead_audit = audit_lookahead_rows(rows)
    groups = defaultdict(dict)
    protocols = set()
    receiver_protocols = set()
    for row in rows:
        key = (row["snr_db"], row["seed"], row["frame"])
        if row["method"] in groups[key]:
            raise ValueError("检测到重复逐帧记录。")
        groups[key][row["method"]] = row
        protocols.add(row.get("traditional_calibration_sha256"))
        if row["environment_seed"] != 50_000 + row["seed"]:
            raise ValueError("统一入口实际环境 seed 与记录索引不匹配。")
        if row["method"] == PROPOSED:
            receiver_protocol = row.get("reward_receiver_input", "pilot_only")
            adapt_protocol = row.get("adapt_receiver_input", "adapt_only")
            if (row.get("data_labels_used_online") is not False
                    or receiver_protocol not in {"pilot_only", "full_rx"}
                    or adapt_protocol not in {"adapt_only", "full_rx"}
                    or row.get("data_waveform_used_online") is not (
                        receiver_protocol == "full_rx" or (
                            adapt_protocol == "full_rx" and row.get("data_waveform_used_for_update") is True))):
                raise ValueError("RL 在线信息隔离声明缺失或失败。")
            receiver_protocols.add((receiver_protocol, adapt_protocol))
        if row["method"] in CALIBRATED_METHODS and (
            row.get("traditional_calibration_applied") is not True
            or row.get("uses_reward_or_data_labels") is not False
            or row.get("uses_neural_network") is not False or row.get("uses_rl") is not False
        ):
            raise ValueError("传统方法缺少校准或信息边界审计。")
    if len(protocols) != 1 or None in protocols:
        raise ValueError("配对记录混合了不同传统校准协议。")
    if len(receiver_protocols) != 1:
        raise ValueError("配对记录混合了不同 Reward 或 Adapt 接收协议。")
    for group in groups.values():
        if set(group) != set(METHODS):
            raise ValueError("同一帧缺少比较方法。")
        hashes = {row["rx_sha256"] for row in group.values()}
        if len(hashes) != 1 or any(len(value) != 64 for value in hashes):
            raise ValueError("同一帧的方法接收波形 SHA256 不一致。")
    snrs = sorted({key[0] for key in groups})
    if snrs != [0, 5, 10, 15]:
        raise ValueError("诊断汇总需要完整的四 SNR 主网格。")
    seed_sets = [{key[1] for key in groups if key[0] == snr} for snr in snrs]
    if any(current != seed_sets[0] for current in seed_sets):
        raise ValueError("四 SNR 的 seed 集合不完整。")
    seeds = sorted(seed_sets[0])
    frame_sets = [{key[2] for key in groups if key[:2] == (snr, seed)}
                  for snr in snrs for seed in seeds]
    if any(current != frame_sets[0] for current in frame_sets) or not frame_sets[0]:
        raise ValueError("各 seed/SNR 的帧集合不一致。")
    frames = sorted(frame_sets[0])
    if frames != list(range(1, len(frames) + 1)):
        raise ValueError("帧编号必须从 1 连续编号。")
    summaries = []
    for snr in snrs:
        seed_means = {}
        late_gains = []
        segment = max(1, len(frames) // 3)
        for seed in seeds:
            seed_means[seed] = {
                method: float(np.mean([groups[(snr, seed, frame)][method]["ber_data"] for frame in frames]))
                for method in METHODS
            }
            gains = [groups[(snr, seed, frame)][FROZEN]["ber_data"]
                     - groups[(snr, seed, frame)][PROPOSED]["ber_data"] for frame in frames]
            late_gains.append(float(np.mean(gains[-segment:]) - np.mean(gains[:segment])))
        selected = [row for row in rows if row["snr_db"] == snr and row["method"] == PROPOSED]
        summaries.append({
            "snr_db": snr,
            "ber_data": {method: paired_interval([seed_means[seed][method] for seed in seeds]) for method in METHODS},
            "online_vs_frozen_gain": paired_interval([seed_means[seed][FROZEN] - seed_means[seed][PROPOSED] for seed in seeds]),
            "online_vs_traditional_gain": {method: paired_interval([
                seed_means[seed][method] - seed_means[seed][PROPOSED] for seed in seeds
            ]) for method in CALIBRATED_METHODS},
            "late_minus_early_online_gain": paired_interval(late_gains),
            "policy_action_frames": dict(Counter(row["policy_action_name"] for row in selected)),
            "peft_attempt_frames": sum(row["policy_action_name"].startswith("peft_") for row in selected),
            "peft_accepted_frames": sum(row["policy_action_name"].startswith("peft_")
                                        and row["peft_frame_accepted"] for row in selected),
            "ppo_update_frames": sum(row.get("policy_loss") is not None for row in selected),
            "closed_windows": sum(bool(row.get("policy_window_closed", False)) for row in selected),
            "positive_reward_windows": sum(bool(row.get("policy_window_closed", False))
                                           and row["reward"] > 0 for row in selected),
            "window_rollbacks": sum(bool(row.get("window_reward_rollback", False)) for row in selected),
            "max_retained_trajectory_peft_delta_norm": (
                max(row["trajectory_peft_delta_norm_after_reward"] for row in selected)
                if all("trajectory_peft_delta_norm_after_reward" in row for row in selected) else None),
            "max_data_logit_abs_change": max(row["data_logits_abs_delta_mean_eval_only"] for row in selected),
        })
    return {"schema_version": "calibrated-compare-diagnostic-v1", "diagnostic_only": True,
            "main_aggregation_allowed": False, "frame_records": len(rows), "paired_frames": len(groups),
            "seeds": seeds, "frames_per_seed": len(frames), "segment_frames": segment,
            "traditional_calibration_sha256": next(iter(protocols)), "per_snr": summaries,
            "information_audit": {"shared_rx_sha256_passed": True,
                                  **lookahead_audit,
                                  "rl_declares_no_data_labels_online": True,
                                  "reward_receiver_input": next(iter(receiver_protocols))[0],
                                  "adapt_receiver_input": next(iter(receiver_protocols))[1],
                                  "data_waveform_used_online": any(
                                      row.get("data_waveform_used_online") is True for row in rows if row["method"] == PROPOSED),
                                  "traditional_declares_no_hidden_labels_nn_or_rl": True},
            "interpretation": "信息声明与哈希需结合隔离测试；短轨迹不证明长期 PPO 学习，零经验区间不等于总体零风险。"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, target = Path(args.input), Path(args.output)
    if target.exists():
        raise ValueError("汇总输出已存在，请指定新路径。")
    path = source / "frame_metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = summarize_rows(rows)
    result["raw_file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    result["comparison_manifest"] = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(target), "frame_records": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
