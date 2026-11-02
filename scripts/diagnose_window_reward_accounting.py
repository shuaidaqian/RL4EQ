# -*- coding: utf-8 -*-
"""只从 Pilot 控制日志复核窗口收益、更新成本与回滚，不读取 Data 指标。"""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


PROPOSED = "RL-Modulated Neural Block Equalizer"


def summarize_reward_windows(rows):
    """重建既有 reward，不改系数、不重选动作，也不使用 Data 选择规则。"""
    groups = defaultdict(list)
    for row in rows:
        if row["method"] == PROPOSED:
            if row.get("data_labels_used_online") is not False:
                raise ValueError("缺少在线 Data 标签隔离声明。")
            groups[(row["snr_db"], row["seed"])].append(row)
    if not groups:
        raise ValueError("没有正式离散 RL 记录。")
    windows, partial = [], []
    for (snr, seed), trajectory in sorted(groups.items()):
        trajectory.sort(key=lambda row: row["frame"])
        if [row["frame"] for row in trajectory] != list(range(1, len(trajectory) + 1)):
            raise ValueError("轨迹必须从首帧连续且没有重复。")
        pending = []
        for row in trajectory:
            pending.append(row)
            if not row["policy_window_closed"]:
                continue
            if len(pending) != row["window_size"] or len({x["policy_action_name"] for x in pending}) != 1:
                raise ValueError("窗口长度或窗口内动作不一致。")
            loss_gain = sum(x["reward_pilot_loss_before"] - x["reward_pilot_loss_after"] for x in pending) / len(pending)
            ber_gain = sum(x["reward_pilot_ber_before"] - x["reward_pilot_ber_after"] for x in pending) / len(pending)
            attempted_motion = sum(x["peft_delta_norm"] for x in pending)
            cost = .005 * (pending[0]["action_delta_norm"] + attempted_motion)
            utility = loss_gain + ber_gain
            reconstructed = utility - cost
            if not math.isfinite(reconstructed) or not math.isclose(reconstructed, row["reward"], abs_tol=1e-6, rel_tol=1e-5):
                raise ValueError("记录 reward 与 Pilot 改善及动作成本不能对齐。")
            windows.append({
                "snr_db": snr, "seed": seed, "first_frame": pending[0]["frame"],
                "last_frame": row["frame"], "action": row["policy_action_name"],
                "reward_loss_objective": row.get("reward_loss_objective", "bce"),
                "pilot_loss_gain_mean": loss_gain, "pilot_ber_gain_mean": ber_gain,
                "pilot_utility_before_cost": utility, "attempted_parameter_motion_sum": attempted_motion,
                "action_cost": cost, "reconstructed_reward": reconstructed,
                "recorded_reward": row["reward"],
                "positive_pilot_utility_cancelled_by_cost": utility > 0 and reconstructed <= 0,
                "accepted_candidate_frames": sum(x["peft_frame_accepted"] for x in pending)
                    if row["policy_action_name"].startswith("peft_") else 0,
                "window_rollback": row["window_reward_rollback"],
                "terminal_retained_delta_norm": row.get("trajectory_peft_delta_norm_after_reward"),
            })
            pending = []
        if pending:
            partial.append({"snr_db": snr, "seed": seed, "unclosed_frames": len(pending)})
    return {
        "schema_version": "pilot-window-reward-accounting-v1", "diagnostic_only": True,
        "data_metrics_read": False, "reward_rule_changed": False,
        "cost_rule": "0.005 * (modulation_delta_norm + sum(attempted_peft_delta_norm))",
        "benefit_rule": "mean(Pilot BER improvement) + mean(Pilot loss improvement)",
        "windows": windows, "partial_windows": partial,
        "interpretation": "逐帧验收与窗口回滚是不同决策；成本抵消正Pilot收益不证明这些候选有Data收益。",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, target = Path(args.input), Path(args.output)
    if target.exists():
        raise ValueError("输出已存在，请指定新的诊断路径。")
    raw = (source / "frame_metrics.jsonl").read_bytes()
    result = summarize_reward_windows([json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()])
    result["raw_file_sha256"] = hashlib.sha256(raw).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(target), "closed_windows": len(result["windows"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
