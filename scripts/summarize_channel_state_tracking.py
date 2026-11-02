# -*- coding: utf-8 -*-
"""核验预先固定的 128 帧状态跟踪实验；仅作事后统计，不生成在线策略。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pilot_transfer import paired_interval, summarize_transfer


def summarize(root):
    summary_path, rows_path = root / "summary.json", root / "frame_metrics.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    manifest = summary["manifest"]
    assert manifest["persistent"] and manifest["frames"] == 128
    assert manifest["seeds"] == [84001, 84002, 84003]
    assert manifest["snrs"] == [0, 5, 10, 15]
    assert len(rows) == 3072
    assert len({(r["branch"], r["snr_db"], r["seed"], r["frame"]) for r in rows}) == 3072
    assert all(not r["data_labels_used_online"] and r["episode_delta_norm_online"] <= .200001 for r in rows)
    assert summary["per_snr"] == summarize_transfer(rows)
    baseline = {(r["snr_db"], r["seed"], r["frame"]): r for r in rows if r["branch"] == "baseline"}
    assert all(r["input_sha256"] == baseline[(r["snr_db"], r["seed"], r["frame"])]["input_sha256"] for r in rows)
    audits = []
    for snr in manifest["snrs"]:
        for branch in ["baseline", "post_adapt"]:
            item = {"snr_db": snr, "branch": branch, "seed_results": []}
            for seed in manifest["seeds"]:
                subset = sorted([r for r in rows if (r["snr_db"], r["branch"], r["seed"]) ==
                                 (snr, branch, seed)], key=lambda r: r["frame"])
                assert [r["frame"] for r in subset] == list(range(128))
                assert subset[0]["episode_delta_norm_before"] == 0
                for prev, cur in zip(subset, subset[1:]):
                    assert prev["episode_delta_norm_online"] == cur["episode_delta_norm_before"]
                gain = lambda values: float(np.mean([r["ber_frozen"] - r["ber_online"] for r in values]))
                accepted = [r for r in subset if r["accepted"]]
                rejected = [r for r in subset if not r["accepted"]]
                item["seed_results"].append({
                    "seed": seed, "gain": gain(subset), "early_gain": gain(subset[:42]),
                    "late_gain": gain(subset[-42:]),
                    "final_delta_norm": subset[-1]["episode_delta_norm_online"],
                    "max_delta_norm": max(r["episode_delta_norm_online"] for r in subset),
                    "radius_bound_frames": sum(r["episode_delta_norm_online"] >= .199999 for r in subset),
                    "accepted": len(accepted),
                    "helpful_acceptances": sum(r["ber_candidate"] < r["ber_before_update"] for r in accepted),
                    "harmful_acceptances": sum(r["ber_candidate"] > r["ber_before_update"] for r in accepted),
                    "helpful_rejections": sum(r["ber_candidate"] < r["ber_before_update"] for r in rejected),
                    "harmful_rejections": sum(r["ber_candidate"] > r["ber_before_update"] for r in rejected),
                    "zero_reward_ber_frames": sum(r["reward_ber_before"] == r["reward_ber_after"] == 0 for r in subset),
                })
            for key in ("early_gain", "late_gain"):
                item[key] = paired_interval([r[key] for r in item["seed_results"]])
            audits.append(item)
    summary["episode_audit"] = audits
    summary["integrity"] = {
        "contexts": 1536, "rows": 3072, "waveforms_matched": True,
        "episode_resets_verified": True, "cross_frame_state_verified": True,
        "radius_bound_verified": True, "online_data_labels_used": False,
        "data_waveform_used_for_update": False, "data_waveform_used_for_guard": True,
        "metadata_correction": "原日志误将信号 loss 的 Pilot 波形范围用于整个 guard 字段；Reward BER 实际读取整帧接收波形。原始文件不改写，标签边界不变。",
    }
    summary["source_files"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [summary_path, rows_path]}
    summary["status"] = "partial_0db_mechanism_only_goal_incomplete"
    summary["interpretation"] = {
        "all_snr_gain_confirmed": False, "late_advantage_confirmed": False,
        "formal_rl_validated": False, "traditional_comparison_completed": False,
        "multiple_comparison_adjusted": False, "wall_clock_seconds": None,
        "next_step": "训练分布上固定首帧动作，诊断跨帧有效期与当前Pilot反馈排序。",
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"per_snr": summary["per_snr"], "episode_audit": summary["episode_audit"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
