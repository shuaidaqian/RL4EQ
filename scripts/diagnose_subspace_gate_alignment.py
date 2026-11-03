# -*- coding: utf-8 -*-
"""只读审计子空间验证Pilot守门与动作后Data变化的对应关系。"""

from __future__ import annotations

import argparse
import hashlib
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.metrics import spearman_reward_data


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _correlation(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    result = spearman_reward_data(x, y, threshold=0.0)
    return float(result.correlation)


def analyze(rows: list[dict]) -> dict:
    """以seed会话为汇总单位，逐帧相关性仅作为会话内描述量。"""
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("role") != "evaluation":
            raise ValueError("只允许读取evaluation轨迹。")
        if row.get("training") != "B_post_adapt":
            continue
        if row.get("data_labels_used_for_control") is not False:
            raise ValueError("发现Data标签参与在线控制的记录。")
        grouped[(int(row["snr_db"]), int(row["seed"]))].append(row)

    sessions = []
    for (snr, seed), session in sorted(grouped.items()):
        session.sort(key=lambda row: int(row["frame"]))
        pilot_candidate_gain = [float(row["validation_loss_before"]) - float(row["validation_loss_candidate"])
                                for row in session]
        pilot_retained_gain = [float(row["validation_loss_before"]) - float(row["validation_loss_retained"])
                               for row in session]
        data_bce_gain = [float(row["frozen_data_loss"]) - float(row["online_data_loss"])
                         for row in session]
        data_ber_gain = [float(row["ber_frozen"]) - float(row["ber_online"])
                         for row in session]
        accepted = [row for row in session if row["candidate_accepted"]]
        rejected = [row for row in session if not row["candidate_accepted"]]
        sessions.append({
            "snr_db": snr,
            "seed": seed,
            "frames": len(session),
            "candidate_accepted": len(accepted),
            "reset_to_origin": sum(bool(row["reset_to_origin"]) for row in session),
            "pilot_candidate_gain_mean": _mean(pilot_candidate_gain),
            "pilot_retained_gain_mean": _mean(pilot_retained_gain),
            "retained_data_bce_gain_mean": _mean(data_bce_gain),
            "retained_data_ber_gain_mean": _mean(data_ber_gain),
            "retained_pilot_to_data_bce_spearman_within_session": _correlation(pilot_retained_gain, data_bce_gain),
            "accepted_retained_data_bce_gain_mean": _mean([
                float(row["frozen_data_loss"]) - float(row["online_data_loss"]) for row in accepted
            ]),
            "rejected_retained_data_bce_gain_mean": _mean([
                float(row["frozen_data_loss"]) - float(row["online_data_loss"]) for row in rejected
            ]),
            "accepted_retained_data_ber_gain_mean": _mean([
                float(row["ber_frozen"]) - float(row["ber_online"]) for row in accepted
            ]),
        })

    by_snr = []
    for snr in sorted({row["snr_db"] for row in sessions}):
        items = [row for row in sessions if row["snr_db"] == snr]
        by_snr.append({
            "snr_db": snr,
            "independent_seed_sessions": len(items),
            "frames": sum(row["frames"] for row in items),
            "candidate_accepted": sum(row["candidate_accepted"] for row in items),
            "reset_to_origin": sum(row["reset_to_origin"] for row in items),
            "mean_seed_pilot_candidate_gain": _mean([row["pilot_candidate_gain_mean"] for row in items]),
            "mean_seed_pilot_retained_gain": _mean([row["pilot_retained_gain_mean"] for row in items]),
            "mean_seed_retained_data_bce_gain": _mean([row["retained_data_bce_gain_mean"] for row in items]),
            "mean_seed_retained_data_ber_gain": _mean([row["retained_data_ber_gain_mean"] for row in items]),
            "mean_seed_session_spearman": _mean([
                row["retained_pilot_to_data_bce_spearman_within_session"]
                for row in items if row["retained_pilot_to_data_bce_spearman_within_session"] is not None
            ]),
            "accepted_retained_data_bce_gain_mean": _mean([
                float(row["accepted_retained_data_bce_gain_mean"])
                for row in items if row["accepted_retained_data_bce_gain_mean"] is not None
            ]),
            "rejected_retained_data_bce_gain_mean": _mean([
                float(row["rejected_retained_data_bce_gain_mean"])
                for row in items if row["rejected_retained_data_bce_gain_mean"] is not None
            ]),
            "accepted_retained_data_ber_gain_mean": _mean([
                float(row["accepted_retained_data_ber_gain_mean"])
                for row in items if row["accepted_retained_data_ber_gain_mean"] is not None
            ]),
        })

    return {
        "schema_version": "subspace-gate-alignment-diagnostic-v1",
        "diagnostic_only": True,
        "source_role": "evaluation",
        "method_arm": "B_post_adapt",
        "snrs_db": sorted({row["snr_db"] for row in sessions}),
        "by_snr": by_snr,
        "sessions": sessions,
        "independent_unit": "seed-session; 3 repeated development seeds per SNR",
        "interpretation_limit": "逐帧相关只在seed会话内计算；同一信道持续状态下帧不视为独立。Data字段仅作事后诊断，不参与在线控制。轨迹未记录Adapt与Data梯度向量，不能从本归档估算梯度方向余弦。",
        "online_control_uses_data_labels": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, target = Path(args.trajectory), Path(args.output)
    if target.exists():
        raise ValueError("诊断输出已存在，请提供新路径。")
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    result = analyze(rows)
    result["source_trajectory"] = str(source)
    result["source_trajectory_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["by_snr"], ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
