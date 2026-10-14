# -*- coding: utf-8 -*-
"""从忽略的原始 replay 日志生成可提交的精简证据文件。"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "replay_protocol_audit_20261014"
OUTPUT = ROOT / "docs" / "results" / "pilot_replay_protocol_audit_20261014.json"
CHECKPOINT = ROOT / "pretrained" / "eme_bce_all_32_20260905_pilot256" / "model_best.pt"
EXPECTED_CHECKPOINT_SHA256 = "8cd3912d1c957c9689c51730aff9d27a8f74942186c802a7483e5781af65aec4"

RUNS = {
    "historical_bce_3seed_8f": "cumulative_aligned_3seed_8f.json",
    "future_bce_3seed_10f": "cumulative_future_reward_3seed_10f.json",
    "future_signal_3seed_10f": "cumulative_future_signal_reward_3seed_10f.json",
    "future_signal_newseed_24f": "cumulative_future_signal_reward_newseed_24f.json",
    "hold_reward_diagnostic_seed4": "cumulative_signal_reward_hold_monitor_s4_24f.json",
    "monitor2_seed3_4_5": "cumulative_future_signal_reward_monitor2_newseed_24f.json",
    "monitor2_seed6_7_8": "cumulative_future_signal_reward_monitor2_freshseed_24f.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _trial_summary(rows: list[dict], arguments: dict) -> dict:
    rows = sorted(rows, key=lambda row: row["test_frame_index"])
    first = rows[0]
    initial_accept = bool(first.get("initial_reward_gate_accepted", first["reward_gate_accepted"]))
    rollback_rows = [row for row in rows if row.get("monitor_reward_rollback", False)]
    test_frames = [row["test_frame_index"] for row in rows]
    pilot_frames = arguments["pilot_total"]
    frame_len = int(first["environment_config"]["frame_len"])
    data_bits_per_frame = frame_len - pilot_frames
    retained = [row["test_frame_index"] for row in rows if row["reward_gate_accepted"]]
    error_totals = {
        label: sum(round(row[key] * data_bits_per_frame) for row in rows)
        for label, key in (
            ("frozen_data_errors", "frozen_data_ber"),
            ("raw_candidate_data_errors", "candidate_data_ber"),
            ("protected_data_errors", "data_ber"),
        )
    }
    one_time_hold_errors = (
        error_totals["raw_candidate_data_errors"] if initial_accept else error_totals["frozen_data_errors"]
    )
    return {
        "snr_db": first["snr_db"],
        "seed": first["seed"],
        "train_frame_count": first["train_frame_count"],
        "train_frame_indices": first["train_frame_indices"],
        "reward_validation_frame_indices": first.get("reward_validation_frame_indices", []),
        "data_test_frame_indices": test_frames,
        "data_test_frames": len(rows),
        "data_bits": len(rows) * data_bits_per_frame,
        **error_totals,
        "initial_reward_accepted": initial_accept,
        "reward_acceptance_mode": first.get("reward_gate_mode", "historical_reward"),
        "reward_score": first.get("reward_score", "bce"),
        "train_reward_improvements": first["train_reward_improvements"],
        "reward_validation_improvements": first.get("validation_reward_improvements", []),
        "validation_reward_error_count_changes": first.get("gate_reward_error_count_changes", []),
        "monitor_reward_window": first.get("monitor_reward_window", 0),
        "monitor_rollback_frame_indices": [row["test_frame_index"] for row in rollback_rows],
        "retained_candidate_frame_indices": retained,
        "retained_candidate_frame_fraction": len(retained) / len(rows),
        "candidate_active_on_last_test_frame": bool(rows[-1]["reward_gate_accepted"]),
        "updated_parameter_count": first["updated_parameter_count"],
        "state_mapping_delta_norm": first["state_mapping_delta_norm"],
        "direct_phase_cfo_delta_norm": first["direct_phase_cfo_delta_norm"],
        "adapt_reconstruction_before": first["adapt_reconstruction_before"],
        "adapt_reconstruction_after": first["adapt_reconstruction_after"],
        "frozen_data_ber": _mean(rows, "frozen_data_ber"),
        "frozen_data_bce": _mean(rows, "frozen_data_bce"),
        "raw_candidate_data_ber": _mean(rows, "candidate_data_ber"),
        "raw_candidate_data_ber_improvement": _mean(rows, "candidate_data_ber_improvement"),
        "protected_data_ber": _mean(rows, "data_ber"),
        "protected_data_ber_improvement": _mean(rows, "data_ber_improvement"),
        "raw_candidate_data_bce": _mean(rows, "candidate_data_bce"),
        "raw_candidate_data_bce_improvement": _mean(rows, "candidate_data_bce_improvement"),
        "protected_data_bce": _mean(rows, "data_bce"),
        "protected_data_bce_improvement": _mean(rows, "data_bce_improvement"),
        "raw_candidate_logits_abs_mean_delta": _mean(rows, "candidate_data_logits_abs_mean_delta"),
        "raw_candidate_logits_direction_change_fraction": _mean(
            rows, "candidate_data_logits_direction_change_fraction"),
        "raw_candidate_soft_abs_mean_delta": _mean(rows, "candidate_data_soft_abs_mean_delta"),
        "one_time_hold_posthoc": {
            "is_independent_experiment": False,
            "tail_mode": "shared_frozen_soft_tail",
            "interpretation": "仅根据已记录的原始候选和初始 Pilot 决定重建一次验收持有，不用于选参。",
            "data_errors": one_time_hold_errors,
            "extra_errors_over_frozen": one_time_hold_errors - error_totals["frozen_data_errors"],
        },
    }


def _snr_summary(trials: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for trial in trials:
        grouped[(trial["train_frame_count"], trial["snr_db"])].append(trial)
    result = []
    for (count, snr), items in sorted(grouped.items()):
        bits = sum(item["data_bits"] for item in items)
        frozen_errors = sum(item["frozen_data_errors"] for item in items)
        candidate_errors = sum(item["raw_candidate_data_errors"] for item in items)
        protected_errors = sum(item["protected_data_errors"] for item in items)
        result.append({
            "train_frame_count": count, "snr_db": snr, "seed_count": len(items),
            "data_test_frames": sum(item["data_test_frames"] for item in items),
            "data_bits": bits,
            "frozen_data_ber": frozen_errors / bits,
            "raw_candidate_data_ber": candidate_errors / bits,
            "protected_data_ber": protected_errors / bits,
            "protected_ber_gain_percentage_points": 100 * (frozen_errors - protected_errors) / bits,
            "initial_accepted_seed_count": sum(item["initial_reward_accepted"] for item in items),
            "rollback_seed_count": sum(bool(item["monitor_rollback_frame_indices"]) for item in items),
            "retained_candidate_frame_fraction": sum(len(item["retained_candidate_frame_indices"]) for item in items)
            / sum(item["data_test_frames"] for item in items),
            "positive_seed_count": sum(item["protected_data_errors"] < item["frozen_data_errors"] for item in items),
            "negative_seed_count": sum(item["protected_data_errors"] > item["frozen_data_errors"] for item in items),
        })
    return result


def _build_run(name: str, filename: str) -> dict:
    source = LOG_DIR / filename
    payload = json.loads(source.read_text(encoding="utf-8"))
    arguments = payload["arguments"]
    grouped: dict[tuple[int, float, int], list[dict]] = defaultdict(list)
    for row in payload["rows"]:
        environment = row["environment_config"]
        if (row["main_aggregation_allowed"] or row["data_labels_used_online"]
                or row["condition_source"] != "pilot_cir_phase"
                or row["tail_mode"] != "shared_frozen_soft_tail"
                or environment["profile_name"] != "eme_long_memory_v2"
                or environment["frame_len"] != 1024 or environment["max_delay"] != 116
                or row["updated_parameter_count"] != 14):
            raise ValueError(f"原始日志不符合本轮固定协议：{filename}")
        grouped[(row["train_frame_count"], row["snr_db"], row["seed"])].append(row)
    trials = [
        _trial_summary(rows, arguments)
        for _, rows in sorted(grouped.items(), key=lambda item: item[0])
    ]
    hold_rows = []
    if name == "hold_reward_diagnostic_seed4":
        hold_rows = [
            {key: row[key] for key in (
                "test_frame_index", "test_reward_loss_frozen", "test_reward_loss_candidate",
                "test_reward_improvement", "test_reward_error_count_frozen",
                "test_reward_error_count_candidate", "frozen_data_ber", "candidate_data_ber",
                "candidate_data_ber_improvement", "candidate_data_bce_improvement",
            )}
            for row in payload["rows"]
        ]
    return {
        "name": name,
        "source_log": f"logs/replay_protocol_audit_20261014/{filename}",
        "source_log_sha256": _sha256(source),
        "metric": payload["metric"],
        "arguments": arguments,
        "main_aggregation_allowed": payload["main_aggregation_allowed"],
        "data_labels_used_online": payload["data_labels_used_online"],
        "test_reward_role": (
            "ongoing_pilot_acceptance_and_rollback" if arguments.get("monitor_reward_window", 0)
            else "post_acceptance_monitor_diagnostic_or_unrecorded"
        ),
        "metadata_note": "作用以 CLI 和逐帧接受/回滚字段为准；较早监测日志的 test_reward_role 字样尚未修正。",
        "trial_count": len(trials),
        "trials": trials,
        "summary": _snr_summary(trials),
        "hold_reward_diagnostic_rows": hold_rows,
    }


def main() -> None:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"找不到固定 checkpoint：{CHECKPOINT}")
    checkpoint_sha256 = _sha256(CHECKPOINT)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("固定 checkpoint 的 SHA256 已变化，不能生成本轮结果证据。")
    baseline = json.loads((LOG_DIR / RUNS["future_signal_newseed_24f"]).read_text(encoding="utf-8"))
    configuration = baseline["experiment_config"]
    output = {
        "schema_version": "pilot-replay-protocol-audit-v1",
        "generated_from_raw_logs": True,
        "checkpoint": {
            "path": "pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt",
            "sha256": checkpoint_sha256,
        },
        "configuration": {
            "path": "configs/online_phase_signal_state_embedding_formal_moderate.json",
            "channel_profile": configuration["channel_profile"],
            "level": configuration["level"],
            "frame_len": configuration["frame_len"],
            "max_delay": configuration["max_delay"],
            "coherence_time_seconds": configuration["coherence_time_seconds"],
            "impairment_profile": configuration["impairment_profile"],
            "model_config": baseline["model_config"],
            "experiment_config_snapshot": configuration,
            "source_config_sha256": _sha256(ROOT / "configs" / "online_phase_signal_state_embedding_formal_moderate.json"),
            "environment_config_snapshot": baseline["rows"][0]["environment_config"],
            "replay_protocol_version": baseline["rows"][0]["replay_protocol_version"],
            "optimizer": "AdamW",
            "learning_rate": configuration["online_adaptation_learning_rate"],
            "update_steps": configuration["online_adaptation_steps"],
            "max_parameter_delta_norm": configuration["online_adaptation_max_delta_norm"],
            "proximal_weight": configuration["online_adaptation_proximal_weight"],
            "pilot_protocol": {
                "layout": configuration["pilot_layout"],
                "pilot_total": configuration["pilot_total"],
                "reward_pilot_total": configuration["reward_pilot_total"],
                "adapt_pilot_total": configuration["pilot_total"] - configuration["reward_pilot_total"],
                "data_symbols_per_frame": configuration["frame_len"] - configuration["pilot_total"],
            },
            "condition_source": "pilot_cir_phase",
            "tail_mode": "shared_frozen_soft_tail",
            "updated_parameter_count": 14,
        },
        "acceptance_rules": {
            "minimum_mean_reward_improvement": configuration["online_adaptation_min_reward_improvement"],
            "cross_frame_tolerance": configuration["online_cross_frame_rollback_tolerance"],
            "freeze_below_snr_db": configuration["online_adaptation_freeze_below_snr_db"],
            "monitor_window_for_monitor_runs": 2,
            "signal_reconstruction_hard_error_guard": True,
            "bce_hard_error_guard": False,
            "data_labels_used_for_acceptance": False,
        },
        "runs": [_build_run(name, filename) for name, filename in RUNS.items()],
        "interpretation": {
            "stable_online_gain_demonstrated": False,
            "monitoring_effect": "可减少负收益，但该两组监测验证均未在 Data 测试阶段保留候选。",
            "scope_limit": "小样本 replay 诊断；不属于正式主实验，不用于主结果平均。",
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"已生成 {OUTPUT.relative_to(ROOT)}，包含 {sum(run['trial_count'] for run in output['runs'])} 条逐 seed 结果。")


if __name__ == "__main__":
    main()
