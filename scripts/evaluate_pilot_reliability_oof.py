# -*- coding: utf-8 -*-
"""按 seed 留出评估 Pilot-only 候选可靠性排序。

输入是候选 replay 的事后诊断 JSON。特征白名单只包含 Pilot、状态、SNR 和候选类型；
Data BER/BCE 只作为离线训练目标和评估指标，不会参与在线更新或候选选择。
该脚本只评估跨 seed 的预测能力，不保存或部署在线模型。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.metrics import spearman_reward_data


FEATURE_NAMES = (
    "snr_db",
    "phase0",
    "cfo_cycles_per_symbol",
    "phase_residual_variance",
    "phase_residual_energy",
    "reconstruction_error",
    "confidence",
    "adapt_bce_improvement",
    "boundary_loss_improvement",
    "selective_boundary_loss_improvement",
    "pilot_reconstruction_improvement",
    "pilot_reconstruction_error_after",
    "peft_delta_norm",
    "is_identity_candidate",
    "is_signal_reconstruction_candidate",
)


def _feature_vector(row: dict) -> np.ndarray:
    """只从允许的 Pilot 可见字段构造特征。"""

    values = [float(row.get(name, 0.0)) for name in FEATURE_NAMES[:-2]]
    values.extend(
        [
            float(row.get("candidate_name") == "identity"),
            float(row.get("objective") == "pilot_signal_reconstruction"),
        ]
    )
    vector = np.asarray(values, dtype=np.float64)
    if not np.isfinite(vector).all():
        raise ValueError("Pilot 特征必须全部为有限数。")
    return vector


def _ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """只用训练 seed 的均值、方差和 L2 线性回归预测留出 seed。"""

    if alpha < 0.0:
        raise ValueError("alpha 不能为负数。")
    if train_x.ndim != 2 or test_x.ndim != 2 or train_x.shape[1] != test_x.shape[1]:
        raise ValueError("训练和测试特征必须是列数一致的二维数组。")
    x_mean = train_x.mean(axis=0)
    x_scale = train_x.std(axis=0)
    x_scale[x_scale < 1e-8] = 1.0
    x_train = (train_x - x_mean) / x_scale
    x_test = (test_x - x_mean) / x_scale
    y_mean = float(train_y.mean())
    y_centered = train_y - y_mean
    gram = x_train.T @ x_train + float(alpha) * np.eye(x_train.shape[1], dtype=np.float64)
    weights = np.linalg.solve(gram, x_train.T @ y_centered)
    return y_mean + x_test @ weights


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return float(np.mean(items)) if items else math.nan


def _metric_summary(rows: list[dict], *, score_name: str, target_name: str) -> dict:
    scores = [float(row[score_name]) for row in rows]
    targets = [float(row[target_name]) for row in rows]
    result = spearman_reward_data(scores, targets, threshold=0.6)
    return {
        "spearman": float(result.correlation),
        "sample_count": int(result.n),
        "spearman_gate_pass": bool(result.passed),
    }


def _select_one_per_trial(rows: list[dict], *, score_name: str, gate_at_zero: bool) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["snr_db"], row["seed"], row["frame_index"])].append(row)

    selected = []
    for key, candidates in sorted(grouped.items()):
        identity = next((row for row in candidates if row["candidate_name"] == "identity"), None)
        if identity is None:
            raise ValueError(f"试验 {key} 缺少 identity 候选，无法评估冻结策略。")
        best = max(candidates, key=lambda row: float(row[score_name]))
        if gate_at_zero and float(best[score_name]) <= 0.0:
            best = identity
        selected.append(best)
    return selected


def evaluate_rows(rows: list[dict], *, alpha: float = 10.0) -> dict:
    """执行留一 seed 交叉验证，并只在 OOF 预测上评估候选选择。"""

    if not rows:
        raise ValueError("候选 replay 不能为空。")
    if any(bool(row.get("data_labels_used_online", False)) for row in rows):
        raise ValueError("输入日志声称在线使用了 Data 标签，不能用于此校准诊断。")
    if any(bool(row.get("data_waveform_used_online", False)) for row in rows):
        raise ValueError("输入日志声称在线使用了 Data 波形，不能用于此校准诊断。")

    normalized = []
    for source in rows:
        row = dict(source)
        for required in ("seed", "snr_db", "frame_index", "candidate_name", "data_ber_improvement", "data_bce_improvement"):
            if required not in row:
                raise ValueError(f"候选 replay 缺少字段：{required}")
        row["seed"] = int(row["seed"])
        row["snr_db"] = float(row["snr_db"])
        row["frame_index"] = int(row["frame_index"])
        row["data_ber_improvement"] = float(row["data_ber_improvement"])
        row["data_bce_improvement"] = float(row["data_bce_improvement"])
        if not math.isfinite(row["data_ber_improvement"]) or not math.isfinite(row["data_bce_improvement"]):
            raise ValueError("Data 诊断目标必须全部为有限数。")
        # Reward 指标不参与可靠性模型拟合；它只留作现有验收策略的离线对照。
        row["pilot_score"] = float(
            row.get("pilot_holdout_loss_improvement", row.get("reward_loss_improvement", 0.0))
        )
        row["features"] = _feature_vector(row)
        normalized.append(row)

    seeds = sorted({row["seed"] for row in normalized})
    if len(seeds) < 2:
        raise ValueError("至少需要两个不同 seed 才能进行留出评估。")

    predictions = []
    fold_summaries = []
    for heldout_seed in seeds:
        train = [row for row in normalized if row["seed"] != heldout_seed]
        test = [row for row in normalized if row["seed"] == heldout_seed]
        if not train or not test:
            continue
        x_train = np.stack([row["features"] for row in train])
        x_test = np.stack([row["features"] for row in test])
        for target_name, prediction_name in (
            ("data_bce_improvement", "predicted_data_bce_improvement"),
            ("data_ber_improvement", "predicted_data_ber_improvement"),
        ):
            y_train = np.asarray([row[target_name] for row in train], dtype=np.float64)
            y_test = np.asarray([row[target_name] for row in test], dtype=np.float64)
            predicted = _ridge_predict(x_train, y_train, x_test, alpha)
            fold_summaries.append(
                {
                    "heldout_seed": int(heldout_seed),
                    "training_seeds": [seed for seed in seeds if seed != heldout_seed],
                    "target": target_name,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "spearman": float(
                        spearman_reward_data(predicted.tolist(), y_test.tolist(), threshold=0.6).correlation
                    ),
                    "mae": float(np.mean(np.abs(predicted - y_test))),
                }
            )
            for row, value in zip(test, predicted):
                if prediction_name == "predicted_data_bce_improvement":
                    item = dict(row)
                    item[prediction_name] = float(value)
                    predictions.append(item)
                else:
                    match = next(
                        item
                        for item in predictions
                        if item["seed"] == row["seed"]
                        and item["snr_db"] == row["snr_db"]
                        and item["frame_index"] == row["frame_index"]
                        and item["candidate_name"] == row["candidate_name"]
                    )
                    match[prediction_name] = float(value)

    ber_rows = [row for row in predictions if "predicted_data_ber_improvement" in row]
    bce_rows = [row for row in predictions if "predicted_data_bce_improvement" in row]
    candidate_ber_rows = [row for row in ber_rows if row["candidate_name"] != "identity"]
    candidate_bce_rows = [row for row in bce_rows if row["candidate_name"] != "identity"]
    pilot_candidate_rows = [row for row in predictions if row["candidate_name"] != "identity"]

    policies = {}
    freeze_selected = [
        next(row for row in candidates if row["candidate_name"] == "identity")
        for _, candidates in _trial_groups(predictions)
    ]
    policies["freeze"] = _policy_summary(freeze_selected)
    policy_specs = (
        ("pilot_reward", "pilot_score"),
        ("oof_reliability", "predicted_data_ber_improvement"),
        ("offline_bce_oracle", "data_bce_improvement"),
    )
    for policy_name, score_name in policy_specs:
        selected = _select_one_per_trial(predictions, score_name=score_name, gate_at_zero=True)
        policies[policy_name] = _policy_summary(selected)

    pilot_ber = _metric_summary(predictions, score_name="pilot_score", target_name="data_ber_improvement")
    reliability_ber = _metric_summary(
        ber_rows, score_name="predicted_data_ber_improvement", target_name="data_ber_improvement"
    )
    reliability_bce = _metric_summary(
        bce_rows,
        score_name="predicted_data_bce_improvement",
        target_name="data_bce_improvement",
    )
    pilot_candidate_ber = _metric_summary(
        pilot_candidate_rows,
        score_name="pilot_score",
        target_name="data_ber_improvement",
    )
    reliability_candidate_ber = _metric_summary(
        candidate_ber_rows,
        score_name="predicted_data_ber_improvement",
        target_name="data_ber_improvement",
    )
    reliability_candidate_bce = _metric_summary(
        candidate_bce_rows,
        score_name="predicted_data_bce_improvement",
        target_name="data_bce_improvement",
    )
    return {
        "schema_version": "pilot-reliability-oof-v1",
        "fold_strategy": "leave_one_seed_out",
        "heldout_seeds": seeds,
        "feature_names": list(FEATURE_NAMES),
        "regularization_alpha": float(alpha),
        "candidate_rows": len(normalized),
        "trial_count": len({(row["snr_db"], row["seed"], row["frame_index"]) for row in normalized}),
        "data_labels_used_online": False,
        "diagnostic_uses_data_labels": True,
        "offline_targets_include_data_labels": True,
        "pilot_reward_vs_data_ber": pilot_ber,
        "pilot_reward_vs_nonidentity_data_ber": pilot_candidate_ber,
        "oof_reliability_vs_data_ber": reliability_ber,
        "oof_reliability_vs_data_bce": reliability_bce,
        "oof_reliability_nonidentity_vs_data_ber": reliability_candidate_ber,
        "oof_reliability_nonidentity_vs_data_bce": reliability_candidate_bce,
        "reliability_policy_target": "data_ber_improvement",
        "oof_policy_comparison": policies,
        "fold_summaries": fold_summaries,
        "oof_predictions": [
            {
                key: value
                for key, value in row.items()
                if key not in {"features"}
            }
            for row in predictions
        ],
    }


def _trial_groups(rows: list[dict]) -> list[tuple[tuple, list[dict]]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["snr_db"], row["seed"], row["frame_index"])].append(row)
    return sorted(grouped.items())


def _policy_summary(selected: list[dict]) -> dict:
    return {
        "trial_count": len(selected),
        "update_coverage": float(np.mean([row["candidate_name"] != "identity" for row in selected])),
        "mean_data_ber_improvement": _mean(row["data_ber_improvement"] for row in selected),
        "mean_data_bce_improvement": _mean(row["data_bce_improvement"] for row in selected),
        "positive_ber_trial_fraction": float(
            np.mean([row["data_ber_improvement"] > 0.0 for row in selected])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="候选 replay JSON 路径。")
    parser.add_argument("--output", required=True, help="留一 seed 结果 JSON 路径。")
    parser.add_argument("--alpha", type=float, default=10.0, help="岭回归 L2 系数。")
    args = parser.parse_args()

    input_path = Path(args.input)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("输入 JSON 必须包含 rows 数组。")
    result = evaluate_rows(rows, alpha=float(args.alpha))
    result["source"] = str(input_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8"
    )
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
