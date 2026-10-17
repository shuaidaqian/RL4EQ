# -*- coding: utf-8 -*-
"""Pilot 子窗口分离 replay。

前一个 Adapt Pilot 子窗口只用于更新 Adapter，后一个 Pilot 子窗口只用于候选验收诊断。
Data 标签仅用于离线比较候选后的 BER/BCE，不参与更新或在线策略。该脚本不修改在线主流程。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.comm_env import CommEnvConfig, CommunicationEnvironment
from env.frame_structure import Frame
from evaluation.metrics import spearman_reward_data
from training.meta_training import _estimate_cir_from_known_frame
from training.rl_modulated_online import _build_equalizer, _load_model_config
from scripts.replay_pilot_candidate_ranking import (
    _candidate_specs,
    _evaluate_candidate,
    _pilot_state,
)
from agent.cir_estimator import condition_from_cir


def split_pilot_frame(frame: Frame, holdout_ratio: float) -> tuple[Frame, torch.Tensor, int, int]:
    """把原 Adapt Pilot 按时间顺序拆成更新子窗和验收子窗。"""

    if not 0.0 < float(holdout_ratio) < 1.0:
        raise ValueError("holdout_ratio 必须位于 (0, 1)。")
    indices = torch.nonzero(frame.adapt_mask, as_tuple=False).reshape(-1)
    if indices.numel() < 2:
        raise ValueError("Adapt Pilot 至少需要两个符号才能拆分。")
    holdout_count = max(1, min(int(indices.numel()) - 1, round(indices.numel() * float(holdout_ratio))))
    train_indices = indices[:-holdout_count]
    holdout_indices = indices[-holdout_count:]
    train_mask = torch.zeros_like(frame.adapt_mask)
    holdout_mask = torch.zeros_like(frame.adapt_mask)
    train_mask[train_indices] = True
    holdout_mask[holdout_indices] = True
    train_frame = Frame(
        frame_index=frame.frame_index,
        bits=frame.bits,
        tx_symbols=frame.tx_symbols,
        rx_symbols=torch.where(
            train_mask,
            frame.rx_symbols,
            torch.zeros_like(frame.rx_symbols),
        ),
        adapt_mask=train_mask,
        reward_mask=torch.zeros_like(frame.reward_mask),
        # Data mask 只供 _evaluate_candidate 输出事后诊断值；更新损失仍严格由 adapt_mask 限定。
        data_mask=frame.data_mask,
        model_region_ids=frame.model_region_ids,
        adapt_block_lengths=(int(train_indices.numel()),),
        tail_symbols=frame.tail_symbols,
        true_cir=frame.true_cir,
    )
    return train_frame, holdout_mask, int(train_indices.numel()), int(holdout_indices.numel())


def _summary(rows: list[dict]) -> list[dict]:
    summaries = []
    for score_name in (
        "pilot_holdout_loss_improvement",
        "adapt_bce_improvement",
        "pilot_reconstruction_improvement",
        "data_bce_improvement",
    ):
        candidates = [row for row in rows if row["candidate_name"] != "identity"]
        scores = [float(row.get(score_name, 0.0)) for row in candidates]
        targets = [float(row["data_ber_improvement"]) for row in candidates]
        result = spearman_reward_data(scores, targets, threshold=0.6)
        summaries.append(
            {
                "score": score_name,
                "spearman": float(result.correlation),
                "n": int(result.n),
                "gate_pass": bool(result.passed),
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--snrs", nargs="+", type=float, default=[5.0, 10.0, 15.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--delay", type=int, default=116)
    parser.add_argument("--pilot-total", type=int, default=256)
    parser.add_argument("--gap-seconds", type=float, default=600.0)
    parser.add_argument("--holdout-ratio", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model_config = _load_model_config(config, Path(args.pretrained))
    candidates = _candidate_specs(args.lr, args.steps)
    selected = [
        item
        for item in candidates
        if item["name"] in {"identity", "phase_trend_bce", "phase_trend_signal_reconstruction"}
    ]
    rows = []
    for snr in args.snrs:
        for seed in args.seeds:
            env = CommunicationEnvironment(
                CommEnvConfig(
                    level="B",
                    max_delay=args.delay,
                    snr_db=float(snr),
                    total_pilot=args.pilot_total,
                    layout="prefix",
                    seed=90_000 + int(seed),
                    acquisition_to_data_gap_seconds=float(args.gap_seconds),
                )
            )
            start = env.reset_episode()
            cir = _estimate_cir_from_known_frame(start.acquisition, args.delay).to(args.device)
            condition = condition_from_cir(cir, float(snr))
            frames = [env.next_frame() for _ in range(args.frames)]
            for frame_index, original in enumerate(frames):
                split, holdout_mask, train_count, holdout_count = split_pilot_frame(original, args.holdout_ratio)
                model = _build_equalizer(model_config, Path(args.pretrained), args.device)
                model.attach_online_physics_residual_adapter()
                model.attach_online_channel_residual_adapter()
                model.attach_online_phase_trend_adapter()
                model.attach_online_input_trend_adapter()
                model.attach_online_input_affine_adapter()
                model.attach_online_input_fir_adapter()
                model.attach_online_logit_affine_adapter()
                model.attach_online_logit_fir_adapter()
                tail = start.initial_soft_tail.to(args.device)
                state = _pilot_state(split, cir, tail, condition)
                base = {
                    "snr_db": float(snr),
                    "seed": int(seed),
                    "frame_index": int(frame_index),
                    "adapt_train_symbols": train_count,
                    "pilot_holdout_symbols": holdout_count,
                    "holdout_ratio": float(args.holdout_ratio),
                    **state,
                    "data_labels_used_online": False,
                    "pilot_split_role": "train_prefix_then_holdout_suffix",
                }
                for candidate in selected:
                    result = _evaluate_candidate(
                        model,
                        original,
                        condition,
                        tail,
                        candidate,
                        pilot_holdout_mask=holdout_mask,
                        pilot_update_frame=split,
                    )
                    rows.append(
                        {
                            **base,
                            **result,
                            "pilot_holdout_loss_improvement": float(
                                result.get("pilot_holdout_loss_improvement", 0.0)
                            ),
                            "pilot_holdout_role": "candidate_acceptance_only",
                            "diagnostic_uses_data_labels": True,
                        }
                    )

    payload = {
        "metric": "pilot_subwindow_split_replay",
        "split": "prefix_adapt_update_suffix_adapt_holdout",
        "holdout_ratio": float(args.holdout_ratio),
        "rows": rows,
        "summary": _summary(rows),
        "data_labels_used_online": False,
        "diagnostic_uses_data_labels": True,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    print(f"saved {target}")


if __name__ == "__main__":
    main()
