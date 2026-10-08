# -*- coding: utf-8 -*-
"""滚动短窗口的 Pilot-only 在线 PEFT replay。

每个窗口第一帧使用 Adapt Pilot 更新，后续帧使用 Reward Pilot 验收；
接受后的参数进入下一窗口，拒绝则恢复窗口开始时的快照。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.cir_estimator import condition_from_cir
from agent.pilot_state import PilotDriftDetector, PilotStateEmbedding, build_pilot_state_summary
from env.comm_env import CommEnvConfig, CommunicationEnvironment
from evaluation.research_diagnostics import evaluate_frozen_peft_reward_window
from training.meta_training import _estimate_cir_from_known_frame
from training.online_adaptation import RewardWindowGate
from training.rl_modulated_online import _build_equalizer, _load_model_config
from training.online_adaptation import _frame_to_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--snrs", nargs="+", type=float, default=[0.0, 5.0, 10.0, 15.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--delay", type=int, default=116)
    parser.add_argument("--pilot-total", type=int, default=256)
    parser.add_argument("--gap-seconds", type=float, default=120.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--drift-threshold", type=float, default=0.25)
    parser.add_argument("--min-drift-confidence", type=float, default=0.15)
    parser.add_argument("--disable-drift-gate", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.frames < args.window_size + 1:
        raise ValueError("frames 必须大于一个更新帧和一个 Reward 窗口。")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model_config = _load_model_config(config, Path(args.pretrained))
    rows = []
    candidates = [
        ("physics_residual_conservative", args.lr * 0.5),
        ("physics_residual", args.lr),
    ]
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
            initial_cir = _estimate_cir_from_known_frame(start.acquisition, args.delay).to(args.device)
            frames = [env.next_frame() for _ in range(args.frames)]
            for candidate_name, lr in candidates:
                model = _build_equalizer(model_config, Path(args.pretrained), args.device)
                model.attach_online_physics_residual_adapter()
                soft_tail = start.initial_soft_tail.to(args.device)
                cir = initial_cir.detach().clone()
                reference_cir = initial_cir.detach().clone()
                drift_detector = PilotDriftDetector(
                    threshold=args.drift_threshold,
                    min_confidence=args.min_drift_confidence,
                )
                state_embedding = PilotStateEmbedding()
                window_index = 0
                cursor = 0
                while cursor + args.window_size < len(frames):
                    window_frames = frames[cursor : cursor + args.window_size + 1]
                    current_frame = window_frames[0]
                    condition = condition_from_cir(cir, float(snr))
                    summary = build_pilot_state_summary(
                        cir=cir,
                        reference_cir=reference_cir,
                        phase0=0.0,
                        cfo_cycles_per_symbol=0.0,
                        noise_variance=condition.noise_variance,
                        confidence=condition.confidence,
                        reconstruction_error=0.0,
                    )
                    embedding = state_embedding(summary)
                    drift_detected, drift_distance = drift_detector.update(embedding, summary.confidence)
                    drift_gate_applied = not args.disable_drift_gate
                    baseline = copy.deepcopy(model)
                    gate = RewardWindowGate(
                        max_parameter_delta_norm=0.5,
                        min_accepted_frames=args.window_size,
                        max_single_frame_regression=0.0,
                    )
                    if drift_gate_applied and not drift_detected:
                        result = evaluate_frozen_peft_reward_window(
                            model=model,
                            adapt_frame=window_frames[0],
                            reward_frames=window_frames[1:],
                            condition=condition,
                            soft_tail=soft_tail,
                            candidate={"name": "identity", "groups": set(), "lr": 0.0, "steps": 0},
                            gate=gate,
                            objective="bce",
                        )
                        result["adaptation_accepted"] = False
                        result["gate_reason"] = "no_pilot_drift"
                    else:
                        result = evaluate_frozen_peft_reward_window(
                            model=model,
                            adapt_frame=window_frames[0],
                            reward_frames=window_frames[1:],
                            condition=condition,
                            soft_tail=soft_tail,
                            candidate={
                                "name": candidate_name,
                                "groups": {"physics_residual"},
                                "lr": lr,
                                "steps": args.steps,
                            },
                            gate=gate,
                            objective="bce",
                        )
                    candidate_model = result.pop("_candidate_model", None)
                    candidate_tail = result.pop("_final_soft_tail", None)
                    baseline_tail = result.pop("_baseline_soft_tail", None)
                    if result["adaptation_accepted"] and candidate_model is not None:
                        model = candidate_model
                        if candidate_tail is not None:
                            soft_tail = candidate_tail
                    else:
                        model = baseline
                        if baseline_tail is not None:
                            soft_tail = baseline_tail
                    result.update(
                        {
                            "snr_db": float(snr),
                            "seed": int(seed),
                            "window_index": int(window_index),
                            "cursor": int(cursor),
                            "candidate_name": candidate_name,
                            "rolling": True,
                            "gap_seconds": float(args.gap_seconds),
                            "pilot_state_embedding": embedding.tolist(),
                            "pilot_state_drift_detected": bool(drift_detected),
                            "pilot_state_drift_distance": float(drift_distance),
                            "pilot_state_drift_gate_applied": bool(drift_gate_applied),
                            "pilot_state_confidence": float(summary.confidence),
                            "data_labels_used_online": False,
                        }
                    )
                    rows.append(result)
                    cursor += args.window_size + 1
                    window_index += 1
                    # 仅使用当前窗口的 Adapt Pilot 更新状态估计，供下一窗口漂移检测。
                    from agent.cir_estimator import pilot_sparse_cir_update

                    cir = pilot_sparse_cir_update(
                        _frame_to_device(current_frame, args.device),
                        cir,
                        soft_tail,
                        max_paths=24,
                        alpha=0.2,
                    ).to(args.device)
    payload = {
        "metric": "rolling_frozen_peft_multi_frame_reward_gate",
        "rows": rows,
        "data_labels_used_online": False,
        "diagnostic_uses_data_labels": True,
        "gap_seconds": float(args.gap_seconds),
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {target}")


if __name__ == "__main__":
    main()
