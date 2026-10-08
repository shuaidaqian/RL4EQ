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
from env.comm_env import CommEnvConfig, CommunicationEnvironment
from evaluation.research_diagnostics import evaluate_frozen_peft_reward_window
from training.meta_training import _estimate_cir_from_known_frame
from training.online_adaptation import RewardWindowGate
from training.rl_modulated_online import _build_equalizer, _load_model_config


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
            cir = _estimate_cir_from_known_frame(start.acquisition, args.delay).to(args.device)
            condition = condition_from_cir(cir, float(snr))
            frames = [env.next_frame() for _ in range(args.frames)]
            for candidate_name, lr in candidates:
                model = _build_equalizer(model_config, Path(args.pretrained), args.device)
                model.attach_online_physics_residual_adapter()
                soft_tail = start.initial_soft_tail.to(args.device)
                window_index = 0
                cursor = 0
                while cursor + args.window_size < len(frames):
                    window_frames = frames[cursor : cursor + args.window_size + 1]
                    baseline = copy.deepcopy(model)
                    gate = RewardWindowGate(
                        max_parameter_delta_norm=0.5,
                        min_accepted_frames=args.window_size,
                        max_single_frame_regression=0.0,
                    )
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
                    if not result["adaptation_accepted"]:
                        model = baseline
                    result.update(
                        {
                            "snr_db": float(snr),
                            "seed": int(seed),
                            "window_index": int(window_index),
                            "cursor": int(cursor),
                            "candidate_name": candidate_name,
                            "rolling": True,
                            "gap_seconds": float(args.gap_seconds),
                        }
                    )
                    rows.append(result)
                    cursor += args.window_size + 1
                    window_index += 1
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
