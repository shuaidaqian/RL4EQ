# -*- coding: utf-8 -*-
"""在线均衡入口；默认使用冻结主干 + Pilot-only 物理状态跟踪。"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--config", default="configs/level_b_state_tracking.json")
    parser.add_argument("--pretrained", default="pretrained/model_best.pt")
    parser.add_argument(
        "--method",
        choices=["state", "pilot", "ppo"],
        default="state",
        help="state 为正式主路线；pilot/ppo 仅用于历史诊断。",
    )
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--update-interval", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--delays", nargs="*", type=int, default=None)
    parser.add_argument("--snrs", nargs="*", type=float, default=None)
    parser.add_argument("--pilot-total", type=int, default=128)
    parser.add_argument("--pilot-layout", default="prefix")
    parser.add_argument("--cir-update", choices=["fixed", "pilot_sparse", "decision_directed"], default=None)
    parser.add_argument("--cir-alpha", type=float, default=0.2)
    parser.add_argument("--state-split", choices=["offline_train", "heldout_edge", "drift"], default=None)
    parser.add_argument("--scheduler", choices=["fixed", "bandit"], default=None)
    parser.add_argument("--output-dir", default="logs/online")
    args = parser.parse_args()
    if args.version:
        print("RL4EQ state-tracking schema-v1")
        return
    if args.method == "state":
        if args.cir_update is not None or args.scheduler is not None:
            parser.error("state 方法的物理状态门控由配置控制，不接受 PEFT/Bandit 兼容参数。")
        from training.state_tracking_online import run_state_tracking_online

        run_state_tracking_online(
            config_path=args.config,
            pretrained=args.pretrained,
            frames=args.frames,
            num_seeds=args.num_seeds,
            output_dir=args.output_dir,
            delays=args.delays,
            snrs=args.snrs,
            pilot_total=args.pilot_total,
            pilot_layout=args.pilot_layout,
            state_split=args.state_split,
        )
        print(f"saved {args.output_dir}")
        return
    if args.pretrained and not Path(args.pretrained).is_file():
        raise FileNotFoundError(f"预训练 checkpoint 不存在：{args.pretrained}")
    common = {
        "config_path": args.config,
        "frames": args.frames,
        "num_seeds": args.num_seeds,
        "output_dir": args.output_dir,
        "delays": args.delays,
        "snrs": args.snrs,
        "pilot_total": args.pilot_total,
        "pilot_layout": args.pilot_layout,
        "pretrained": args.pretrained,
        "cir_update_mode": args.cir_update or "fixed",
        "cir_update_alpha": args.cir_alpha,
        "state_split": args.state_split,
    }
    if args.method == "pilot":
        from training.online_adaptation import run_pilot_driven_online

        run_pilot_driven_online(**common, scheduler=args.scheduler or "fixed")
    else:
        from training.windowed_discrete_ppo import run_windowed_discrete_online

        run_windowed_discrete_online(
            **common,
            window_size=args.window_size,
            update_interval=args.update_interval,
        )
    print(f"saved {args.output_dir}")


if __name__ == "__main__":
    main()
