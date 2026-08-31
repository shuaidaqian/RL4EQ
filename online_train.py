# -*- coding: utf-8 -*-
"""在线均衡入口；默认使用 Pilot 驱动在线适配，PPO 仅作可选消融。"""

import argparse
from pathlib import Path

from training.windowed_discrete_ppo import run_windowed_discrete_online
from training.online_adaptation import run_pilot_driven_online


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--config", default="configs/continual_ppo.json")
    parser.add_argument("--pretrained", default="pretrained/model_best.pt")
    parser.add_argument("--phase", choices=["offline", "continual", "all"], default="all")
    parser.add_argument("--method", choices=["pilot", "ppo"], default="pilot")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--update-interval", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--delays", nargs="*", type=int, default=None)
    parser.add_argument("--snrs", nargs="*", type=float, default=None)
    parser.add_argument("--pilot-total", type=int, default=128)
    parser.add_argument("--pilot-layout", default="prefix")
    parser.add_argument("--cir-update", choices=["fixed", "pilot_sparse", "decision_directed"], default="fixed")
    parser.add_argument("--cir-alpha", type=float, default=0.2)
    parser.add_argument("--state-split", choices=["offline_train", "heldout_edge", "drift"], default=None)
    parser.add_argument("--scheduler", choices=["fixed", "bandit"], default="fixed")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output-dir", default="logs/online")
    args = parser.parse_args()
    if args.version:
        print("RL4EQ continual-ppo schema-v1")
        return
    del args.amp, args.phase, args.resume
    common = {
        "config_path": args.config,
        "frames": args.frames,
        "num_seeds": args.num_seeds,
        "output_dir": args.output_dir,
        "delays": args.delays,
        "snrs": args.snrs,
        "pilot_total": args.pilot_total,
        "pilot_layout": args.pilot_layout,
        "pretrained": args.pretrained if args.pretrained and Path(args.pretrained).exists() else None,
        "cir_update_mode": args.cir_update,
        "cir_update_alpha": args.cir_alpha,
        "state_split": args.state_split,
    }
    if args.method == "pilot":
        run_pilot_driven_online(**common, scheduler=args.scheduler)
    else:
        run_windowed_discrete_online(
            **common,
            window_size=args.window_size,
            update_interval=args.update_interval,
        )
    print(f"saved {args.output_dir}")


if __name__ == "__main__":
    main()
