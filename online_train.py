# -*- coding: utf-8 -*-
"""部署期间持续更新 PPO 入口。"""

import argparse

from training.continual_ppo import run_online_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--config", default="configs/continual_ppo.json")
    parser.add_argument("--pretrained", default="pretrained/model_best.pt")
    parser.add_argument("--phase", choices=["offline", "continual", "all"], default="all")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--update-interval", type=int, default=32)
    parser.add_argument("--delays", nargs="*", type=int, default=None)
    parser.add_argument("--snrs", nargs="*", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output-dir", default="logs/online")
    args = parser.parse_args()
    if args.version:
        print("RL4EQ continual-ppo schema-v1")
        return
    del args.amp
    run_online_training(
        config_path=args.config,
        pretrained=args.pretrained,
        frames=args.frames,
        num_seeds=args.num_seeds,
        update_interval=args.update_interval,
        output_dir=args.output_dir,
        phase=args.phase,
        resume=args.resume,
        delays=args.delays,
        snrs=args.snrs,
    )
    print(f"saved {args.output_dir}")


if __name__ == "__main__":
    main()
