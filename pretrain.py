# -*- coding: utf-8 -*-
"""Pilot 条件课程监督预训练入口。"""

import argparse
import torch

from training.curriculum import CurriculumTrainer, load_config
from training.meta_training import MetaTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--config", default="configs/level_b_state_tracking.json")
    parser.add_argument("--stage", default="all")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save-dir", default="pretrained")
    args = parser.parse_args()
    if args.version:
        print("RL4EQ state-tracking schema-v1")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    if args.stage == "meta":
        if config.get("channel_profile") in {"eme_measurement_v1", "eme_long_memory_v2"}:
            raise ValueError(
                "EME 物理信道尚未接入 meta 阶段，禁止静默训练单位信道："
                f"profile_name={config.get('profile_name', config.get('channel_profile'))}。"
            )
        trainer = MetaTrainer(config, device=device, save_dir=args.save_dir)
        resume_loaded = trainer.load_resume(args.resume)
        metrics = trainer.train(steps=args.steps, batch_size=args.batch_size, smoke=args.steps <= 2, resume_loaded=resume_loaded)
    elif args.stage == "online_meta":
        raise ValueError(
            "online_meta 已移除；当前在线路线是冻结均衡主干和 Pilot-only 物理状态跟踪。"
        )
    else:
        trainer = CurriculumTrainer(config, device=device)
        resume_loaded = trainer.load_resume(args.resume)
        metrics = trainer.train(
            stage=args.stage,
            steps=args.steps,
            batch_size=args.batch_size,
            accumulation_steps=args.accumulation_steps,
            use_amp=args.amp,
        )
        metrics["resume_loaded"] = resume_loaded
    trainer.save(args.save_dir, metrics)
    print(f"saved {args.save_dir}")


if __name__ == "__main__":
    main()
