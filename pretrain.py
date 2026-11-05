# -*- coding: utf-8 -*-
"""Pilot 条件课程监督预训练入口。"""

import argparse
import torch

from training.curriculum import CurriculumTrainer, load_config
from training.meta_training import MetaTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--config", default="configs/continual_ppo.json")
    parser.add_argument("--stage", default="all")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save-dir", default="pretrained")
    parser.add_argument("--transfer-objective", choices=["baseline", "post_adapt", "trajectory"], default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--subspace-manifest", default=None, help="固定信道子空间的开发训练清单。")
    parser.add_argument("--inspection-dir", default=None, help="已通过几何检查的子空间审计目录。")
    args = parser.parse_args()
    if args.version:
        print("RL4EQ continual-ppo schema-v1")
        return
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.stage == "detection_subspace":
        import sys
        from evaluation.subspace_development import run_training
        allowed = {"--stage", "--subspace-manifest", "--inspection-dir", "--save-dir", "--device"}
        if any(item.startswith("--") and item.split("=",1)[0] not in allowed for item in sys.argv[1:]):
            parser.error("子空间训练条件只能由清单指定，禁止混用旧训练覆盖参数。")
        if not args.subspace_manifest or not args.inspection_dir:
            parser.error("子空间训练必须提供清单和已有检查目录。")
        torch.set_num_threads(1)
        run_training(args.subspace_manifest, args.inspection_dir, args.save_dir, device)
        print(f"saved {args.save_dir}")
        return
    if args.subspace_manifest or args.inspection_dir:
        parser.error("子空间清单参数仅用于--stage detection_subspace。")
    config = load_config(args.config)
    if int(config.get("receiver_lookahead_len", 0)) and args.stage != "pilot_transfer":
        raise ValueError("跨帧观测离线训练请使用 --stage pilot_transfer；其他训练入口尚未支持该协议。")
    if args.stage == "pilot_transfer":
        from training.pilot_transfer import PilotTransferTrainer

        trainer = PilotTransferTrainer(config, device, objective=args.transfer_objective)
        trainer.load_resume(args.resume)
        metrics = trainer.train(args.steps, args.batch_size, args.accumulation_steps, args.amp)
    elif args.stage == "meta":
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
            "online_meta 已移除；当前路线为离线整帧 BCE_all + 安全 Contextual Bandit 在线调度。"
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
