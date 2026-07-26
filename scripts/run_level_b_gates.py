"""运行 Level B 三个前置门槛。

该脚本只负责复现实验门槛，不参与在线控制器训练：

1. Perfect-CIR 块检测器逐主配置 BER_data < 0.01。
2. Best Fixed 搜索逐主配置 BER_data < 0.1，候选选择只看 Reward Pilot。
3. Reward/Data paired samples 的 Spearman >= 0.6。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.curriculum import CurriculumTrainer, load_config
from training.meta_training import evaluate_best_fixed_level_b, evaluate_reward_data_alignment


def _print_event(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/continual_ppo.json")
    parser.add_argument("--output-dir", default="logs/gates_level_b")
    parser.add_argument("--frames-per-config", type=int, default=5)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--pilot-total", type=int, default=128)
    parser.add_argument("--pilot-layout", default="two_block")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _print_event({"event": "start", "device": str(device), "time": time.strftime("%Y-%m-%d %H:%M:%S")})

    trainer = CurriculumTrainer(config, device=device)
    validation = trainer._validate_level_b()
    (output_dir / "perfect_cir_gate.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _print_event(
        {
            "event": "perfect_cir_done",
            "gate_pass": validation["gate_pass"],
            "mean_ber_data": validation["mean_ber_data"],
            "max_config_ber_data": max(row["ber_data"] for row in validation["per_config"]),
        }
    )
    if not validation["gate_pass"]:
        raise SystemExit("Perfect-CIR gate failed")

    best = evaluate_best_fixed_level_b(
        config,
        output_dir=output_dir,
        frames_per_config=args.frames_per_config,
        seeds=args.seeds,
        cg_grid=[32, 64],
        refine_grid=[1, 2],
    )
    _print_event(
        {
            "event": "best_fixed_done",
            "gate_pass": best["gate_pass"],
            "selected": best["selected"],
            "max_config_ber_data": max(row["ber_data"] for row in best["per_config"]),
        }
    )
    if not best["gate_pass"]:
        raise SystemExit("Best Fixed gate failed")

    alignment = evaluate_reward_data_alignment(
        config,
        output_dir=output_dir,
        frames_per_config=args.frames_per_config,
        seeds=args.seeds,
        pilot_total=args.pilot_total,
        pilot_layout=args.pilot_layout,
    )
    _print_event(
        {
            "event": "spearman_done",
            "gate_pass": alignment["gate_pass"],
            "spearman": alignment["spearman"],
            "num_pairs": alignment["num_pairs"],
        }
    )
    if not alignment["gate_pass"]:
        raise SystemExit("Spearman gate failed")

    _print_event({"event": "all_gates_passed", "time": time.strftime("%Y-%m-%d %H:%M:%S")})


if __name__ == "__main__":
    main()
