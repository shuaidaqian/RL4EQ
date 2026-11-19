# -*- coding: utf-8 -*-
"""在指定离线状态划分上估计 EME CIR 的 AR(1) 动态统计。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.comm_env import CommunicationEnvironment
from env.experiment_config import build_comm_env_config
from evaluation.channel_dynamics import estimate_ar1_statistics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eme_long_memory_v2_online_dd_tail_v1.json")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--frames", type=int, default=11719)
    parser.add_argument("--state-split", default="offline_train", choices=["offline_train"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.frames < 2 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("frames 必须至少为 2，seed 必须互不重复。")

    experiment = json.loads(Path(args.config).read_text(encoding="utf-8"))
    trajectories = []
    metadata = []
    for seed in args.seeds:
        env_config = build_comm_env_config(
            experiment, level="B", snr_db=0.0, seed=seed, state_split=args.state_split
        )
        env = CommunicationEnvironment(env_config)
        env.reset_episode()
        env.channel.advance_after_acquisition()
        states = [env.channel.true_cir()]
        for _ in range(args.frames - 1):
            env.channel._evolve_eme_taps()
            states.append(env.channel.true_cir())
        trajectories.append(torch.stack(states))
        metadata.append(env.state_metadata())

    estimate = estimate_ar1_statistics(trajectories)
    per_seed_rho = estimate["per_trajectory_rho"]
    mean_rho = float(sum(per_seed_rho) / len(per_seed_rho))
    if len(per_seed_rho) > 1:
        sample_variance = sum((value - mean_rho) ** 2 for value in per_seed_rho) / (len(per_seed_rho) - 1)
        standard_error = math.sqrt(sample_variance / len(per_seed_rho))
        critical = float(student_t.ppf(0.975, len(per_seed_rho) - 1))
        rho_ci_95 = [mean_rho - critical * standard_error, mean_rho + critical * standard_error]
    else:
        rho_ci_95 = None
    result = {
        "state_split": args.state_split,
        "seeds": args.seeds,
        "frames_per_seed": args.frames,
        "trajectory_metadata": metadata,
        "rho_estimate": mean_rho,
        "rho_95_t_interval_across_seeds": rho_ci_95,
        "rho_energy_weighted_diagnostic": estimate["rho_energy_weighted"],
        "per_seed_rho_estimate": per_seed_rho,
        "stationary_power_by_tap": estimate["stationary_power"].tolist(),
        "transition_count": estimate["transition_count"],
        "state_count": estimate["state_count"],
        "interpretation": "离线/开发信道统计估计；不得用当前测试会话 CIR 更新或在线重估。",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "state_split", "seeds", "frames_per_seed", "rho_estimate",
        "transition_count", "state_count",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
