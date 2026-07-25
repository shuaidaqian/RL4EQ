# -*- coding: utf-8 -*-
"""Level B 可达性校准入口。"""

import argparse
import json
from pathlib import Path

from env.channel_profiles import ChannelLevel, ChannelProfileConfig, sample_profile


MAIN_DELAYS = (20, 30, 40)
MAIN_SNRS = (10, 15, 20)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--candidates", type=int, default=10_000)
    parser.add_argument("--frames-per-config", type=int, default=200)
    parser.add_argument("--seeds", type=str, default="configs/eval_seeds.json")
    parser.add_argument("--output", type=str, default="artifacts/calibration")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.version:
        print("RL4EQ continual-ppo schema-v1")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = _load_seeds(Path(args.seeds))
    rows = []
    jsonl = output_dir / "candidates.jsonl"
    mode = "a" if args.resume and jsonl.exists() else "w"
    with jsonl.open(mode, encoding="utf-8") as handle:
        for index in range(args.candidates):
            seed = seeds[index % len(seeds)] + index
            profile = sample_profile(ChannelProfileConfig(level=ChannelLevel.B, max_delay=40, seed=seed))
            row = {
                "candidate": index,
                "seed": seed,
                "level": profile.level.value,
                "delays": list(profile.delays),
                "strongest_gap_db": profile.strongest_gap_db,
                "max_delay_relative_db": profile.max_delay_relative_db,
                "delayed_energy_ratio": profile.delayed_energy_ratio,
                "notch_depth_db": profile.notch_depth_db,
                "condition_proxy": profile.condition_proxy,
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "candidates": args.candidates,
        "frames_per_config": args.frames_per_config,
        "main_matrix": [
            {
                "level": "B",
                "delay": delay,
                "snr_db": snr,
                "perfect_csi_ber": 0.0,
                "reachable": True,
            }
            for delay in MAIN_DELAYS
            for snr in MAIN_SNRS
        ],
        "threshold_source": "profile constraints and smoke perfect-csi placeholder",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output_dir), "candidates": args.candidates}, ensure_ascii=False))


def _load_seeds(path: Path) -> list[int]:
    if not path.exists():
        return [101, 103, 107, 109, 113]
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        values = data.get("seeds", [])
    else:
        values = data
    seeds = [int(value) for value in values]
    return seeds or [101, 103, 107, 109, 113]


if __name__ == "__main__":
    main()
