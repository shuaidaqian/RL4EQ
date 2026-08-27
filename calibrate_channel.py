# -*- coding: utf-8 -*-
"""Level B 可达性校准入口。"""

import argparse
import json
from pathlib import Path

import torch

from baseline.block_equalizers import bit_error_rate, perfect_csi_cg_detect
from env.channel_profiles import ChannelLevel, ChannelProfileConfig, sample_profile
from env.linear_operator import LinearChannelOperator


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
    jsonl = output_dir / "candidates.jsonl"
    start_index = _resume_start(jsonl) if args.resume else 0
    mode = "a" if start_index > 0 else "w"
    with jsonl.open(mode, encoding="utf-8") as handle:
        for index in range(start_index, args.candidates):
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
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    matrix = [
        _simulate_perfect_csi_ber(delay=delay, snr_db=snr, seeds=seeds, frames=args.frames_per_config)
        for delay in MAIN_DELAYS
        for snr in MAIN_SNRS
    ]
    summary = {
        "candidates_target": args.candidates,
        "candidates_existing_before_run": start_index,
        "candidates_total_after_run": max(args.candidates, start_index),
        "frames_per_config": args.frames_per_config,
        "main_matrix": matrix,
        "threshold_source": "measured perfect-csi cg smoke",
        "reachable_threshold": 0.01,
        "gate_enforced": args.frames_per_config >= 200,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output_dir), "candidates": args.candidates}, ensure_ascii=False))
    if summary["gate_enforced"] and any(not row["reachable"] for row in matrix):
        raise SystemExit(2)


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


def _resume_start(jsonl: Path) -> int:
    if not jsonl.exists():
        return 0
    last_index = -1
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        last_index = max(last_index, int(json.loads(line)["candidate"]))
    return last_index + 1


def _simulate_perfect_csi_ber(delay: int, snr_db: int, seeds: list[int], frames: int) -> dict:
    frame_len = 512
    max_frames = max(1, frames)
    errors = []
    operator = LinearChannelOperator(frame_len=frame_len, max_delay=delay)
    noise_variance = float(10.0 ** (-snr_db / 10.0))
    for frame_index in range(max_frames):
        seed = seeds[frame_index % len(seeds)] + delay * 10_000 + snr_db * 100 + frame_index
        generator = torch.Generator(device="cpu").manual_seed(seed)
        bits = torch.randint(0, 2, (1, frame_len), generator=generator, dtype=torch.int64).bool()
        tx = torch.complex(bits.to(torch.float32) * 2.0 - 1.0, torch.zeros(1, frame_len))
        profile = sample_profile(ChannelProfileConfig(level=ChannelLevel.B, max_delay=delay, seed=seed))
        cir = torch.zeros(1, delay + 1, dtype=torch.complex64)
        cir[0, list(profile.delays)] = torch.as_tensor(profile.taps, dtype=torch.complex64)
        tail = torch.complex(
            torch.randint(0, 2, (1, delay), generator=generator, dtype=torch.int64).to(torch.float32) * 2.0 - 1.0,
            torch.zeros(1, delay),
        )
        clean = operator.forward(tx, cir, tail)
        std = (noise_variance / 2.0) ** 0.5
        noise = torch.complex(
            torch.randn(1, frame_len, generator=generator) * std,
            torch.randn(1, frame_len, generator=generator) * std,
        )
        result = perfect_csi_cg_detect(clean + noise, cir, tail, torch.tensor(noise_variance), iterations=96)
        errors.append(bit_error_rate(result.logits, bits))
    ber = float(sum(errors) / len(errors))
    return {
        "level": "B",
        "delay": delay,
        "snr_db": snr_db,
        "perfect_csi_ber": ber,
        "reachable": ber < 0.01,
        "frames": max_frames,
    }


if __name__ == "__main__":
    main()
