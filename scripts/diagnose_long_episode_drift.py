# -*- coding: utf-8 -*-
"""长 episode tail/CIR/drift 诊断入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.long_episode_diagnostics import run_long_episode_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/continual_ppo.json")
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--delay", type=int, default=40)
    parser.add_argument("--snr", type=float, default=10.0)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--tail-modes", nargs="*", default=["soft", "hard", "oracle", "zero"])
    parser.add_argument("--cir-modes", nargs="*", default=["fixed", "decision_directed", "oracle"])
    parser.add_argument("--cir-alpha", type=float, default=0.2)
    parser.add_argument("--cir-confidence-threshold", type=float, default=None)
    parser.add_argument("--rhos", nargs="*", type=float, default=[0.99, 1.0])
    parser.add_argument("--pilot-total", type=int, default=128)
    parser.add_argument("--pilot-layout", default="prefix")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    payload = run_long_episode_diagnostic(
        config_path=args.config,
        pretrained=args.pretrained,
        output_dir=args.output_dir,
        delay=args.delay,
        snr_db=args.snr,
        frames=args.frames,
        seeds=args.seeds,
        tail_modes=args.tail_modes,
        cir_modes=args.cir_modes,
        rhos=args.rhos,
        pilot_total=args.pilot_total,
        pilot_layout=args.pilot_layout,
        cir_alpha=args.cir_alpha,
        cir_confidence_threshold=args.cir_confidence_threshold,
        device=args.device or "cuda",
    )
    print(f"saved {args.output_dir}")
    print(f"rows {len(payload['rows'])}")


if __name__ == "__main__":
    main()
