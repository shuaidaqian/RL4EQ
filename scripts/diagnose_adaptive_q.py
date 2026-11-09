# -*- coding: utf-8 -*-
"""只读诊断：过程噪声 Q 的网格扫描。

复用 `diagnose_bayesian_cir._run` 的 Bayes/Frozen 轨迹，只改变跨帧过程噪声 Q，
回答两个问题：
1. 当前确认配置的 Q=1e-4 是否已在开发集上接近最优；
2. 最优 Q 是否随 SNR 变化（若变化，说明固定 Q 不是最优，但仍需合法在线估计）。

Data 标签只用于事后 BER，真实 CIR 只用于事后 NMSE。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_bayesian_cir import _run
from evaluation.pilot_transfer import _load_branch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[110001, 110002, 110003])
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--q-values", nargs="+", type=float,
                        default=[0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if (config.get("main_level") != "B" or config.get("pilot_layout") != "prefix"
            or config.get("max_delay") != 116 or config.get("pilot_total") != 256
            or config.get("reward_pilot_total") != 32
            or config.get("main_snrs") != [0, 5, 10, 15]):
        raise ValueError("诊断只允许固定 Level B、116 符号长回波、前缀 Pilot=256 和四个指定 SNR。")
    output = Path(args.output)
    if output.exists():
        raise ValueError("输出文件已存在，请使用新路径。")

    device = torch.device(args.device)
    model, _ = _load_branch(Path(args.checkpoint), config, device)
    model.eval()

    result = {
        "schema_version": "adaptive-q-grid-diagnostic-v1",
        "diagnostic_only": True,
        "config": str(Path(args.config)),
        "checkpoint": str(Path(args.checkpoint)),
        "seeds": args.seeds,
        "frames": args.frames,
        "ridge": args.ridge,
        "q_values": args.q_values,
        "per_snr": {},
    }
    for snr in config["main_snrs"]:
        per_q = {}
        for q in args.q_values:
            rows = [
                row for seed in args.seeds
                for row in _run(model, config, seed, snr, args.frames, q, args.ridge, device)
            ]
            count = len(rows)
            per_q[str(q)] = {
                "bayes_ber": sum(row["bayes_ber"] for row in rows) / count,
                "bayes_nmse": sum(row["bayes_nmse"] for row in rows) / count,
                "frozen_ber": sum(row["frozen_ber"] for row in rows) / count,
                "frozen_nmse": sum(row["frozen_nmse"] for row in rows) / count,
                "frames": count,
            }
        result["per_snr"][str(snr)] = per_q
        best = min(per_q.items(), key=lambda item: item[1]["bayes_nmse"])
        print(
            f"snr={snr:2d} best_q={best[0]} nmse={best[1]['bayes_nmse']:.4f} "
            f"ber={best[1]['bayes_ber']:.6f} | "
            + " ".join(f"q={q:g}:{per_q[str(q)]['bayes_nmse']:.4f}"
                       for q in args.q_values),
            flush=True,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"written {output}", flush=True)


if __name__ == "__main__":
    main()
