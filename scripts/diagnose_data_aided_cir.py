# -*- coding: utf-8 -*-
"""只读诊断：数据辅助 CIR 精化相对 Frozen / Pilot-only / Oracle 的增益。

四个物理状态模式：
- frozen：始终使用 acquisition CIR；
- pilot_cir：每帧用 Adapt Pilot 的已知符号做 Pilot-only CIR 更新；
- data_cir：每帧用 Data 位置的模型 soft 判决做加权 LS（不读 Data 标签）；
- oracle_cir：该帧使用真实 CIR（仅诊断，不可部署）。

CIR 更新只用于下一帧，保持因果。Data 标签只用于事后 BER；真实 CIR 只用于
隔离 oracle 与 NMSE 诊断，不回流任何在线状态。
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

from agent.cir_estimator import decision_directed_cir_update
from evaluation.pilot_transfer import _load_branch
from training.data_aided_cir import data_aided_cir_update
from training.pilot_transfer import context_logits, make_context, make_episode, next_pilot_tail


def _normalized_nmse(estimate: torch.Tensor, truth: torch.Tensor) -> float:
    """对齐全局复标量后的归一化 NMSE，消除相位/尺度歧义。"""

    estimate = estimate.reshape(-1).to(torch.complex128)
    truth = truth.reshape(-1).to(torch.complex128)
    denominator = torch.vdot(estimate, estimate).real.clamp_min(1e-12)
    scale = torch.vdot(estimate, truth) / denominator
    residual = truth - scale * estimate
    return float((residual.abs().square().sum() / truth.abs().square().sum().clamp_min(1e-12)).real)


def _run(model, config, seed, snr, frames, mode, alpha, threshold, device):
    env, cir, cfo, tail, received = make_episode(
        config, model, seed=seed, snr=snr, frames=frames,
        state_split="offline_train", device=device,
    )
    max_delay = int(config["max_delay"])
    rows = []
    for frame in received:
        context = make_context(frame, cir, cfo, tail, snr)
        with torch.no_grad():
            logits = context_logits(model, context, None, visible="data")
            mask = frame.data_mask.bool()
            ber = ((logits[mask] >= 0) != frame.bits[mask].bool()).float().mean()
        truth = frame.true_cir if frame.true_cir is not None else None
        nmse = _normalized_nmse(cir, truth) if truth is not None else None
        rows.append({
            "seed": int(seed), "snr_db": int(snr), "mode": mode,
            "frame_index": int(frame.frame_index),
            "ber_data": float(ber),
            "cir_nmse": nmse,
        })
        tail = next_pilot_tail(
            model, context, alpha=float(config.get("tail_update_alpha", 0.5)), mode="dd_soft",
        )
        if mode == "pilot_cir":
            cir = decision_directed_cir_update(frame, logits, max_delay, cir, alpha=alpha)
        elif mode == "data_cir":
            cir = data_aided_cir_update(
                frame, logits, max_delay, cir, alpha=alpha, confidence_threshold=threshold,
            )
        elif mode == "oracle_cir":
            if truth is None:
                raise ValueError("oracle_cir 需要 frame.true_cir。")
            cir = truth.to(device=device, dtype=torch.complex64).reshape(-1)
        elif mode != "frozen":
            raise ValueError(f"未知模式：{mode}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[92001, 92002, 92003])
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--modes", nargs="+", default=["frozen", "pilot_cir", "data_cir", "oracle_cir"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model, _ = _load_branch(Path(args.checkpoint), config, device)
    model.eval()

    result = {"config": str(Path(args.config)), "checkpoint": str(Path(args.checkpoint)),
              "seeds": args.seeds, "frames": args.frames, "alpha": args.alpha,
              "confidence_threshold": args.confidence_threshold, "rows": []}
    for mode in args.modes:
        for snr in config["main_snrs"]:
            for seed in args.seeds:
                rows = _run(model, config, seed, snr, args.frames, mode, args.alpha,
                            args.confidence_threshold, device)
                result["rows"].extend(rows)
                values = [row["ber_data"] for row in rows]
                print(f"{mode:9s} snr={snr:2d} seed={seed} "
                      f"ber={sum(values)/len(values):.6f} first4={sum(values[:4])/4:.6f} "
                      f"last4={sum(values[-4:])/4:.6f}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
