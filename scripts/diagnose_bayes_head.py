# -*- coding: utf-8 -*-
"""只读诊断：共享 Bayes CIR 物理跟踪下，在线 head 微调是否还有神经增量。

Frozen 与 Online 共用同一 Bayes CIR、同一 dd_soft tail 与同一波形；
Online 每帧用 Adapt Pilot BCE 更新 head（Reward Pilot 硬门控），
再对同一帧 Data 判决。Data 标签只用于事后 BER 与 oracle，不进入 head 更新。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pilot_transfer import _load_branch
from training.bayesian_cir import acquisition_prior, add_process_noise, data_posterior, pilot_posterior
from training.pilot_transfer import context_logits, make_context, make_episode, next_pilot_tail


def _metrics(logits, frame, mask_name):
    mask = getattr(frame, f"{mask_name}_mask").bool()
    labels = frame.bits[mask].bool()
    bce = F.binary_cross_entropy_with_logits(logits[mask].float(), labels.float())
    ber = ((logits[mask] >= 0) != labels).float().mean()
    return float(bce), float(ber)


def _project(weight, bias, weight0, bias0, radius):
    delta = torch.cat([(weight - weight0).reshape(-1), (bias - bias0).reshape(-1)])
    norm = delta.norm().clamp_min(1e-12)
    if float(norm) <= radius:
        return weight, bias
    scale = radius / norm
    return weight0 + scale * (weight - weight0), bias0 + scale * (bias - bias0)


def _adapt_head(model, context, steps, lr, radius):
    weight = model.head.weight.detach().clone().requires_grad_(True)
    bias = model.head.bias.detach().clone().requires_grad_(True)
    weight0, bias0 = weight.detach().clone(), bias.detach().clone()
    mw = torch.zeros_like(weight); vw = torch.zeros_like(weight)
    mb = torch.zeros_like(bias); vb = torch.zeros_like(bias)
    for step in range(1, int(steps) + 1):
        logits = context_logits(model, context, {"head.weight": weight, "head.bias": bias}, visible="data")
        mask = context.frame.adapt_mask.bool()
        loss = F.binary_cross_entropy_with_logits(logits[mask].float(), context.frame.bits[mask].float())
        gw, gb = torch.autograd.grad(loss, [weight, bias])
        mw = 0.9 * mw + 0.1 * gw; vw = 0.999 * vw + 0.001 * gw.square()
        mb = 0.9 * mb + 0.1 * gb; vb = 0.999 * vb + 0.001 * gb.square()
        mw_hat = mw / (1.0 - 0.9 ** step); vw_hat = vw / (1.0 - 0.999 ** step)
        mb_hat = mb / (1.0 - 0.9 ** step); vb_hat = vb / (1.0 - 0.999 ** step)
        weight = weight - lr * mw_hat / (vw_hat.sqrt() + 1e-8)
        bias = bias - lr * mb_hat / (vb_hat.sqrt() + 1e-8)
        weight, bias = _project(weight, bias, weight0, bias0, radius)
    return weight.detach(), bias.detach()


def _run(model, config, seed, snr, frames, steps, lr, radius, process_noise, ridge, gate, device):
    env, cir, cfo, tail, received, acquisition = make_episode(
        config, model, seed=seed, snr=snr, frames=frames,
        state_split="offline_train", device=device, return_acquisition=True,
    )
    max_delay = int(config["max_delay"])
    noise_variance = 10.0 ** (-float(snr) / 10.0)
    bayes_cir, bayes_cov = acquisition_prior(acquisition, cfo, max_delay, noise_variance, ridge=ridge)
    tail_state = tail.clone()
    rows = []
    for frame in received:
        context = make_context(frame, bayes_cir, cfo, tail_state, snr)
        with torch.no_grad():
            frozen_logits = context_logits(model, context, None, visible="data")
            frozen_bce, frozen_ber = _metrics(frozen_logits, frame, "data")
            frozen_reward_bce, frozen_reward_ber = _metrics(frozen_logits, frame, "reward")
        weight, bias = _adapt_head(model, context, steps, lr, radius)
        with torch.no_grad():
            online_logits = context_logits(
                model, context, {"head.weight": weight, "head.bias": bias}, visible="data",
            )
            online_bce, online_ber = _metrics(online_logits, frame, "data")
            online_reward_bce, online_reward_ber = _metrics(online_logits, frame, "reward")
        accepted = online_reward_bce <= frozen_reward_bce + gate
        committed_ber = online_ber if accepted else frozen_ber
        rows.append({
            "seed": int(seed), "snr_db": int(snr), "frame_index": int(frame.frame_index),
            "frozen_ber": frozen_ber, "online_ber": online_ber,
            "committed_ber": committed_ber, "accepted": bool(accepted),
            "frozen_data_bce": frozen_bce, "online_data_bce": online_bce,
            "frozen_reward_bce": frozen_reward_bce, "online_reward_bce": online_reward_bce,
        })
        tail_state = next_pilot_tail(model, context, alpha=1.0, mode="dd_soft")
        latent = context.condition.latent_residual
        phase0 = float(latent[0, 0].detach().cpu()) if latent is not None else 0.0
        cfo_est = float(latent[0, 1].detach().cpu()) if latent is not None else float(cfo)
        bayes_cir, bayes_cov = pilot_posterior(
            frame, bayes_cir, bayes_cov, max_delay, noise_variance, phase0, cfo_est,
        )
        bayes_cir, bayes_cov = data_posterior(
            frame, frozen_logits, bayes_cir, bayes_cov, max_delay, noise_variance, phase0, cfo_est,
        )
        bayes_cov = add_process_noise(bayes_cov, process_noise)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[96001, 96002, 96003])
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--process-noise", type=float, default=1e-4)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--gate", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model, _ = _load_branch(Path(args.checkpoint), config, device)
    model.eval()

    result = {"config": str(Path(args.config)), "checkpoint": str(Path(args.checkpoint)),
              "seeds": args.seeds, "frames": args.frames, "steps": args.steps,
              "lr": args.lr, "radius": args.radius, "process_noise": args.process_noise,
              "ridge": args.ridge, "gate": args.gate, "per_snr": {}, "rows": []}
    for snr in config["main_snrs"]:
        rows = [row for seed in args.seeds
                for row in _run(model, config, seed, snr, args.frames, args.steps, args.lr,
                                args.radius, args.process_noise, args.ridge, args.gate, device)]
        result["rows"].extend(rows)
        n = len(rows)
        frozen = sum(r["frozen_ber"] for r in rows) / n
        online = sum(r["committed_ber"] for r in rows) / n
        raw = sum(r["online_ber"] for r in rows) / n
        accepted = sum(1 for r in rows if r["accepted"])
        result["per_snr"][str(snr)] = {
            "frozen_ber": frozen, "online_ber": online, "raw_online_ber": raw,
            "gain": frozen - online, "accepted": accepted, "frames": n,
        }
        print(f"snr={snr:2d} frozen={frozen:.6f} online={online:.6f} gain={frozen-online:+.6f} "
              f"raw={raw:.6f} accepted={accepted}/{n}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
