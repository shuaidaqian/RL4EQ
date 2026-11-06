# -*- coding: utf-8 -*-
"""只读诊断：同帧 Data 软边缘化 ELBO head 适配能否吃到同帧 oracle 增益。

每帧独立：
- Frozen：离线 head + 共享物理状态；
- ELBO：用当前帧 Data 位置的 ELBO 更新 head K 步，再对同一帧重新判决；
- Oracle：用 Data BCE（仅诊断）更新 head K 步，作为同帧上界。

ELBO 只读接收波形、模型 soft 输出与估计 CIR；Data 标签只用于事后 BER 和
oracle 方向。head 适配不跨帧持久，物理状态（CIR/tail）保持共享。
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
from training.data_aided import data_aided_elbo
from training.pilot_transfer import context_logits, make_context, make_episode, next_pilot_tail


def _head_params(model):
    return (model.head.weight.detach().clone().requires_grad_(True),
            model.head.bias.detach().clone().requires_grad_(True))


def _as_dict(weight, bias):
    return {"head.weight": weight, "head.bias": bias}


def _metrics(logits, frame, mask_name="data"):
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


def _adapt(model, context, objective, steps, lr, radius, variance_weight, anchor_weight):
    weight, bias = _head_params(model)
    weight0, bias0 = weight.detach().clone(), bias.detach().clone()
    mw = torch.zeros_like(weight); vw = torch.zeros_like(weight)
    mb = torch.zeros_like(bias); vb = torch.zeros_like(bias)
    for step in range(1, int(steps) + 1):
        if objective == "elbo":
            result = data_aided_elbo(
                model, context, _as_dict(weight, bias),
                variance_weight=variance_weight, anchor_weight=anchor_weight,
            )
            loss = result["elbo"]
        else:
            logits = context_logits(model, context, _as_dict(weight, bias), visible="data")
            mask = context.frame.data_mask.bool()
            loss = F.binary_cross_entropy_with_logits(
                logits[mask].float(), context.frame.bits[mask].float(),
            )
        gw, gb = torch.autograd.grad(loss, [weight, bias], retain_graph=False)
        mw = 0.9 * mw + 0.1 * gw; vw = 0.999 * vw + 0.001 * gw.square()
        mb = 0.9 * mb + 0.1 * gb; vb = 0.999 * vb + 0.001 * gb.square()
        mw_hat = mw / (1.0 - 0.9 ** step); vw_hat = vw / (1.0 - 0.999 ** step)
        mb_hat = mb / (1.0 - 0.9 ** step); vb_hat = vb / (1.0 - 0.999 ** step)
        weight = weight - lr * mw_hat / (vw_hat.sqrt() + 1e-8)
        bias = bias - lr * mb_hat / (vb_hat.sqrt() + 1e-8)
        weight, bias = _project(weight, bias, weight0, bias0, radius)
    return weight.detach(), bias.detach()


def _evaluate(model, context, weight, bias):
    with torch.no_grad():
        logits = context_logits(model, context, _as_dict(weight, bias), visible="data")
        data_bce, data_ber = _metrics(logits, context.frame, "data")
        reward_bce, reward_ber = _metrics(logits, context.frame, "reward")
    return {"data_bce": data_bce, "data_ber": data_ber,
            "reward_bce": reward_bce, "reward_ber": reward_ber}


def _run(model, config, seed, snr, frames, steps, lr, radius, variance_weight, anchor_weight, device):
    env, cir, cfo, tail, received = make_episode(
        config, model, seed=seed, snr=snr, frames=frames,
        state_split="offline_train", device=device,
    )
    rows = []
    for frame in received:
        context = make_context(frame, cir, cfo, tail, snr)
        with torch.no_grad():
            base = _evaluate(model, context, model.head.weight, model.head.bias)
        elbo_w, elbo_b = _adapt(
            model, context, "elbo", steps, lr, radius, variance_weight, anchor_weight,
        )
        oracle_w, oracle_b = _adapt(model, context, "oracle", steps, lr, radius, 1.0, 0.0)
        elbo = _evaluate(model, context, elbo_w, elbo_b)
        oracle = _evaluate(model, context, oracle_w, oracle_b)
        rows.append({
            "seed": int(seed), "snr_db": int(snr), "frame_index": int(frame.frame_index),
            "frozen_ber": base["data_ber"], "elbo_ber": elbo["data_ber"], "oracle_ber": oracle["data_ber"],
            "frozen_bce": base["data_bce"], "elbo_bce": elbo["data_bce"], "oracle_bce": oracle["data_bce"],
            "frozen_reward_ber": base["reward_ber"], "elbo_reward_ber": elbo["reward_ber"],
            "frozen_reward_bce": base["reward_bce"], "elbo_reward_bce": elbo["reward_bce"],
        })
        tail = next_pilot_tail(
            model, context, alpha=float(config.get("tail_update_alpha", 0.5)), mode="dd_soft",
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[93001, 93002])
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--variance-weight", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model, _ = _load_branch(Path(args.checkpoint), config, device)
    model.eval()

    result = {"config": str(Path(args.config)), "checkpoint": str(Path(args.checkpoint)),
              "seeds": args.seeds, "frames": args.frames, "steps": args.steps,
              "lr": args.lr, "radius": args.radius,
              "variance_weight": args.variance_weight, "anchor_weight": args.anchor_weight,
              "per_snr": {}, "rows": []}
    for snr in config["main_snrs"]:
        rows = [row for seed in args.seeds
                for row in _run(model, config, seed, snr, args.frames, args.steps, args.lr,
                                args.radius, args.variance_weight, args.anchor_weight, device)]
        result["rows"].extend(rows)
        n = len(rows)
        frozen = sum(r["frozen_ber"] for r in rows) / n
        elbo = sum(r["elbo_ber"] for r in rows) / n
        oracle = sum(r["oracle_ber"] for r in rows) / n
        gate_reject = sum(1 for r in rows if r["elbo_reward_bce"] > r["frozen_reward_bce"])
        result["per_snr"][str(snr)] = {
            "frozen_ber": frozen, "elbo_ber": elbo, "oracle_ber": oracle,
            "elbo_gain": frozen - elbo, "oracle_gain": frozen - oracle,
            "reward_bce_reject_frames": gate_reject, "frames": n,
        }
        print(f"snr={snr:2d} frozen={frozen:.6f} elbo={elbo:.6f} "
              f"gain={frozen-elbo:+.6f} oracle={oracle:.6f} oracle_gain={frozen-oracle:+.6f} "
              f"reward_bce_reject={gate_reject}/{n}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
