# -*- coding: utf-8 -*-
"""只读诊断：神经 PEFT 的 prompt（Adapt Pilot）与 query（Data）方向对齐。

两臂共享 acquisition CIR + dd_soft tail；不引入 Bayes CIR。
对每个 session 计算：
- 每帧 Adapt BCE 与 Data BCE 对 head 的梯度；
- per-frame cos(Adapt, Data)；
- 跨帧方向稳定性：相邻帧梯度的余弦（Adapt 与 Data 各自）；
- 子空间捕获：Adapt 梯度 top-k 子空间能解释多少 Data 梯度能量，反之亦然。

Data 标签只用于事后 query 梯度诊断，不进入任何在线更新或门控。
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


def _head_grad(loss, model):
    grads = torch.autograd.grad(
        loss, [model.head.weight, model.head.bias], retain_graph=True, allow_unused=True,
    )
    return torch.cat([
        (g if g is not None else torch.zeros_like(p)).reshape(-1)
        for g, p in zip(grads, [model.head.weight, model.head.bias])
    ]).detach()


def _unit(value):
    return value / value.norm().clamp_min(1e-12)


def _cos(left, right):
    return float(torch.dot(_unit(left), _unit(right)))


def _subspace_capture(source, target, rank):
    """target 梯度能量被 source top-rank 子空间解释的比例。"""

    source = torch.stack([_unit(row) for row in source])
    target = torch.stack([_unit(row) for row in target])
    _, _, vh = torch.linalg.svd(source, full_matrices=False)
    basis = vh[:rank].transpose(0, 1)
    projected = target @ basis
    return float(projected.square().sum() / target.square().sum().clamp_min(1e-12))


def _stability(grads):
    if len(grads) < 2:
        return None
    return sum(_cos(grads[i], grads[i + 1]) for i in range(len(grads) - 1)) / (len(grads) - 1)


def _run(model, config, seed, snr, frames, rank, device):
    env, cir, cfo, tail, received = make_episode(
        config, model, seed=seed, snr=snr, frames=frames,
        state_split="offline_train", device=device,
    )
    adapt_grads, data_grads, elbo_grads = [], [], []
    for frame in received:
        context = make_context(frame, cir, cfo, tail, snr)
        logits = context_logits(model, context, None, visible="data")
        adapt_mask = frame.adapt_mask.bool()
        data_mask = frame.data_mask.bool()
        adapt_loss = F.binary_cross_entropy_with_logits(
            logits[adapt_mask].float(), frame.bits[adapt_mask].float(),
        )
        data_loss = F.binary_cross_entropy_with_logits(
            logits[data_mask].float(), frame.bits[data_mask].float(),
        )
        elbo = data_aided_elbo(model, context, None, variance_weight=1.0, anchor_weight=0.0)
        adapt_grads.append(_head_grad(adapt_loss, model))
        data_grads.append(_head_grad(data_loss, model))
        elbo_grads.append(_head_grad(elbo["elbo"], model))
        tail = next_pilot_tail(model, context, alpha=1.0, mode="dd_soft")

    per_frame = [_cos(adapt_grads[i], data_grads[i]) for i in range(frames)]
    elbo_data = [_cos(elbo_grads[i], data_grads[i]) for i in range(frames)]
    return {
        "seed": int(seed), "snr_db": int(snr),
        "cos_adapt_data": sum(per_frame) / len(per_frame),
        "cos_elbo_data": sum(elbo_data) / len(elbo_data),
        "cos_elbo_adapt": sum(_cos(elbo_grads[i], adapt_grads[i]) for i in range(frames)) / frames,
        "adapt_stability": _stability(adapt_grads),
        "data_stability": _stability(data_grads),
        "elbo_stability": _stability(elbo_grads),
        "adapt_subspace_captures_data": _subspace_capture(adapt_grads, data_grads, rank),
        "data_subspace_captures_adapt": _subspace_capture(data_grads, adapt_grads, rank),
        "elbo_subspace_captures_data": _subspace_capture(elbo_grads, data_grads, rank),
        "data_subspace_captures_elbo": _subspace_capture(data_grads, elbo_grads, rank),
        "adapt_grad_norm": sum(float(g.norm()) for g in adapt_grads) / frames,
        "data_grad_norm": sum(float(g.norm()) for g in data_grads) / frames,
        "elbo_grad_norm": sum(float(g.norm()) for g in elbo_grads) / frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[110001, 110002, 110003])
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model, _ = _load_branch(Path(args.checkpoint), config, device)
    model.eval()
    for parameter in model.head.parameters():
        parameter.requires_grad_(True)

    result = {"config": str(Path(args.config)), "checkpoint": str(Path(args.checkpoint)),
              "seeds": args.seeds, "frames": args.frames, "rank": args.rank,
              "per_snr": {}, "rows": []}
    for snr in config["main_snrs"]:
        rows = [_run(model, config, seed, snr, args.frames, args.rank, device) for seed in args.seeds]
        result["rows"].extend(rows)
        n = len(rows)
        keys = ["cos_adapt_data", "cos_elbo_data", "cos_elbo_adapt",
                "adapt_stability", "data_stability", "elbo_stability",
                "adapt_subspace_captures_data", "data_subspace_captures_adapt",
                "elbo_subspace_captures_data", "data_subspace_captures_elbo",
                "adapt_grad_norm", "data_grad_norm", "elbo_grad_norm"]
        result["per_snr"][str(snr)] = {key: sum(row[key] for row in rows) / n for key in keys}
        result["per_snr"][str(snr)]["rows"] = rows
        print(f"snr={snr:2d} cos_adapt_data={result['per_snr'][str(snr)]['cos_adapt_data']:+.4f} "
              f"cos_elbo_data={result['per_snr'][str(snr)]['cos_elbo_data']:+.4f} "
              f"elbo_stab={result['per_snr'][str(snr)]['elbo_stability']:+.4f} "
              f"adapt_to_data={result['per_snr'][str(snr)]['adapt_subspace_captures_data']:.4f} "
              f"elbo_to_data={result['per_snr'][str(snr)]['elbo_subspace_captures_data']:.4f} "
              f"data_stab={result['per_snr'][str(snr)]['data_stability']:+.4f} "
              f"norms={result['per_snr'][str(snr)]['adapt_grad_norm']:.3g}/{result['per_snr'][str(snr)]['data_grad_norm']:.3g}/{result['per_snr'][str(snr)]['elbo_grad_norm']:.3g}",
              flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
