# -*- coding: utf-8 -*-
"""开发 smoke：query-aligned 低秩 Adapter + ELBO 跨帧更新。

- 两臂共享 acquisition CIR + dd_soft tail；不引入 Bayes CIR。
- Frozen：head 参数 = theta0（c=0）。
- Online：head 参数 = theta0 + B c；B 由开发 seed 的 ELBO 梯度 top-K 子空间得到；
  c 用当前帧 ELBO 梯度更新（Adam + 信赖域 + Reward Pilot 门控），作用于下一帧。
- ELBO 只读接收波形、模型 soft 输出与估计 CIR；Data 标签只用于事后 BER。
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


def _flat_head(model):
    return torch.cat([model.head.weight.detach().reshape(-1), model.head.bias.detach().reshape(-1)])


def _head_params(model, theta0, basis, coefficient):
    theta = theta0 + basis @ coefficient
    weight = theta[:model.head.weight.numel()].reshape_as(model.head.weight)
    bias = theta[model.head.weight.numel():].reshape_as(model.head.bias)
    return {"head.weight": weight, "head.bias": bias}


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


def _ber(logits, frame):
    mask = frame.data_mask.bool()
    return float(((logits[mask] >= 0) != frame.bits[mask].bool()).float().mean())


def _reward_bce(logits, frame):
    mask = frame.reward_mask.bool()
    return float(F.binary_cross_entropy_with_logits(
        logits[mask].float(), frame.bits[mask].float(),
    ))


def _collect_elbo_basis(model, config, seeds, frames, rank, device):
    grads = []
    for seed in seeds:
        for snr in config["main_snrs"]:
            _, cir, cfo, tail, received = make_episode(
                config, model, seed=seed, snr=snr, frames=frames,
                state_split="offline_train", device=device,
            )
            for frame in received:
                context = make_context(frame, cir, cfo, tail, snr)
                elbo = data_aided_elbo(model, context, None, variance_weight=1.0, anchor_weight=0.0)
                grads.append(_unit(_head_grad(elbo["elbo"], model)))
                tail = next_pilot_tail(model, context, alpha=1.0, mode="dd_soft")
    matrix = torch.stack(grads)
    _, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    return vh[:rank].transpose(0, 1).contiguous()


def _run(model, config, seed, snr, frames, theta0, basis, lr, radius, gate, steps, device):
    _, cir, cfo, tail, received = make_episode(
        config, model, seed=seed, snr=snr, frames=frames,
        state_split="offline_train", device=device,
    )
    rank = basis.shape[1]
    zero = torch.zeros(rank, device=device)
    coefficient = torch.zeros(rank, device=device, requires_grad=True)
    m = torch.zeros(rank, device=device)
    v = torch.zeros(rank, device=device)
    rows = []
    for frame in received:
        context = make_context(frame, cir, cfo, tail, snr)
        with torch.no_grad():
            frozen_logits = context_logits(
                model, context, _head_params(model, theta0, basis, zero), visible="data",
            )
            online_logits = context_logits(
                model, context, _head_params(model, theta0, basis, coefficient.detach()), visible="data",
            )
            frozen_ber = _ber(frozen_logits, frame)
            online_ber = _ber(online_logits, frame)
        tail = next_pilot_tail(model, context, alpha=1.0, mode="dd_soft")

        accepted = 0
        for step in range(1, int(steps) + 1):
            elbo = data_aided_elbo(
                model, context, _head_params(model, theta0, basis, coefficient),
                variance_weight=1.0, anchor_weight=0.0,
            )
            gradient = torch.autograd.grad(elbo["elbo"], coefficient)[0]
            m = 0.9 * m + 0.1 * gradient
            v = 0.999 * v + 0.001 * gradient.square()
            m_hat = m / (1.0 - 0.9 ** step)
            v_hat = v / (1.0 - 0.999 ** step)
            candidate = coefficient.detach() - lr * m_hat / (v_hat.sqrt() + 1e-8)
            norm = candidate.norm().clamp_min(1e-12)
            if float(norm) > radius:
                candidate = candidate * (radius / norm)
            with torch.no_grad():
                reward_before = _reward_bce(
                    context_logits(model, context, _head_params(model, theta0, basis, coefficient.detach()), visible="data"),
                    frame,
                )
                reward_after = _reward_bce(
                    context_logits(model, context, _head_params(model, theta0, basis, candidate), visible="data"),
                    frame,
                )
            if reward_after <= reward_before + gate:
                coefficient = candidate.detach().requires_grad_(True)
                accepted += 1
            else:
                coefficient = coefficient.detach().requires_grad_(True)

        rows.append({
            "seed": int(seed), "snr_db": int(snr), "frame_index": int(frame.frame_index),
            "frozen_ber": frozen_ber, "online_ber": online_ber,
            "gain": frozen_ber - online_ber, "accepted_steps": accepted,
            "coefficient_norm": float(coefficient.detach().norm()),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--basis-seeds", nargs="+", type=int, default=[110001, 110002, 110003])
    parser.add_argument("--basis-frames", type=int, default=8)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[120001, 120002])
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--gate", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model, _ = _load_branch(Path(args.checkpoint), config, device)
    model.eval()
    for parameter in model.head.parameters():
        parameter.requires_grad_(True)
    theta0 = _flat_head(model)
    basis = _collect_elbo_basis(
        model, config, args.basis_seeds, args.basis_frames, args.rank, device,
    ).to(device=device, dtype=theta0.dtype)

    result = {"config": str(Path(args.config)), "checkpoint": str(Path(args.checkpoint)),
              "basis_seeds": args.basis_seeds, "basis_frames": args.basis_frames,
              "eval_seeds": args.eval_seeds, "frames": args.frames, "rank": args.rank,
              "lr": args.lr, "radius": args.radius, "gate": args.gate, "steps": args.steps,
              "per_snr": {}, "rows": []}
    for snr in config["main_snrs"]:
        rows = [row for seed in args.eval_seeds
                for row in _run(model, config, seed, snr, args.frames, theta0, basis,
                                args.lr, args.radius, args.gate, args.steps, device)]
        result["rows"].extend(rows)
        n = len(rows)
        frozen = sum(r["frozen_ber"] for r in rows) / n
        online = sum(r["online_ber"] for r in rows) / n
        result["per_snr"][str(snr)] = {
            "frozen_ber": frozen, "online_ber": online, "gain": frozen - online,
            "accepted_steps": sum(r["accepted_steps"] for r in rows),
            "coefficient_norm_mean": sum(r["coefficient_norm"] for r in rows) / n,
            "frames": n,
        }
        print(f"snr={snr:2d} frozen={frozen:.6f} online={online:.6f} gain={frozen-online:+.6f} "
              f"accepted={result['per_snr'][str(snr)]['accepted_steps']} "
              f"|c|={result['per_snr'][str(snr)]['coefficient_norm_mean']:.4f}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
