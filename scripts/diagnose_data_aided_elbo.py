# -*- coding: utf-8 -*-
"""只读诊断：Data 软边缘化 ELBO 是否提供可迁移到 Data 判决的更新方向。

对每个帧比较三条 head 梯度：
- adapt：Adapt Pilot BCE（当前在线目标）；
- data_oracle：Data BCE（仅离线诊断，不可部署）；
- elbo：Data 位置软边缘化模型似然（只用接收波形、模型输出和估计 CIR）。

并沿每个方向做一次固定半径的 head 步进，测量动作后的 Data BCE/BER 变化。
Data 标签只用于事后评分与 oracle 方向，不进入 ELBO 或在线状态。
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


def _bce(logits: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits[mask].float(), labels[mask].float())


def _head_parameters(model) -> dict[str, torch.Tensor]:
    return {"head.weight": model.head.weight, "head.bias": model.head.bias}


def _flat_grad(loss: torch.Tensor, model) -> torch.Tensor:
    grads = torch.autograd.grad(
        loss, [model.head.weight, model.head.bias], retain_graph=True, allow_unused=True,
    )
    return torch.cat([
        (grad if grad is not None else torch.zeros_like(param)).reshape(-1)
        for grad, param in zip(grads, [model.head.weight, model.head.bias])
    ]).detach()


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.dot(left, right) / (left.norm() * right.norm()).clamp_min(1e-12))


def _step_effect(model, context, direction, radius):
    direction = direction / direction.norm().clamp_min(1e-12)
    weight = model.head.weight.detach() - radius * direction[:model.head.weight.numel()].reshape_as(model.head.weight)
    bias = model.head.bias.detach() - radius * direction[model.head.weight.numel():].reshape_as(model.head.bias)
    with torch.no_grad():
        logits = context_logits(
            model, context, {"head.weight": weight, "head.bias": bias}, visible="data",
        )
        mask = context.frame.data_mask.bool()
        bce = _bce(logits, mask, context.frame.bits)
        ber = ((logits[mask] >= 0) != context.frame.bits[mask].bool()).float().mean()
    return float(bce), float(ber)


def _run(model, config, seed, snr, frames, radius, device):
    env, cir, cfo, tail, received = make_episode(
        config, model, seed=seed, snr=snr, frames=frames,
        state_split="offline_train", device=device,
    )
    rows = []
    for frame in received:
        context = make_context(frame, cir, cfo, tail, snr)
        logits = context_logits(model, context, None, visible="data")
        adapt_mask = frame.adapt_mask.bool()
        data_mask = frame.data_mask.bool()
        adapt_loss = _bce(logits, adapt_mask, frame.bits)
        data_loss = _bce(logits, data_mask, frame.bits)
        elbo = data_aided_elbo(model, context, None, variance_weight=1.0, anchor_weight=0.0)
        adapt_grad = _flat_grad(adapt_loss, model)
        data_grad = _flat_grad(data_loss, model)
        elbo_grad = _flat_grad(elbo["elbo"], model)
        base_bce, base_ber = _step_effect(model, context, torch.zeros_like(adapt_grad), radius)
        adapt_step = _step_effect(model, context, adapt_grad, radius)
        oracle_step = _step_effect(model, context, data_grad, radius)
        elbo_step = _step_effect(model, context, elbo_grad, radius)
        rows.append({
            "frame_index": int(frame.frame_index),
            "adapt_bce": float(adapt_loss),
            "data_bce": float(data_loss),
            "elbo": float(elbo["elbo"].detach()),
            "elbo_reconstruction": float(elbo["reconstruction"].detach()),
            "elbo_variance": float(elbo["variance"].detach()),
            "cos_adapt_data": _cosine(adapt_grad, data_grad),
            "cos_elbo_data": _cosine(elbo_grad, data_grad),
            "cos_elbo_adapt": _cosine(elbo_grad, adapt_grad),
            "adapt_grad_norm": float(adapt_grad.norm()),
            "data_grad_norm": float(data_grad.norm()),
            "elbo_grad_norm": float(elbo_grad.norm()),
            "step_adapt_bce_delta": adapt_step[0] - base_bce,
            "step_adapt_ber_delta": adapt_step[1] - base_ber,
            "step_oracle_bce_delta": oracle_step[0] - base_bce,
            "step_oracle_ber_delta": oracle_step[1] - base_ber,
            "step_elbo_bce_delta": elbo_step[0] - base_bce,
            "step_elbo_ber_delta": elbo_step[1] - base_ber,
        })
        tail = next_pilot_tail(
            model, context, alpha=float(config.get("tail_update_alpha", 0.5)), mode="dd_soft",
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[90001, 90002])
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--radius", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model, _ = _load_branch(Path(args.checkpoint), config, device)
    model.eval()
    # 基础模型默认冻结；本诊断只对 head 反传，显式打开 head 梯度。
    for parameter in model.head.parameters():
        parameter.requires_grad_(True)

    result = {"config": str(Path(args.config)), "checkpoint": str(Path(args.checkpoint)),
              "seeds": args.seeds, "frames": args.frames, "radius": args.radius, "per_snr": {}}
    keys = ["adapt_bce", "data_bce", "elbo", "elbo_reconstruction", "elbo_variance",
            "cos_adapt_data", "cos_elbo_data", "cos_elbo_adapt",
            "adapt_grad_norm", "data_grad_norm", "elbo_grad_norm",
            "step_adapt_bce_delta", "step_adapt_ber_delta",
            "step_oracle_bce_delta", "step_oracle_ber_delta",
            "step_elbo_bce_delta", "step_elbo_ber_delta"]
    for snr in config["main_snrs"]:
        rows = [
            row for seed in args.seeds
            for row in _run(model, config, seed, snr, args.frames, args.radius, device)
        ]
        count = len(rows)
        result["per_snr"][str(snr)] = {
            key: sum(row[key] for row in rows) / count for key in keys
        }
        result["per_snr"][str(snr)]["frames"] = count
        print(f"snr={snr:2d} cos_elbo_data={result['per_snr'][str(snr)]['cos_elbo_data']:+.4f} "
              f"cos_adapt_data={result['per_snr'][str(snr)]['cos_adapt_data']:+.4f} "
              f"elbo_step_ber={result['per_snr'][str(snr)]['step_elbo_ber_delta']:+.6f} "
              f"oracle_step_ber={result['per_snr'][str(snr)]['step_oracle_ber_delta']:+.6f}",
              flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
