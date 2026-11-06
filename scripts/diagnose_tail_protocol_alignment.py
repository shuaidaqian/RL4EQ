# -*- coding: utf-8 -*-
"""只读诊断：比较 pilot_only 与 dd_soft 两种跨帧 soft-tail 协议。

dd_soft 使用当前帧最后 max_delay 个位置的模型软判决递推下一帧 tail，
与离线 curriculum 的 online_soft 一致；它只读取接收波形和模型自身输出，
不读取 Data 发送符号或 Data 比特。

本脚本的 Data 标签只用于事后 BER/BCE 与梯度对齐评分，不回流任何在线
状态、门控或参数更新。输出仅作开发机制诊断，不是正式确认。
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

from evaluation.pilot_transfer import _load_branch
from training.pilot_transfer import (
    TAIL_UPDATE_MODES,
    context_logits,
    make_context,
    make_episode,
    next_pilot_tail,
)


def _bce(logits: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits[mask].float(), labels[mask].float(),
    )


def _flat_head_grad(model, loss: torch.Tensor) -> torch.Tensor:
    """返回 head 的展平梯度；共享计算图，供同帧多个诊断损失复用。"""

    parameters = [model.head.weight, model.head.bias]
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([
        (grad if grad is not None else torch.zeros_like(param)).reshape(-1)
        for grad, param in zip(grads, parameters)
    ]).detach()


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.dot(left, right) / (left.norm() * right.norm()).clamp_min(1e-12))


def _run_session(model, config, seed, snr, frames, mode, device):
    env, cir, cfo, tail, received = make_episode(
        config, model, seed=seed, snr=snr, frames=frames,
        state_split="offline_train", device=device,
    )
    frame_rows = []
    for frame in received:
        context = make_context(frame, cir, cfo, tail, snr)
        logits = context_logits(model, context, None, visible="data")
        adapt_mask = frame.adapt_mask.bool()
        data_mask = frame.data_mask.bool()
        adapt_bce = _bce(logits, adapt_mask, frame.bits)
        data_bce = _bce(logits, data_mask, frame.bits)
        adapt_grad = _flat_head_grad(model, adapt_bce)
        data_grad = _flat_head_grad(model, data_bce)
        positions = torch.arange(frame.bits.numel(), device=frame.bits.device)
        head_mask = adapt_mask & (positions < tail.numel())
        rest_mask = adapt_mask & (positions >= tail.numel())
        head_grad = _flat_head_grad(model, _bce(logits, head_mask, frame.bits))
        rest_grad = _flat_head_grad(model, _bce(logits, rest_mask, frame.bits))
        with torch.no_grad():
            data_ber = ((logits[data_mask] >= 0) != frame.bits[data_mask].bool()).float().mean()
        frame_rows.append({
            "frame_index": int(frame.frame_index),
            "data_ber": float(data_ber),
            "adapt_bce": float(adapt_bce),
            "data_bce": float(data_bce),
            "adapt_data_cosine": _cosine(adapt_grad, data_grad),
            "head116_data_cosine": _cosine(head_grad, data_grad),
            "rest_data_cosine": _cosine(rest_grad, data_grad),
            "adapt_grad_norm": float(adapt_grad.norm()),
            "data_grad_norm": float(data_grad.norm()),
        })
        tail = next_pilot_tail(model, context, alpha=float(config.get("tail_update_alpha", 0.5)),
                               mode=mode)
    return frame_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[90001, 90002])
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model, _ = _load_branch(Path(args.checkpoint), config, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    result = {"config": str(Path(args.config)), "checkpoint": str(Path(args.checkpoint)),
              "seeds": args.seeds, "frames": args.frames, "per_snr": {}}
    for snr in config["main_snrs"]:
        result["per_snr"][str(snr)] = {}
        for mode in TAIL_UPDATE_MODES:
            rows = [
                row
                for seed in args.seeds
                for row in _run_session(model, config, seed, snr, args.frames, mode, device)
            ]
            count = len(rows)
            result["per_snr"][str(snr)][mode] = {
                "frames": count,
                "data_ber": sum(row["data_ber"] for row in rows) / count,
                "adapt_bce": sum(row["adapt_bce"] for row in rows) / count,
                "data_bce": sum(row["data_bce"] for row in rows) / count,
                "adapt_data_cosine": sum(row["adapt_data_cosine"] for row in rows) / count,
                "head116_data_cosine": sum(row["head116_data_cosine"] for row in rows) / count,
                "rest_data_cosine": sum(row["rest_data_cosine"] for row in rows) / count,
                "adapt_grad_norm": sum(row["adapt_grad_norm"] for row in rows) / count,
                "data_grad_norm": sum(row["data_grad_norm"] for row in rows) / count,
            }
            print(f"snr={snr:2d} {mode:9s} "
                  f"BER={result['per_snr'][str(snr)][mode]['data_ber']:.6f} "
                  f"adapt_bce={result['per_snr'][str(snr)][mode]['adapt_bce']:.4f} "
                  f"cos={result['per_snr'][str(snr)][mode]['adapt_data_cosine']:+.4f}",
                  flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
