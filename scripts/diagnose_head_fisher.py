# -*- coding: utf-8 -*-
"""用合法Adapt Pilot审计强离线网络共享输出头的信息几何。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pilot_transfer import _load_branch
from training.pilot_transfer import make_context, make_episode, context_logits


def head_fisher(logit_jacobian: torch.Tensor, logits: torch.Tensor,
                adapt_mask: torch.Tensor) -> torch.Tensor:
    """由完整均衡前向对head参数的Jacobian构造Pilot logistic Fisher。"""
    if logit_jacobian.ndim != 2 or logits.ndim != 1 or adapt_mask.ndim != 1:
        raise ValueError("head Jacobian、logit与Adapt掩码必须为二维/一维/一维。")
    if logit_jacobian.shape[0] != logits.numel() or logits.shape != adapt_mask.shape:
        raise ValueError("head Jacobian、logit与Adapt掩码长度必须一致。")
    if not torch.isfinite(logit_jacobian).all() or not torch.isfinite(logits).all():
        raise ValueError("head Fisher输入包含非有限值。")
    selected = adapt_mask.bool()
    if not bool(selected.any()):
        raise ValueError("Adapt Pilot不能为空。")
    probability = torch.sigmoid(logits[selected].detach())
    weight = (probability * (1.0 - probability)).clamp_min(1e-12)
    design = logit_jacobian[selected]
    fisher = design.T @ (weight[:, None] * design) / int(selected.sum())
    return (fisher + fisher.T) * 0.5


def summarize_fisher(fisher: torch.Tensor, *, relative_tolerance: float = 1e-6) -> dict:
    """汇总Fisher有效秩、谱熵和非零子空间条件数。"""
    if fisher.ndim != 2 or fisher.shape[0] != fisher.shape[1]:
        raise ValueError("Fisher矩阵必须为方阵。")
    eigenvalues_asc, eigenvectors_asc = torch.linalg.eigh(fisher.double())
    eigenvalues = eigenvalues_asc.clamp_min(0).flip(0)
    principal = eigenvectors_asc[:, -1]
    maximum = float(eigenvalues[0]) if eigenvalues.numel() else 0.0
    threshold = maximum * relative_tolerance
    active = eigenvalues[eigenvalues > threshold]
    rank = int(active.numel())
    total = float(eigenvalues.sum())
    probabilities = eigenvalues / max(total, 1e-300)
    positive = probabilities[probabilities > 0]
    entropy = float(-(positive * positive.log()).sum()) if positive.numel() else 0.0
    effective_rank = float(torch.exp(torch.tensor(entropy, dtype=torch.float64)))
    return {
        "dimension": int(fisher.shape[0]),
        "numerical_rank": rank,
        "relative_tolerance": relative_tolerance,
        "eigenvalue_max": maximum,
        "eigenvalue_min_active": float(active[-1]) if rank else 0.0,
        "active_condition_number": float(active[0] / active[-1]) if rank else None,
        "effective_rank": effective_rank,
        "principal_direction": [float(value) for value in principal],
        "eigenvalues": [float(value) for value in eigenvalues],
    }


def run(config: dict, checkpoint: str | Path, seeds: list[int], device: str, output: str | Path):
    target = Path(output)
    if target.exists():
        raise ValueError("输出文件已存在，请使用新路径。")
    if (config.get("main_level") != "B" or config.get("pilot_layout") != "prefix"
            or config.get("max_delay") != 116 or config.get("pilot_total") != 256
            or config.get("reward_pilot_total") != 32
            or config.get("main_snrs") != [0, 5, 10, 15]):
        raise ValueError("诊断只允许固定Level B、116符号长回波、前缀Pilot=256和四个指定SNR。")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("诊断seed必须非空且不重复。")
    model, payload = _load_branch(Path(checkpoint), config, device)
    training_seeds = {row["seed"] for row in payload.get("metrics", {}).get("history", []) if "seed" in row}
    if training_seeds.intersection(seeds):
        raise ValueError("诊断seed与来源checkpoint训练seed重叠。")
    model.eval()
    rows = []
    for snr in config["main_snrs"]:
        for seed in seeds:
            _, cir, cfo, tail, frames = make_episode(
                config, model, seed=seed, snr=snr, frames=1,
                state_split="offline_train", device=device,
            )
            frame = frames[0]
            context = make_context(frame, cir, cfo, tail, snr)
            theta = torch.cat((model.head.weight.detach().reshape(-1), model.head.bias.detach().reshape(-1)))
            def head_logits(value):
                parameters = {
                    "head.weight": value[:model.head.weight.numel()].reshape_as(model.head.weight),
                    "head.bias": value[model.head.weight.numel():].reshape_as(model.head.bias),
                }
                return context_logits(model, context, parameters, visible="adapt")[frame.adapt_mask]
            theta.requires_grad_(True)
            with torch.enable_grad():
                jacobian = torch.autograd.functional.jacobian(head_logits, theta, vectorize=True)
                logits = head_logits(theta)
            fisher = head_fisher(jacobian.detach(), logits.detach(), torch.ones_like(logits, dtype=torch.bool))
            summary = summarize_fisher(fisher)
            rows.append({
                "seed": seed, "snr_db": snr, "frame_index": frame.frame_index,
                "adapt_symbols": int(frame.adapt_mask.sum()),
                "parameter_count": int(model.head.weight.numel() + model.head.bias.numel()),
                "input_sha256": hashlib.sha256(frame.rx_symbols.detach().cpu().numpy().tobytes()).hexdigest(),
                "fisher": summary,
            })
            print(json.dumps({"seed": seed, "snr_db": snr, "rank": summary["numerical_rank"],
                              "effective_rank": summary["effective_rank"]}, ensure_ascii=False), flush=True)
    result = {
        "schema_version": "strong-offline-head-adapt-fisher-diagnostic-v1",
        "diagnostic_only": True,
        "main_aggregation_allowed": False,
        "state_split": "offline_train",
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "config": config, "seeds": seeds,
        "snrs_db": list(config["main_snrs"]),
        "data_labels_used": False,
        "data_labels_used_for_control": False,
        "online_updates_executed": False,
        "parameter_group": "head",
        "parameter_count": int(model.head.weight.numel() + model.head.bias.numel()),
        "rows": rows,
        "selection_rule": "仅按Adapt logistic Fisher的谱与数值秩确定候选子空间维数；不读取Data指标。",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--seed-start", type=int, default=62000)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.num_seeds < 1:
        parser.error("num-seeds必须为正数。")
    run(json.loads(Path(args.config).read_text(encoding="utf-8")), args.pretrained,
        list(range(args.seed_start, args.seed_start + args.num_seeds)), args.device, args.output)


if __name__ == "__main__":
    main()
