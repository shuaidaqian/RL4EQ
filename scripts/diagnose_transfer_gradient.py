# -*- coding: utf-8 -*-
"""仅在训练范围诊断更新后目标与普通 Data 重加权的梯度差异。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.pilot_transfer import _load_branch, paired_interval
from training.pilot_transfer import (
    context_logits, differentiable_adapt, make_context, make_episode, next_pilot_tail,
)


def compare_gradients(model, context, settings):
    """比较精确外层梯度；不执行优化、验收、动作选择或 checkpoint 写入。"""
    named = [(name, parameter) for name, parameter in model.named_parameters()
             if parameter.requires_grad]
    if not named:
        raise ValueError("梯度诊断需要可微的基础模型参数。")
    with sdpa_kernel(SDPBackend.MATH):
        logits = context_logits(model, context)
        mask, bits = context.frame.data_mask, context.frame.bits.float()
        base_loss = F.binary_cross_entropy_with_logits(logits.float(), bits)
        before_loss = F.binary_cross_entropy_with_logits(logits[mask].float(), bits[mask])
        fast, update = differentiable_adapt(
            model, [context], learning_rate=float(settings["inner_learning_rate"]),
            steps=int(settings["inner_steps"]), max_delta_norm=float(settings["max_delta_norm"]),
            groups=settings["inner_groups"], objective=settings["inner_objective"], create_graph=True,
        )
        after = context_logits(model, context, fast)
        after_loss = F.binary_cross_entropy_with_logits(after[mask].float(), bits[mask])
        gradients = []
        for index, loss in enumerate((base_loss, before_loss, after_loss)):
            values = torch.autograd.grad(loss, [parameter for _, parameter in named],
                                         retain_graph=index < 2, allow_unused=True)
            gradients.append([torch.zeros_like(parameter) if value is None else value
                              for (_, parameter), value in zip(named, values)])
    selected = {name for name, _ in model.peft.named_group_parameters(settings["inner_groups"])}
    result = {"base_bce": float(base_loss.detach()), "data_bce_before": float(before_loss.detach()),
              "data_bce_after": float(after_loss.detach()), **update}
    for label, indices in (("all_base", list(range(len(named)))),
                           ("selected_group", [i for i, (name, _) in enumerate(named) if name in selected])):
        if not indices:
            raise ValueError("选择的参数组不属于可微基础模型。")
        base, before, post = [torch.cat([values[i].detach().flatten() for i in indices]).float()
                              for values in gradients]
        if not all(bool(torch.isfinite(value).all()) for value in (base, before, post)):
            raise FloatingPointError("梯度诊断遇到非有限值。")
        weight0 = float(settings.get("base_weight", 1))
        weight1 = float(settings.get("post_weight", 1))
        reweighted = weight0 * base + weight1 * before
        updated = weight0 * base + weight1 * post
        residual = updated - reweighted
        result[label] = {
            "parameter_count": base.numel(),
            "post_minus_before_relative_norm": float((post - before).norm() / before.norm().clamp_min(1e-12)),
            "meta_residual_relative_to_reweighted": float(residual.norm() / reweighted.norm().clamp_min(1e-12)),
            "post_before_cosine": float(F.cosine_similarity(post, before, dim=0)),
            "updated_reweighted_cosine": float(F.cosine_similarity(updated, reweighted, dim=0)),
            "before_data_gradient_norm": float(before.norm()),
            "meta_residual_norm": float(residual.norm()),
        }
    return result


def run_diagnostic(config, checkpoint, seeds, frames, device, output):
    target = Path(output)
    if target.exists():
        raise ValueError("诊断输出已存在，请指定新路径。")
    if config.get("offline_state_split") != "offline_train" or not seeds or frames < 1:
        raise ValueError("梯度机制诊断只允许 offline_train、非空 seed 和正帧数。")
    model, _ = _load_branch(Path(checkpoint), config, device)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith("online_phase_trend_adapter."))
    settings = config["pilot_transfer"]
    rows = []
    for snr in config["main_snrs"]:
        for seed in seeds:
            _, cir, cfo, tail, received = make_episode(
                config, model, seed=seed, snr=snr, frames=frames,
                state_split="offline_train", device=device,
            )
            for index, frame in enumerate(received):
                context = make_context(frame, cir, cfo, tail, snr)
                rows.append({"snr_db": snr, "seed": seed, "frame": index,
                             "input_sha256": hashlib.sha256(frame.rx_symbols.detach().cpu().numpy().tobytes()).hexdigest(),
                             **compare_gradients(model, context, settings)})
                tail = next_pilot_tail(model, context, alpha=float(config.get("tail_update_alpha", 0.5)))
            print(json.dumps({"snr_db": snr, "seed": seed, "frames_complete": frames}), flush=True)
    summary = []
    for snr in config["main_snrs"]:
        item = {"snr_db": snr}
        for label in ("all_base", "selected_group"):
            item[label] = {}
            for field in ("post_minus_before_relative_norm", "meta_residual_relative_to_reweighted",
                          "post_before_cosine", "updated_reweighted_cosine"):
                item[label][field] = paired_interval([
                    np.mean([row[label][field] for row in rows if row["snr_db"] == snr and row["seed"] == seed])
                    for seed in seeds
                ])
        summary.append(item)
    payload = {"diagnostic_only": True, "main_aggregation_allowed": False,
               "state_split": "offline_train", "config": config, "seeds": seeds, "frames": frames,
               "uses_offline_data_labels": True, "performs_online_control": False,
               "updates_model_weights": False, "attention_backend": "math",
               "linear_operator_backend": os.environ.get("RL4EQ_LINEAR_OPERATOR_BACKEND", "direct"),
               "checkpoint": str(checkpoint),
               "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
               "per_snr": summary, "rows": rows}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--seed-start", type=int, default=38000)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.num_seeds < 1 or args.frames < 1:
        parser.error("num-seeds 与 frames 必须为正数。")
    run_diagnostic(json.loads(Path(args.config).read_text(encoding="utf-8")), args.pretrained,
                   list(range(args.seed_start, args.seed_start + args.num_seeds)),
                   args.frames, args.device, args.output)


if __name__ == "__main__":
    main()
