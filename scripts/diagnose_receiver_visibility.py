# -*- coding: utf-8 -*-
"""离线审计接收波形置零对 Pilot 梯度的影响，不改变在线控制器。"""

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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.pilot_transfer import _load_branch, paired_interval
from training.pilot_transfer import context_logits, make_context, make_episode, next_pilot_tail


GRADIENT_RESOLUTION = 1e-8


def _resolved_cosine(first, second):
    """接近零的梯度没有可解释方向，不把数值零余弦当成正交证据。"""
    if min(float(first.norm()), float(second.norm())) <= GRADIENT_RESOLUTION:
        return None
    return float(F.cosine_similarity(first, second, dim=0, eps=1e-12))


def _gradient(model, context, named, *, visible, mask, labels):
    logits = context_logits(model, context, visible=visible)
    loss = F.binary_cross_entropy_with_logits(logits[mask].float(), labels[mask].float())
    values = torch.autograd.grad(loss, [parameter for _, parameter in named], allow_unused=True)
    gradient = {
        name: torch.zeros_like(parameter) if value is None else value.detach()
        for (name, parameter), value in zip(named, values)
    }
    ber = float(((logits[mask] >= 0) != labels[mask].bool()).float().mean().detach())
    return float(loss.detach()), gradient, ber


def pilot_gradients(model, context, named):
    """只用各自已知 Pilot 标签；完整接收波形不提供未知发送符号。"""
    frame = context.frame
    view = frame.receiver_view()
    adapt_labels = (view.adapt_symbols.real > 0).float()
    reward_labels = (view.reward_symbols.real > 0).float()
    results = {}
    for name, visible, mask, labels in (
        ("adapt_only", "adapt", frame.adapt_mask, adapt_labels),
        ("adapt_full_rx", "data", frame.adapt_mask, adapt_labels),
        ("reward_pilot_rx", "pilot", frame.reward_mask, reward_labels),
        ("reward_full_rx", "data", frame.reward_mask, reward_labels),
    ):
        results[name] = _gradient(model, context, named, visible=visible, mask=mask, labels=labels)
    return results


def compare_visibility(model, context, groups=("head", "phase")):
    """Data 梯度只作事后离线参照，不输出更新候选或选择动作。"""
    named = list(model.peft.named_group_parameters(set(groups)))
    if not named or any(not parameter.requires_grad for _, parameter in named):
        raise ValueError("诊断需要指定可微的已有神经参数组。")
    pilot = pilot_gradients(model, context, named)
    frame = context.frame
    data_loss, data_gradient, data_ber = _gradient(
        model, context, named, visible="data", mask=frame.data_mask, labels=frame.bits,
    )
    result = {"pilot_bce": {name: values[0] for name, values in pilot.items()},
              "pilot_ber": {name: values[2] for name, values in pilot.items()},
              "data_bce": data_loss, "data_ber": data_ber, "groups": {}}
    for group in groups:
        names = [name for name, _ in model.peft.named_group_parameters({group})]
        if not names:
            raise ValueError("指定的已有参数组不存在。")
        vectors = {key: torch.cat([values[1][name].flatten() for name in names]).float()
                   for key, values in pilot.items()}
        vectors["data"] = torch.cat([data_gradient[name].flatten() for name in names]).float()
        if any(not bool(torch.isfinite(value).all()) for value in vectors.values()):
            raise FloatingPointError("可见性诊断出现非有限梯度。")
        item = {"parameter_count": vectors["data"].numel(),
                "data_gradient_norm": float(vectors["data"].norm())}
        for key in pilot:
            item[f"{key}_data_cosine"] = _resolved_cosine(vectors[key], vectors["data"])
            item[f"{key}_gradient_norm"] = float(vectors[key].norm())
        for prefix, masked, complete in (("adapt", "adapt_only", "adapt_full_rx"),
                                          ("reward", "reward_pilot_rx", "reward_full_rx")):
            item[f"{prefix}_masked_full_cosine"] = _resolved_cosine(vectors[masked], vectors[complete])
            item[f"{prefix}_relative_gradient_gap"] = (float(
                (vectors[masked] - vectors[complete]).norm() / vectors[complete].norm())
                if float(vectors[complete].norm()) > GRADIENT_RESOLUTION else None)
            angles = [item[f"{key}_data_cosine"] for key in (complete, masked)]
            item[f"{prefix}_data_alignment_change"] = (angles[0] - angles[1]
                                                        if all(value is not None for value in angles) else None)
        result["groups"][group] = item
    return result


def run_diagnostic(config, checkpoint, seeds, frames, device, output):
    target = Path(output)
    if target.exists():
        raise ValueError("诊断输出已存在，请指定新路径。")
    if config.get("offline_state_split") != "offline_train" or not seeds or frames < 1:
        raise ValueError("接收可见性诊断只允许 offline_train、非空 seed 和正帧数。")
    model, _ = _load_branch(Path(checkpoint), config, device)
    named = list(model.peft.named_group_parameters({"head", "phase"}))
    for _, parameter in named:
        parameter.requires_grad_(True)
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
                             **compare_visibility(model, context)})
                tail = next_pilot_tail(model, context, alpha=float(config.get("tail_update_alpha", 0.5)))
            print(json.dumps({"snr_db": snr, "seed": seed, "frames_complete": frames}), flush=True)
    summary = []
    fields = ("adapt_only_data_cosine", "adapt_full_rx_data_cosine", "reward_pilot_rx_data_cosine",
              "reward_full_rx_data_cosine", "adapt_masked_full_cosine", "reward_masked_full_cosine",
              "adapt_relative_gradient_gap", "reward_relative_gradient_gap",
              "adapt_data_alignment_change", "reward_data_alignment_change")
    for snr in config["main_snrs"]:
        item = {"snr_db": snr, "groups": {}}
        for group in ("head", "phase"):
            item["groups"][group] = {}
            for field in fields:
                seed_means, count = [], 0
                for seed in seeds:
                    values = [row["groups"][group][field] for row in rows
                              if row["snr_db"] == snr and row["seed"] == seed
                              and row["groups"][group][field] is not None]
                    count += len(values)
                    if values:
                        seed_means.append(float(np.mean(values)))
                stats = paired_interval(seed_means) if seed_means else {"mean": None, "ci95": None, "seeds": 0}
                item["groups"][group][field] = {**stats, "resolved_frames": count}
        item["pilot_metrics"] = {}
        for metric in ("pilot_ber", "pilot_bce"):
            item["pilot_metrics"][metric] = {view: paired_interval([
                np.mean([row[metric][view] for row in rows if row["snr_db"] == snr and row["seed"] == seed])
                for seed in seeds
            ]) for view in ("adapt_only", "adapt_full_rx", "reward_pilot_rx", "reward_full_rx")}
        summary.append(item)
    payload = {"diagnostic_only": True, "main_aggregation_allowed": False,
               "state_split": "offline_train", "config": config, "seeds": seeds, "frames": frames,
               "uses_offline_data_labels_for_gradient_reference": True,
               "pilot_gradients_use_data_labels": False, "performs_online_control": False,
               "updates_model_weights": False,
               "gradient_resolution_absolute_norm": GRADIENT_RESOLUTION,
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
    parser.add_argument("--seed-start", type=int, default=39000)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.frames < 1 or args.num_seeds < 1:
        parser.error("frames 与 num-seeds 必须为正数。")
    run_diagnostic(json.loads(Path(args.config).read_text(encoding="utf-8")), args.pretrained,
                   list(range(args.seed_start, args.seed_start + args.num_seeds)),
                   args.frames, args.device, args.output)


if __name__ == "__main__":
    main()
