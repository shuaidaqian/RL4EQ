# -*- coding: utf-8 -*-
"""仅离线分析 Pilot 与 Data 梯度方向，不据 Data 标签生成在线候选。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pilot_transfer import _load_branch, paired_interval
from training.pilot_transfer import (
    _AdaptObjective, context_logits, differentiable_adapt, make_context, make_episode,
    next_pilot_tail, receiver_inputs, receiver_tasks, task_context,
)


def gradient_alignment(model, context, settings, *, history=None):
    """候选只由 Adapt 产生；离线标签只用于解释固定候选的局部方向。"""
    inner_task, reward_task = receiver_tasks(settings)
    if receiver_inputs(settings) != ("full_rx", "full_rx") or settings.get("inner_objective") != "bce":
        raise ValueError("本诊断只支持窗内完整接收波形与 Adapt BCE。")
    contexts = [context] if history is None else list(history)
    if not contexts or contexts[-1] is not context:
        raise ValueError("聚合历史必须非空且以当前上下文结束。")
    inner_views = [task_context(item, inner_task, role="adapt") for item in contexts]
    start = inner_views[0][1]
    if any(offset != start for _, offset in inner_views):
        raise ValueError("聚合窗内的 Adapt 评分起点必须一致。")
    inner_contexts = [view for view, _ in inner_views]
    reward, _ = task_context(context, reward_task, role="reward")
    selected = list(model.peft.named_group_parameters(settings["inner_groups"]))
    parameters = [p for _, p in selected]
    original_flags = [p.requires_grad for p in parameters]
    try:
        for p in parameters:
            p.requires_grad_(True)
        # 先固定合法候选，再开始任何 Data 标签分析；不对模型执行原位更新。
        fast, update = differentiable_adapt(model, inner_contexts,
            learning_rate=settings["inner_learning_rate"], steps=settings["inner_steps"],
            max_delta_norm=settings["max_delta_norm"], groups=settings["inner_groups"],
            objective="bce", receiver_input="full_rx", score_start=start)
        delta = torch.cat([(fast[n] - p).detach().flatten() for n, p in selected])
        gradients = {}
        losses = {}

        def collect(name, loss):
            values = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
            gradients[name] = torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                                         for p, g in zip(parameters, values)])
            losses[name] = float(loss.detach())

        objective = _AdaptObjective(model, start, "bce", "full_rx")
        collect("adapt", torch.stack([objective(item) for item in inner_contexts]).mean())
        logits = context_logits(model, reward, visible="full_rx")
        mask = reward.frame.reward_mask
        collect("reward", F.binary_cross_entropy_with_logits(logits[mask], reward.frame.bits[mask].float()))
        logits = context_logits(model, context, visible="full_rx")
        frame = context.frame
        positions = torch.arange(logits.numel(), device=logits.device)
        tail = positions >= logits.numel() - context.tail.numel()
        for name, mask in (("data", frame.data_mask), ("data_tail", frame.data_mask & tail),
                           ("data_interior", frame.data_mask & ~tail)):
            collect(name, F.binary_cross_entropy_with_logits(logits[mask], frame.bits[mask].float()))
        result = {"losses": losses, **update, "parameter_count": delta.numel(), "history_frames": len(contexts)}
        for name, gradient in gradients.items():
            result[name] = {"gradient_norm": float(gradient.norm()),
                            "predicted_loss_improvement": -float(torch.dot(gradient, delta))}
            if name != "adapt":
                result[name]["cosine_with_adapt"] = float(F.cosine_similarity(gradient, gradients["adapt"], dim=0))
        with torch.no_grad():
            after = context_logits(model, context, fast, visible="full_rx")
            mask = frame.data_mask
            result["actual_data_bce_improvement"] = losses["data"] - float(
                F.binary_cross_entropy_with_logits(after[mask], frame.bits[mask].float()))
            result["actual_data_ber_improvement"] = float(
                (((logits[mask] >= 0) != frame.bits[mask]).float().mean()
                 - ((after[mask] >= 0) != frame.bits[mask]).float().mean()).cpu())
        return result
    finally:
        for p, flag in zip(parameters, original_flags):
            p.requires_grad_(flag)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--seed-start", type=int, default=73000)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--history-windows", type=int, nargs="+", default=[1])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    target = Path(args.output)
    if target.exists() or args.num_seeds < 1 or args.frames < 1:
        raise ValueError("需要新的输出文件和正预算。")
    if not args.history_windows or min(args.history_windows) < 1 or max(args.history_windows) > args.frames:
        raise ValueError("聚合窗必须为正且不超过会话帧数。")
    if (config.get("offline_state_split") != "offline_train" or config.get("main_level") != "B"
            or config.get("pilot_layout") != "prefix"):
        raise ValueError("梯度解释只允许 Level B、prefix 和离线训练分布。")
    model, payload = _load_branch(args.pretrained, config, args.device)
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    if {r["seed"] for r in payload["metrics"]["history"]}.intersection(seeds):
        raise ValueError("诊断 seed 与训练重叠。")
    rows = []
    for snr in config["main_snrs"]:
        for seed in seeds:
            _, cir, cfo, tail, frames = make_episode(config, model, seed=seed, snr=snr,
                frames=args.frames, state_split="offline_train", device=args.device)
            history = []
            for index, frame in enumerate(frames):
                context = make_context(frame, cir, cfo, tail, snr)
                history.append(context)
                # 等最大聚合窗装满后，各窗口在相同当前帧上成对解释。
                if len(history) >= max(args.history_windows):
                    for window in args.history_windows:
                        rows.append({"snr_db": snr, "seed": seed, "frame": index, "window": window,
                            **gradient_alignment(model, context, config["pilot_transfer"], history=history[-window:])})
                tail = next_pilot_tail(model, context, alpha=config.get("tail_update_alpha", .5))
        print(json.dumps({"snr_db": snr, "complete": True}), flush=True)
    summary = []
    for snr in config["main_snrs"]:
        for window in args.history_windows:
            subset = [r for r in rows if r["snr_db"] == snr and r["window"] == window]
            count = args.frames - max(args.history_windows) + 1
            item = {"snr_db": snr, "window": window}
            for name in ("adapt", "reward", "data", "data_tail", "data_interior"):
                item[name] = {}
                for field in rows[0][name]:
                    item[name][field] = paired_interval([
                        sum(r[name][field] for r in subset if r["seed"] == seed) / count for seed in seeds])
            for field in ("actual_data_bce_improvement", "actual_data_ber_improvement"):
                item[field] = paired_interval([sum(r[field] for r in subset if r["seed"] == seed) / count for seed in seeds])
            summary.append(item)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"diagnostic_only": True, "main_aggregation_allowed": False,
        "uses_offline_data_labels": True, "data_labels_used_for_candidates": False,
        "updates_model_weights": False, "state_split": "offline_train", "config": config,
        "checkpoint": args.pretrained, "checkpoint_sha256": hashlib.sha256(Path(args.pretrained).read_bytes()).hexdigest(),
        "seeds": seeds, "frames": args.frames, "history_windows": args.history_windows,
        "per_snr": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
