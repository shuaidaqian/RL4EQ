# -*- coding: utf-8 -*-
"""事后诊断现有 head/相位校正参数的作用能力，不生成在线控制器。"""

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

from evaluation.pilot_transfer import _load_branch, _metrics, paired_interval
from training.pilot_transfer import (
    _AdaptObjective, context_logits, differentiable_adapt, make_context, make_episode,
    next_pilot_tail,
)
from training.windowed_discrete_ppo import should_accept_reward_guard


def run_diagnostic(config, checkpoint, seeds, frames, device, output, *, groups=None, score_start=0):
    """先生成候选并完成 Pilot 验收，随后才用 Data 标签做离线梯度分析。"""
    target = Path(output)
    if target.exists():
        raise ValueError("诊断输出已存在，请指定新路径。")
    settings = config["pilot_transfer"]
    groups = list(groups if groups is not None else settings.get("inner_groups", ["head"]))
    if groups not in [["head"], ["phase"], ["phase_trend"]] or settings.get("inner_objective") != "bce":
        raise ValueError("此诊断仅支持 head、phase 或 phase_trend 单组的 Adapt BCE 更新。")
    if score_start < 0:
        raise ValueError("Pilot 评分起点不能为负。")
    model, _ = _load_branch(Path(checkpoint), config, device)
    selected = list(model.peft.named_group_parameters(set(groups)))
    if not selected:
        raise ValueError("模型中不存在指定的参数组。")
    for _, parameter in selected:
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
                fast, update = differentiable_adapt(
                    model, [context], learning_rate=settings["inner_learning_rate"],
                    steps=settings["inner_steps"], max_delta_norm=settings["max_delta_norm"],
                    groups=groups, objective="bce", score_start=score_start,
                )
                fast = {name: value.detach() for name, value in fast.items()}
                with torch.no_grad():
                    pilot_before = context_logits(model, context, visible="pilot")
                    pilot_after = context_logits(model, context, fast, visible="pilot")
                    ber0, loss0 = _metrics(pilot_before, frame, frame.reward_mask)
                    ber1, loss1 = _metrics(pilot_after, frame, frame.reward_mask)
                accepted = should_accept_reward_guard(
                    reward_loss_before=loss0, reward_loss_after=loss1,
                    reward_ber_before=ber0, reward_ber_after=ber1,
                    allow_loss_only=True, peft_delta_norm=update["delta_norm"],
                )
                # 此后才读取 Data 标签；得到的梯度只写入诊断文件，禁止用于动作选择。
                pilot_loss = _AdaptObjective(model, score_start=score_start, objective="bce")(context)
                pilot_gradients = torch.autograd.grad(
                    pilot_loss, [p for _, p in selected], allow_unused=True,
                )
                pilot_grad = torch.cat([
                    (torch.zeros_like(parameter) if gradient is None else gradient).flatten()
                    for (_, parameter), gradient in zip(selected, pilot_gradients)
                ])
                before = context_logits(model, context)
                mask = frame.data_mask
                data_loss = F.binary_cross_entropy_with_logits(before[mask], frame.bits[mask].float())
                data_gradients = torch.autograd.grad(
                    data_loss, [p for _, p in selected], allow_unused=True,
                )
                data_grad = torch.cat([
                    (torch.zeros_like(parameter) if gradient is None else gradient).flatten()
                    for (_, parameter), gradient in zip(selected, data_gradients)
                ])
                delta = torch.cat([(fast[name] - parameter).detach().flatten() for name, parameter in selected])
                cosine = F.cosine_similarity(pilot_grad, data_grad, dim=0, eps=1e-12)
                with torch.no_grad():
                    after = context_logits(model, context, fast)
                    frozen_ber, frozen_bce = _metrics(before.detach(), frame, mask)
                    candidate_ber, candidate_bce = _metrics(after, frame, mask)
                    change = after[mask] - before.detach()[mask]
                    frozen_wrong = (before.detach()[mask] >= 0) != frame.bits[mask]
                    candidate_wrong = (after[mask] >= 0) != frame.bits[mask]
                    rows.append({
                        "snr_db": snr, "seed": seed, "frame": index, "accepted": accepted,
                        "ber_frozen": frozen_ber, "ber_candidate": candidate_ber,
                        "ber_committed": candidate_ber if accepted else frozen_ber,
                        "bce_frozen": frozen_bce, "bce_candidate": candidate_bce,
                        "pilot_data_gradient_cosine": float(cosine),
                        "data_first_order_predicted_improvement": -float(torch.dot(data_grad, delta)),
                        "data_logit_abs_change": float(change.abs().mean()),
                        "data_logit_max_change": float(change.abs().max()),
                        "fixed_errors": int((frozen_wrong & ~candidate_wrong).sum()),
                        "introduced_errors": int((~frozen_wrong & candidate_wrong).sum()),
                        "data_bits": int(mask.sum()), **update,
                        "input_sha256": hashlib.sha256(frame.rx_symbols.detach().cpu().numpy().tobytes()).hexdigest(),
                    })
                tail = next_pilot_tail(model, context, alpha=float(config.get("tail_update_alpha", 0.5)))
            print(json.dumps({"snr_db": snr, "seed": seed, "frames_complete": frames}), flush=True)
    summary = []
    for snr in config["main_snrs"]:
        subset = [row for row in rows if row["snr_db"] == snr]
        item = {"snr_db": snr, "accepted": sum(row["accepted"] for row in subset),
                "frames": len(subset), "fixed_errors": sum(row["fixed_errors"] for row in subset),
                "introduced_errors": sum(row["introduced_errors"] for row in subset)}
        for field in ("pilot_data_gradient_cosine", "data_first_order_predicted_improvement",
                      "data_logit_abs_change", "data_logit_max_change"):
            item[field] = paired_interval([
                np.mean([row[field] for row in subset if row["seed"] == seed]) for seed in seeds
            ])
        for label, field in (("candidate_gain", "ber_candidate"), ("committed_gain", "ber_committed")):
            item[label] = paired_interval([
                np.mean([row["ber_frozen"] - row[field] for row in subset if row["seed"] == seed])
                for seed in seeds
            ])
        item["harmful_acceptances"] = sum(row["accepted"] and row["ber_candidate"] > row["ber_frozen"] for row in subset)
        summary.append(item)
    payload = {"diagnostic_only": True, "main_aggregation_allowed": False,
               "uses_data_labels_for_posthoc_diagnostic_only": True,
               "data_labels_used_for_online_decisions": False,
               "groups": groups, "score_start": score_start,
               "selected_parameter_count": sum(p.numel() for _, p in selected),
               "candidate_only_no_rl_action_registered": groups != ["head"] or score_start != 0,
               "state_split": "offline_train", "checkpoint": str(checkpoint),
               "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
               "linear_operator_backend": os.environ.get("RL4EQ_LINEAR_OPERATOR_BACKEND", "direct"),
               "seeds": seeds, "frames": frames, "config": config,
               "per_snr": summary, "rows": rows}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--seed-start", type=int, default=35000)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups", nargs="+", choices=["head", "phase", "phase_trend"], default=None)
    parser.add_argument("--score-start", type=int, default=0)
    args = parser.parse_args()
    if args.frames < 1 or args.num_seeds < 1:
        parser.error("frames 与 num-seeds 必须为正数。")
    run_diagnostic(json.loads(Path(args.config).read_text(encoding="utf-8")), args.pretrained,
                   list(range(args.seed_start, args.seed_start + args.num_seeds)),
                   args.frames, args.device, args.output, groups=args.groups, score_start=args.score_start)


if __name__ == "__main__":
    main()
