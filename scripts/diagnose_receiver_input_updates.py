# -*- coding: utf-8 -*-
"""离线组件实验：分开检验候选更新输入与 Reward 验收输入，不注册正式方法。"""

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
from training.pilot_transfer import context_logits, differentiable_adapt, make_context, make_episode, next_pilot_tail
from training.windowed_discrete_ppo import should_accept_reward_guard


def _reward_metrics(logits, frame):
    """只从接收端已知 Reward 符号构造标签，不读取 Data 标签。"""
    labels = (frame.receiver_view().reward_symbols.real > 0).float()
    mask = frame.reward_mask
    loss = F.binary_cross_entropy_with_logits(logits[mask].float(), labels[mask])
    ber = ((logits[mask] >= 0) != labels[mask].bool()).float().mean()
    return float(ber), float(loss)


def pilot_candidates_and_decisions(model, context, settings, groups):
    """所有候选及其验收先结束；此函数不计算任何 Data 标签指标。"""
    with torch.no_grad():
        before_logits = {"pilot_only": context_logits(model, context, visible="pilot"),
                         "full_rx": context_logits(model, context, visible="full_rx")}
        before_metrics = {key: _reward_metrics(value, context.frame) for key, value in before_logits.items()}
    decisions, candidates = [], {}
    for group in groups:
        for receiver_input in ("adapt_only", "full_rx"):
            fast, update = differentiable_adapt(
                model, [context], learning_rate=float(settings["inner_learning_rate"]),
                steps=int(settings["inner_steps"]), max_delta_norm=float(settings["max_delta_norm"]),
                groups={group}, objective="bce", receiver_input=receiver_input,
            )
            fast = {name: value.detach() for name, value in fast.items()}
            key = f"{group}:{receiver_input}"
            candidates[key] = fast
            for reward_input in ("pilot_only", "full_rx"):
                with torch.no_grad():
                    after = context_logits(model, context, fast,
                                           visible="pilot" if reward_input == "pilot_only" else "full_rx")
                    ber1, loss1 = _reward_metrics(after, context.frame)
                ber0, loss0 = before_metrics[reward_input]
                accepted = should_accept_reward_guard(
                    reward_loss_before=loss0, reward_loss_after=loss1,
                    reward_ber_before=ber0, reward_ber_after=ber1,
                    allow_loss_only=True, peft_delta_norm=update["delta_norm"],
                )
                decisions.append({"group": group, "receiver_input": receiver_input, "reward_input": reward_input,
                                  "candidate_key": key, "accepted": accepted, **update,
                                  "reward_ber_before": ber0, "reward_ber_after": ber1,
                                  "reward_loss_before": loss0, "reward_loss_after": loss1,
                                  "reward_loss_improvement": loss0 - loss1})
    return decisions, candidates, before_logits["full_rx"]


def summarize(rows, seeds, snrs, groups):
    result = []
    for snr in snrs:
        for group in groups:
            reference = [row for row in rows if row["snr_db"] == snr and row["group"] == group
                         and row["receiver_input"] == "adapt_only" and row["reward_input"] == "pilot_only"]
            for receiver_input in ("adapt_only", "full_rx"):
                for reward_input in ("pilot_only", "full_rx"):
                    subset = [row for row in rows if row["snr_db"] == snr and row["group"] == group
                              and row["receiver_input"] == receiver_input and row["reward_input"] == reward_input]
                    if not subset or len(subset) != len(reference):
                        raise ValueError("输入协议组件配对不完整。")
                    item = {"snr_db": snr, "group": group, "receiver_input": receiver_input,
                            "reward_input": reward_input, "frames": len(subset),
                            "ber_frozen": float(np.mean([row["ber_frozen"] for row in subset])),
                            "ber_committed": float(np.mean([row["ber_committed"] for row in subset])),
                            "accepted": sum(row["accepted"] for row in subset),
                            "harmful_acceptances": sum(row["accepted"] and row["ber_candidate"] > row["ber_frozen"] for row in subset),
                            "fixed_errors": sum(row["fixed_errors"] for row in subset if row["accepted"]),
                            "introduced_errors": sum(row["introduced_errors"] for row in subset if row["accepted"])}
                    for metric, candidate in (("committed_gain", "ber_committed"), ("candidate_gain", "ber_candidate"),
                                              ("committed_bce_gain", "bce_committed")):
                        frozen = "bce_frozen" if "bce" in metric else "ber_frozen"
                        item[metric] = paired_interval([
                            np.mean([row[frozen] - row[candidate] for row in subset if row["seed"] == seed]) for seed in seeds
                        ])
                    item["gain_vs_strict_reference"] = paired_interval([
                        np.mean([row["ber_committed"] for row in reference if row["seed"] == seed])
                        - np.mean([row["ber_committed"] for row in subset if row["seed"] == seed]) for seed in seeds
                    ])
                    result.append(item)
    return result


def run_diagnostic(config, checkpoint, seeds, frames, device, output, groups=("head", "phase")):
    target = Path(output)
    if target.exists():
        raise ValueError("组件实验输出已存在，请指定新路径。")
    if (config.get("offline_state_split") != "offline_train" or config.get("main_level") != "B"
            or config.get("pilot_layout") != "prefix" or not seeds or frames < 1):
        raise ValueError("首轮输入组件实验只允许 offline_train、Level B、前缀 Pilot 和正预算。")
    if not groups or len(set(groups)) != len(groups) or not set(groups) <= {"head", "phase"}:
        raise ValueError("组件实验只支持不重复的已有 head/phase 参数组。")
    if len(set(seeds)) != len(seeds):
        raise ValueError("组件实验的 seed 不允许重复。")
    settings = config["pilot_transfer"]
    if settings.get("inner_objective") != "bce":
        raise ValueError("输入组件实验仅支持 Adapt BCE。")
    model, payload = _load_branch(Path(checkpoint), config, device)
    training_seeds = {item["seed"] for item in payload.get("metrics", {}).get("history", []) if "seed" in item}
    if training_seeds.intersection(seeds):
        raise ValueError("组件诊断 seed 与当前 checkpoint 的训练 seed 重叠。")
    rows = []
    for snr in config["main_snrs"]:
        for seed in seeds:
            _, cir, cfo, tail, received = make_episode(
                config, model, seed=seed, snr=snr, frames=frames, state_split="offline_train", device=device,
            )
            for index, frame in enumerate(received):
                context = make_context(frame, cir, cfo, tail, snr)
                decisions, candidates, frozen_logits = pilot_candidates_and_decisions(model, context, settings, groups)
                # 所有协议的 Pilot 更新和验收已结束；此后 Data 标签只用于离线仿真评分。
                with torch.no_grad():
                    ber0, bce0 = _metrics(frozen_logits, frame, frame.data_mask)
                    wrong0 = (frozen_logits[frame.data_mask] >= 0) != frame.bits[frame.data_mask]
                    metrics = {}
                    for key, fast in candidates.items():
                        logits = context_logits(model, context, fast, visible="full_rx")
                        ber1, bce1 = _metrics(logits, frame, frame.data_mask)
                        wrong1 = (logits[frame.data_mask] >= 0) != frame.bits[frame.data_mask]
                        metrics[key] = {"ber_candidate": ber1, "bce_candidate": bce1,
                                        "fixed_errors": int((wrong0 & ~wrong1).sum()),
                                        "introduced_errors": int((~wrong0 & wrong1).sum()),
                                        "data_logit_abs_change": float((logits - frozen_logits)[frame.data_mask].abs().mean())}
                fingerprint = hashlib.sha256(frame.rx_symbols.detach().cpu().numpy().tobytes()).hexdigest()
                for decision in decisions:
                    candidate = metrics[decision["candidate_key"]]
                    rows.append({"snr_db": snr, "seed": seed, "frame": index, **decision, **candidate,
                                 "ber_frozen": ber0, "bce_frozen": bce0,
                                 "ber_committed": candidate["ber_candidate"] if decision["accepted"] else ber0,
                                 "bce_committed": candidate["bce_candidate"] if decision["accepted"] else bce0,
                                 "data_labels_used_for_control": False,
                                 "data_waveform_used_for_candidate": decision["receiver_input"] == "full_rx",
                                 "data_waveform_used_for_guard": decision["reward_input"] == "full_rx",
                                 "input_sha256": fingerprint})
                # 不原地提交任何候选；历史规则也不随实验输入协议变化。
                tail = next_pilot_tail(model, context, alpha=float(config.get("tail_update_alpha", 0.5)))
            print(json.dumps({"snr_db": snr, "seed": seed, "frames_complete": frames}), flush=True)
    payload = {"schema_version": "receiver-input-candidate-prototype-v1", "diagnostic_only": True,
               "main_aggregation_allowed": False, "candidate_protocol_only_not_formal": True,
               "state_split": "offline_train", "config": config, "seeds": list(seeds), "frames": frames,
               "groups": list(groups), "parameter_counts": {
                   group: sum(value.numel() for _, value in model.peft.named_group_parameters({group})) for group in groups},
               "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
               "linear_operator_backend": os.environ.get("RL4EQ_LINEAR_OPERATOR_BACKEND", "direct"),
               "base_weights_modified": False, "data_labels_used_for_control": False,
               "per_snr": summarize(rows, seeds, config["main_snrs"], groups), "rows": rows}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["per_snr"], ensure_ascii=False), flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--seed-start", type=int, default=42000)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--groups", choices=["head", "phase"], nargs="+", default=["head", "phase"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.frames < 1 or args.num_seeds < 1:
        parser.error("frames 与 num-seeds 必须为正数。")
    run_diagnostic(json.loads(Path(args.config).read_text(encoding="utf-8")), args.pretrained,
                   list(range(args.seed_start, args.seed_start + args.num_seeds)), args.frames,
                   args.device, args.output, groups=args.groups)


if __name__ == "__main__":
    main()
