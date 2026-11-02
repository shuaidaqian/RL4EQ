# -*- coding: utf-8 -*-
"""按实际 CIR 变化审计神经状态 head 的作用幅度；只在训练分布运行。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.func import functional_call

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pilot_transfer import _load_branch, _metrics, paired_interval
from scripts.diagnose_receiver_input_updates import _reward_metrics
from training.pilot_transfer import (channel_state_reward_loss, context_logits, differentiable_adapt,
    make_context, make_episode)
from training.windowed_discrete_ppo import _adapt_receiver_view, should_accept_reward_guard


ACTIONS = (("small", .005, 1, .05), ("medium", .02, 3, .2), ("large", .05, 3, .5))


class _StateOutput(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, context):
        view = _adapt_receiver_view(context.frame)
        iq = torch.stack((view.rx_symbols.real, view.rx_symbols.imag), dim=-1).unsqueeze(0).float()
        corrected = self.model._apply_phase_correction(iq, context.condition)
        return self.model._channel_state_condition(corrected, context.condition,
            view.adapt_symbols.unsqueeze(0), view.adapt_mask.unsqueeze(0), context.frame.frame_index).complex_cir


@torch.no_grad()
def state_output(model, context, parameters=None):
    captured = []
    hook = model.channel_state.head.register_forward_hook(lambda module, args, result: captured.append(result.detach()))
    try:
        wrapper = _StateOutput(model)
        cir = (wrapper(context) if parameters is None else functional_call(wrapper,
            {"model." + name: value for name, value in parameters.items()}, (context,)))
    finally:
        hook.remove()
    gates = 2 * torch.tanh(captured[-1])
    return cir, {"gate_mean": float(gates.mean()), "gate_abs_max": float(gates.abs().max()),
                 "source_cir_delta_norm": float((cir - context.condition.complex_cir).norm())}


def candidates_and_guards(model, context, objectives=("channel_state_signal",), actions=ACTIONS):
    """不访问 Data 标签；幅度档位固定，不根据事后指标排序。"""
    with torch.no_grad():
        cir0, base_info = state_output(model, context)
        logits0 = context_logits(model, context)
        ber0, _ = _reward_metrics(logits0, context.frame)
        loss0 = float(channel_state_reward_loss(model, context))
    decisions, candidates = [], {}
    for objective in objectives:
        for name, lr, steps, radius in actions:
            fast, info = differentiable_adapt(model, [context], learning_rate=lr, steps=steps,
                max_delta_norm=radius, groups={"channel_state_head"}, objective=objective, receiver_input="full_rx")
            fast = {key: value.detach() for key, value in fast.items()}
            with torch.no_grad():
                cir1, after_info = state_output(model, context, fast)
                logits1 = context_logits(model, context, fast)
                ber1, _ = _reward_metrics(logits1, context.frame)
                loss1 = float(channel_state_reward_loss(model, context, fast))
                accepted = should_accept_reward_guard(reward_loss_before=loss0, reward_loss_after=loss1,
                    reward_ber_before=ber0, reward_ber_after=ber1, allow_loss_only=True, peft_delta_norm=info["delta_norm"])
            candidates[name + ":" + objective] = logits1
            decisions.append({"action": name + ":" + objective, "objective": objective, "accepted": accepted, "learning_rate": lr, "steps": steps,
                "radius": radius, "cir_step_norm": float((cir1 - cir0).norm()),
                "before": base_info, "after": after_info, "reward_loss_before": loss0,
                "reward_loss_after": loss1, "reward_ber_before": ber0, "reward_ber_after": ber1, **info})
    return logits0, candidates, decisions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--seed-start", type=int, default=77000)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--objectives", nargs="+", choices=["channel_state_signal", "channel_state_map"], default=["channel_state_signal"])
    parser.add_argument("--ages", nargs="+", type=int, default=[0, 7, 31, 63, 127])
    parser.add_argument("--actions", nargs="+", choices=[action[0] for action in ACTIONS], default=[action[0] for action in ACTIONS])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    target = Path(args.output)
    if (target.exists() or args.num_seeds < 1 or config.get("offline_state_split") != "offline_train"
            or config.get("main_level") != "B" or config.get("pilot_layout") != "prefix"
            or not args.ages or args.ages != sorted(set(args.ages)) or min(args.ages) < 0
            or len(set(args.actions)) != len(args.actions)):
        raise ValueError("只允许新的输出文件、正预算、训练分布、Level B 与前缀。")
    model, payload = _load_branch(args.pretrained, config, args.device)
    if model.channel_state is None:
        raise ValueError("需要已启用的神经信道状态分支。")
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    if {r["seed"] for r in payload["metrics"]["history"]}.intersection(seeds):
        raise ValueError("诊断 seed 与训练重叠。")
    ages = args.ages
    actions = tuple(action for action in ACTIONS if action[0] in args.actions)
    rows = []
    started = time.monotonic()
    for snr in config["main_snrs"]:
        for seed in seeds:
            _, cir, cfo, tail, frames = make_episode(config, model, seed=seed, snr=snr,
                frames=max(ages) + 1, state_split="offline_train", device=args.device)
            if any(bool(frame.adapt_mask[-tail.numel():].any()) for frame in frames):
                raise ValueError("本固定年龄诊断要求前缀不覆盖帧尾，才能保留相同合法 tail 状态。")
            for index in ages:
                frame = frames[index]
                context = make_context(frame, cir, cfo, tail, snr)
                frozen, candidates, decisions = candidates_and_guards(model, context, args.objectives, actions)
                with torch.no_grad():
                    ber0, bce0 = _metrics(frozen, frame, frame.data_mask)
                    for decision in decisions:
                        ber1, bce1 = _metrics(candidates[decision["action"]], frame, frame.data_mask)
                        rows.append({"snr_db": snr, "seed": seed, "frame": index, **decision,
                            "ber_frozen": ber0, "ber_candidate": ber1, "bce_frozen": bce0, "bce_candidate": bce1,
                            "ber_committed": ber1 if decision["accepted"] else ber0, "data_labels_used_for_control": False})
        print(json.dumps({"snr_db": snr, "complete": True}), flush=True)
    summary = []
    for snr in config["main_snrs"]:
        for objective in args.objectives:
            for action, *_ in actions:
                subset = [r for r in rows if r["snr_db"] == snr and r["action"] == action + ":" + objective]
                item = {"snr_db": snr, "action": action, "objective": objective, "accepted": sum(r["accepted"] for r in subset), "frames": len(subset),
                        "cir_step_norm_mean": float(np.mean([r["cir_step_norm"] for r in subset]))}
                for key in ("ber_candidate", "ber_committed", "bce_candidate"):
                    before = "bce_frozen" if key.startswith("bce") else "ber_frozen"
                    item[key + "_gain"] = paired_interval([
                        np.mean([r[before] - r[key] for r in subset if r["seed"] == seed]) for seed in seeds])
                summary.append(item)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"diagnostic_only": True, "main_aggregation_allowed": False,
        "state_split": "offline_train", "data_labels_used_for_control": False,
        "checkpoint": args.pretrained, "checkpoint_sha256": hashlib.sha256(Path(args.pretrained).read_bytes()).hexdigest(),
        "config": config, "seeds": seeds, "ages": ages, "actions": actions, "objectives": args.objectives,
        "elapsed_seconds": time.monotonic() - started, "per_snr": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
