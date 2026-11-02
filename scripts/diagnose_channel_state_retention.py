# -*- coding: utf-8 -*-
"""训练分布上的首帧固定动作有效期诊断；不改正式在线控制器。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pilot_transfer import _load_branch, _metrics, paired_interval
from training.pilot_transfer import (channel_state_reward_loss, context_logits, differentiable_adapt,
                                    make_context, make_episode)
from training.windowed_discrete_ppo import should_accept_reward_guard


@torch.no_grad()
def pilot_comparison(model, context, retained, delta_norm):
    """比较两个已生成的状态；只有 Reward 标签进入评分，不读取 Data 标签。"""
    logits, scores = {}, {}
    for name, parameters in (("frozen", None), ("retained", retained)):
        logits[name] = context_logits(model, context, parameters)
        ber, bce = _metrics(logits[name], context.frame, context.frame.reward_mask)
        scores[name] = {"ber": ber, "bce": bce,
                        "signal_loss": float(channel_state_reward_loss(model, context, parameters))}

    def accepts(before, after):
        return should_accept_reward_guard(reward_loss_before=scores[before]["signal_loss"],
            reward_loss_after=scores[after]["signal_loss"], reward_ber_before=scores[before]["ber"],
            reward_ber_after=scores[after]["ber"], allow_loss_only=True, peft_delta_norm=delta_norm)

    return logits, {"reward": scores, "revalidate_keep": accepts("frozen", "retained"),
                    "rollback": accepts("retained", "frozen")}


def run(config, checkpoint, seeds, device, output):
    output = Path(output)
    if (output.exists() or config.get("offline_state_split") != "offline_train"
            or config.get("main_level") != "B" or config.get("pilot_layout") != "prefix"
            or not seeds or len(set(seeds)) != len(seeds)):
        raise ValueError("要求训练分布、Level B、前缀、独立 seed 和新的输出文件。")
    model, payload = _load_branch(checkpoint, config, device)
    if model.channel_state is None:
        raise ValueError("需要启用神经状态分支。")
    if {r["seed"] for r in payload["metrics"]["history"]}.intersection(seeds):
        raise ValueError("诊断 seed 与训练重叠。")
    ages = [0, 7, 31, 63, 127]
    settings = {"learning_rate": .02, "steps": 3, "max_delta_norm": .2,
                "groups": {"channel_state_head"}, "objective": "channel_state_signal", "receiver_input": "full_rx"}
    rows = []
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    started = time.monotonic()
    for snr in config["main_snrs"]:
        for seed in seeds:
            _, cir, cfo, tail, frames = make_episode(config, model, seed=seed, snr=snr,
                frames=128, state_split="offline_train", device=device)
            if any(bool(frame.adapt_mask[-tail.numel():].any()) for frame in frames):
                raise ValueError("固定 tail 诊断要求前缀不覆盖帧尾。")
            context0 = make_context(frames[0], cir, cfo, tail, snr)
            candidate, update = differentiable_adapt(model, [context0], **settings)
            candidate = {name: value.detach() for name, value in candidate.items()}
            _, first = pilot_comparison(model, context0, candidate, update["delta_norm"])
            first_accepted = first["revalidate_keep"]
            # 首帧接受后固定参数；拒绝时整个会话保留离线参数，不做后续梯度。
            retained = candidate if first_accepted else None
            committed_norm = update["delta_norm"] if first_accepted else 0.
            for age in ages:
                context = make_context(frames[age], cir, cfo, tail, snr)
                logits, decisions = pilot_comparison(model, context, retained, committed_norm)
                # 两种规则只形成当前帧只读分支，不改变下一帧的固定状态。
                selections = {"retained": "retained",
                    "revalidated": "retained" if decisions["revalidate_keep"] else "frozen",
                    "rollback_guarded": "frozen" if decisions["rollback"] else "retained"}
                with torch.no_grad():
                    data = {name: dict(zip(("ber", "bce"), _metrics(value, context.frame, context.frame.data_mask)))
                            for name, value in logits.items()}
                row = {"snr_db": snr, "seed": seed, "frame": age, "first_accepted": first_accepted,
                    "first_update_norm": update["delta_norm"], "retained_norm": committed_norm,
                    "input_sha256": hashlib.sha256(context.frame.rx_symbols.cpu().numpy().tobytes()).hexdigest(),
                    "data_labels_used_for_control": False, **decisions, "data": data, "selections": selections}
                row["data_gain"] = {method: data["frozen"]["ber"] - data[selected]["ber"]
                                    for method, selected in selections.items()}
                rows.append(row)
        print(json.dumps({"snr_db": snr, "complete": True}), flush=True)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    summary = []
    for snr in config["main_snrs"]:
        subset = [r for r in rows if r["snr_db"] == snr]
        item = {"snr_db": snr, "first_acceptances": sum(r["first_accepted"] for r in subset if r["frame"] == 0),
                "frames": len(subset), "nonzero_retained_frames": sum(r["retained_norm"] > 0 for r in subset),
                "methods": {}, "feedback_sign_counts": {}}
        for method in ["retained", "revalidated", "rollback_guarded"]:
            item["methods"][method] = {"gain": paired_interval([
                np.mean([r["data_gain"][method] for r in subset if r["seed"] == seed]) for seed in seeds]),
                "age127_minus_age0_gain": paired_interval([
                    next(r["data_gain"][method] for r in subset if r["seed"] == seed and r["frame"] == 127) -
                    next(r["data_gain"][method] for r in subset if r["seed"] == seed and r["frame"] == 0) for seed in seeds]),
                "keep_frames": sum(r["selections"][method] == "retained" for r in subset),
                "effective_keep_frames": sum(r["selections"][method] == "retained" and r["retained_norm"] > 0 for r in subset)}
        # 事后符号表只用于解释；不学习阈值或选择在线规则。
        for metric in ["ber", "bce", "signal_loss"]:
            counts = {}
            for row in subset:
                reward_gain = row["reward"]["frozen"][metric] - row["reward"]["retained"][metric]
                data_gain = row["data_gain"]["retained"]
                sign = lambda x: "positive" if x > 1e-8 else "negative" if x < -1e-8 else "tie"
                key = sign(reward_gain) + "_reward__" + sign(data_gain) + "_data"
                counts[key] = counts.get(key, 0) + 1
            item["feedback_sign_counts"][metric] = counts
        summary.append(item)
    result = {"diagnostic_only": True, "main_aggregation_allowed": False, "state_split": "offline_train",
        "data_labels_used_for_control": False, "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "seeds": seeds, "ages": ages, "config": config, "elapsed_seconds": elapsed,
        "action": {**settings, "groups": sorted(settings["groups"])}, "per_snr": summary, "rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--seed-start", type=int, default=79000)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.num_seeds < 1:
        raise ValueError("seed 数必须为正。")
    result = run(json.loads(Path(args.config).read_text(encoding="utf-8")), args.pretrained,
                 list(range(args.seed_start, args.seed_start + args.num_seeds)), args.device, args.output)
    print(json.dumps({"elapsed_seconds": result["elapsed_seconds"], "per_snr": result["per_snr"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
