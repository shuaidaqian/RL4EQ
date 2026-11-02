# -*- coding: utf-8 -*-
"""用训练分布分离 Adapt 聚合与 Reward 聚合；不是正式 RL 或连续神经更新。"""

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
from training.pilot_transfer import channel_state_reward_loss, context_logits, differentiable_adapt, make_context, make_episode
from training.windowed_discrete_ppo import should_accept_reward_guard


def window_candidates(model, contexts, windows=(1, 8)):
    """所有候选与验收完成后才返回给 Data 评分；不修改模型参数。"""
    if (not windows or min(windows) < 1 or len(set(windows)) != len(windows)
            or len(contexts) < max(windows)):
        raise ValueError("窗口必须为正且唯一，历史必须足够。")
    indices = [c.frame.frame_index for c in contexts]
    if indices != list(range(indices[0], indices[0] + len(contexts))):
        raise ValueError("窗口必须由连续已接收帧组成。")

    def score(parameters):
        logits, metrics = [], []
        with torch.no_grad():
            for context in contexts:
                value = context_logits(model, context, parameters)
                ber, _ = _metrics(value, context.frame, context.frame.reward_mask)
                metrics.append((ber, float(channel_state_reward_loss(model, context, parameters))))
                logits.append(value)
        return logits, metrics

    base_logits, before = score(None)
    candidates, decisions = {}, []
    for adapt_window in windows:
        fast, update = differentiable_adapt(model, contexts[-adapt_window:], learning_rate=.02, steps=3,
            max_delta_norm=.2, groups={"channel_state_head"}, objective="channel_state_signal", receiver_input="full_rx")
        fast = {name: value.detach() for name, value in fast.items()}
        logits, after = score(fast)
        candidates[adapt_window] = logits[-1]
        for reward_window in windows:
            ber0, loss0 = np.mean(before[-reward_window:], axis=0)
            ber1, loss1 = np.mean(after[-reward_window:], axis=0)
            accepted = should_accept_reward_guard(reward_loss_before=loss0, reward_loss_after=loss1,
                reward_ber_before=ber0, reward_ber_after=ber1, allow_loss_only=True, peft_delta_norm=update["delta_norm"])
            decisions.append({"adapt_window": adapt_window, "reward_window": reward_window, "accepted": accepted,
                "reward_ber_before": float(ber0), "reward_ber_after": float(ber1),
                "reward_loss_before": float(loss0), "reward_loss_after": float(loss1), **update})
    return base_logits[-1], candidates, decisions


def summarize(rows, seeds):
    result, contrasts = [], []
    for snr, age in sorted({(r["snr_db"], r["frame"]) for r in rows}):
        current = [r for r in rows if (r["snr_db"], r["frame"]) == (snr, age)]
        for aw in [1, 8]:
            for rw in [1, 8]:
                subset = [r for r in current if (r["adapt_window"], r["reward_window"]) == (aw, rw)]
                if sorted(r["seed"] for r in subset) != sorted(seeds):
                    raise ValueError("每个窗口组合必须包含全部独立 seed，且不能重复。")
                result.append({"snr_db": snr, "frame": age, "adapt_window": aw, "reward_window": rw,
                    "accepted": sum(r["accepted"] for r in subset), "contexts": len(subset),
                    "ber_frozen": float(np.mean([r["ber_frozen"] for r in subset])),
                    "candidate_gain": paired_interval([r["ber_frozen"] - r["ber_candidate"] for r in subset]),
                    "committed_gain": paired_interval([r["ber_frozen"] - r["ber_committed"] for r in subset])})
        lookup = {(r["seed"], r["adapt_window"], r["reward_window"]): r for r in current}
        for factor, left, right in [("reward_only_at_adapt1", (1, 1), (1, 8)),
                                    ("reward_only_at_adapt8", (8, 1), (8, 8)),
                                    ("adapt_only_at_reward1", (1, 1), (8, 1)),
                                    ("adapt_only_at_reward8", (1, 8), (8, 8))]:
            contrasts.append({"snr_db": snr, "frame": age, "contrast": factor,
                "gain": paired_interval([lookup[(seed, *left)]["ber_committed"] -
                                          lookup[(seed, *right)]["ber_committed"] for seed in seeds])})
    return result, contrasts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--seed-start", type=int, default=92000)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if (args.output.exists() or args.num_seeds < 1 or config.get("offline_state_split") != "offline_train"
            or config.get("main_level") != "B" or config.get("pilot_layout") != "prefix"):
        raise ValueError("要求新的输出、正 seed 预算、训练分布、Level B 和前缀 Pilot。")
    model, payload = _load_branch(args.pretrained, config, args.device)
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    if {r["seed"] for r in payload["metrics"]["history"]}.intersection(seeds):
        raise ValueError("诊断 seed 与训练重叠。")
    ages, rows = [127, 1023, 2047, 4095], []
    started = time.monotonic()
    for snr in config["main_snrs"]:
        for seed in seeds:
            _, cir, cfo, tail, frames = make_episode(config, model, seed=seed, snr=snr,
                frames=4096, state_split="offline_train", device=args.device)
            if any(bool(frame.adapt_mask[-tail.numel():].any()) for frame in frames):
                raise ValueError("窗口诊断要求 prefix 不覆盖帧尾，保持合法 tail 不变。")
            for age in ages:
                contexts = [make_context(frame, cir, cfo, tail, snr) for frame in frames[age - 7:age + 1]]
                frozen, candidates, decisions = window_candidates(model, contexts)
                frame = frames[age]
                with torch.no_grad():
                    ber0, bce0 = _metrics(frozen, frame, frame.data_mask)
                    for decision in decisions:
                        ber1, bce1 = _metrics(candidates[decision["adapt_window"]], frame, frame.data_mask)
                        rows.append({"snr_db": snr, "seed": seed, "frame": age, **decision,
                            "ber_frozen": ber0, "bce_frozen": bce0, "ber_candidate": ber1, "bce_candidate": bce1,
                            "ber_committed": ber1 if decision["accepted"] else ber0,
                            "input_sha256": hashlib.sha256(frame.rx_symbols.cpu().numpy().tobytes()).hexdigest(),
                            "data_labels_used_for_control": False})
            print(json.dumps({"snr_db": snr, "seed": seed, "complete": True}), flush=True)
    per_age, contrasts = summarize(rows, seeds)
    result = {"diagnostic_only": True, "main_aggregation_allowed": False, "state_split": "offline_train",
        "checkpoint": args.pretrained, "checkpoint_sha256": hashlib.sha256(Path(args.pretrained).read_bytes()).hexdigest(),
        "config": config, "seeds": seeds, "ages": ages, "windows": [1, 8], "data_labels_used_for_control": False,
        "elapsed_seconds": time.monotonic() - started, "per_snr_age": per_age, "contrasts": contrasts, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
