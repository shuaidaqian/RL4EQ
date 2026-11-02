# -*- coding: utf-8 -*-
"""将前缀构造成真实短接收窗，检验边界任务能否迁移到整帧 Data。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.pilot_transfer import _load_branch, _metrics
from scripts.diagnose_receiver_input_updates import _reward_metrics, summarize
from training.pilot_transfer import context_logits, differentiable_adapt, make_context, make_episode, next_pilot_tail, prefix_context
from training.windowed_discrete_ppo import should_accept_reward_guard


def candidates_and_guards(model, context, settings):
    frame = context.frame
    adapt_end = int(torch.nonzero(frame.adapt_mask).max()) + 1
    reward_end = int(torch.nonzero(frame.reward_mask).max()) + 1
    adapt = prefix_context(context, adapt_end)
    reward = prefix_context(context, reward_end)
    views = {"full_frame": context, "prefix_boundary": reward}
    with torch.no_grad():
        before = {name: context_logits(model, view, visible="full_rx") for name, view in views.items()}
        metrics = {name: _reward_metrics(before[name], view.frame) for name, view in views.items()}
    candidates, decisions = {}, []
    for task, view, start in (("full_frame", context, 0), ("prefix_boundary", adapt, model.config.max_delay)):
        fast, update = differentiable_adapt(model, [view], learning_rate=settings["inner_learning_rate"],
            steps=settings["inner_steps"], max_delta_norm=settings["max_delta_norm"],
            groups={"symbol_residual_head"}, objective="bce", receiver_input="full_rx", score_start=start)
        fast = {name: value.detach() for name, value in fast.items()}
        candidates[task] = fast
        for guard, guard_view in views.items():
            with torch.no_grad():
                logits = context_logits(model, guard_view, fast, visible="full_rx")
                ber1, loss1 = _reward_metrics(logits, guard_view.frame)
            ber0, loss0 = metrics[guard]
            accepted = should_accept_reward_guard(reward_loss_before=loss0, reward_loss_after=loss1,
                reward_ber_before=ber0, reward_ber_after=ber1, allow_loss_only=True, peft_delta_norm=update["delta_norm"])
            decisions.append({"update_task": task, "guard_task": guard, "group": "symbol_residual_head",
                "accepted": accepted, "reward_ber_before": ber0, "reward_ber_after": ber1,
                "reward_loss_before": loss0, "reward_loss_after": loss1, **update})
    return decisions, candidates, before["full_frame"]


def run_diagnostic(config, checkpoint, seeds, frames, device, output):
    target = Path(output)
    if target.exists():
        raise ValueError("输出已存在，请指定新路径。")
    if (config.get("offline_state_split") != "offline_train" or config.get("main_level") != "B"
            or config.get("pilot_layout") != "prefix" or not seeds or len(set(seeds)) != len(seeds) or frames < 1):
        raise ValueError("短窗机制实验仅支持 offline_train、Level B、prefix 和正预算。")
    model, payload = _load_branch(Path(checkpoint), config, device)
    if not model.config.enable_symbol_residual_branch:
        raise ValueError("本轮要求已训练的神经符号残差分支。")
    if {item["seed"] for item in payload["metrics"]["history"]}.intersection(seeds):
        raise ValueError("实验 seed 与离线训练重叠。")
    rows = []
    for snr in config["main_snrs"]:
        for seed in seeds:
            _, cir, cfo, tail, received = make_episode(config, model, seed=seed, snr=snr, frames=frames,
                                                     state_split="offline_train", device=device)
            for index, frame in enumerate(received):
                context = make_context(frame, cir, cfo, tail, snr)
                decisions, candidates, frozen = candidates_and_guards(model, context, config["pilot_transfer"])
                # 所有 Pilot 决策结束后才访问 Data 标签；不将这些指标回传到控制。
                with torch.no_grad():
                    ber0, bce0 = _metrics(frozen, frame, frame.data_mask)
                    wrong0 = (frozen[frame.data_mask] >= 0) != frame.bits[frame.data_mask]
                    evaluated = {}
                    for task, fast in candidates.items():
                        logits = context_logits(model, context, fast, visible="full_rx")
                        ber1, bce1 = _metrics(logits, frame, frame.data_mask)
                        wrong1 = (logits[frame.data_mask] >= 0) != frame.bits[frame.data_mask]
                        evaluated[task] = {"ber_candidate": ber1, "bce_candidate": bce1,
                            "fixed_errors": int((wrong0 & ~wrong1).sum()),
                            "introduced_errors": int((~wrong0 & wrong1).sum())}
                fingerprint = hashlib.sha256(frame.rx_symbols.detach().cpu().numpy().tobytes()).hexdigest()
                for decision in decisions:
                    result = evaluated[decision["update_task"]]
                    rows.append({"snr_db": snr, "seed": seed, "frame": index, "input_sha256": fingerprint,
                        **decision, **result, "ber_frozen": ber0, "bce_frozen": bce0,
                        "ber_committed": result["ber_candidate"] if decision["accepted"] else ber0,
                        "bce_committed": result["bce_candidate"] if decision["accepted"] else bce0,
                        "data_labels_used_for_control": False})
                tail = next_pilot_tail(model, context, alpha=float(config.get("tail_update_alpha", 0.5)))
            print(json.dumps({"snr_db": snr, "seed": seed, "frames_complete": frames}), flush=True)
    # 仅在调用既有配对统计函数时转换键，导出始终使用真实的任务名称。
    indexed = [{**row, "receiver_input": "adapt_only" if row["update_task"] == "full_frame" else "full_rx",
                "reward_input": "pilot_only" if row["guard_task"] == "full_frame" else "full_rx"} for row in rows]
    summary = summarize(indexed, seeds, config["main_snrs"], ["symbol_residual_head"])
    for item in summary:
        item["update_task"] = "full_frame" if item.pop("receiver_input") == "adapt_only" else "prefix_boundary"
        item["guard_task"] = "full_frame" if item.pop("reward_input") == "pilot_only" else "prefix_boundary"
        item["gain_vs_full_frame_reference"] = item.pop("gain_vs_strict_reference")
    result = {"schema_version": "prefix-task-v1", "diagnostic_only": True, "main_aggregation_allowed": False,
        "config": config, "seeds": list(seeds), "frames": frames, "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "data_labels_used_for_control": False, "weights_modified": False, "per_snr": summary, "rows": rows}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--seed-start", type=int, default=47000)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_diagnostic(json.loads(Path(args.config).read_text(encoding="utf-8")), args.pretrained,
                   list(range(args.seed_start, args.seed_start + args.num_seeds)), args.frames, args.device, args.output)


if __name__ == "__main__":
    main()
