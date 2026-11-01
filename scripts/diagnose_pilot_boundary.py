# -*- coding: utf-8 -*-
"""诊断长回波前缀边界对 Adapt 信号重构更新的影响，不改变正式动作。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.pilot_transfer import _load_branch, _metrics, paired_interval
from training.pilot_transfer import context_logits, differentiable_adapt, make_context, make_episode
from training.windowed_discrete_ppo import should_accept_reward_guard


def run_diagnostic(config, checkpoint, seeds, frames, device, output):
    """两个预先规定的固定动作都只用 Adapt 更新，Data 仅用于事后统计。"""
    model, _ = _load_branch(Path(checkpoint), config, device)
    settings = config["pilot_transfer"]
    rows = []
    for snr in config["main_snrs"]:
        for seed in seeds:
            env, cir, cfo, tail, received = make_episode(
                config, model, seed=seed, snr=snr, frames=frames,
                state_split="offline_train", device=device,
            )
            for index, frame in enumerate(received):
                context = make_context(frame, cir, cfo, tail, snr)
                outcomes = []
                with torch.no_grad():
                    pilot_before = context_logits(model, context, visible="pilot")
                    reward_ber_before, reward_loss_before = _metrics(pilot_before, frame, frame.reward_mask)
                for score_start in (0, model.config.max_delay):
                    parameters, update = differentiable_adapt(
                        model, [context], learning_rate=settings["inner_learning_rate"],
                        steps=settings["inner_steps"], max_delta_norm=settings["max_delta_norm"],
                        score_start=score_start,
                    )
                    parameters = {name: value.detach() for name, value in parameters.items()}
                    with torch.no_grad():
                        pilot_after = context_logits(model, context, parameters, visible="pilot")
                        reward_ber_after, reward_loss_after = _metrics(pilot_after, frame, frame.reward_mask)
                    accepted = should_accept_reward_guard(
                        reward_loss_before=reward_loss_before, reward_loss_after=reward_loss_after,
                        reward_ber_before=reward_ber_before, reward_ber_after=reward_ber_after,
                        allow_loss_only=True, peft_delta_norm=update["delta_norm"],
                    )
                    outcomes.append((score_start, parameters, accepted, update))
                # 两个固定动作的 Pilot 决策已经全部结束，再读取 Data 标签。
                with torch.no_grad():
                    frozen = context_logits(model, context)
                    frozen_ber, frozen_bce = _metrics(frozen, frame, frame.data_mask)
                    for score_start, parameters, accepted, update in outcomes:
                        after = context_logits(model, context, parameters)
                        ber, bce = _metrics(after, frame, frame.data_mask)
                        positions = torch.arange(frame.adapt_mask.numel(), device=frame.adapt_mask.device)
                        rows.append({"snr_db": snr, "seed": seed, "frame": index,
                            "score_start": score_start,
                            "scored_pilot_count": int((frame.adapt_mask & (positions >= score_start)).sum()),
                            "ber_frozen": frozen_ber, "ber_candidate": ber,
                            "ber_committed": ber if accepted else frozen_ber,
                            "bce_frozen": frozen_bce, "bce_candidate": bce, "accepted": accepted,
                            "tail_matches_previous_transmission": index == 0,
                            "data_labels_used_online": False, "data_waveform_used_online": False, **update})
            print(json.dumps({"snr_db": snr, "seed": seed, "frames_complete": frames}), flush=True)
    summary = []
    for snr in config["main_snrs"]:
        for score_start in (0, model.config.max_delay):
            selected = [row for row in rows if row["snr_db"] == snr and row["score_start"] == score_start]
            gains = [sum(row["ber_frozen"] - row["ber_committed"] for row in selected if row["seed"] == seed) / frames for seed in seeds]
            summary.append({"snr_db": snr, "score_start": score_start, "gain": paired_interval(gains),
                            "accepted": sum(row["accepted"] for row in selected),
                            "harmful_acceptances": sum(row["accepted"] and row["ber_candidate"] > row["ber_frozen"] for row in selected)})
    payload = {"diagnostic_only": True, "main_aggregation_allowed": False,
               "linear_operator_backend": os.environ.get("RL4EQ_LINEAR_OPERATOR_BACKEND", "direct"),
               "state_split": "offline_train", "checkpoint": str(checkpoint),
               "seed_start": min(seeds), "num_seeds": len(seeds), "frames": frames,
               "per_snr": summary, "rows": rows,
               "protocol": "同一 acquisition CIR、同一 Adapt-only 条件；仅改变重构损失评分窗，已知 Pilot 卷积输入保持完整"}
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--seed-start", type=int, default=32000)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.num_seeds < 1 or args.frames < 1:
        parser.error("num-seeds 与 frames 必须为正数。")
    run_diagnostic(json.loads(Path(args.config).read_text(encoding="utf-8")), args.pretrained,
                   list(range(args.seed_start, args.seed_start + args.num_seeds)),
                   args.frames, args.device, args.output)


if __name__ == "__main__":
    main()
