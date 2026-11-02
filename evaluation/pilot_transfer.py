# -*- coding: utf-8 -*-
"""离线训练目标的四臂配对机制实验；不作为正式 PPO 主结果。"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t

from training.pilot_transfer import (
    context_logits, differentiable_adapt, make_context, make_episode, next_pilot_tail, receiver_inputs,
    receiver_tasks, task_context,
)
from training.rl_modulated_online import _build_equalizer
from training.windowed_discrete_ppo import should_accept_reward_guard
from agent.unfolded_equalizer import UnfoldedConfig


def paired_interval(values):
    """独立信道 seed 为统计单位；单 seed 不生成推断区间。"""

    sample = np.asarray(values, dtype=float)
    if not sample.size or not np.isfinite(sample).all():
        raise ValueError("配对统计需要非空有限样本。")
    mean = float(sample.mean())
    if sample.size < 2:
        return {"mean": mean, "ci95": None, "seeds": int(sample.size)}
    margin = float(t.ppf(0.975, len(sample) - 1) * sample.std(ddof=1) / np.sqrt(len(sample)))
    return {"mean": mean, "ci95": [mean - margin, mean + margin], "seeds": int(sample.size)}


def summarize_transfer(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["branch"], row["snr_db"], row["seed"])].append(row)
    seed_values = {}
    for key, frames in grouped.items():
        frames.sort(key=lambda item: item["frame"])
        segment = max(1, len(frames) // 3)
        gains = [item["ber_frozen"] - item["ber_online"] for item in frames]
        seed_values[key] = {
            "frozen": float(np.mean([item["ber_frozen"] for item in frames])),
            "online": float(np.mean([item["ber_online"] for item in frames])),
            "gain": float(np.mean(gains)),
            "late_minus_early_gain": float(np.mean(gains[-segment:]) - np.mean(gains[:segment])),
        }
    result = []
    snrs = sorted({row["snr_db"] for row in rows})
    for snr in snrs:
        branch_result = {}
        for branch in ("baseline", "post_adapt"):
            values = [value for (name, current, _), value in seed_values.items() if name == branch and current == snr]
            subset = [row for row in rows if row["branch"] == branch and row["snr_db"] == snr]
            accepted = [row for row in subset if row["accepted"]]
            branch_result[branch] = {
                "ber_frozen": float(np.mean([value["frozen"] for value in values])),
                "ber_online": float(np.mean([value["online"] for value in values])),
                "gain": paired_interval([value["gain"] for value in values]),
                "late_minus_early_gain": paired_interval([value["late_minus_early_gain"] for value in values]),
                "positive_seeds": sum(value["gain"] > 0 for value in values),
                "accepted": len(accepted), "frames": len(subset),
                "harmful_acceptances": sum(row["ber_candidate"] > row["ber_before_update"] for row in accepted),
            }
        seeds = sorted(seed for branch, current, seed in seed_values if branch == "baseline" and current == snr)
        if any(("post_adapt", snr, seed) not in seed_values for seed in seeds):
            raise ValueError("四臂配对缺少分支或 seed。")
        interaction = [seed_values[("post_adapt", snr, seed)]["gain"] - seed_values[("baseline", snr, seed)]["gain"] for seed in seeds]
        absolute = [seed_values[("baseline", snr, seed)]["online"] - seed_values[("post_adapt", snr, seed)]["online"] for seed in seeds]
        result.append({"snr_db": snr, **branch_result,
                       "adaptation_gain_interaction": paired_interval(interaction),
                       "post_vs_baseline_online_gain": paired_interval(absolute)})
    return result


def _metrics(logits, frame, mask):
    selected = logits[mask]
    labels = frame.bits[mask]
    ber = float(((selected >= 0) != labels).float().mean().cpu())
    loss = float(torch.nn.functional.binary_cross_entropy_with_logits(selected.float(), labels.float()).cpu())
    return ber, loss


def _load_branch(path, config, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model_config = UnfoldedConfig.from_dict(payload["model_config"])
    if model_config.to_dict() != UnfoldedConfig.from_dict(config["model"]).to_dict():
        raise ValueError("配对实验禁止静默覆盖 checkpoint 的模型配置。")
    model = _build_equalizer(model_config, path, device)
    model.attach_online_phase_trend_adapter()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def run_transfer_comparison(config, *, baseline_checkpoint, post_checkpoint, seeds,
                            snrs, frames, device, output_dir, state_split="heldout_edge",
                            persistent=False):
    """先完成当前帧 Pilot 验收，再首次计算 Data 指标。"""

    if frames < 1 or not seeds or len(set(seeds)) != len(seeds) or state_split != "heldout_edge":
        raise ValueError("首阶段机制实验需要正帧数、独立 seed 与 heldout_edge。")
    settings = config["pilot_transfer"]
    inner_input, reward_input = receiver_inputs(settings)
    inner_task, reward_task = receiver_tasks(settings)
    update_uses_data_waveform = inner_input == "full_rx" and inner_task == "full_frame"
    guard_uses_data_waveform = reward_input == "full_rx" and reward_task == "full_frame"
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    output = target / "frame_metrics.jsonl"
    if output.exists() and output.stat().st_size:
        raise ValueError("机制实验输出已存在，请指定新的 output-dir，避免覆盖或混合实验。")
    branches = {"baseline": Path(baseline_checkpoint), "post_adapt": Path(post_checkpoint)}
    manifest = {"schema_version": "pilot-transfer-four-arm-v1", "diagnostic_only": True,
                "linear_operator_backend": os.environ.get("RL4EQ_LINEAR_OPERATOR_BACKEND", "direct"),
                "main_aggregation_allowed": False, "state_split": state_split,
                "seeds": list(seeds), "snrs": list(snrs), "frames": frames,
                "persistent": persistent, "config": config, "checkpoints": {},
                "inner_receiver_input": inner_input, "reward_receiver_input": reward_input,
                "inner_task": inner_task, "reward_task": reward_task}
    rows = []
    fingerprints = {}
    training_signature = None
    with output.open("w", encoding="utf-8") as handle:
        for branch, checkpoint in branches.items():
            model, payload = _load_branch(checkpoint, config, device)
            training_seeds = {item["seed"] for item in payload.get("metrics", {}).get("history", []) if "seed" in item}
            if training_seeds.intersection(seeds):
                raise ValueError("测试 seed 与离线训练 seed 重叠。")
            metrics = payload.get("metrics", {})
            if metrics.get("stage") == "pilot_transfer" and receiver_inputs(metrics.get("settings", {})) != (inner_input, reward_input):
                raise ValueError("四臂实验的训练与在线接收输入协议不匹配。")
            if metrics.get("stage") == "pilot_transfer" and receiver_tasks(metrics.get("settings", {})) != (inner_task, reward_task):
                raise ValueError("四臂实验的训练与在线接收任务不匹配。")
            signature = {
                "source_sha256": (metrics.get("source_checkpoint") or {}).get("sha256"),
                "training_settings": {key: value for key, value in metrics.get("settings", {}).items()
                                      if key != "objective"},
                "samples": [(item.get("seed"), item.get("snr_db"), item.get("channel_state"))
                            for item in metrics.get("history", [])],
            }
            if branch == "baseline":
                training_signature = signature
            elif signature != training_signature:
                raise ValueError("配对训练分支的初始化、训练 seed/SNR 或实际信道状态不一致。")
            manifest["checkpoints"][branch] = {"path": str(checkpoint.resolve()),
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "training_objective": payload.get("metrics", {}).get("objective")}
            for snr in snrs:
                for seed in seeds:
                    env, cir, cfo, tail, episode_frames = make_episode(
                        config, model, seed=seed, snr=snr, frames=frames,
                        state_split=state_split, device=device,
                    )
                    initial = {name: value.detach().clone() for name, value in model.peft.named_group_parameters(
                        settings.get("inner_groups", ["phase_trend"]),
                    )}
                    active = dict(initial)
                    for index, frame in enumerate(episode_frames):
                        fingerprint = hashlib.sha256(frame.rx_symbols.detach().cpu().numpy().tobytes()).hexdigest()
                        key = (snr, seed, index)
                        if branch == "baseline":
                            fingerprints[key] = fingerprint
                        elif fingerprints[key] != fingerprint:
                            raise ValueError("两个训练分支的测试接收波形不一致。")
                        context = make_context(frame, cir, cfo, tail, snr)
                        before = active if persistent else initial
                        before_norm = float(torch.sqrt(sum((before[name] - value).square().sum()
                                                          for name, value in initial.items())).detach().cpu())
                        inner_context, score_start = task_context(context, inner_task, role="adapt")
                        reward_context, _ = task_context(context, reward_task, role="reward")
                        candidate, update = differentiable_adapt(
                            model, [inner_context], learning_rate=float(settings["inner_learning_rate"]),
                            steps=int(settings["inner_steps"]), max_delta_norm=float(settings["max_delta_norm"]),
                            initial_parameters=before,
                            groups=settings.get("inner_groups", ["phase_trend"]),
                            objective=settings.get("inner_objective", "pilot_signal_reconstruction"),
                            receiver_input=inner_input,
                            score_start=score_start,
                        )
                        # 与主配置一致地约束相对离线参数起点的累计偏移。
                        delta = torch.sqrt(sum((candidate[name] - value).square().sum() for name, value in initial.items()) + 1e-24)
                        scale = (float(settings["max_delta_norm"]) / delta).clamp(max=1.0)
                        for name, value in initial.items():
                            candidate[name] = value + (candidate[name] - value) * scale
                        step_norm = float(torch.sqrt(sum((candidate[name] - before[name]).square().sum() for name in initial)).detach().cpu())
                        candidate = {name: value.detach() for name, value in candidate.items()}
                        with torch.no_grad():
                            visible = "full_rx" if reward_input == "full_rx" else "pilot"
                            pilot_before = context_logits(model, reward_context, before, visible=visible)
                            pilot_candidate = context_logits(model, reward_context, candidate, visible=visible)
                            reward_frame = reward_context.frame
                            reward_before, loss_before = _metrics(pilot_before, reward_frame, reward_frame.reward_mask)
                            reward_after, loss_after = _metrics(pilot_candidate, reward_frame, reward_frame.reward_mask)
                            accepted = should_accept_reward_guard(
                                reward_loss_before=loss_before, reward_loss_after=loss_after,
                                reward_ber_before=reward_before, reward_ber_after=reward_after,
                                allow_loss_only=True, peft_delta_norm=step_norm,
                            )
                            active = {name: (candidate[name] if accepted else before[name]).detach() for name in initial}
                            # 所有更新与验收都已结束；Data 仅用于事后仿真评估。
                            frozen_logits = context_logits(model, context, initial)
                            before_logits = context_logits(model, context, before) if persistent else frozen_logits
                            candidate_logits = context_logits(model, context, candidate)
                            online_logits = candidate_logits if accepted else before_logits
                            ber_frozen, bce_frozen = _metrics(frozen_logits, frame, frame.data_mask)
                            ber_before, _ = _metrics(before_logits, frame, frame.data_mask)
                            ber_candidate, bce_candidate = _metrics(candidate_logits, frame, frame.data_mask)
                            ber_online, bce_online = _metrics(online_logits, frame, frame.data_mask)
                        row = {"branch": branch, "snr_db": snr, "seed": seed, "frame": index,
                               "ber_frozen": ber_frozen, "ber_online": ber_online,
                               "ber_before_update": ber_before, "ber_candidate": ber_candidate,
                               "bce_frozen": bce_frozen, "bce_online": bce_online, "bce_candidate": bce_candidate,
                               "accepted": accepted, "reward_loss_improvement": loss_before - loss_after,
                               "reward_ber_improvement": reward_before - reward_after,
                               "reward_loss_before": loss_before, "reward_loss_after": loss_after,
                               "reward_ber_before": reward_before, "reward_ber_after": reward_after,
                               "input_sha256": fingerprint, "episode_final_channel_state": env.state_metadata(), **update,
                               "data_labels_used_online": False,
                               "data_waveform_used_online": update_uses_data_waveform or guard_uses_data_waveform,
                               "data_waveform_used_for_update": update_uses_data_waveform,
                               "data_waveform_used_for_guard": guard_uses_data_waveform,
                               "inner_task": inner_task, "reward_task": reward_task}
                        row.update(episode_delta_norm_before=before_norm,
                                   episode_delta_norm_candidate=float(delta.detach().clamp(max=float(settings["max_delta_norm"])).cpu()),
                                   episode_delta_norm_online=float(torch.sqrt(sum(
                                       (active[name] - value).square().sum() for name, value in initial.items())).cpu()))
                        rows.append(row)
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        handle.flush()
                        tail = next_pilot_tail(model, context, alpha=float(config.get("tail_update_alpha", 0.5)))
                    print(json.dumps({"branch": branch, "snr_db": snr, "seed": seed,
                                      "frames_complete": frames}, ensure_ascii=False), flush=True)
            del model
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
    summary = {"manifest": manifest, "per_snr": summarize_transfer(rows)}
    (target / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["per_snr"], ensure_ascii=False), flush=True)
    return summary
