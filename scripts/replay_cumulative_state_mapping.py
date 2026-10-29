# -*- coding: utf-8 -*-
"""累计历史 Adapt Pilot 更新状态映射，在统一未来帧检验候选和验收效果。"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compare import _build_equalizer, _load_model_config, _reward_signal_reconstruction_loss
from evaluation.pilot_replay_protocol import build_pilot_replay_episode, pilot_state_embedding
from evaluation.research_diagnostics import (
    _frame_to_device, _model_logits, _next_soft_tail_from_logits,
    _pilot_signal_reconstruction_loss_for_diagnostic,
)


def _adapt_only_frame(frame):
    """只保留 Adapt Pilot 接收波形，供条件、状态与梯度路径使用。"""

    mask = frame.adapt_mask.bool()
    return replace(
        frame,
        rx_symbols=torch.where(mask, frame.rx_symbols, torch.zeros_like(frame.rx_symbols)),
    )


def _pilot_visible_frame(frame):
    """只保留 Adapt/Reward Pilot 接收波形，供留出 Pilot 验收使用。"""

    mask = frame.adapt_mask.bool() | frame.reward_mask.bool()
    return replace(
        frame,
        rx_symbols=torch.where(mask, frame.rx_symbols, torch.zeros_like(frame.rx_symbols)),
    )


def _adapt_mapping(model, contexts: list[dict], *, lr: float, steps: int,
                   max_delta_norm: float, proximal_weight: float,
                   objective: str = "signal_reconstruction", joint_bce_weight: float = 0.5,
                   update_scope: str = "state_mapping") -> dict:
    """梯度仅来自历史 Adapt Pilot；不访问 Reward/Data 标签。"""

    if not contexts or steps < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("累计更新需要非空历史窗口、正 steps 和有限正学习率。")
    if objective not in {"signal_reconstruction", "bce", "joint"}:
        raise ValueError("累计目标必须为 signal_reconstruction、bce 或 joint。")
    if update_scope not in {"state_mapping", "direct_phase_cfo", "mapping_plus_phase_cfo", "channel_residual"}:
        raise ValueError("更新对象必须为 state_mapping、direct_phase_cfo、mapping_plus_phase_cfo 或 channel_residual。")
    if not math.isfinite(joint_bce_weight) or not 0 < joint_bce_weight < 1:
        raise ValueError("联合目标 BCE 权重必须在 (0, 1) 内。")
    if not math.isfinite(max_delta_norm) or max_delta_norm <= 0:
        raise ValueError("参数增量范数上限必须为有限正数。")
    adapter = model.online_phase_trend_adapter
    if adapter is None:
        raise ValueError("累计更新需要已挂载的 phase_trend Adapter。")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    mapping_parameters = list(adapter.state_to_phase_cfo.parameters())
    direct_parameters = [adapter.raw_phase, adapter.raw_cfo]
    channel_adapter = model.online_channel_residual_adapter
    channel_parameters = [] if channel_adapter is None else list(channel_adapter.parameters())
    if update_scope == "state_mapping":
        trainable = mapping_parameters
    elif update_scope == "direct_phase_cfo":
        trainable = direct_parameters
    elif update_scope == "mapping_plus_phase_cfo":
        trainable = mapping_parameters + direct_parameters
    else:
        if channel_adapter is None:
            raise ValueError("channel_residual 更新对象需要先挂载 CIR 残差 Adapter。")
        trainable = channel_parameters
    mapping_before = [parameter.detach().clone() for parameter in mapping_parameters]
    channel_before = [parameter.detach().clone() for parameter in channel_parameters]
    for parameter in trainable:
        parameter.requires_grad_(True)
    before = [parameter.detach().clone() for parameter in trainable]
    direct_before = torch.stack((adapter.raw_phase.detach(), adapter.raw_cfo.detach()))
    optimizer = torch.optim.AdamW(trainable, lr=lr)

    def reconstruction():
        return torch.stack([
            _pilot_signal_reconstruction_loss_for_diagnostic(
                model, item.get("adapt_frame", _adapt_only_frame(item["frame"])),
                item["condition"], item["tail"],
                state_embedding=item["embedding"],
            ) for item in contexts
        ]).mean()

    def delta_squared():
        return sum((parameter - initial).float().pow(2).sum()
                   for parameter, initial in zip(trainable, before))

    def mapping_delta_squared():
        return sum((parameter - initial).float().pow(2).sum()
                   for parameter, initial in zip(mapping_parameters, mapping_before))

    def channel_delta_squared():
        if not channel_parameters:
            return mapping_parameters[0].new_zeros(())
        return sum((parameter - initial).float().pow(2).sum()
                   for parameter, initial in zip(channel_parameters, channel_before))

    before_loss = float(reconstruction().detach().cpu())
    # 联合目标以更新前的 Adapt 损失为尺度，常数不读取 Reward/Data，也不随梯度更新。
    with torch.no_grad():
        before_bce = float(torch.stack([_adapt_bce(model, item) for item in contexts]).mean().cpu())
    reconstruction_scale = max(before_loss, 1e-4)
    bce_scale = max(before_bce, 1e-4)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        if objective == "signal_reconstruction":
            # 保留原始纯重构路径，以便复现已提交的对照。
            loss = reconstruction() + proximal_weight * delta_squared()
            loss.backward()
        else:
            # 逐帧累积梯度，避免五帧非因果注意力图同时占用显存。
            for item in contexts:
                loss = _adapt_bce(model, item)
                if objective == "joint":
                    signal = _pilot_signal_reconstruction_loss_for_diagnostic(
                        model, item.get("adapt_frame", _adapt_only_frame(item["frame"])),
                        item["condition"], item["tail"],
                        state_embedding=item["embedding"],
                    )
                    loss = (joint_bce_weight * loss / bce_scale
                            + (1 - joint_bce_weight) * signal / reconstruction_scale)
                (loss / len(contexts)).backward()
            (proximal_weight * delta_squared()).backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        with torch.no_grad():
            norm = torch.sqrt(delta_squared())
            scale = min(1.0, max_delta_norm / max(float(norm), 1e-12))
            for parameter, initial in zip(trainable, before):
                parameter.copy_(initial + scale * (parameter - initial))
    with torch.no_grad():
        result = {
            "adapt_reconstruction_before": before_loss,
            "adapt_reconstruction_after": float(reconstruction().cpu()),
            "state_mapping_delta_norm": float(torch.sqrt(mapping_delta_squared()).cpu()),
            "channel_residual_delta_norm": float(torch.sqrt(channel_delta_squared()).cpu()),
            "direct_phase_cfo_delta_norm": float(torch.linalg.vector_norm(
                torch.stack((adapter.raw_phase, adapter.raw_cfo)) - direct_before
            ).cpu()),
            "updated_parameter_count": sum(parameter.numel() for parameter in trainable),
            "updated_parameter_scope": update_scope,
            "max_parameter_delta_norm": max_delta_norm,
            "adapt_objective": objective,
            "joint_bce_weight": joint_bce_weight if objective == "joint" else None,
            "joint_reconstruction_scale": reconstruction_scale if objective == "joint" else None,
            "joint_bce_scale": bce_scale if objective == "joint" else None,
            "adapt_bce_before": before_bce,
            "adapt_bce_after": float(torch.stack([_adapt_bce(model, item) for item in contexts]).mean().cpu()),
        }
    model.set_trainable_groups(set())
    return result


def _adapt_bce(model, item: dict) -> torch.Tensor:
    """用接收端可见的 Adapt 已知符号监督均衡器输出，保留参数梯度。"""

    frame = item.get("adapt_frame", _adapt_only_frame(item["frame"]))
    view = frame.receiver_view()
    mask = view.adapt_mask.bool()
    if not bool(mask.any()):
        raise ValueError("累计 BCE 更新需要非空 Adapt Pilot。")
    logits = _model_logits(
        model, frame, item["condition"], item["tail"].reshape(1, -1).to(torch.complex64),
        state_embedding=item["embedding"],
    )
    targets = (view.adapt_symbols[mask].real > 0).float()
    return torch.nn.functional.binary_cross_entropy_with_logits(logits[mask], targets)


def _logits(model, item: dict) -> torch.Tensor:
    with torch.no_grad():
        return _model_logits(
            model, item.get("pilot_frame", _pilot_visible_frame(item["frame"])), item["condition"],
            item["tail"].reshape(1, -1).to(torch.complex64),
            state_embedding=item["embedding"],
        ).detach()


def _data_logits(model, item: dict) -> torch.Tensor:
    """仅在 Pilot 决策完成后对完整帧做事后 Data 评估。"""

    with torch.no_grad():
        return _model_logits(
            model, item["frame"], item["condition"],
            item["tail"].reshape(1, -1).to(torch.complex64),
            state_embedding=item["embedding"],
        ).detach()


def _reward_bce(logits: torch.Tensor, frame) -> float:
    """验收只读留出 Reward Pilot 的已知符号。"""

    view = frame.receiver_view()
    mask = frame.reward_mask.bool()
    targets = (view.reward_symbols[mask].real > 0).float()
    if not bool(mask.any()):
        raise ValueError("累计验收需要非空 Reward Pilot。")
    return float(torch.nn.functional.binary_cross_entropy_with_logits(logits[mask], targets).cpu())


def _reward_loss(model, item: dict, score: str) -> float:
    """信号验收以噪声方差归一化，避免高 SNR 下 BCE 饱和掩盖连续变化。"""

    if score == "bce":
        return _reward_bce(_logits(model, item), item["frame"])
    if score in {"bce_relative", "joint_relative"}:
        raise ValueError(f"{score} 必须通过一对冻结/候选模型计算改善量。")
    if score != "signal_reconstruction":
        raise ValueError("未知 Reward 评分。")
    loss = _reward_signal_reconstruction_loss(
        model, item.get("pilot_frame", _pilot_visible_frame(item["frame"])),
        item["condition"], item["tail"], state_embedding=item["embedding"],
    )
    noise = item["condition"].noise_variance.reshape(-1).mean().clamp_min(1e-8)
    return float((loss / noise).cpu())


def _reward_gain(reference_model, candidate_model, item: dict, score: str) -> float:
    """从 Reward Pilot 的 BCE 与信号误差相对变化组成无标签验收分数。"""

    if score not in {"joint_relative", "bce_relative"}:
        return _reward_loss(reference_model, item, score) - _reward_loss(candidate_model, item, score)
    if score == "bce_relative":
        reference_bce = _reward_loss(reference_model, item, "bce")
        candidate_bce = _reward_loss(candidate_model, item, "bce")
        return (reference_bce - candidate_bce) / max(reference_bce, 1e-3)
    reference_signal = _reward_loss(reference_model, item, "signal_reconstruction")
    candidate_signal = _reward_loss(candidate_model, item, "signal_reconstruction")
    reference_bce = _reward_loss(reference_model, item, "bce")
    candidate_bce = _reward_loss(candidate_model, item, "bce")
    signal_gain = (reference_signal - candidate_signal) / max(reference_signal, 1e-3)
    bce_gain = (reference_bce - candidate_bce) / max(reference_bce, 1e-3)
    return 0.5 * (signal_gain + bce_gain)


def _reward_error_count(logits: torch.Tensor, frame) -> int:
    view = frame.receiver_view()
    mask = frame.reward_mask.bool()
    return int(((logits[mask] >= 0) != (view.reward_symbols[mask].real > 0)).sum().cpu())


def _reward_gate(gains: list[float], *, min_improvement: float,
                 cross_frame_tolerance: float, snr: float, freeze_below_snr: float,
                 aggregation: str = "per_frame") -> bool:
    """使用预设改善门限；可逐帧守门，或只检验独立窗口的平均改善。"""

    if aggregation not in {"per_frame", "aggregate"}:
        raise ValueError("Reward 聚合方式必须为 per_frame 或 aggregate。")
    return bool(gains) and snr >= freeze_below_snr and all(math.isfinite(g) for g in gains) and (
        sum(gains) / len(gains) > min_improvement
        and (aggregation == "aggregate" or min(gains) >= -cross_frame_tolerance)
    )


def _reward_error_gate(error_changes: list[int], aggregation: str) -> bool:
    """不允许监测窗的 Reward 硬错误总数增加。"""

    if aggregation == "aggregate":
        return sum(error_changes) <= 0
    if aggregation == "per_frame":
        return all(change <= 0 for change in error_changes)
    raise ValueError("Reward 聚合方式必须为 per_frame 或 aggregate。")


def _data_metrics(before: torch.Tensor, after: torch.Tensor, frame) -> dict:
    """事后连续指标和硬 BER；返回值不参与更新或验收。"""

    mask = frame.data_mask.bool()
    targets = frame.bits[mask].float()
    first, second = before[mask], after[mask]
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    frozen_ber = float(((first >= 0) != (targets >= 0.5)).float().mean().cpu())
    candidate_ber = float(((second >= 0) != (targets >= 0.5)).float().mean().cpu())
    frozen_bce, candidate_bce = float(bce(first, targets).cpu()), float(bce(second, targets).cpu())
    return {
        "frozen_data_ber": frozen_ber, "data_ber": candidate_ber,
        "data_ber_improvement": frozen_ber - candidate_ber,
        "frozen_data_bce": frozen_bce, "data_bce": candidate_bce,
        "data_bce_improvement": frozen_bce - candidate_bce,
        "data_logits_abs_mean_delta": float((second - first).abs().mean().cpu()),
        "data_logits_direction_change_fraction": float(((second >= 0) != (first >= 0)).float().mean().cpu()),
        "data_soft_abs_mean_delta": float((second.sigmoid() - first.sigmoid()).abs().mean().cpu()),
    }


def _run_trial(*, experiment: dict, model_config, checkpoint: Path, device: str,
               snr: float, seed: int, frames_count: int, train_counts: list[int],
               delay: int, pilot_total: int, reward_pilot_total: int,
               gap_seconds: float | None, lr: float, steps: int,
               condition_source: str | None = None, validation_frames: int = 0,
               reward_score: str = "bce", monitor_reward_window: int = 0,
               adapt_objective: str = "signal_reconstruction", joint_bce_weight: float = 0.5,
               update_scope: str = "state_mapping", reward_aggregation: str = "per_frame",
               min_reward_improvement: float | None = None,
               max_delta_norm: float | None = None) -> list[dict]:
    if monitor_reward_window < 0 or monitor_reward_window > validation_frames:
        raise ValueError("持续监测需要足够的独立 Reward 历史，不能使用训练帧补足窗口。")
    reward_threshold = float(
        min_reward_improvement if min_reward_improvement is not None
        else experiment.get("online_adaptation_min_reward_improvement", 0.0005)
    )
    if not math.isfinite(reward_threshold) or reward_threshold < 0:
        raise ValueError("Reward 最低改善门限必须是有限非负数。")
    delta_cap = float(
        max_delta_norm if max_delta_norm is not None
        else experiment.get("online_adaptation_max_delta_norm", 0.05)
    )
    if not math.isfinite(delta_cap) or delta_cap <= 0:
        raise ValueError("参数增量范数上限必须为有限正数。")
    episode = build_pilot_replay_episode(
        experiment, model_config, snr=snr, seed=seed, delay=delay,
        pilot_total=pilot_total, reward_pilot_total=reward_pilot_total,
        gap_seconds=gap_seconds, device=device, condition_source=condition_source,
    )
    model = _build_equalizer(model_config, checkpoint, device)
    model.attach_online_phase_trend_adapter()
    if update_scope == "channel_residual":
        model.attach_online_channel_residual_adapter()
    frozen = copy.deepcopy(model).eval()
    tail_alpha = float(experiment.get("tail_update_alpha", 0.5))
    if not math.isfinite(tail_alpha) or not 0 <= tail_alpha <= 1:
        raise ValueError("tail_update_alpha 必须为 [0, 1] 内的有限数。")
    tail = episode.start.initial_soft_tail.to(device)
    contexts = []
    for _ in range(frames_count):
        frame = _frame_to_device(episode.env.next_frame(), device)
        adapt_frame = _adapt_only_frame(frame)
        pilot_frame = _pilot_visible_frame(frame)
        condition = episode.condition(adapt_frame, tail)
        embedding = pilot_state_embedding(adapt_frame, episode.cir, tail, condition)
        item = {
            "frame": frame,
            "adapt_frame": adapt_frame,
            "pilot_frame": pilot_frame,
            "condition": condition,
            "tail": tail.detach().clone(),
            "embedding": embedding,
        }
        item["frozen_logits"] = _logits(frozen, item)
        contexts.append(item)
        detected = _next_soft_tail_from_logits(item["frozen_logits"], tail.numel())
        visible_tail = (frame.adapt_mask | frame.reward_mask)[-tail.numel():]
        detected = torch.where(visible_tail, detected, tail)
        tail = (1.0 - tail_alpha) * tail + tail_alpha * detected

    # 所有累计长度都使用同一未来窗口；较短长度不会多占测试帧。
    validation_start = max(train_counts)
    test_start = validation_start + validation_frames
    rows = []
    for train_count in train_counts:
        model.load_state_dict(frozen.state_dict(), strict=True)
        adaptation = _adapt_mapping(
            model, contexts[:train_count], lr=lr, steps=steps,
            max_delta_norm=delta_cap,
            proximal_weight=float(experiment.get("online_adaptation_proximal_weight", 0.01)),
            objective=adapt_objective, joint_bce_weight=joint_bce_weight,
            update_scope=update_scope,
        )
        history_gains = [_reward_gain(frozen, model, item, reward_score)
                         for item in contexts[:train_count]]
        validation_contexts = contexts[validation_start:test_start]
        validation_gains = [_reward_gain(frozen, model, item, reward_score)
                            for item in validation_contexts]
        gains = validation_gains if validation_frames else history_gains
        gate_contexts = validation_contexts if validation_frames else contexts[:train_count]
        reward_error_changes = [_reward_error_count(_logits(model, item), item["frame"])
                               - _reward_error_count(item["frozen_logits"], item["frame"])
                               for item in gate_contexts]
        accepted = _reward_gate(
            gains, min_improvement=reward_threshold,
            cross_frame_tolerance=float(experiment.get("online_cross_frame_rollback_tolerance", 1e-5)),
            snr=snr, freeze_below_snr=float(experiment.get("online_adaptation_freeze_below_snr_db", 5)),
            aggregation=reward_aggregation,
        )
        accepted = accepted and _reward_error_gate(reward_error_changes, reward_aggregation)
        active = accepted
        monitor_gains = list(gains)
        monitor_error_changes = list(reward_error_changes)
        # 先确定只依赖 Pilot 的接受决定，再读取未来 Data 标签做诊断。
        for index in range(test_start, frames_count):
            item = contexts[index]
            before = item["frozen_logits"]
            candidate = _logits(model, item)
            # 记录持有期的 Pilot 变化；可选持续监测只据这些 Pilot 指标决定回滚。
            test_reward_bce_improvement = (
                _reward_loss(frozen, item, "bce") - _reward_loss(model, item, "bce")
            )
            test_reward_signal_improvement = (
                _reward_loss(frozen, item, "signal_reconstruction")
                - _reward_loss(model, item, "signal_reconstruction")
            )
            test_reward_improvement = _reward_gain(frozen, model, item, reward_score)
            if reward_score == "bce":
                test_reward_before = _reward_bce(before, item["frame"])
                test_reward_after = _reward_bce(candidate, item["frame"])
            elif reward_score in {"bce_relative", "joint_relative"}:
                test_reward_before = 1.0
                test_reward_after = 1.0 - test_reward_improvement
            else:
                test_reward_before = _reward_loss(frozen, item, reward_score)
                test_reward_after = _reward_loss(model, item, reward_score)
            test_reward_errors_before = _reward_error_count(before, item["frame"])
            test_reward_errors_after = _reward_error_count(candidate, item["frame"])
            monitor_rollback = False
            if monitor_reward_window:
                # 接收机整帧缓冲，因此当前前缀 Pilot 可先验收，再输出本帧 Data。
                monitor_gains.append(test_reward_improvement)
                monitor_error_changes.append(test_reward_errors_after - test_reward_errors_before)
                keep = _reward_gate(
                    monitor_gains[-monitor_reward_window:],
                    min_improvement=reward_threshold,
                    cross_frame_tolerance=float(experiment.get("online_cross_frame_rollback_tolerance", 1e-5)),
                    snr=snr, freeze_below_snr=float(experiment.get("online_adaptation_freeze_below_snr_db", 5)),
                    aggregation=reward_aggregation,
                )
                keep = keep and _reward_error_gate(
                    monitor_error_changes[-monitor_reward_window:], reward_aggregation,
                )
                monitor_rollback = active and not keep
                # 同一候选回滚后不再激活；原始候选保留在诊断副本中以供配对评估。
                active = active and keep
            # 先完整结束当前帧的 Pilot 验收/回滚，再首次读取 Data 波形与标签。
            data_before = _data_logits(frozen, item)
            data_candidate = _data_logits(model, item)
            raw = _data_metrics(data_before, data_candidate, item["frame"])
            committed = _data_metrics(
                data_before,
                data_candidate if active else data_before,
                item["frame"],
            )
            rows.append({
                **episode.metadata(), **adaptation,
                "snr_db": snr, "seed": seed, "train_frame_count": train_count,
                "train_frame_indices": list(range(train_count)),
                "test_frame_index": index, "common_test_start": test_start,
                "tail_mode": "shared_frozen_soft_tail", "tail_update_alpha": tail_alpha,
                "learning_rate": lr, "steps": steps, "optimizer": "AdamW",
                "reward_gate_accepted": active, "initial_reward_gate_accepted": accepted,
                "monitor_reward_window": monitor_reward_window,
                "monitor_reward_rollback": monitor_rollback,
                "monitor_reward_frame_indices": (
                    list(range(max(validation_start, index + 1 - monitor_reward_window), index + 1))
                    if monitor_reward_window else []
                ),
                "train_reward_improvements": history_gains,
                "mean_train_reward_improvement": sum(history_gains) / len(history_gains),
                "reward_gate_mode": "future_reward_only" if validation_frames else "historical_reward",
                "reward_score": reward_score,
                "reward_aggregation": reward_aggregation,
                "minimum_reward_improvement": reward_threshold,
                "reward_hard_ber_guard": True,
                "gate_reward_error_count_changes": reward_error_changes,
                "reward_validation_frame_indices": list(range(validation_start, test_start)),
                "validation_reward_improvements": validation_gains,
                "gate_reward_improvements": gains,
                "test_reward_loss_frozen": test_reward_before,
                "test_reward_loss_candidate": test_reward_after,
                "test_reward_improvement": test_reward_improvement,
                "test_reward_signal_reconstruction_improvement": test_reward_signal_improvement,
                "test_reward_bce_improvement": test_reward_bce_improvement,
                "test_reward_error_count_frozen": test_reward_errors_before,
                "test_reward_error_count_candidate": test_reward_errors_after,
                "test_reward_role": (
                    "ongoing_pilot_acceptance_and_rollback" if monitor_reward_window
                    else "post_acceptance_monitor_diagnostic"
                ),
                **{f"candidate_{key}": value for key, value in raw.items()},
                **committed, "data_labels_used_online": False,
                "data_waveform_used_online": False,
                "diagnostic_uses_data_labels": True,
            })
    return rows


def _summarize(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["train_frame_count"], row["snr_db"])].append(row)
    result = []
    for (count, snr), values in sorted(grouped.items()):
        by_seed = defaultdict(list)
        for row in values:
            by_seed[row["seed"]].append(row)
        seed_gains = [sum(row["data_ber_improvement"] for row in items) / len(items)
                      for items in by_seed.values()]
        candidate_seed_gains = [sum(row["candidate_data_ber_improvement"] for row in items) / len(items)
                                for items in by_seed.values()]
        entry = {"train_frame_count": count, "snr_db": snr, "test_rows": len(values),
                 "seed_count": len(by_seed), "positive_seed_count": sum(g > 0 for g in seed_gains),
                 "negative_seed_count": sum(g < 0 for g in seed_gains),
                 "candidate_positive_seed_count": sum(g > 0 for g in candidate_seed_gains),
                 "candidate_negative_seed_count": sum(g < 0 for g in candidate_seed_gains),
                 "candidate_worst_seed_data_ber_improvement": min(candidate_seed_gains),
                 "worst_seed_data_ber_improvement": min(seed_gains),
                 "initial_accepted_seed_count": sum(
                     bool(items[0].get("initial_reward_gate_accepted", items[0]["reward_gate_accepted"]))
                     for items in by_seed.values()),
                 "monitor_rollback_seed_count": sum(
                     any(row.get("monitor_reward_rollback", False) for row in items)
                     for items in by_seed.values()),
                 "reward_acceptance_rate": sum(row["reward_gate_accepted"] for row in values) / len(values)}
        for key in ("frozen_data_ber", "data_ber", "data_ber_improvement", "data_bce_improvement",
                    "candidate_data_ber_improvement", "candidate_data_bce_improvement",
                    "candidate_data_logits_abs_mean_delta", "candidate_data_soft_abs_mean_delta"):
            entry[f"mean_{key}"] = sum(row[key] for row in values) / len(values)
        result.append(entry)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "pretrained", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--snrs", nargs="+", type=float, default=[0, 5, 10, 15])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--train-frame-counts", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--validation-frames", type=int, default=0,
                        help="固定未来 Reward 验收帧数；这些帧不训练、不纳入 Data 测试。0 表示历史 Reward 验收。")
    parser.add_argument("--reward-score", choices=["bce", "bce_relative", "signal_reconstruction", "joint_relative"], default="bce",
                        help="bce_relative 使用相对 Reward BCE 改善；joint_relative 合并 BCE 与信号重构的相对改善。")
    parser.add_argument("--adapt-objective", choices=["signal_reconstruction", "bce", "joint"],
                        default="signal_reconstruction", help="仅改变历史 Adapt 训练目标，更新参数和 Reward 验收规则保持一致。")
    parser.add_argument("--joint-bce-weight", type=float, default=0.5,
                        help="联合目标使用更新前 Adapt 损失归一化；此权重在看 Data 前固定。")
    parser.add_argument("--update-scope", choices=["state_mapping", "direct_phase_cfo", "mapping_plus_phase_cfo", "channel_residual"],
                        default="state_mapping", help="选择 Adapter 状态映射层、相位/CFO 参数或 CIR 残差 Adapter。")
    parser.add_argument("--monitor-reward-window", type=int, default=0,
                        help="持有期逐帧复查最近 N 帧 Reward，失效后回滚且不重新激活该候选；0 表示一次验收。")
    parser.add_argument("--reward-aggregation", choices=["per_frame", "aggregate"], default="per_frame",
                        help="aggregate 对独立窗口的 Pilot 改善和硬错数求和，减轻单帧 Reward 小样本波动。")
    parser.add_argument("--min-reward-improvement", type=float, default=None,
                        help="验收所需的最小平均 Reward 改善，未指定时沿用固定配置。")
    parser.add_argument("--max-delta-norm", type=float, default=None,
                        help="PEFT 参数总增量范数上限；未指定时沿用配置 0.05。")
    parser.add_argument("--delay", type=int, default=116)
    parser.add_argument("--pilot-total", type=int, default=256)
    parser.add_argument("--reward-pilot-total", type=int, default=32)
    parser.add_argument("--gap-seconds", type=float, default=None, help="不指定时继承配置；非零仅作老化诊断。")
    parser.add_argument("--condition-source", choices=["acquisition", "pilot_phase", "pilot_cir_phase"],
                        default=None, help="继承配置；主诊断显式使用 pilot_cir_phase，acquisition 仅作消融。")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    counts = sorted(set(args.train_frame_counts))
    if (args.frames < 2 or args.validation_frames < 0 or args.monitor_reward_window < 0
            or any(count <= 0 for count in counts)
            or max(counts) + args.validation_frames >= args.frames):
        parser.error("累计帧数必须为正，验收帧数必须非负，且至少保留一帧统一未来测试。")
    if args.monitor_reward_window and args.monitor_reward_window > args.validation_frames:
        parser.error("持有期监测窗口不能大于独立 Reward 验收帧数，禁止用训练帧补足监测历史。")
    experiment = json.loads(Path(args.config).read_text(encoding="utf-8"))
    lr = float(args.lr if args.lr is not None else experiment.get("online_adaptation_learning_rate", 0.005))
    steps = int(args.steps if args.steps is not None else experiment.get("online_adaptation_steps", 5))
    if steps < 1 or not math.isfinite(lr) or lr <= 0 or any(seed < 0 for seed in args.seeds):
        parser.error("steps、学习率必须为正，seed 必须非负。")
    if not math.isfinite(args.joint_bce_weight) or not 0 < args.joint_bce_weight < 1:
        parser.error("joint-bce-weight 必须为 (0, 1) 内的有限数。")
    if (args.min_reward_improvement is not None
            and (not math.isfinite(args.min_reward_improvement) or args.min_reward_improvement < 0)):
        parser.error("min-reward-improvement 必须为有限非负数。")
    if (args.max_delta_norm is not None
            and (not math.isfinite(args.max_delta_norm) or args.max_delta_norm <= 0)):
        parser.error("max-delta-norm 必须为有限正数。")
    for field, default in (("online_adaptation_max_delta_norm", 0.05),
                           ("online_adaptation_proximal_weight", 0.01),
                           ("online_adaptation_min_reward_improvement", 0.0005),
                           ("online_cross_frame_rollback_tolerance", 1e-5)):
        value = float(experiment.get(field, default))
        if not math.isfinite(value) or value < 0:
            parser.error(f"{field} 必须为有限非负数。")
    checkpoint = Path(args.pretrained)
    model_config = _load_model_config(experiment, checkpoint)
    rows = []
    for snr in args.snrs:
        for seed in args.seeds:
            rows.extend(_run_trial(
                experiment=experiment, model_config=model_config, checkpoint=checkpoint, device=args.device,
                snr=snr, seed=seed, frames_count=args.frames, train_counts=counts,
                delay=args.delay, pilot_total=args.pilot_total, reward_pilot_total=args.reward_pilot_total,
                gap_seconds=args.gap_seconds, lr=lr, steps=steps, condition_source=args.condition_source,
                validation_frames=args.validation_frames,
                reward_score=args.reward_score,
                monitor_reward_window=args.monitor_reward_window,
                adapt_objective=args.adapt_objective, joint_bce_weight=args.joint_bce_weight,
                update_scope=args.update_scope,
                reward_aggregation=args.reward_aggregation,
                min_reward_improvement=args.min_reward_improvement,
                max_delta_norm=args.max_delta_norm,
            ))
    payload = {"metric": "cumulative_pilot_state_mapping_future_frame_replay",
               "checkpoint": str(checkpoint.resolve()), "model_config": model_config.to_dict(),
               "experiment_config": experiment, "arguments": vars(args),
               "main_aggregation_allowed": False, "data_labels_used_online": False,
               "diagnostic_uses_data_labels": True, "rows": rows, "summary": _summarize(rows)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"已保存 {output}，共 {len(rows)} 条统一未来帧记录。")


if __name__ == "__main__":
    main()
