# -*- coding: utf-8 -*-
"""研究假设诊断工具。

本模块只用于离线诊断：Data 标签可以用于分析 reward 是否预测 Data BER、
动作空间是否存在潜在有效动作，但这些信息不得进入在线 observation、
reward、动作选择或 PPO 更新。
"""

from __future__ import annotations

import json
import copy
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from agent.cir_estimator import CIRCondition, condition_from_cir
from agent.modulation import ModulationConfig, ModulationState
from agent.unfolded_equalizer import UnfoldedEqualizer
from baseline.block_equalizers import bit_error_rate
from baseline.traditional_equalizers import TRADITIONAL_BASELINES, TraditionalPhaseState, estimate_acquisition_cir_with_cfo, run_traditional_equalizer
from env.comm_env import CommEnvConfig, CommunicationEnvironment, ReceiverState
from evaluation.metrics import spearman_reward_data
from training.meta_training import _estimate_cir_from_known_frame


def summarize_reward_data_correlation(
    rows: Iterable[dict],
    threshold: float = 0.6,
    surrogate_name: str = "reward_loss_delta",
) -> dict:
    """汇总 Reward surrogate 与 Data BER 改善的 Spearman 相关性。"""

    normalized = list(rows)
    reward = [_surrogate_score(row, surrogate_name) for row in normalized]
    data = [float(row["data_ber_improvement"]) for row in normalized]
    spearman = spearman_reward_data(reward, data, threshold=threshold)
    return {
        "metric": f"{surrogate_name}_vs_data_ber_delta",
        "gate_threshold": float(threshold),
        "gate_pass": bool(spearman.passed),
        "spearman": float(spearman.correlation),
        "num_pairs": int(spearman.n),
        "surrogate_name": str(surrogate_name),
        "rows": normalized,
    }


def summarize_reward_surrogates(rows: Iterable[dict], threshold: float = 0.6) -> dict:
    """比较多个不使用 Data 标签的 reward surrogate 对 Data BER 改善的预测性。"""

    normalized = list(rows)
    data = [float(row["data_ber_improvement"]) for row in normalized]
    definitions = {
        "reward_loss_delta": lambda row: float(row["reward_loss_improvement"]),
        "reward_ber_delta": lambda row: float(row.get("reward_ber_improvement", 0.0)),
        "reward_margin_delta": lambda row: float(row.get("reward_margin_improvement", 0.0)),
        "loss_plus_ber": lambda row: float(row["reward_loss_improvement"]) + 0.5 * float(row.get("reward_ber_improvement", 0.0)),
        "loss_plus_margin": lambda row: float(row["reward_loss_improvement"]) + 0.01 * float(row.get("reward_margin_improvement", 0.0)),
        "loss_minus_0.005_delta": lambda row: float(row["reward_loss_improvement"]) - 0.005 * float(row.get("peft_delta_norm", row.get("action_delta_norm", 0.0))),
        "loss_minus_0.01_delta": lambda row: float(row["reward_loss_improvement"]) - 0.01 * float(row.get("peft_delta_norm", row.get("action_delta_norm", 0.0))),
        "ber_plus_margin": lambda row: float(row.get("reward_ber_improvement", 0.0)) + 0.01 * float(row.get("reward_margin_improvement", 0.0)),
        "ber_plus_0.1_margin": lambda row: float(row.get("reward_ber_improvement", 0.0)) + 0.1 * float(row.get("reward_margin_improvement", 0.0)),
        "ber_plus_0.25_margin": lambda row: float(row.get("reward_ber_improvement", 0.0)) + 0.25 * float(row.get("reward_margin_improvement", 0.0)),
    }
    surrogates = []
    for name, scorer in definitions.items():
        scores = [scorer(row) for row in normalized]
        spearman = spearman_reward_data(scores, data, threshold=threshold)
        surrogates.append(
            {
                "name": name,
                "spearman": float(spearman.correlation),
                "num_pairs": int(spearman.n),
                "gate_pass": bool(spearman.passed),
            }
        )
    best = max(surrogates, key=lambda item: float("-inf") if item["spearman"] != item["spearman"] else item["spearman"])
    action_level = _summarize_action_level_surrogates(normalized, definitions, threshold)
    return {
        "metric": "reward_surrogate_vs_data_ber_delta",
        "gate_threshold": float(threshold),
        "gate_pass": bool(best["gate_pass"]),
        "best": best,
        "surrogates": surrogates,
        "action_level_surrogates": action_level,
    }


def summarize_reward_selected_actions(rows: Iterable[dict], surrogate_name: str = "reward_ber_delta") -> dict:
    """按每帧 Reward surrogate 选择离散动作，Data 只用于诊断评估。"""

    normalized = list(rows)
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in normalized:
        key = (
            row.get("delay"),
            row.get("snr_db"),
            row.get("pilot_total"),
            row.get("pilot_layout"),
            row.get("seed"),
            row.get("frame"),
        )
        grouped[key].append(row)
    selected = []
    for key, values in sorted(grouped.items()):
        identity = next((row for row in values if row.get("action_name") == "identity"), None)
        best = max(
            values,
            key=lambda row: (
                _surrogate_score(row, surrogate_name),
                float(row.get("reward_loss_improvement", 0.0)),
                -float(row.get("action_delta_norm", 0.0)),
            ),
        )
        if _surrogate_score(best, surrogate_name) <= 0.0 and identity is not None:
            best = identity
        selected.append(
            {
                "key": key,
                "action_name": best.get("action_name", "unknown"),
                "surrogate_score": float(_surrogate_score(best, surrogate_name)),
                "data_ber_improvement": float(best["data_ber_improvement"]),
                "reward_loss_improvement": float(best["reward_loss_improvement"]),
                "reward_ber_improvement": float(best.get("reward_ber_improvement", 0.0)),
                "peft_delta_norm": float(best.get("peft_delta_norm", best.get("action_delta_norm", 0.0))),
            }
        )
    action_counts: dict[str, int] = defaultdict(int)
    for row in selected:
        action_counts[str(row["action_name"])] += 1
    improvements = [float(row["data_ber_improvement"]) for row in selected]
    return {
        "metric": "reward_selected_discrete_action",
        "surrogate_name": surrogate_name,
        "selected_frames": len(selected),
        "mean_data_ber_improvement": float(sum(improvements) / max(1, len(improvements))),
        "fraction_data_improved": float(sum(1 for value in improvements if value > 0.0) / max(1, len(improvements))),
        "selection_uses_data_labels": False,
        "diagnostic_uses_data_labels": True,
        "action_counts": dict(sorted(action_counts.items())),
        "selected_rows": selected,
    }


def summarize_windowed_reward_data_correlation(
    rows: Iterable[dict],
    window_size: int = 4,
    threshold: float = 0.6,
    surrogate_name: str = "reward_loss_delta",
) -> dict:
    """按短窗口聚合 Reward surrogate 与 Data BER 改善的相关性。

    该函数仍然只用于离线诊断。窗口聚合后的 Data BER 改善只用于判断
    reward 设计是否值得进入 PPO，不进入在线 observation 或 reward。
    """

    window_rows = _aggregate_action_windows(list(rows), int(window_size))
    reward = [_surrogate_score(row, surrogate_name) for row in window_rows]
    data = [float(row["data_ber_improvement"]) for row in window_rows]
    spearman = spearman_reward_data(reward, data, threshold=threshold)
    return {
        "metric": f"windowed_{surrogate_name}_vs_data_ber_delta",
        "window_size": int(window_size),
        "gate_threshold": float(threshold),
        "gate_pass": bool(spearman.passed),
        "spearman": float(spearman.correlation),
        "num_pairs": int(spearman.n),
        "surrogate_name": str(surrogate_name),
        "diagnostic_uses_data_labels": True,
        "online_policy_uses_data_labels": False,
        "rows": window_rows,
    }


def summarize_windowed_selected_actions(
    rows: Iterable[dict],
    window_size: int = 4,
    surrogate_name: str = "reward_loss_delta",
) -> dict:
    """按每个窗口选择 reward surrogate 最优动作，Data 只用于离线评估。"""

    window_rows = _aggregate_action_windows(list(rows), int(window_size))
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in window_rows:
        key = (
            row.get("delay"),
            row.get("snr_db"),
            row.get("pilot_total"),
            row.get("pilot_layout"),
            row.get("seed"),
            row.get("window_index"),
        )
        grouped[key].append(row)
    selected = []
    for key, values in sorted(grouped.items()):
        identity = next((row for row in values if row.get("action_name") == "identity"), None)
        best = max(
            values,
            key=lambda row: (
                _surrogate_score(row, surrogate_name),
                float(row.get("reward_loss_improvement", 0.0)),
                -float(row.get("peft_delta_norm", row.get("action_delta_norm", 0.0))),
            ),
        )
        if _surrogate_score(best, surrogate_name) <= 0.0 and identity is not None:
            best = identity
        selected.append(
            {
                "key": key,
                "action_name": best.get("action_name", "unknown"),
                "surrogate_score": float(_surrogate_score(best, surrogate_name)),
                "data_ber_improvement": float(best["data_ber_improvement"]),
                "reward_loss_improvement": float(best["reward_loss_improvement"]),
                "reward_ber_improvement": float(best.get("reward_ber_improvement", 0.0)),
                "peft_delta_norm": float(best.get("peft_delta_norm", best.get("action_delta_norm", 0.0))),
            }
        )
    action_counts: dict[str, int] = defaultdict(int)
    for row in selected:
        action_counts[str(row["action_name"])] += 1
    improvements = [float(row["data_ber_improvement"]) for row in selected]
    return {
        "metric": "windowed_reward_selected_discrete_action",
        "window_size": int(window_size),
        "surrogate_name": str(surrogate_name),
        "selected_windows": len(selected),
        "mean_data_ber_improvement": float(sum(improvements) / max(1, len(improvements))),
        "fraction_data_improved": float(sum(1 for value in improvements if value > 0.0) / max(1, len(improvements))),
        "selection_uses_data_labels": False,
        "diagnostic_uses_data_labels": True,
        "action_counts": dict(sorted(action_counts.items())),
        "selected_rows": selected,
    }


def _aggregate_action_windows(rows: list[dict], window_size: int) -> list[dict]:
    if rows and all("window_index" in row and "frame" not in row for row in rows):
        return rows
    size = max(1, int(window_size))
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        frame = int(row.get("frame", 1))
        window_index = (frame - 1) // size
        key = (
            row.get("action_name"),
            row.get("delay"),
            row.get("snr_db"),
            row.get("pilot_total"),
            row.get("pilot_layout"),
            row.get("seed"),
            window_index,
        )
        grouped[key].append(row)
    aggregated = []
    for key, values in sorted(grouped.items()):
        action_name, delay, snr_db, pilot_total, pilot_layout, seed, window_index = key
        aggregated.append(
            {
                "action_name": action_name,
                "delay": delay,
                "snr_db": snr_db,
                "pilot_total": pilot_total,
                "pilot_layout": pilot_layout,
                "seed": seed,
                "window_index": int(window_index),
                "window_size": int(size),
                "frames": [int(row.get("frame", 0)) for row in sorted(values, key=lambda item: int(item.get("frame", 0)))],
                "reward_loss_improvement": _mean([float(row.get("reward_loss_improvement", 0.0)) for row in values]),
                "reward_ber_improvement": _mean([float(row.get("reward_ber_improvement", 0.0)) for row in values]),
                "reward_margin_improvement": _mean([float(row.get("reward_margin_improvement", 0.0)) for row in values]),
                "data_ber_improvement": _mean([float(row.get("data_ber_improvement", 0.0)) for row in values]),
                "peft_delta_norm": _mean([float(row.get("peft_delta_norm", row.get("action_delta_norm", 0.0))) for row in values]),
                "diagnostic_uses_data_labels": True,
                "online_policy_uses_data_labels": False,
            }
        )
    return aggregated


def _surrogate_score(row: dict, surrogate_name: str) -> float:
    if surrogate_name == "reward_loss_delta":
        return float(row["reward_loss_improvement"])
    if surrogate_name in {"reward_ber_delta", "reward_ber_improvement"}:
        return float(row.get("reward_ber_improvement", 0.0))
    if surrogate_name in {"reward_margin_delta", "reward_margin_improvement"}:
        return float(row.get("reward_margin_improvement", 0.0))
    if surrogate_name == "loss_plus_ber":
        return float(row["reward_loss_improvement"]) + 0.5 * float(row.get("reward_ber_improvement", 0.0))
    if surrogate_name == "loss_plus_margin":
        return float(row["reward_loss_improvement"]) + 0.01 * float(row.get("reward_margin_improvement", 0.0))
    if surrogate_name == "loss_minus_0.005_delta":
        return float(row["reward_loss_improvement"]) - 0.005 * float(row.get("peft_delta_norm", row.get("action_delta_norm", 0.0)))
    if surrogate_name == "loss_minus_0.01_delta":
        return float(row["reward_loss_improvement"]) - 0.01 * float(row.get("peft_delta_norm", row.get("action_delta_norm", 0.0)))
    if surrogate_name == "ber_plus_margin":
        return float(row.get("reward_ber_improvement", 0.0)) + 0.01 * float(row.get("reward_margin_improvement", 0.0))
    if surrogate_name == "ber_plus_0.1_margin":
        return float(row.get("reward_ber_improvement", 0.0)) + 0.1 * float(row.get("reward_margin_improvement", 0.0))
    if surrogate_name == "ber_plus_0.25_margin":
        return float(row.get("reward_ber_improvement", 0.0)) + 0.25 * float(row.get("reward_margin_improvement", 0.0))
    raise ValueError(f"未知 reward surrogate：{surrogate_name}")


def _summarize_action_level_surrogates(rows: list[dict], definitions: dict, threshold: float) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("action_name", "all"))].append(row)
    if len(grouped) < 2:
        return []
    action_means = []
    for action_name, values in sorted(grouped.items()):
        item = {
            "action_name": action_name,
            "data_ber_improvement": sum(float(row["data_ber_improvement"]) for row in values) / len(values),
        }
        for name, scorer in definitions.items():
            item[name] = sum(float(scorer(row)) for row in values) / len(values)
        action_means.append(item)
    result = []
    data = [row["data_ber_improvement"] for row in action_means]
    for name in definitions:
        spearman = spearman_reward_data([row[name] for row in action_means], data, threshold=threshold)
        result.append(
            {
                "name": name,
                "spearman": float(spearman.correlation),
                "num_actions": int(spearman.n),
                "gate_pass": bool(spearman.passed),
            }
        )
    return result


def structured_modulation_candidates(num_blocks: int) -> list[tuple[str, ModulationState]]:
    """构造围绕 identity 的低维 modulation 候选动作。"""

    config = ModulationConfig(num_adapter_gates=int(num_blocks), num_lora_scales=int(num_blocks))
    identity = ModulationState.identity(config)
    candidates: list[tuple[str, ModulationState]] = [("identity", identity)]

    def from_values(name: str, *, adapter=1.0, film=0.0, lora=1.0, temperature=1.0, bias=0.0, confidence=0.0):
        vector = torch.cat(
            (
                torch.full((config.num_adapter_gates,), float(adapter)),
                torch.tensor([float(film)]),
                torch.full((config.num_lora_scales,), float(lora)),
                torch.tensor([float(temperature), float(bias), float(confidence)]),
            )
        )
        candidates.append((name, ModulationState.from_vector(vector, config)))

    from_values("adapter_gate_0.8", adapter=0.8)
    from_values("adapter_gate_1.2", adapter=1.2)
    from_values("lora_scale_0.8", lora=0.8)
    from_values("lora_scale_1.2", lora=1.2)
    from_values("film_minus_0.1", film=-0.1)
    from_values("film_plus_0.1", film=0.1)
    from_values("head_temperature_0.8", temperature=0.8)
    from_values("head_temperature_1.2", temperature=1.2)
    from_values("head_bias_minus_0.2", bias=-0.2)
    from_values("head_bias_plus_0.2", bias=0.2)
    return candidates


@torch.no_grad()
def evaluate_modulation_candidates(
    model: UnfoldedEqualizer,
    frame,
    condition: CIRCondition,
    soft_tail: torch.Tensor,
    candidates: Iterable[tuple[str, ModulationState]],
) -> list[dict]:
    """在同一帧上扫描低维 modulation 候选，输出 Reward/Data paired 改善。"""

    model.eval()
    device = next(model.parameters()).device
    frame = _frame_to_device(frame, device)
    condition = _condition_to_device(condition, device)
    tail = soft_tail.to(device).unsqueeze(0).to(torch.complex64)
    rx_iq = torch.stack((frame.rx_symbols.real, frame.rx_symbols.imag), dim=-1).unsqueeze(0).float()
    region_ids = frame.model_region_ids.unsqueeze(0).long()
    base_logits, _ = model(rx_iq, condition, region_ids, tail, modulation=ModulationState.identity(ModulationConfig(len(model.blocks), len(model.blocks)), device=device))
    base = base_logits.squeeze(0)
    base_reward_loss = _masked_bce(base, frame.bits, frame.reward_mask)
    base_reward_ber = bit_error_rate(base[frame.reward_mask], frame.bits[frame.reward_mask])
    base_reward_margin = _masked_margin(base, frame.reward_mask)
    base_data_ber = bit_error_rate(base[frame.data_mask], frame.bits[frame.data_mask])

    rows = []
    identity_vector = ModulationState.identity(ModulationConfig(len(model.blocks), len(model.blocks)), device=device).to_vector()
    for name, modulation in candidates:
        modulation_device = _modulation_to_device(modulation, device)
        logits, _ = model(rx_iq, condition, region_ids, tail, modulation=modulation_device)
        candidate = logits.squeeze(0)
        reward_loss = _masked_bce(candidate, frame.bits, frame.reward_mask)
        reward_ber = bit_error_rate(candidate[frame.reward_mask], frame.bits[frame.reward_mask])
        reward_margin = _masked_margin(candidate, frame.reward_mask)
        data_ber = bit_error_rate(candidate[frame.data_mask], frame.bits[frame.data_mask])
        rows.append(
            {
                "action_name": name,
                "action_delta_norm": float(torch.norm(modulation_device.to_vector() - identity_vector).detach().cpu()),
                "reward_loss_before": float(base_reward_loss.detach().cpu()),
                "reward_loss_after": float(reward_loss.detach().cpu()),
                "reward_loss_improvement": float((base_reward_loss - reward_loss).detach().cpu()),
                "reward_ber_before": float(base_reward_ber),
                "reward_ber_after": float(reward_ber),
                "reward_ber_improvement": float(base_reward_ber - reward_ber),
                "reward_margin_before": float(base_reward_margin.detach().cpu()),
                "reward_margin_after": float(reward_margin.detach().cpu()),
                "reward_margin_improvement": float((reward_margin - base_reward_margin).detach().cpu()),
                "data_ber_before": float(base_data_ber),
                "data_ber_after": float(data_ber),
                "data_ber_improvement": float(base_data_ber - data_ber),
                "diagnostic_uses_data_labels": True,
                "online_policy_uses_data_labels": False,
            }
        )
    return rows


def apply_adapt_only_peft_update(
    model: UnfoldedEqualizer,
    frame,
    condition: CIRCondition,
    soft_tail: torch.Tensor,
    groups: set[str],
    lr: float,
    steps: int,
) -> dict:
    """只用 Adapt Pilot loss 对指定 PEFT 参数做真实梯度更新。"""

    device = next(model.parameters()).device
    frame = _frame_to_device(frame, device)
    condition = _condition_to_device(condition, device)
    tail = soft_tail.to(device).unsqueeze(0).to(torch.complex64)
    model.eval()
    before_logits = _model_logits(model, frame, condition, tail)
    reward_before = _masked_bce(before_logits, frame.bits, frame.reward_mask)
    reward_ber_before = bit_error_rate(before_logits[frame.reward_mask], frame.bits[frame.reward_mask])
    reward_margin_before = _masked_margin(before_logits, frame.reward_mask)
    data_before = bit_error_rate(before_logits[frame.data_mask], frame.bits[frame.data_mask])

    model.train()
    model.set_trainable_groups(groups)
    if not model.trainable_parameters():
        _retag_unfolded_peft_groups(model)
        model.set_trainable_groups(groups)
    trainable = model.trainable_parameters()
    if not trainable:
        raise ValueError(f"PEFT 组 {sorted(groups)} 没有可训练参数。")
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if getattr(parameter, "_peft_group", None) in model.peft.resolve(set(groups))
    }
    optimizer = torch.optim.AdamW(trainable, lr=float(lr))
    for _ in range(int(steps)):
        logits = _model_logits(model, frame, condition, tail)
        loss = _masked_bce(logits, frame.bits, frame.adapt_mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

    model.eval()
    model.set_trainable_groups(set())
    after_logits = _model_logits(model, frame, condition, tail)
    reward_after = _masked_bce(after_logits, frame.bits, frame.reward_mask)
    reward_ber_after = bit_error_rate(after_logits[frame.reward_mask], frame.bits[frame.reward_mask])
    reward_margin_after = _masked_margin(after_logits, frame.reward_mask)
    data_after = bit_error_rate(after_logits[frame.data_mask], frame.bits[frame.data_mask])
    peft_delta_sq = torch.zeros((), device=device)
    named = dict(model.named_parameters())
    for name, value in before.items():
        peft_delta_sq = peft_delta_sq + (named[name].detach() - value).float().pow(2).sum()
    peft_delta_norm = float(torch.sqrt(peft_delta_sq).detach().cpu())
    return {
        "method": "adapt_only_peft",
        "updated_groups": sorted(groups),
        "adapt_steps": int(steps),
        "reward_loss_before": float(reward_before.detach().cpu()),
        "reward_loss_after": float(reward_after.detach().cpu()),
        "reward_loss_improvement": float((reward_before - reward_after).detach().cpu()),
        "reward_ber_before": float(reward_ber_before),
        "reward_ber_after": float(reward_ber_after),
        "reward_ber_improvement": float(reward_ber_before - reward_ber_after),
        "reward_margin_before": float(reward_margin_before.detach().cpu()),
        "reward_margin_after": float(reward_margin_after.detach().cpu()),
        "reward_margin_improvement": float((reward_margin_after - reward_margin_before).detach().cpu()),
        "data_ber_before": float(data_before),
        "data_ber_after": float(data_after),
        "data_ber_improvement": float(data_before - data_after),
        "peft_delta_norm": float(peft_delta_norm),
        "action_delta_norm": float(peft_delta_norm),
        "diagnostic_uses_data_labels": True,
        "online_update_uses_data_labels": False,
    }


def evaluate_peft_window_candidates(
    model: UnfoldedEqualizer,
    frames: list,
    condition: CIRCondition,
    soft_tail: torch.Tensor,
    candidates: Iterable[dict],
    window_index: int = 0,
) -> list[dict]:
    """扫描“同一 PEFT 动作持续作用一个窗口”的真实候选效果。

    每个候选动作从同一个窗口初始模型和 tail 出发；窗口内模型参数和
    soft tail 会持续演化。Data BER 只用于离线诊断动作有效性。
    """

    rows = []
    frames_local = list(frames)
    device = next(model.parameters()).device
    for candidate in candidates:
        action_name = str(candidate["name"])
        groups = set(candidate.get("groups", set()))
        lr = float(candidate.get("lr", 0.0))
        steps = int(candidate.get("steps", 0))
        candidate_model = copy.deepcopy(model)
        candidate_tail = soft_tail.detach().clone().to(device)
        frame_results = []
        peft_delta_norms = []
        for frame in frames_local:
            if groups and steps > 0:
                result = apply_adapt_only_peft_update(
                    model=candidate_model,
                    frame=frame,
                    condition=condition,
                    soft_tail=candidate_tail,
                    groups=groups,
                    lr=lr,
                    steps=steps,
                )
            else:
                result = _identity_peft_result(candidate_model, frame, condition, candidate_tail)
            frame_results.append(result)
            peft_delta_norms.append(float(result.get("peft_delta_norm", 0.0)))
            with torch.no_grad():
                frame_device = _frame_to_device(frame, device)
                condition_device = _condition_to_device(condition, device)
                tail = candidate_tail.to(device).unsqueeze(0).to(torch.complex64)
                logits = _model_logits(candidate_model, frame_device, condition_device, tail)
                candidate_tail = _next_soft_tail_from_logits(logits, candidate_tail.numel()).detach()
        rows.append(
            {
                "method": "windowed_adapt_only_peft",
                "action_name": action_name,
                "updated_groups": sorted(groups) if groups else ["identity"],
                "adapt_steps": int(steps),
                "peft_lr": float(lr),
                "window_index": int(window_index),
                "window_size": len(frames_local),
                "reward_loss_improvement": _mean([float(item["reward_loss_improvement"]) for item in frame_results]),
                "reward_ber_improvement": _mean([float(item.get("reward_ber_improvement", 0.0)) for item in frame_results]),
                "reward_margin_improvement": _mean([float(item.get("reward_margin_improvement", 0.0)) for item in frame_results]),
                "data_ber_improvement": _mean([float(item["data_ber_improvement"]) for item in frame_results]),
                "data_ber_before": _mean([float(item["data_ber_before"]) for item in frame_results]),
                "data_ber_after": _mean([float(item["data_ber_after"]) for item in frame_results]),
                "peft_delta_norm": _mean(peft_delta_norms),
                "action_delta_norm": _mean(peft_delta_norms),
                "diagnostic_uses_data_labels": True,
                "online_update_uses_data_labels": False,
            }
        )
    return rows


def _identity_peft_result(
    model: UnfoldedEqualizer,
    frame,
    condition: CIRCondition,
    soft_tail: torch.Tensor,
) -> dict:
    device = next(model.parameters()).device
    frame = _frame_to_device(frame, device)
    condition = _condition_to_device(condition, device)
    tail = soft_tail.to(device).unsqueeze(0).to(torch.complex64)
    with torch.no_grad():
        logits = _model_logits(model, frame, condition, tail)
        reward_loss = _masked_bce(logits, frame.bits, frame.reward_mask)
        reward_ber = bit_error_rate(logits[frame.reward_mask], frame.bits[frame.reward_mask])
        reward_margin = _masked_margin(logits, frame.reward_mask)
        data_ber = bit_error_rate(logits[frame.data_mask], frame.bits[frame.data_mask])
    return {
        "method": "identity",
        "updated_groups": ["identity"],
        "adapt_steps": 0,
        "reward_loss_before": float(reward_loss.detach().cpu()),
        "reward_loss_after": float(reward_loss.detach().cpu()),
        "reward_loss_improvement": 0.0,
        "reward_ber_before": float(reward_ber),
        "reward_ber_after": float(reward_ber),
        "reward_ber_improvement": 0.0,
        "reward_margin_before": float(reward_margin.detach().cpu()),
        "reward_margin_after": float(reward_margin.detach().cpu()),
        "reward_margin_improvement": 0.0,
        "data_ber_before": float(data_ber),
        "data_ber_after": float(data_ber),
        "data_ber_improvement": 0.0,
        "peft_delta_norm": 0.0,
        "action_delta_norm": 0.0,
        "diagnostic_uses_data_labels": True,
        "online_update_uses_data_labels": False,
    }


def _next_soft_tail_from_logits(logits: torch.Tensor, tail_len: int) -> torch.Tensor:
    next_tail_real = torch.tanh(logits[-int(tail_len):] / 2.0)
    return torch.complex(next_tail_real, torch.zeros_like(next_tail_real)).to(torch.complex64)


def run_level_b_difficulty_scan(
    delays: list[int],
    snrs: list[float],
    pilot_totals: list[int],
    pilot_layouts: list[str],
    seeds: list[int],
    frames: int,
    output_dir: str | Path,
    methods: tuple[str, ...] = TRADITIONAL_BASELINES,
    seed_offset: int = 80_000,
    impairment_profile: str = "clean",
) -> dict:
    """扫描 Level B 下传统 baseline 难度，不包含神经网络或 RL。"""

    rows = []
    for delay in delays:
        for snr_db in snrs:
            for pilot_total in pilot_totals:
                for layout in pilot_layouts:
                    for seed in seeds:
                        env = CommunicationEnvironment(
                            CommEnvConfig(
                                level="B",
                                max_delay=int(delay),
                                snr_db=float(snr_db),
                                total_pilot=int(pilot_total),
                                layout=str(layout),
                                seed=int(seed_offset) + int(seed),
                                impairment_profile=str(impairment_profile),
                            )
                        )
                        start = env.reset_episode()
                        if str(impairment_profile) == "clean":
                            cir = _estimate_cir_from_known_frame(start.acquisition, int(delay))
                            acquisition_cfo_hat = 0.0
                        else:
                            cir, acquisition_cfo_hat = estimate_acquisition_cir_with_cfo(start.acquisition, int(delay))
                        states = {method: ReceiverState(start.initial_soft_tail.clone()) for method in methods}
                        phase_states = {method: TraditionalPhaseState() for method in methods}
                        for frame_index in range(1, int(frames) + 1):
                            frame = env.next_frame()
                            for method in methods:
                                result = run_traditional_equalizer(
                                    method,
                                    frame.receiver_view(),
                                    cir,
                                    states[method].soft_tail,
                                    float(snr_db),
                                    phase_state=phase_states[method],
                                )
                                states[method].update_tail(result.soft_tail)
                                rows.append(
                                    {
                                        "method": method,
                                        "level": "B",
                                        "delay": int(delay),
                                        "snr_db": float(snr_db),
                                        "pilot_total": int(pilot_total),
                                        "pilot_layout": str(layout),
                                        "impairment_profile": str(impairment_profile),
                                        "acquisition_cfo_hat": float(acquisition_cfo_hat),
                                        "seed": int(seed),
                                        "frame": int(frame_index),
                                        "ber_data": bit_error_rate(result.logits[frame.data_mask], frame.bits[frame.data_mask]),
                                        "ber_reward_pilot": bit_error_rate(result.logits[frame.reward_mask], frame.bits[frame.reward_mask]),
                                        "uses_neural_network": False,
                                        "uses_rl": False,
                                    }
                                )
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["method"],
                row["delay"],
                row["snr_db"],
                row["pilot_total"],
                row["pilot_layout"],
                row["impairment_profile"],
            )
        ].append(float(row["ber_data"]))
    summary = [
        {
            "method": key[0],
            "delay": key[1],
            "snr_db": key[2],
            "pilot_total": key[3],
            "pilot_layout": key[4],
            "impairment_profile": key[5],
            "mean_ber_data": float(sum(values) / max(1, len(values))),
            "frames": len(values),
        }
        for key, values in sorted(grouped.items())
    ]
    payload = {"metric": "level_b_traditional_difficulty", "rows": rows, "summary": summary}
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "level_b_difficulty.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _model_logits(model: UnfoldedEqualizer, frame, condition: CIRCondition, tail: torch.Tensor) -> torch.Tensor:
    rx_iq = torch.stack((frame.rx_symbols.real, frame.rx_symbols.imag), dim=-1).unsqueeze(0).float()
    region_ids = frame.model_region_ids.unsqueeze(0).long()
    logits, _ = model(rx_iq, condition, region_ids, tail)
    return logits.squeeze(0)


def _masked_bce(logits: torch.Tensor, bits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if int(mask.sum().item()) == 0:
        return torch.zeros((), device=logits.device)
    return F.binary_cross_entropy_with_logits(logits[mask].float(), bits[mask].float())


def _masked_margin(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if int(mask.sum().item()) == 0:
        return torch.zeros((), device=logits.device)
    return torch.abs(logits[mask].float()).mean()


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _condition_to_device(condition: CIRCondition, device: torch.device | str) -> CIRCondition:
    return CIRCondition(
        complex_cir=condition.complex_cir.to(device),
        support_probability=condition.support_probability.to(device),
        noise_variance=condition.noise_variance.to(device),
        confidence=condition.confidence.to(device),
        latent_residual=condition.latent_residual.to(device),
    )


def _frame_to_device(frame, device: torch.device | str):
    from dataclasses import replace

    return replace(
        frame,
        bits=frame.bits.to(device),
        tx_symbols=frame.tx_symbols.to(device),
        rx_symbols=frame.rx_symbols.to(device),
        adapt_mask=frame.adapt_mask.to(device),
        reward_mask=frame.reward_mask.to(device),
        data_mask=frame.data_mask.to(device),
        model_region_ids=frame.model_region_ids.to(device),
        tail_symbols=frame.tail_symbols.to(device) if frame.tail_symbols is not None else None,
        true_cir=frame.true_cir.to(device) if frame.true_cir is not None else None,
    )


def _modulation_to_device(modulation: ModulationState, device: torch.device | str) -> ModulationState:
    return ModulationState(
        adapter_gates=modulation.adapter_gates.to(device),
        film_residual_scale=modulation.film_residual_scale.to(device),
        lora_scales=modulation.lora_scales.to(device),
        head_temperature=modulation.head_temperature.to(device),
        head_bias=modulation.head_bias.to(device),
        confidence_threshold=modulation.confidence_threshold.to(device),
    )


def _retag_unfolded_peft_groups(model: UnfoldedEqualizer) -> None:
    """为 deepcopy 后丢失自定义属性的参数恢复 PEFT 分组标记。"""

    for name, parameter in model.named_parameters():
        group = None
        if name.startswith("conditioner."):
            group = "conditioner_film"
        elif name.startswith("head."):
            group = "head"
        elif ".attn_lora." in name:
            group = "attention_lora"
        elif ".ffn_lora." in name:
            group = "ffn_lora"
        elif ".adapter." in name:
            group = "adapter"
        if group is not None:
            setattr(parameter, "_peft_group", group)
