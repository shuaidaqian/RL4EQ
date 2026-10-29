# -*- coding: utf-8 -*-
"""用互不重叠的 Pilot 子窗口生成、排序并验收 PEFT 候选。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.pilot_replay_protocol import build_pilot_replay_episode
from evaluation.research_diagnostics import (
    _frame_to_device,
    _model_logits,
    _pilot_signal_reconstruction_loss_for_diagnostic,
    _retag_unfolded_peft_groups,
    apply_adapt_only_peft_update,
)
from compare import _build_equalizer, _load_model_config


_CANDIDATES = (
    {"name": "identity", "groups": (), "objective": "bce"},
    {"name": "phase_trend_bce", "groups": ("phase_trend",), "objective": "bce"},
    {
        "name": "phase_trend_signal_reconstruction",
        "groups": ("phase_trend",),
        "objective": "pilot_signal_reconstruction",
    },
    {"name": "head_bce", "groups": ("head",), "objective": "bce"},
)


def candidate_specs(bank: str, *, lr: float, steps: int,
                    max_delta_norm: float = 0.05) -> tuple[dict, ...]:
    """在观察本轮 Data 指标前固定候选集合与更新幅度。"""

    if bank == "objective":
        return tuple({**item, "lr": float(lr), "steps": int(steps), "max_delta_norm": float(max_delta_norm)}
                     for item in _CANDIDATES)
    if bank == "multiscale_signal":
        scales = (
            ("phase_signal_conservative", 0.125, 0.25),
            ("phase_signal_standard", 0.25, 1.0),
            ("phase_signal_moderate", 0.5, 1.0),
        )
        return (
            {"name": "identity", "groups": (), "objective": "bce", "lr": 0.0,
             "steps": 0, "max_delta_norm": 0.0},
            *(
                {
                    "name": name,
                    "groups": ("phase_trend",),
                    "objective": "pilot_signal_reconstruction",
                    "lr": float(lr) * lr_scale,
                    "steps": int(steps),
                    "max_delta_norm": float(max_delta_norm) * delta_scale,
                }
                for name, lr_scale, delta_scale in scales
            ),
        )
    if bank == "fixed_moderate_signal":
        return (
            {"name": "identity", "groups": (), "objective": "bce", "lr": 0.0,
             "steps": 0, "max_delta_norm": 0.0},
            {
                "name": "phase_signal_fixed_moderate",
                "groups": ("phase_trend",),
                "objective": "pilot_signal_reconstruction",
                "lr": float(lr) * 0.5,
                "steps": int(steps),
                "max_delta_norm": float(max_delta_norm),
            },
        )
    raise ValueError(f"未知 candidate bank：{bank}")


def _masked_pilot_view(frame, visible_mask: torch.Tensor, *, adapt_mask: torch.Tensor,
                       reward_mask: torch.Tensor):
    """创建只暴露指定 Pilot 接收波形的均衡器视图。"""

    visible = visible_mask.to(device=frame.rx_symbols.device, dtype=torch.bool)
    return replace(
        frame,
        rx_symbols=torch.where(visible, frame.rx_symbols, torch.zeros_like(frame.rx_symbols)),
        adapt_mask=adapt_mask.to(device=frame.adapt_mask.device, dtype=torch.bool),
        reward_mask=reward_mask.to(device=frame.reward_mask.device, dtype=torch.bool),
        data_mask=torch.zeros_like(frame.data_mask, dtype=torch.bool),
    )


def split_adapt_subwindows(frame, holdout_fraction: float = 0.25):
    """按时间顺序将 Adapt Pilot 拆成更新前缀和独立排序后缀。"""

    fraction = float(holdout_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("holdout_fraction 必须是 (0, 1) 内的有限数。")
    indices = torch.nonzero(frame.adapt_mask.bool(), as_tuple=False).reshape(-1)
    if indices.numel() < 2:
        raise ValueError("至少需要两个 Adapt Pilot 符号才能分窗。")
    holdout_count = max(1, min(int(indices.numel()) - 1, round(indices.numel() * fraction)))
    update_indices = indices[:-holdout_count]
    rank_indices = indices[-holdout_count:]
    update_mask = torch.zeros_like(frame.adapt_mask, dtype=torch.bool)
    rank_mask = torch.zeros_like(frame.adapt_mask, dtype=torch.bool)
    update_mask[update_indices] = True
    rank_mask[rank_indices] = True
    no_mask = torch.zeros_like(frame.adapt_mask, dtype=torch.bool)
    update_view = _masked_pilot_view(
        frame, update_mask, adapt_mask=update_mask, reward_mask=no_mask,
    )
    rank_view = _masked_pilot_view(
        frame, rank_mask, adapt_mask=rank_mask, reward_mask=no_mask,
    )
    reward_visible = frame.adapt_mask.bool() | frame.reward_mask.bool()
    reward_view = _masked_pilot_view(
        frame, reward_visible, adapt_mask=frame.adapt_mask, reward_mask=frame.reward_mask,
    )
    if bool(torch.any(update_mask & rank_mask)):
        raise RuntimeError("更新窗和排序窗发生重叠。")
    return update_view, rank_view, reward_view, update_mask, rank_mask


def _logits(model, frame, condition, tail: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return _model_logits(
            model,
            frame,
            condition,
            tail.reshape(1, -1).to(torch.complex64),
        ).detach()


def _masked_bce(logits: torch.Tensor, frame, mask: torch.Tensor) -> float:
    selected = mask.bool()
    if not bool(selected.any()):
        raise ValueError("Pilot 评分窗口不能为空。")
    loss = F.binary_cross_entropy_with_logits(logits[selected].float(), frame.bits[selected].float())
    return float(loss.detach().cpu())


def _error_count(logits: torch.Tensor, frame, mask: torch.Tensor) -> int:
    selected = mask.bool()
    return int(((logits[selected] >= 0) != (frame.bits[selected] >= 0.5)).sum().detach().cpu())


def _reward_gate(*, snr: float, freeze_below_snr_db: float,
                 reward_bce_improvement: float, min_reward_improvement: float,
                 reward_errors_before: int, reward_errors_after: int,
                 reward_signal_improvement: float,
                 require_signal_improvement: bool) -> bool:
    """只用 Reward Pilot 的预设 BCE、硬错数和可选重构门限验收。"""

    signal_ok = not require_signal_improvement or (
        math.isfinite(reward_signal_improvement) and reward_signal_improvement > 0.0
    )
    return bool(
        snr >= freeze_below_snr_db
        and math.isfinite(reward_bce_improvement)
        and reward_bce_improvement > min_reward_improvement
        and reward_errors_after <= reward_errors_before
        and signal_ok
    )


def _posthoc_data_metrics(before: torch.Tensor, after: torch.Tensor, frame) -> dict:
    """仅在 Pilot 选择和 Reward 验收结束后计算事后 Data 指标。"""

    mask = frame.data_mask.bool()
    if not bool(mask.any()):
        raise ValueError("事后 Data 评估窗口不能为空。")
    targets = frame.bits[mask].float()
    first, second = before[mask], after[mask]
    before_ber = float(((first >= 0) != (targets >= 0.5)).float().mean().cpu())
    after_ber = float(((second >= 0) != (targets >= 0.5)).float().mean().cpu())
    before_bce = float(F.binary_cross_entropy_with_logits(first.float(), targets).cpu())
    after_bce = float(F.binary_cross_entropy_with_logits(second.float(), targets).cpu())
    return {
        "frozen_data_ber": before_ber,
        "candidate_data_ber": after_ber,
        "data_ber_improvement": before_ber - after_ber,
        "frozen_data_bce": before_bce,
        "candidate_data_bce": after_bce,
        "data_bce_improvement": before_bce - after_bce,
        "data_logits_abs_mean_delta": float((second - first).abs().mean().cpu()),
        "data_logits_direction_change_fraction": float(((second >= 0) != (first >= 0)).float().mean().cpu()),
        "data_soft_abs_mean_delta": float((second.sigmoid() - first.sigmoid()).abs().mean().cpu()),
    }


def _rank_candidates(rows: list[dict], rank_score: str = "bce") -> dict:
    """按独立 Adapt Pilot 分数排序；并列时优先 identity 和较小更新。"""

    score_field = {
        "bce": "rank_pilot_bce_improvement",
        "signal_reconstruction": "rank_pilot_signal_improvement",
    }.get(rank_score)
    if score_field is None:
        raise ValueError("rank_score 必须是 bce 或 signal_reconstruction。")
    if not rows:
        raise ValueError("候选列表不能为空。")
    return max(
        rows,
        key=lambda row: (
            float(row[score_field]),
            row["candidate_name"] == "identity",
            -float(row["peft_delta_norm"]),
            row["candidate_name"],
        ),
    )


def _make_candidate_models(frozen, update_frame, rank_frame, condition, tail,
                           *, candidate_bank: tuple[dict, ...], lr: float,
                           steps: int, rank_score: str = "bce") -> tuple[list[dict], dict]:
    """只从更新窗学习参数，并只在独立排序窗计算候选分数。"""

    update_mask = update_frame.adapt_mask.bool()
    rank_mask = rank_frame.adapt_mask.bool()
    frozen_rank_logits = _logits(frozen, rank_frame, condition, tail)
    rank_before = _masked_bce(frozen_rank_logits, rank_frame, rank_mask)
    rank_signal_before = float(_pilot_signal_reconstruction_loss_for_diagnostic(
        frozen, rank_frame, condition, tail,
    ).detach().cpu())
    results = []
    models = {}
    for specification in candidate_bank:
        candidate_model = copy.deepcopy(frozen).eval()
        _retag_unfolded_peft_groups(candidate_model)
        if specification["groups"]:
            group_names = set(specification["groups"])
            before_parameters = {
                name: parameter.detach().clone()
                for name, parameter in candidate_model.named_parameters()
                if getattr(parameter, "_peft_group", None) in group_names
            }
            update = apply_adapt_only_peft_update(
                model=candidate_model,
                frame=update_frame,
                condition=condition,
                soft_tail=tail,
                groups=set(specification["groups"]),
                lr=float(specification.get("lr", lr)),
                steps=int(specification.get("steps", steps)),
                objective=str(specification["objective"]),
                include_evaluation_metrics=False,
            )
            delta_norm = float(update["peft_delta_norm"])
            delta_limit = float(specification.get("max_delta_norm", math.inf))
            if not math.isfinite(delta_limit) and delta_limit != math.inf:
                raise ValueError("max_delta_norm 必须是正有限数或未限制。")
            if delta_limit <= 0.0:
                raise ValueError("可训练候选的 max_delta_norm 必须为正数。")
            if delta_norm > delta_limit:
                scale = delta_limit / max(delta_norm, 1e-12)
                with torch.no_grad():
                    named_parameters = dict(candidate_model.named_parameters())
                    for name, before in before_parameters.items():
                        current = named_parameters[name]
                        current.copy_(before + scale * (current - before))
                delta_norm = float(delta_limit)
        else:
            delta_norm = 0.0
        candidate_model.eval()
        rank_logits = _logits(candidate_model, rank_frame, condition, tail)
        rank_after = _masked_bce(rank_logits, rank_frame, rank_mask)
        rank_signal_after = float(_pilot_signal_reconstruction_loss_for_diagnostic(
            candidate_model, rank_frame, condition, tail,
        ).detach().cpu())
        result = {
            "candidate_name": str(specification["name"]),
            "objective": str(specification["objective"]),
            "updated_groups": list(specification["groups"]),
            "learning_rate": float(specification.get("lr", lr)),
            "adapt_steps": int(specification.get("steps", steps)),
            "max_delta_norm": float(specification.get("max_delta_norm", math.inf)),
            "peft_delta_norm": delta_norm,
            "rank_pilot_bce_before": rank_before,
            "rank_pilot_bce_after": rank_after,
            "rank_pilot_bce_improvement": rank_before - rank_after,
            "rank_pilot_signal_before": rank_signal_before,
            "rank_pilot_signal_after": rank_signal_after,
            "rank_pilot_signal_improvement": rank_signal_before - rank_signal_after,
            "pilot_rank_score_name": str(rank_score),
            "pilot_rank_score_improvement": (
                rank_before - rank_after if rank_score == "bce"
                else rank_signal_before - rank_signal_after
            ),
            "rank_pilot_error_count_before": _error_count(frozen_rank_logits, rank_frame, rank_mask),
            "rank_pilot_error_count_after": _error_count(rank_logits, rank_frame, rank_mask),
        }
        results.append(result)
        models[str(specification["name"])] = candidate_model
    selected = _rank_candidates(results, rank_score=rank_score)
    return results, {"models": models, "selected": selected, "update_count": int(update_mask.sum())}


def _run_trial(*, experiment: dict, model_config, checkpoint: Path, device: str,
               snr: float, seed: int, frames_count: int, delay: int, pilot_total: int,
               reward_pilot_total: int, gap_seconds: float, holdout_fraction: float,
               lr: float, steps: int, min_reward_improvement: float,
               freeze_below_snr_db: float,
               candidate_bank: tuple[dict, ...], rank_score: str,
               require_reward_signal_improvement: bool) -> list[dict]:
    episode = build_pilot_replay_episode(
        experiment,
        model_config,
        snr=float(snr),
        seed=int(seed),
        delay=int(delay),
        pilot_total=int(pilot_total),
        reward_pilot_total=int(reward_pilot_total),
        gap_seconds=float(gap_seconds),
        device=device,
        condition_source="pilot_cir_phase",
    )
    frozen = _build_equalizer(model_config, checkpoint, device)
    frozen.attach_online_phase_trend_adapter()
    frozen.eval()
    tail = episode.start.initial_soft_tail.to(device).to(torch.complex64)
    rows = []
    for frame_index in range(int(frames_count)):
        frame = _frame_to_device(episode.env.next_frame(), device)
        update_frame, rank_frame, reward_frame, update_mask, rank_mask = split_adapt_subwindows(
            frame,
            holdout_fraction,
        )
        # 状态和物理条件仅由训练 Pilot 窗口估计，排序窗和 Reward Pilot 不参与更新条件。
        condition = episode.condition(update_frame, tail)
        condition = type(condition)(
            complex_cir=condition.complex_cir.to(device),
            support_probability=condition.support_probability.to(device),
            noise_variance=condition.noise_variance.to(device),
            confidence=condition.confidence.to(device),
            latent_residual=condition.latent_residual.to(device),
        )
        candidate_results, candidate_state = _make_candidate_models(
            frozen,
            update_frame,
            rank_frame,
            condition,
            tail,
            candidate_bank=candidate_bank,
            lr=lr,
            steps=steps,
            rank_score=rank_score,
        )
        selected = candidate_state["selected"]
        selected_model = candidate_state["models"][selected["candidate_name"]]

        # Reward Pilot 只检查排序窗选出的单个候选，完全不参与候选排序。
        reward_mask = reward_frame.reward_mask.bool()
        frozen_reward_logits = _logits(frozen, reward_frame, condition, tail)
        selected_reward_logits = _logits(selected_model, reward_frame, condition, tail)
        reward_bce_before = _masked_bce(frozen_reward_logits, reward_frame, reward_mask)
        reward_bce_after = _masked_bce(selected_reward_logits, reward_frame, reward_mask)
        reward_improvement = reward_bce_before - reward_bce_after
        reward_errors_before = _error_count(frozen_reward_logits, reward_frame, reward_mask)
        reward_errors_after = _error_count(selected_reward_logits, reward_frame, reward_mask)
        no_mask = torch.zeros_like(reward_frame.adapt_mask, dtype=torch.bool)
        reward_signal_frame = _masked_pilot_view(
            frame,
            frame.reward_mask,
            adapt_mask=frame.reward_mask,
            reward_mask=no_mask,
        )
        reward_signal_before = float(_pilot_signal_reconstruction_loss_for_diagnostic(
            frozen, reward_signal_frame, condition, tail,
        ).detach().cpu())
        reward_signal_after = float(_pilot_signal_reconstruction_loss_for_diagnostic(
            selected_model, reward_signal_frame, condition, tail,
        ).detach().cpu())
        reward_signal_improvement = reward_signal_before - reward_signal_after
        accepted = _reward_gate(
            snr=float(snr),
            freeze_below_snr_db=float(freeze_below_snr_db),
            reward_bce_improvement=reward_improvement,
            min_reward_improvement=float(min_reward_improvement),
            reward_errors_before=reward_errors_before,
            reward_errors_after=reward_errors_after,
            reward_signal_improvement=reward_signal_improvement,
            require_signal_improvement=bool(require_reward_signal_improvement),
        )

        # 到这里所有在线可见的排序、验收和保留/拒绝决定均已完成；此后才首次查看 Data。
        frozen_data_logits = _logits(frozen, frame, condition, tail)
        committed_model = selected_model if accepted else frozen
        committed_data_logits = _logits(committed_model, frame, condition, tail)
        committed_metrics = _posthoc_data_metrics(frozen_data_logits, committed_data_logits, frame)
        for result in candidate_results:
            candidate_model = candidate_state["models"][result["candidate_name"]]
            candidate_data_logits = _logits(candidate_model, frame, condition, tail)
            result.update(_posthoc_data_metrics(frozen_data_logits, candidate_data_logits, frame))
            result.update({
                "seed": int(seed),
                "snr_db": float(snr),
                "frame_index": int(frame_index),
                "rank_update_pilot_count": int(update_mask.sum()),
                "rank_holdout_pilot_count": int(rank_mask.sum()),
                "pilot_total": int(pilot_total),
                "reward_pilot_total": int(reward_pilot_total),
                "candidate_selected_by_holdout": result["candidate_name"] == selected["candidate_name"],
                "candidate_accepted_by_reward": bool(
                    accepted and result["candidate_name"] == selected["candidate_name"]
                ),
                "data_waveform_used_online": False,
                "data_labels_used_online": False,
                "diagnostic_uses_data_labels": True,
            })
        common = {
            "seed": int(seed),
            "snr_db": float(snr),
            "frame_index": int(frame_index),
            "selected_candidate": str(selected["candidate_name"]),
            "selected_rank_pilot_bce_improvement": float(selected["rank_pilot_bce_improvement"]),
            "selected_pilot_rank_score_name": str(rank_score),
            "selected_pilot_rank_score_improvement": float(selected["pilot_rank_score_improvement"]),
            "reward_bce_before": reward_bce_before,
            "reward_bce_after": reward_bce_after,
            "reward_bce_improvement": reward_improvement,
            "reward_signal_error_before": reward_signal_before,
            "reward_signal_error_after": reward_signal_after,
            "reward_signal_improvement": reward_signal_improvement,
            "reward_signal_guard_required": bool(require_reward_signal_improvement),
            "reward_error_count_before": reward_errors_before,
            "reward_error_count_after": reward_errors_after,
            "reward_gate_accepted": accepted,
            "minimum_reward_improvement": float(min_reward_improvement),
            "freeze_below_snr_db": float(freeze_below_snr_db),
            "committed_candidate": str(selected["candidate_name"] if accepted else "identity"),
            **committed_metrics,
            "data_waveform_used_online": False,
            "data_labels_used_online": False,
            "diagnostic_uses_data_labels": True,
        }
        rows.append({"trial": common, "candidates": candidate_results})
        print(
            f"seed={seed} snr={snr:g} frame={frame_index} "
            f"rank={selected['candidate_name']} reward={'accept' if accepted else 'reject'} "
            f"committed_ber_gain={committed_metrics['data_ber_improvement']:.6f}",
            flush=True,
        )
    return rows


def _pairwise_agreement(rows: list[dict], target: str) -> dict:
    concordant = discordant = tied = 0
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            score_delta = float(left["rank_pilot_bce_improvement"]) - float(right["rank_pilot_bce_improvement"])
            target_delta = float(left[target]) - float(right[target])
            if score_delta == 0.0 or target_delta == 0.0:
                tied += 1
            elif (score_delta > 0.0) == (target_delta > 0.0):
                concordant += 1
            else:
                discordant += 1
    comparable = concordant + discordant
    return {
        "comparable_pairs": comparable,
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
        "tied_pairs": tied,
        "agreement": concordant / comparable if comparable else math.nan,
    }


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2.0
        for offset in range(start, end):
            ranks[order[offset]] = average
        start = end
    return ranks


def _rank_correlation(rows: list[dict], target: str) -> float:
    """对单个 seed/SNR/frame 内的候选分数和事后目标计算 Spearman。"""

    if len(rows) < 2:
        return math.nan
    x = _average_ranks([float(row["pilot_rank_score_improvement"]) for row in rows])
    y = _average_ranks([float(row[target]) for row in rows])
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    covariance = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    variance_x = sum((value - mean_x) ** 2 for value in x)
    variance_y = sum((value - mean_y) ** 2 for value in y)
    if variance_x <= 0.0 or variance_y <= 0.0:
        return math.nan
    return covariance / math.sqrt(variance_x * variance_y)


def _pooled_within_group_rank_correlation(groups: list[list[dict]], target: str) -> float:
    """先在每个 trial 内转成名次，再合并名次计算相关，避免跨 trial 基线混淆。"""

    x_all, y_all = [], []
    for rows in groups:
        if len(rows) < 2:
            continue
        x = _average_ranks([float(row["pilot_rank_score_improvement"]) for row in rows])
        y = _average_ranks([float(row[target]) for row in rows])
        if len(set(x)) < 2 or len(set(y)) < 2:
            continue
        x_all.extend(x)
        y_all.extend(y)
    if len(x_all) < 2:
        return math.nan
    mean_x, mean_y = sum(x_all) / len(x_all), sum(y_all) / len(y_all)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_all, y_all))
    variance_x = sum((x - mean_x) ** 2 for x in x_all)
    variance_y = sum((y - mean_y) ** 2 for y in y_all)
    if variance_x <= 0.0 or variance_y <= 0.0:
        return math.nan
    return covariance / math.sqrt(variance_x * variance_y)


def summarize(rows: list[dict]) -> dict:
    flattened = [candidate for trial in rows for candidate in trial["candidates"]]
    grouped = defaultdict(list)
    for candidate in flattened:
        grouped[(candidate["seed"], candidate["snr_db"], candidate["frame_index"])].append(candidate)
    rank_groups = []
    for (seed, snr, frame_index), values in sorted(grouped.items()):
        rank_groups.append({
            "seed": seed,
            "snr_db": snr,
            "frame_index": frame_index,
            "pilot_rank_vs_data_ber": _pairwise_agreement(values, "data_ber_improvement"),
            "pilot_rank_vs_data_bce": _pairwise_agreement(values, "data_bce_improvement"),
            "pilot_rank_spearman_vs_data_ber": _rank_correlation(values, "data_ber_improvement"),
            "pilot_rank_spearman_vs_data_bce": _rank_correlation(values, "data_bce_improvement"),
            "best_by_independent_pilot": max(values, key=lambda row: row["pilot_rank_score_improvement"])["candidate_name"],
            "best_by_data_ber_posthoc": max(values, key=lambda row: row["data_ber_improvement"])["candidate_name"],
        })
    ber_spearman = _pooled_within_group_rank_correlation(list(grouped.values()), "data_ber_improvement")
    bce_spearman = _pooled_within_group_rank_correlation(list(grouped.values()), "data_bce_improvement")
    trial_rows = [item["trial"] for item in rows]
    by_snr = []
    for snr in sorted({item["snr_db"] for item in trial_rows}):
        values = [item for item in trial_rows if item["snr_db"] == snr]
        by_snr.append({
            "snr_db": snr,
            "trial_count": len(values),
            "reward_acceptance_fraction": sum(bool(row["reward_gate_accepted"]) for row in values) / len(values),
            "mean_committed_data_ber_improvement": sum(row["data_ber_improvement"] for row in values) / len(values),
            "positive_committed_data_ber_fraction": sum(row["data_ber_improvement"] > 0 for row in values) / len(values),
        })
    by_seed = []
    for seed in sorted({item["seed"] for item in trial_rows}):
        values = [item for item in trial_rows if item["seed"] == seed]
        by_seed.append({
            "seed": seed,
            "trial_count": len(values),
            "mean_committed_data_ber_improvement": sum(row["data_ber_improvement"] for row in values) / len(values),
            "positive_committed_data_ber_fraction": sum(row["data_ber_improvement"] > 0 for row in values) / len(values),
        })
    return {
        "schema_version": "pilot-independent-subwindows-v1",
        "trial_count": len(trial_rows),
        "candidate_count": len(flattened),
        "online_data_waveform_isolation_passed": all(
            not row["data_waveform_used_online"] and not row["data_labels_used_online"]
            for row in flattened + trial_rows
        ),
        "ranking_gate_threshold_spearman": 0.6,
        "within_trial_rank_vs_data_ber": {
            "spearman": ber_spearman,
            "gate_pass": bool(math.isfinite(ber_spearman) and ber_spearman >= 0.6),
            "aggregation": "先在每个 seed/SNR/frame 内转成候选名次，再合并名次；不跨 trial 拼接原始分数。",
            "comparable_pairs": sum(row["pilot_rank_vs_data_ber"]["comparable_pairs"] for row in rank_groups),
            "concordant_pairs": sum(row["pilot_rank_vs_data_ber"]["concordant_pairs"] for row in rank_groups),
            "discordant_pairs": sum(row["pilot_rank_vs_data_ber"]["discordant_pairs"] for row in rank_groups),
            "tied_pairs": sum(row["pilot_rank_vs_data_ber"]["tied_pairs"] for row in rank_groups),
        },
        "within_trial_rank_vs_data_bce": {
            "spearman": bce_spearman,
            "gate_pass": bool(math.isfinite(bce_spearman) and bce_spearman >= 0.6),
            "aggregation": "先在每个 seed/SNR/frame 内转成候选名次，再合并名次；不跨 trial 拼接原始分数。",
        },
        "ranking_groups": rank_groups,
        "selection_policy": (
            f"独立 Adapt Pilot {trial_rows[0]['selected_pilot_rank_score_name'] if trial_rows else 'BCE'} "
            "选出单个候选；Reward Pilot 仅对该候选做 BCE 改善与硬错数不增加验收；"
            "验收后 Data 只作事后评估。"
        ),
        "mean_committed_data_ber_improvement": sum(row["data_ber_improvement"] for row in trial_rows) / max(1, len(trial_rows)),
        "positive_committed_data_ber_fraction": sum(row["data_ber_improvement"] > 0 for row in trial_rows) / max(1, len(trial_rows)),
        "reward_acceptance_fraction": sum(bool(row["reward_gate_accepted"]) for row in trial_rows) / max(1, len(trial_rows)),
        "by_snr": by_snr,
        "by_seed": by_seed,
        "trials": trial_rows,
        "candidate_rows": flattened,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--snrs", nargs="+", type=float, default=[5.0, 10.0, 15.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[63, 64, 65])
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--delay", type=int, default=116)
    parser.add_argument("--pilot-total", type=int, default=256)
    parser.add_argument("--reward-pilot-total", type=int, default=32)
    parser.add_argument("--gap-seconds", type=float, default=600.0)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument(
        "--candidate-bank",
        choices=["objective", "multiscale_signal", "fixed_moderate_signal"],
        default="objective",
    )
    parser.add_argument("--rank-score", choices=["bce", "signal_reconstruction"], default="bce")
    parser.add_argument("--require-reward-signal-improvement", action="store_true")
    parser.add_argument("--min-reward-improvement", type=float, default=None)
    parser.add_argument("--freeze-below-snr-db", type=float, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.frames <= 0 or args.delay <= 0 or args.steps <= 0 or args.lr <= 0:
        parser.error("frames、delay、steps 和 lr 必须为正数。")
    if not math.isfinite(args.gap_seconds) or args.gap_seconds < 0:
        parser.error("gap-seconds 必须为有限非负数。")
    if not Path(args.pretrained).is_file():
        parser.error(f"checkpoint 不存在：{args.pretrained}")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model_config = _load_model_config(config, Path(args.pretrained))
    min_reward = float(
        args.min_reward_improvement
        if args.min_reward_improvement is not None
        else config.get("online_adaptation_min_reward_improvement", 0.0005)
    )
    freeze_below = float(
        args.freeze_below_snr_db
        if args.freeze_below_snr_db is not None
        else config.get("online_adaptation_freeze_below_snr_db", 5.0)
    )
    max_delta_norm = float(config.get("online_adaptation_max_delta_norm", 0.05))
    selected_candidates = candidate_specs(
        args.candidate_bank,
        lr=float(args.lr),
        steps=int(args.steps),
        max_delta_norm=max_delta_norm,
    )
    all_rows = []
    for snr in args.snrs:
        for seed in args.seeds:
            all_rows.extend(_run_trial(
                experiment=config,
                model_config=model_config,
                checkpoint=Path(args.pretrained),
                device=args.device,
                snr=float(snr),
                seed=int(seed),
                frames_count=args.frames,
                delay=args.delay,
                pilot_total=args.pilot_total,
                reward_pilot_total=args.reward_pilot_total,
                gap_seconds=args.gap_seconds,
                holdout_fraction=args.holdout_fraction,
                lr=args.lr,
                steps=args.steps,
                min_reward_improvement=min_reward,
                freeze_below_snr_db=freeze_below,
                candidate_bank=selected_candidates,
                rank_score=args.rank_score,
                require_reward_signal_improvement=args.require_reward_signal_improvement,
            ))
    result = summarize(all_rows)
    result.update({
        "checkpoint": str(Path(args.pretrained).resolve()),
        "checkpoint_sha256": hashlib.sha256(Path(args.pretrained).read_bytes()).hexdigest().upper(),
        "checkpoint_modified_or_retrained": False,
        "fixed_condition": {
            "channel_level": "B",
            "profile": "eme_long_memory_v2",
            "delay_symbols": int(args.delay),
            "pilot_layout": "prefix",
            "pilot_total": int(args.pilot_total),
            "reward_pilot_total": int(args.reward_pilot_total),
            "acquisition_gap_seconds": float(args.gap_seconds),
            "snrs_db": [float(value) for value in args.snrs],
            "seeds": [int(value) for value in args.seeds],
            "frames_per_seed_snr": int(args.frames),
            "adapt_holdout_fraction": float(args.holdout_fraction),
            "condition_source": "pilot_cir_phase_from_update_window",
            "learning_rate": float(args.lr),
            "steps": int(args.steps),
            "minimum_reward_improvement": min_reward,
            "freeze_below_snr_db": freeze_below,
            "device": args.device,
            "candidate_bank": args.candidate_bank,
            "rank_score": args.rank_score,
            "require_reward_signal_improvement": args.require_reward_signal_improvement,
            "candidate_specs": selected_candidates,
        },
        "data_metrics_used_online": False,
        "diagnostic_uses_data_labels": True,
    })
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    print(
        f"已保存独立 Pilot 子窗口 replay：{target}；trial={result['trial_count']}，"
        f"候选={result['candidate_count']}，BER 正收益比例={result['positive_committed_data_ber_fraction']:.1%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
