# -*- coding: utf-8 -*-
"""检验 Pilot-only PEFT 更新在连续帧之间持久保留时的收益。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compare import _build_equalizer, _load_model_config
from evaluation.pilot_replay_protocol import build_pilot_replay_episode
from evaluation.research_diagnostics import (
    _frame_to_device,
    _pilot_reconstruction_loss_for_diagnostic,
    _pilot_signal_reconstruction_loss_for_diagnostic,
    _retag_unfolded_peft_groups,
    apply_adapt_only_peft_update,
)
from scripts.replay_pilot_independent_subwindows import (
    _error_count,
    _logits,
    _masked_bce,
    _masked_pilot_view,
    _posthoc_data_metrics,
    _reward_gate,
    split_adapt_subwindows,
)


def _group_parameters(model, group: str) -> dict[str, torch.nn.Parameter]:
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if getattr(parameter, "_peft_group", None) == group
    }


def _parameter_delta_norm(model, reference, group: str) -> float:
    current, baseline = _group_parameters(model, group), _group_parameters(reference, group)
    if set(current) != set(baseline):
        raise ValueError("PEFT 参数结构与冻结参照不一致。")
    if not current:
        return 0.0
    total = sum((current[name].detach() - baseline[name].detach()).float().pow(2).sum() for name in current)
    return float(torch.sqrt(total).cpu())


def _clip_candidate_delta(model, before: dict[str, torch.Tensor], group: str,
                          max_delta_norm: float) -> float:
    """限制单次梯度更新的 PEFT 总增量，超限时沿原方向缩放。"""

    if not math.isfinite(max_delta_norm) or max_delta_norm <= 0.0:
        raise ValueError("max_delta_norm 必须为正有限数。")
    parameters = _group_parameters(model, group)
    if set(parameters) != set(before):
        raise ValueError("更新前后 PEFT 参数集合不一致。")
    delta_sq = sum((parameters[name].detach() - value).float().pow(2).sum() for name, value in before.items())
    delta_norm = float(torch.sqrt(delta_sq).cpu())
    if delta_norm > max_delta_norm:
        scale = max_delta_norm / max(delta_norm, 1e-12)
        with torch.no_grad():
            for name, value in before.items():
                parameter = parameters[name]
                parameter.copy_(value + scale * (parameter - value))
        delta_norm = float(max_delta_norm)
    return delta_norm


def _clip_candidate_to_reference(model, reference, group: str,
                                max_delta_norm: float) -> float:
    """限制 PEFT 参数相对离线参考 checkpoint 的累计总偏移。"""

    reference_parameters = {
        name: parameter.detach().clone()
        for name, parameter in _group_parameters(reference, group).items()
    }
    if not reference_parameters:
        raise ValueError(f"参考模型没有 {group} 参数组。")
    return _clip_candidate_delta(model, reference_parameters, group, max_delta_norm)


def _aggregate_reward_history(scores: list[dict[str, float | int | bool]]) -> dict[str, float | int]:
    """汇总同一候选在最近几帧 Reward Pilot 上的分数。"""

    if not scores:
        raise ValueError("Reward Pilot 历史不能为空。")
    count = len(scores)
    return {
        "frame_count": count,
        "bce_improvement": sum(float(item["bce_improvement"]) for item in scores) / count,
        "signal_improvement": sum(float(item["signal_improvement"]) for item in scores) / count,
        "signal_positive_fraction": sum(float(item["signal_improvement"]) > 0.0 for item in scores) / count,
        "error_nonincrease_fraction": sum(
            int(item["errors_after"]) <= int(item["errors_before"]) for item in scores
        ) / count,
    }


def _reward_history_is_consistent(aggregate: dict[str, float | int],
                                  minimum_fraction: float = 2.0 / 3.0) -> bool:
    """要求多数历史 Reward Pilot 的信号分数和硬错数都不恶化。"""

    threshold = float(minimum_fraction)
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("minimum_fraction 必须在 (0, 1] 范围内。")
    return bool(
        float(aggregate["signal_positive_fraction"]) >= threshold
        and float(aggregate["error_nonincrease_fraction"]) >= threshold
    )


def _score_reward_pair(model, reward_view, reward_signal_view, condition,
                       tail: torch.Tensor, loss_function) -> dict[str, float | int]:
    """只在 Pilot 波形上评分同一候选的前后状态。"""

    reward_mask = reward_view.reward_mask.bool()
    before_logits = _logits(model["before"], reward_view, condition, tail)
    after_logits = _logits(model["after"], reward_view, condition, tail)
    bce_before = _masked_bce(before_logits, reward_view, reward_mask)
    bce_after = _masked_bce(after_logits, reward_view, reward_mask)
    errors_before = _error_count(before_logits, reward_view, reward_mask)
    errors_after = _error_count(after_logits, reward_view, reward_mask)
    signal_before = float(loss_function(
        model["before"], reward_signal_view, condition, tail,
    ).detach().cpu())
    signal_after = float(loss_function(
        model["after"], reward_signal_view, condition, tail,
    ).detach().cpu())
    return {
        "bce_improvement": bce_before - bce_after,
        "signal_improvement": signal_before - signal_after,
        "errors_before": errors_before,
        "errors_after": errors_after,
    }


def _run_trial(*, experiment: dict, model_config, checkpoint: Path, device: str,
               snr: float, seed: int, frames_count: int, delay: int,
               pilot_total: int, reward_pilot_total: int, gap_seconds: float,
               holdout_fraction: float, learning_rate: float, steps: int,
               max_delta_norm: float, min_reward_improvement: float,
               freeze_below_snr_db: float,
               reward_history_frames: int, peft_group: str,
               max_cumulative_delta_norm: float | None,
               state_split: str | None) -> list[dict]:
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
        state_split=state_split,
    )
    frozen = _build_equalizer(model_config, checkpoint, device)
    if peft_group == "phase_trend":
        frozen.attach_online_phase_trend_adapter()
        update_objective = "pilot_signal_reconstruction"
        reconstruction_loss = _pilot_signal_reconstruction_loss_for_diagnostic
    elif peft_group == "channel_residual":
        frozen.attach_online_channel_residual_adapter()
        update_objective = "pilot_reconstruction"
        reconstruction_loss = _pilot_reconstruction_loss_for_diagnostic
    else:
        raise ValueError(f"不支持的 PEFT 参数组：{peft_group}")
    frozen.eval()
    active = copy.deepcopy(frozen).eval()
    _retag_unfolded_peft_groups(active)
    tail = episode.start.initial_soft_tail.to(device).to(torch.complex64)
    rows = []
    reward_history = []

    for frame_index in range(int(frames_count)):
        frame = _frame_to_device(episode.env.next_frame(), device)
        update_view, rank_view, reward_view, _, _ = split_adapt_subwindows(
            frame,
            holdout_fraction,
        )
        # 物理条件只从 Adapt 更新窗估计，不查看独立排序窗、Reward 或 Data 波形。
        condition = episode.condition(update_view, tail)
        condition = type(condition)(
            complex_cir=condition.complex_cir.to(device),
            support_probability=condition.support_probability.to(device),
            noise_variance=condition.noise_variance.to(device),
            confidence=condition.confidence.to(device),
            latent_residual=condition.latent_residual.to(device),
        )

        # 每帧都从当前已接受参数出发；拒绝时 active 始终指向上一帧参数。
        active_before = active
        candidate = copy.deepcopy(active_before).eval()
        _retag_unfolded_peft_groups(candidate)
        peft_parameters_before = {
            name: parameter.detach().clone()
            for name, parameter in _group_parameters(candidate, peft_group).items()
        }
        update_result = apply_adapt_only_peft_update(
            model=candidate,
            frame=update_view,
            condition=condition,
            soft_tail=tail,
            groups={peft_group},
            lr=float(learning_rate),
            steps=int(steps),
            objective=update_objective,
            include_evaluation_metrics=False,
        )
        step_delta_norm = _clip_candidate_delta(
            candidate,
            peft_parameters_before,
            peft_group,
            float(max_delta_norm),
        )
        if max_cumulative_delta_norm is not None:
            _clip_candidate_to_reference(
                candidate,
                frozen,
                peft_group,
                float(max_cumulative_delta_norm),
            )
        candidate_total_delta_norm = _parameter_delta_norm(candidate, frozen, peft_group)
        candidate.eval()

        # 独立 Adapt Pilot 留出窗先决定是否把候选送入 Reward Pilot 验收。
        rank_mask = rank_view.adapt_mask.bool()
        rank_before = float(reconstruction_loss(
            active_before, rank_view, condition, tail,
        ).detach().cpu())
        rank_after = float(reconstruction_loss(
            candidate, rank_view, condition, tail,
        ).detach().cpu())
        rank_improvement = rank_before - rank_after
        reward_checked = bool(math.isfinite(rank_improvement) and rank_improvement > 0.0)

        reward_bce_before = reward_bce_after = 0.0
        reward_bce_improvement = 0.0
        reward_signal_before = reward_signal_after = 0.0
        reward_signal_improvement = 0.0
        reward_history_bce_improvement = 0.0
        reward_history_signal_improvement = 0.0
        reward_history_signal_positive_fraction = 0.0
        reward_history_error_nonincrease_fraction = 0.0
        reward_history_frame_count = 0
        reward_errors_before = reward_errors_after = 0
        accepted = False
        reward_signal_view = _masked_pilot_view(
            frame,
            frame.reward_mask,
            adapt_mask=frame.reward_mask,
            reward_mask=torch.zeros_like(frame.adapt_mask, dtype=torch.bool),
        )
        if reward_checked:
            reward_mask = reward_view.reward_mask.bool()
            before_reward_logits = _logits(active_before, reward_view, condition, tail)
            after_reward_logits = _logits(candidate, reward_view, condition, tail)
            reward_bce_before = _masked_bce(before_reward_logits, reward_view, reward_mask)
            reward_bce_after = _masked_bce(after_reward_logits, reward_view, reward_mask)
            reward_bce_improvement = reward_bce_before - reward_bce_after
            reward_errors_before = _error_count(before_reward_logits, reward_view, reward_mask)
            reward_errors_after = _error_count(after_reward_logits, reward_view, reward_mask)
            reward_signal_before = float(reconstruction_loss(
                active_before, reward_signal_view, condition, tail,
            ).detach().cpu())
            reward_signal_after = float(reconstruction_loss(
                candidate, reward_signal_view, condition, tail,
            ).detach().cpu())
            reward_signal_improvement = reward_signal_before - reward_signal_after
            history_limit = max(0, int(reward_history_frames) - 1)
            history = reward_history[-history_limit:] if history_limit else []
            score_models = {"before": active_before, "after": candidate}
            reward_scores = [
                _score_reward_pair(
                    score_models,
                    reward_view,
                    reward_signal_view,
                    condition,
                    tail,
                    reconstruction_loss,
                )
            ]
            for history_item in history:
                reward_scores.append(_score_reward_pair(
                    score_models,
                    history_item["reward_view"],
                    history_item["reward_signal_view"],
                    history_item["condition"],
                    history_item["tail"],
                    reconstruction_loss,
                ))
            reward_aggregate = _aggregate_reward_history(reward_scores)
            reward_history_bce_improvement = float(reward_aggregate["bce_improvement"])
            reward_history_signal_improvement = float(reward_aggregate["signal_improvement"])
            reward_history_signal_positive_fraction = float(reward_aggregate["signal_positive_fraction"])
            reward_history_error_nonincrease_fraction = float(reward_aggregate["error_nonincrease_fraction"])
            reward_history_frame_count = int(reward_aggregate["frame_count"])
            accepted = _reward_gate(
                snr=float(snr),
                freeze_below_snr_db=float(freeze_below_snr_db),
                reward_bce_improvement=reward_history_bce_improvement,
                min_reward_improvement=float(min_reward_improvement),
                reward_errors_before=reward_errors_before,
                reward_errors_after=reward_errors_after,
                reward_signal_improvement=reward_history_signal_improvement,
                require_signal_improvement=True,
            )
            history_consistent = _reward_history_is_consistent(reward_aggregate)
            accepted = bool(accepted and history_consistent)
        if accepted:
            active = candidate

        # 只缓存屏蔽 Data 的 Reward Pilot 视图，供后续候选做多帧一致性检查。
        reward_history.append({
            "reward_view": reward_view,
            "reward_signal_view": reward_signal_view,
            "condition": condition,
            "tail": tail.detach().clone(),
        })
        reward_history = reward_history[-int(reward_history_frames):]

        # Adapt 排序及 Reward 接受/拒绝决定已结束；Data 才首次进入诊断前向。
        frozen_data_logits = _logits(frozen, frame, condition, tail)
        previous_data_logits = _logits(active_before, frame, condition, tail)
        online_data_logits = _logits(active, frame, condition, tail)
        cumulative_metrics = _posthoc_data_metrics(frozen_data_logits, online_data_logits, frame)
        frame_metrics = _posthoc_data_metrics(previous_data_logits, online_data_logits, frame)
        rows.append({
            **episode.metadata(),
            "seed": int(seed),
            "snr_db": float(snr),
            "frame_index": int(frame_index),
            "pilot_total": int(pilot_total),
            "adapt_update_pilot_count": int(update_view.adapt_mask.sum()),
            "adapt_rank_pilot_count": int(rank_mask.sum()),
            "reward_pilot_total": int(reward_pilot_total),
            "update_group": peft_group,
            "update_objective": update_objective,
            "update_learning_rate": float(learning_rate),
            "update_steps": int(steps),
            "per_step_delta_norm": float(step_delta_norm),
            "optimizer_reported_delta_norm": float(update_result["peft_delta_norm"]),
            "candidate_peft_delta_norm_before_reward": float(candidate_total_delta_norm),
            "max_cumulative_delta_norm": (
                None if max_cumulative_delta_norm is None else float(max_cumulative_delta_norm)
            ),
            "cumulative_peft_delta_norm": _parameter_delta_norm(active, frozen, peft_group),
            **({
                "cumulative_phase_adapter_delta_norm": _parameter_delta_norm(active, frozen, peft_group),
            } if peft_group == "phase_trend" else {}),
            "rank_pilot_signal_before": rank_before,
            "rank_pilot_signal_after": rank_after,
            "rank_pilot_signal_improvement": rank_improvement,
            "reward_checked_after_rank": reward_checked,
            "reward_bce_before": reward_bce_before,
            "reward_bce_after": reward_bce_after,
            "reward_bce_improvement": reward_bce_improvement,
            "reward_signal_before": reward_signal_before,
            "reward_signal_after": reward_signal_after,
            "reward_signal_improvement": reward_signal_improvement,
            "reward_history_frames": int(reward_history_frame_count),
            "reward_history_bce_improvement": reward_history_bce_improvement,
            "reward_history_signal_improvement": reward_history_signal_improvement,
            "reward_history_signal_positive_fraction": reward_history_signal_positive_fraction,
            "reward_history_error_nonincrease_fraction": reward_history_error_nonincrease_fraction,
            "reward_error_count_before": reward_errors_before,
            "reward_error_count_after": reward_errors_after,
            "reward_gate_accepted": accepted,
            "minimum_reward_improvement": float(min_reward_improvement),
            "freeze_below_snr_db": float(freeze_below_snr_db),
            **{f"cumulative_{key}": value for key, value in cumulative_metrics.items()},
            **{f"frame_update_{key}": value for key, value in frame_metrics.items()},
            "data_waveform_used_for_update_or_decision": False,
            "data_labels_used_for_update_or_decision": False,
            "diagnostic_uses_data_labels": True,
        })
        print(
            f"seed={seed} snr={snr:g} frame={frame_index} "
            f"rank={'pass' if reward_checked else 'reject'} "
            f"reward={'accept' if accepted else 'reject'} "
            f"cumulative_ber_gain={cumulative_metrics['data_ber_improvement']:.6f}",
            flush=True,
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("连续 Pilot replay 结果不能为空。")
    by_snr = []
    by_seed = []
    grouped: dict[tuple[int, float], list[dict]] = {}
    for row in rows:
        grouped.setdefault((int(row["seed"]), float(row["snr_db"])), []).append(row)
    for snr in sorted({float(row["snr_db"]) for row in rows}):
        values = [row for row in rows if float(row["snr_db"]) == snr]
        by_snr.append({
            "snr_db": snr,
            "trial_count": len(values),
            "mean_cumulative_data_ber_improvement": sum(row["cumulative_data_ber_improvement"] for row in values) / len(values),
            "positive_cumulative_data_ber_fraction": sum(row["cumulative_data_ber_improvement"] > 0 for row in values) / len(values),
            "update_acceptance_fraction": sum(bool(row["reward_gate_accepted"]) for row in values) / len(values),
        })
    for seed in sorted({int(row["seed"]) for row in rows}):
        values = [row for row in rows if int(row["seed"]) == seed]
        by_seed.append({
            "seed": seed,
            "trial_count": len(values),
            "mean_cumulative_data_ber_improvement": sum(row["cumulative_data_ber_improvement"] for row in values) / len(values),
            "positive_cumulative_data_ber_fraction": sum(row["cumulative_data_ber_improvement"] > 0 for row in values) / len(values),
        })
    by_seed_snr = []
    for (seed, snr), values in sorted(grouped.items()):
        by_seed_snr.append({
            "seed": seed,
            "snr_db": snr,
            "frames": len(values),
            "mean_cumulative_data_ber_improvement": sum(row["cumulative_data_ber_improvement"] for row in values) / len(values),
            "accepted_updates": sum(bool(row["reward_gate_accepted"]) for row in values),
            "rank_passes": sum(bool(row["reward_checked_after_rank"]) for row in values),
        })
    gains = [float(row["cumulative_data_ber_improvement"]) for row in rows]
    return {
        "schema_version": "pilot-persistent-peft-replay-v1",
        "trial_count": len(rows),
        "online_data_waveform_isolation_passed": all(
            not row["data_waveform_used_for_update_or_decision"]
            and not row["data_labels_used_for_update_or_decision"]
            for row in rows
        ),
        "mean_cumulative_data_ber_improvement": sum(gains) / len(gains),
        "positive_cumulative_data_ber_fraction": sum(value > 0 for value in gains) / len(gains),
        "negative_cumulative_data_ber_fraction": sum(value < 0 for value in gains) / len(gains),
        "update_acceptance_fraction": sum(bool(row["reward_gate_accepted"]) for row in rows) / len(rows),
        "holdout_rank_pass_fraction": sum(bool(row["reward_checked_after_rank"]) for row in rows) / len(rows),
        "by_snr": by_snr,
        "by_seed": by_seed,
        "by_seed_snr": by_seed_snr,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--snrs", nargs="+", type=float, default=[5.0, 10.0, 15.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1050, 1051, 1052])
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--delay", type=int, default=116)
    parser.add_argument("--pilot-total", type=int, default=256)
    parser.add_argument("--reward-pilot-total", type=int, default=32)
    parser.add_argument("--gap-seconds", type=float, default=600.0)
    parser.add_argument("--state-split", default=None)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=0.0025)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--max-delta-norm", type=float, default=0.05)
    parser.add_argument("--min-reward-improvement", type=float, default=0.0005)
    parser.add_argument("--freeze-below-snr-db", type=float, default=5.0)
    parser.add_argument("--reward-history-frames", type=int, default=1)
    parser.add_argument("--peft-group", choices=["phase_trend", "channel_residual"], default="phase_trend")
    parser.add_argument("--max-cumulative-delta-norm", type=float, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not Path(args.pretrained).is_file():
        parser.error(f"checkpoint 不存在：{args.pretrained}")
    if args.frames <= 0 or args.steps <= 0 or args.learning_rate <= 0 or args.reward_history_frames <= 0:
        parser.error("frames、steps、learning-rate 和 reward-history-frames 必须为正数。")
    if not math.isfinite(args.gap_seconds) or args.gap_seconds < 0:
        parser.error("gap-seconds 必须为有限非负数。")
    if args.max_cumulative_delta_norm is not None and (
        not math.isfinite(args.max_cumulative_delta_norm) or args.max_cumulative_delta_norm <= 0.0
    ):
        parser.error("max-cumulative-delta-norm 必须为正有限数。")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    state_split = args.state_split if args.state_split is not None else config.get("state_split")
    model_config = _load_model_config(config, Path(args.pretrained))
    max_delta_norm = float(args.max_delta_norm)
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
                learning_rate=args.learning_rate,
                steps=args.steps,
                max_delta_norm=max_delta_norm,
                min_reward_improvement=args.min_reward_improvement,
                freeze_below_snr_db=args.freeze_below_snr_db,
                reward_history_frames=args.reward_history_frames,
                peft_group=args.peft_group,
                max_cumulative_delta_norm=args.max_cumulative_delta_norm,
                state_split=state_split,
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
            "state_split": state_split,
            "snrs_db": [float(value) for value in args.snrs],
            "seeds": [int(value) for value in args.seeds],
            "frames_per_seed_snr": int(args.frames),
            "adapt_holdout_fraction": float(args.holdout_fraction),
            "condition_source": "pilot_cir_phase_from_adapt_update_window",
            "update_group": args.peft_group,
            "update_objective": "pilot_signal_reconstruction" if args.peft_group == "phase_trend" else "pilot_reconstruction",
            "learning_rate": float(args.learning_rate),
            "steps": int(args.steps),
            "max_delta_norm": max_delta_norm,
            "max_cumulative_delta_norm": (
                None if args.max_cumulative_delta_norm is None else float(args.max_cumulative_delta_norm)
            ),
            "min_reward_improvement": float(args.min_reward_improvement),
            "freeze_below_snr_db": float(args.freeze_below_snr_db),
            "reward_history_frames": int(args.reward_history_frames),
            "reward_history_consistency_fraction": 2.0 / 3.0,
            "reward_requires_signal_reconstruction_improvement": True,
            "device": args.device,
            "adapter_state_persists_after_reward_acceptance": True,
        },
        "data_metrics_used_online": False,
    })
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    print(
        f"已保存连续 PEFT replay：{target}；帧={result['trial_count']}，"
        f"平均 BER 改善={result['mean_cumulative_data_ber_improvement']:.6f}，"
        f"接受率={result['update_acceptance_fraction']:.1%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
