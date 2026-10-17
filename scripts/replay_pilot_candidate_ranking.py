# -*- coding: utf-8 -*-
"""固定轨迹下的 Pilot-only PEFT 候选排序 replay。

候选更新和排序特征只使用当前 Adapt Pilot；Data 标签仅用于输出事后相关性，
不能参与候选选择、更新或回滚。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.cir_estimator import pilot_sparse_cir_update
from agent.pilot_state import PilotStateEmbedding, build_pilot_state_summary
from baseline.traditional_equalizers import estimate_phase_residual_vector
from env.linear_operator import LinearChannelOperator
from evaluation.metrics import spearman_reward_data
from evaluation.pilot_replay_protocol import build_pilot_replay_episode
from evaluation.research_diagnostics import (
    _condition_to_device,
    _boundary_adapt_loss,
    _selective_boundary_adapt_loss,
    _frame_to_device,
    _model_logits,
    _pilot_reconstruction_loss_for_diagnostic,
    _pilot_signal_reconstruction_loss_for_diagnostic,
    apply_adapt_only_peft_update,
)
from training.online_adaptation import PilotResidualRLSAdapter
from compare import _build_equalizer, _load_model_config


def _pilot_reconstruction_error(frame, cir: torch.Tensor, soft_tail: torch.Tensor) -> float:
    view = frame.receiver_view()
    device = cir.device
    rx = view.rx_symbols.to(device).to(torch.complex64).reshape(-1)
    tx = view.adapt_symbols.to(device).to(torch.complex64).reshape(-1)
    mask = view.adapt_mask.to(device).bool().reshape(-1)
    operator = LinearChannelOperator(frame_len=tx.numel(), max_delay=cir.numel() - 1)
    predicted = operator.forward(tx, cir, soft_tail.to(device).to(torch.complex64)).reshape(-1)
    residual = predicted[mask] - rx[mask]
    return float((torch.mean(torch.abs(residual) ** 2) / torch.mean(torch.abs(rx[mask]) ** 2).clamp_min(1e-8)).detach().cpu())


def _pilot_state(frame, cir: torch.Tensor, soft_tail: torch.Tensor, condition) -> dict[str, float]:
    phase = torch.as_tensor(
        estimate_phase_residual_vector(frame.receiver_view(), cir, soft_tail, blocks=4),
        dtype=torch.float32,
    ).reshape(-1)
    return {
        "phase0": float(phase[0].item()),
        "cfo_cycles_per_symbol": float(phase[1].item()),
        "phase_residual_variance": float(phase[2::4].mean().item()),
        "phase_residual_energy": float(phase[3::4].mean().item()),
        "reconstruction_error": _pilot_reconstruction_error(frame, cir, soft_tail),
        "confidence": float(torch.as_tensor(condition.confidence).float().mean().cpu()),
    }


def _candidate_specs(lr: float, steps: int) -> list[dict]:
    return [
        {"name": "identity", "groups": set(), "lr": 0.0, "steps": 0, "objective": "bce"},
        {"name": "physics_bce", "groups": {"physics_residual"}, "lr": lr, "steps": steps, "objective": "bce"},
        {"name": "physics_boundary", "groups": {"physics_residual"}, "lr": lr, "steps": steps, "objective": "boundary"},
        {"name": "physics_selective_boundary", "groups": {"physics_residual"}, "lr": lr, "steps": steps, "objective": "selective_boundary"},
        {"name": "rls_residual", "groups": {"online_residual"}, "lr": 0.0, "steps": 1, "objective": "bce", "algorithm": "rls"},
        {"name": "channel_reconstruction", "groups": {"channel_residual"}, "lr": lr, "steps": steps, "objective": "pilot_reconstruction"},
        {"name": "channel_joint", "groups": {"channel_residual"}, "lr": lr, "steps": steps, "objective": "joint"},
        {"name": "phase_trend_bce", "groups": {"phase_trend"}, "lr": lr, "steps": steps, "objective": "bce"},
        {"name": "phase_trend_signal_reconstruction", "groups": {"phase_trend"}, "lr": lr, "steps": steps, "objective": "pilot_signal_reconstruction"},
        {"name": "phase_trend_head_signal_reconstruction", "groups": {"phase_trend", "head"}, "lr": lr, "steps": steps, "objective": "pilot_signal_reconstruction"},
        {"name": "phase_trend_head_bce", "groups": {"phase_trend", "head"}, "lr": lr, "steps": steps, "objective": "bce"},
        {"name": "input_trend_bce", "groups": {"input_trend"}, "lr": lr, "steps": steps, "objective": "bce"},
        {"name": "input_affine_bce", "groups": {"input_affine"}, "lr": lr, "steps": steps, "objective": "bce"},
        {"name": "input_fir_bce", "groups": {"input_fir"}, "lr": lr, "steps": steps, "objective": "bce"},
        {"name": "logit_affine_bce", "groups": {"logit_affine"}, "lr": lr, "steps": steps, "objective": "bce"},
        {"name": "logit_fir_bce", "groups": {"logit_fir"}, "lr": lr, "steps": steps, "objective": "bce"},
        {"name": "conditioner_film_bce", "groups": {"conditioner_film"}, "lr": lr, "steps": steps, "objective": "bce"},
        {"name": "adapter_bce", "groups": {"adapter"}, "lr": lr, "steps": steps, "objective": "bce"},
        {"name": "attention_lora_bce", "groups": {"attention_lora"}, "lr": lr, "steps": steps, "objective": "bce"},
        {"name": "ffn_lora_bce", "groups": {"ffn_lora"}, "lr": lr, "steps": steps, "objective": "bce"},
        {"name": "pilot_encoder_bce", "groups": {"pilot_encoder"}, "lr": lr, "steps": steps, "objective": "bce"},
        {"name": "head_bce", "groups": {"head"}, "lr": lr, "steps": steps, "objective": "bce"},
        {"name": "head_boundary", "groups": {"head"}, "lr": lr, "steps": steps, "objective": "boundary"},
        {"name": "head_selective_boundary", "groups": {"head"}, "lr": lr, "steps": steps, "objective": "selective_boundary"},
    ]


def _split_adapt_pilot(frame, holdout_fraction: float):
    """把 Adapt Pilot 尾部留出，只允许前部 Pilot 产生梯度。"""

    fraction = float(holdout_fraction)
    if not math.isfinite(fraction) or not 0.0 <= fraction < 1.0:
        raise ValueError("adapt holdout fraction 必须位于 [0, 1)。")
    adapt_positions = torch.nonzero(frame.adapt_mask.reshape(-1), as_tuple=False).reshape(-1)
    if fraction <= 0.0 or adapt_positions.numel() < 2:
        return frame, torch.zeros_like(frame.adapt_mask, dtype=torch.bool)
    holdout_count = max(1, int(round(adapt_positions.numel() * fraction)))
    holdout_count = min(holdout_count, int(adapt_positions.numel()) - 1)
    holdout_positions = adapt_positions[-holdout_count:]
    update_mask = frame.adapt_mask.clone()
    update_mask[holdout_positions] = False
    holdout_mask = torch.zeros_like(frame.adapt_mask, dtype=torch.bool)
    holdout_mask[holdout_positions] = True
    # 更新前向只能看到 Adapt Pilot 的接收波形；留出 Pilot、Reward Pilot
    # 和 Data 的波形全部清零，避免信道重构目标通过整帧 rx_symbols 读取它们。
    update_rx = torch.where(
        update_mask,
        frame.rx_symbols,
        torch.zeros_like(frame.rx_symbols),
    )
    return replace(frame, adapt_mask=update_mask, rx_symbols=update_rx), holdout_mask


def _evaluate_candidate(
    model,
    frame,
    condition,
    soft_tail,
    candidate: dict,
    *,
    pilot_holdout_mask=None,
    pilot_update_frame=None,
    state_embedding: torch.Tensor | None = None,
    trainable_parameter_scope: str | None = None,
) -> dict:
    device = next(model.parameters()).device
    frame_device = _frame_to_device(frame, device)
    condition_device = _condition_to_device(condition, device)
    candidate_model = copy.deepcopy(model)
    baseline_model = copy.deepcopy(model)
    pilot_context_mask = frame_device.adapt_mask.bool() | frame_device.reward_mask.bool()
    pilot_frame = replace(
        frame_device,
        rx_symbols=torch.where(
            pilot_context_mask,
            frame_device.rx_symbols,
            torch.zeros_like(frame_device.rx_symbols),
        ),
    )
    before_tail = soft_tail.unsqueeze(0).to(device).to(torch.complex64)
    before = _model_logits(
        candidate_model,
        pilot_frame,
        condition_device,
        before_tail,
        state_embedding=state_embedding,
    )
    if candidate["objective"] == "pilot_signal_reconstruction":
        reconstruction_before = _pilot_signal_reconstruction_loss_for_diagnostic(
            candidate_model,
            pilot_frame,
            condition_device,
            soft_tail,
            state_embedding=state_embedding,
        )
    elif candidate["objective"] in {"pilot_reconstruction", "joint"} and candidate_model.online_channel_residual_adapter is not None:
        reconstruction_before = _pilot_reconstruction_loss_for_diagnostic(
            candidate_model, pilot_frame, condition_device, soft_tail
        )
    else:
        reconstruction_before = torch.tensor(
            _pilot_reconstruction_error(frame_device, condition_device.complex_cir, soft_tail),
            device=device,
        )
    update_frame = pilot_update_frame if pilot_update_frame is not None else frame
    update_frame_device = _frame_to_device(update_frame, device)
    if candidate.get("algorithm") == "rls":
        updater = PilotResidualRLSAdapter(
            candidate_model,
            forgetting_factor=0.99,
            ridge=1.0,
            warmup_symbols=0,
            target_logit=2.0,
            max_delta_norm=5.0,
            max_total_delta_norm=5.0,
        )
        adaptation = updater.adapt(update_frame_device, condition_device, soft_tail)
        result = {"peft_delta_norm": float(adaptation.parameter_delta_norm)}
    elif candidate["groups"]:
        result = apply_adapt_only_peft_update(
            model=candidate_model,
            frame=update_frame_device,
            condition=condition_device,
            soft_tail=soft_tail,
            groups=set(candidate["groups"]),
            lr=float(candidate["lr"]),
            steps=int(candidate["steps"]),
            objective=str(candidate["objective"]),
            state_embedding=state_embedding,
            trainable_parameter_scope=trainable_parameter_scope,
            include_evaluation_metrics=False,
        )
    else:
        result = {
        "reward_loss_improvement": 0.0,
        "data_ber_improvement": 0.0,
        "data_bce_improvement": 0.0,
        "data_logits_abs_mean_delta": 0.0,
        "data_soft_abs_mean_delta": 0.0,
        "peft_delta_norm": 0.0,
        }
    after = _model_logits(
        candidate_model,
        pilot_frame,
        condition_device,
        before_tail,
        state_embedding=state_embedding,
    )
    if candidate["objective"] == "pilot_signal_reconstruction":
        reconstruction_after = _pilot_signal_reconstruction_loss_for_diagnostic(
            candidate_model,
            pilot_frame,
            condition_device,
            soft_tail,
            state_embedding=state_embedding,
        )
    elif candidate["objective"] in {"pilot_reconstruction", "joint"} and candidate_model.online_channel_residual_adapter is not None:
        reconstruction_after = _pilot_reconstruction_loss_for_diagnostic(
            candidate_model, pilot_frame, condition_device, soft_tail
        )
    else:
        reconstruction_after = torch.tensor(
            _pilot_reconstruction_error(frame_device, condition_device.complex_cir, soft_tail),
            device=device,
        )
    update_mask = update_frame_device.adapt_mask.bool()
    adapt_before = torch.nn.functional.binary_cross_entropy_with_logits(
        before[update_mask], frame_device.bits[update_mask].float()
    )
    adapt_after = torch.nn.functional.binary_cross_entropy_with_logits(
        after[update_mask], frame_device.bits[update_mask].float()
    )
    boundary_before = _boundary_adapt_loss(
        before[update_mask], frame_device.bits[update_mask].float()
    )
    boundary_after = _boundary_adapt_loss(
        after[update_mask], frame_device.bits[update_mask].float()
    )
    selective_boundary_before = _selective_boundary_adapt_loss(
        before[update_mask], frame_device.bits[update_mask].float()
    )
    selective_boundary_after = _selective_boundary_adapt_loss(
        after[update_mask], frame_device.bits[update_mask].float()
    )
    pilot_holdout_improvement = 0.0
    if pilot_holdout_mask is not None:
        holdout_mask = torch.as_tensor(pilot_holdout_mask, device=device).bool().reshape(-1)
        if holdout_mask.numel() != before.numel() or not bool(holdout_mask.any()):
            raise ValueError("pilot_holdout_mask 必须与帧长度一致且至少包含一个符号。")
        holdout_targets = frame_device.bits[holdout_mask].float()
        holdout_before = torch.nn.functional.binary_cross_entropy_with_logits(before[holdout_mask], holdout_targets)
        holdout_after = torch.nn.functional.binary_cross_entropy_with_logits(
            after[holdout_mask], holdout_targets
        )
        pilot_holdout_improvement = float((holdout_before - holdout_after).detach().cpu())
    # Data 用于更新完成后的离线评估，不进入 Pilot 候选分数或在线验收。
    with torch.no_grad():
        data_before_logits = _model_logits(
            baseline_model,
            frame_device,
            condition_device,
            before_tail,
            state_embedding=state_embedding,
        )
        data_after_logits = _model_logits(
            candidate_model,
            frame_device,
            condition_device,
            before_tail,
            state_embedding=state_embedding,
        )
    data_mask = frame_device.data_mask.bool()
    if bool(data_mask.any()):
        data_before_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            data_before_logits[data_mask], frame_device.bits[data_mask].float()
        )
        data_after_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            data_after_logits[data_mask], frame_device.bits[data_mask].float()
        )
        data_before_ber = torch.mean(
            (
                torch.sign(data_before_logits[data_mask])
                != torch.where(frame_device.bits[data_mask] > 0.5, 1.0, -1.0)
            ).float()
        )
        data_after_ber = torch.mean(
            (
                torch.sign(data_after_logits[data_mask])
                != torch.where(frame_device.bits[data_mask] > 0.5, 1.0, -1.0)
            ).float()
        )
    else:
        data_before_bce = data_after_bce = torch.zeros((), device=device)
        data_before_ber = data_after_ber = torch.zeros((), device=device)
    result = {
        "candidate_name": candidate["name"],
        "objective": candidate["objective"],
        "updated_groups": sorted(candidate["groups"]),
        "trainable_parameter_scope": trainable_parameter_scope or "all",
        "adapt_bce_improvement": float((adapt_before - adapt_after).detach().cpu()),
        "boundary_loss_improvement": float((boundary_before - boundary_after).detach().cpu()),
        "selective_boundary_loss_improvement": float(
            (selective_boundary_before - selective_boundary_after).detach().cpu()
        ),
        "pilot_holdout_loss_improvement": float(pilot_holdout_improvement),
        "pilot_reconstruction_improvement": float((reconstruction_before - reconstruction_after).detach().cpu()),
        "pilot_reconstruction_error_after": float(reconstruction_after.detach().cpu()),
        "reward_loss_improvement": 0.0,
        "data_ber_improvement": float((data_before_ber - data_after_ber).detach().cpu()),
        "data_bce_improvement": float((data_before_bce - data_after_bce).detach().cpu()),
        "data_logits_abs_mean_delta": float(
            torch.mean(torch.abs(data_after_logits[data_mask] - data_before_logits[data_mask])).detach().cpu()
            if bool(data_mask.any())
            else 0.0
        ),
        "data_logits_direction_change_fraction": float(
            torch.mean(
                (torch.sign(data_before_logits[data_mask]) != torch.sign(data_after_logits[data_mask])).float()
            ).detach().cpu()
            if bool(data_mask.any())
            else 0.0
        ),
        "data_soft_abs_mean_delta": float(
            torch.mean(
                torch.abs(torch.sigmoid(data_after_logits[data_mask]) - torch.sigmoid(data_before_logits[data_mask]))
            ).detach().cpu()
            if bool(data_mask.any())
            else 0.0
        ),
        "peft_delta_norm": float(result.get("peft_delta_norm", 0.0)),
        "data_labels_used_online": False,
        "data_waveform_used_online": False,
        "online_update_uses_data_waveform": False,
        "diagnostic_uses_data_labels": True,
    }
    if bool(frame_device.reward_mask.any()):
        result["reward_loss_improvement"] = float(
            (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    before[frame_device.reward_mask],
                    frame_device.bits[frame_device.reward_mask].float(),
                )
                - torch.nn.functional.binary_cross_entropy_with_logits(
                    after[frame_device.reward_mask],
                    frame_device.bits[frame_device.reward_mask].float(),
                )
            ).detach().cpu()
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--snrs", nargs="+", type=float, default=[5.0, 10.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--delay", type=int, default=116)
    parser.add_argument(
        "--pilot-total",
        nargs="+",
        type=int,
        default=[256],
        help="Pilot 总长度；可一次提供多个长度用于诊断扫描。",
    )
    parser.add_argument("--gap-seconds", type=float, default=None)
    parser.add_argument("--condition-source", choices=["acquisition", "pilot_phase", "pilot_cir_phase"],
                        default=None, help="继承配置；acquisition 只作独立消融。")
    parser.add_argument("--reward-pilot-total", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument(
        "--adapt-holdout-fraction",
        type=float,
        default=0.0,
        help="从 Adapt Pilot 尾部留出一部分，只用于事后 Pilot 留出诊断。",
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=None,
        help="只运行指定候选；省略时运行全部作用位置和监督目标。",
    )
    parser.add_argument("--state-conditioned", action="store_true", help="把 Adapt Pilot 状态 embedding 传入 phase_trend Adapter。")
    parser.add_argument(
        "--state-mapping-only",
        action="store_true",
        help="只更新 phase_trend 的 state_to_phase_cfo 映射参数。",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not math.isfinite(args.adapt_holdout_fraction) or not 0.0 <= args.adapt_holdout_fraction < 1.0:
        parser.error("--adapt-holdout-fraction 必须位于 [0, 1)。")
    all_candidates = _candidate_specs(args.lr, args.steps)
    if args.candidates is None:
        selected_candidates = all_candidates
    else:
        known = {str(item["name"]): item for item in all_candidates}
        unknown = sorted(set(args.candidates) - set(known))
        if unknown:
            raise ValueError(f"未知候选：{unknown}；可选候选为 {sorted(known)}")
        selected_candidates = [known[name] for name in args.candidates]
    if args.state_mapping_only:
        if not args.state_conditioned:
            parser.error("--state-mapping-only 必须与 --state-conditioned 一起使用。")
        incompatible = [
            candidate["name"]
            for candidate in selected_candidates
            if candidate["groups"] != {"phase_trend"}
            or candidate["objective"] != "pilot_signal_reconstruction"
        ]
        if incompatible:
            parser.error(
                "--state-mapping-only 仅支持 phase_trend 的 pilot_signal_reconstruction 候选；"
                f"不兼容候选：{incompatible}。"
            )
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model_config = _load_model_config(config, Path(args.pretrained))
    rows = []
    for pilot_total in args.pilot_total:
        for snr in args.snrs:
            for seed in args.seeds:
                episode = build_pilot_replay_episode(
                    config, model_config, snr=float(snr), seed=int(seed), delay=args.delay,
                    pilot_total=int(pilot_total), reward_pilot_total=args.reward_pilot_total,
                    gap_seconds=args.gap_seconds, device=args.device, condition_source=args.condition_source,
                )
                start, cir = episode.start, episode.cir
                frames = [episode.env.next_frame() for _ in range(args.frames)]
                for frame_index, frame in enumerate(frames):
                    model = _build_equalizer(model_config, Path(args.pretrained), args.device)
                    model.attach_online_physics_residual_adapter()
                    model.attach_online_channel_residual_adapter()
                    model.attach_online_phase_trend_adapter()
                    model.attach_online_input_trend_adapter()
                    model.attach_online_input_affine_adapter()
                    model.attach_online_input_fir_adapter()
                    model.attach_online_logit_affine_adapter()
                    model.attach_online_logit_fir_adapter()
                    tail = start.initial_soft_tail.to(args.device)
                    condition = episode.condition(_frame_to_device(frame, args.device), tail)
                    update_frame, holdout_mask = _split_adapt_pilot(
                        _frame_to_device(frame, args.device), args.adapt_holdout_fraction
                    )
                    state = _pilot_state(frame, cir, tail, condition)
                    state_embedding = None
                    if args.state_conditioned:
                        embedding_cir = pilot_sparse_cir_update(
                            _frame_to_device(frame, args.device),
                            cir,
                            tail.to(args.device),
                            max_paths=24,
                            alpha=1.0,
                            cfo_hint=float(state["cfo_cycles_per_symbol"]),
                        ).to(args.device)
                        state_summary = build_pilot_state_summary(
                            cir=embedding_cir,
                            reference_cir=cir,
                            phase0=float(state["phase0"]),
                            cfo_cycles_per_symbol=float(state["cfo_cycles_per_symbol"]),
                            noise_variance=condition.noise_variance,
                            confidence=float(state["confidence"]),
                            reconstruction_error=float(state["reconstruction_error"]),
                        )
                        state_embedding = PilotStateEmbedding()(state_summary).to(args.device)
                    base = {
                        **episode.metadata(),
                        "tail_mode": "fixed_initial_tail_diagnostic",
                        "pilot_total": int(pilot_total),
                        "state_conditioned": bool(args.state_conditioned),
                        "snr_db": float(snr), "seed": int(seed), "frame_index": int(frame_index),
                        **state, "data_labels_used_online": False,
                        "adapt_holdout_fraction": float(args.adapt_holdout_fraction),
                        "adapt_update_pilot_count": int(update_frame.adapt_mask.sum().item()),
                        "adapt_holdout_pilot_count": int(holdout_mask.sum().item()),
                        "data_waveform_used_online": False,
                        "online_update_uses_data_waveform": False,
                    }
                    for candidate in selected_candidates:
                        rows.append({
                            **base,
                            **_evaluate_candidate(
                                model,
                                frame,
                                condition,
                                tail,
                                candidate,
                                pilot_holdout_mask=holdout_mask,
                                pilot_update_frame=update_frame,
                                state_embedding=state_embedding,
                                trainable_parameter_scope=(
                                    "state_mapping" if args.state_mapping_only else None
                                ),
                            ),
                        })
    summary = []
    for score_name in (
        "adapt_bce_improvement",
        "pilot_holdout_loss_improvement",
        "boundary_loss_improvement",
        "selective_boundary_loss_improvement",
        "reward_loss_improvement",
        "pilot_reconstruction_improvement",
    ):
        by_trial = {}
        for row in rows:
            key = (row["pilot_total"], row["snr_db"], row["seed"], row["frame_index"])
            by_trial.setdefault(key, []).append(row)
        scores, targets = [], []
        for values in by_trial.values():
            values = [item for item in values if item["candidate_name"] != "identity"]
            if len(values) < 2:
                continue
            scores.extend(float(item[score_name]) for item in values)
            targets.extend(float(item["data_ber_improvement"]) for item in values)
        result = spearman_reward_data(scores, targets, threshold=0.6)
        summary.append({"score": score_name, "spearman": float(result.correlation), "n": int(result.n), "gate_pass": bool(result.passed)})
    payload = {
        "metric": "pilot_candidate_ranking_replay",
        "checkpoint": str(Path(args.pretrained).resolve()),
        "model_config": model_config.to_dict(),
        "experiment_config": config,
        "arguments": vars(args),
        "main_aggregation_allowed": False,
        "pilot_totals": [int(value) for value in args.pilot_total],
        "rows": rows,
        "summary": summary,
        "data_labels_used_online": False,
        "diagnostic_uses_data_labels": True,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    print(f"saved {target}")


if __name__ == "__main__":
    main()
