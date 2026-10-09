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
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.cir_estimator import condition_from_cir
from baseline.traditional_equalizers import estimate_phase_residual_vector
from env.comm_env import CommEnvConfig, CommunicationEnvironment
from env.linear_operator import LinearChannelOperator
from evaluation.metrics import spearman_reward_data
from evaluation.research_diagnostics import (
    _condition_to_device,
    _frame_to_device,
    _model_logits,
    _pilot_reconstruction_loss_for_diagnostic,
    apply_adapt_only_peft_update,
)
from training.meta_training import _estimate_cir_from_known_frame
from training.online_adaptation import PilotResidualRLSAdapter
from training.rl_modulated_online import _build_equalizer, _load_model_config


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
        {"name": "rls_residual", "groups": {"online_residual"}, "lr": 0.0, "steps": 1, "objective": "bce", "algorithm": "rls"},
        {"name": "channel_reconstruction", "groups": {"channel_residual"}, "lr": lr, "steps": steps, "objective": "pilot_reconstruction"},
        {"name": "channel_joint", "groups": {"channel_residual"}, "lr": lr, "steps": steps, "objective": "joint"},
        {"name": "head_bce", "groups": {"head"}, "lr": lr, "steps": steps, "objective": "bce"},
    ]


def _evaluate_candidate(model, frame, condition, soft_tail, candidate: dict) -> dict:
    device = next(model.parameters()).device
    frame_device = _frame_to_device(frame, device)
    condition_device = _condition_to_device(condition, device)
    candidate_model = copy.deepcopy(model)
    before_tail = soft_tail.unsqueeze(0).to(device).to(torch.complex64)
    before = _model_logits(candidate_model, frame_device, condition_device, before_tail)
    if candidate_model.online_channel_residual_adapter is not None:
        reconstruction_before = _pilot_reconstruction_loss_for_diagnostic(
            candidate_model, frame_device, condition_device, soft_tail
        )
    else:
        reconstruction_before = torch.tensor(
            _pilot_reconstruction_error(frame_device, condition_device.complex_cir, soft_tail),
            device=device,
        )
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
        adaptation = updater.adapt(frame_device, condition_device, soft_tail)
        result = {"peft_delta_norm": float(adaptation.parameter_delta_norm)}
    elif candidate["groups"]:
        result = apply_adapt_only_peft_update(
            model=candidate_model,
            frame=frame,
            condition=condition_device,
            soft_tail=soft_tail,
            groups=set(candidate["groups"]),
            lr=float(candidate["lr"]),
            steps=int(candidate["steps"]),
            objective=str(candidate["objective"]),
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
    after = _model_logits(candidate_model, frame_device, condition_device, before_tail)
    if candidate_model.online_channel_residual_adapter is not None:
        reconstruction_after = _pilot_reconstruction_loss_for_diagnostic(
            candidate_model, frame_device, condition_device, soft_tail
        )
    else:
        reconstruction_after = torch.tensor(
            _pilot_reconstruction_error(frame_device, condition_device.complex_cir, soft_tail),
            device=device,
        )
    adapt_before = torch.nn.functional.binary_cross_entropy_with_logits(
        before[frame_device.adapt_mask], frame_device.bits[frame_device.adapt_mask].float()
    )
    adapt_after = torch.nn.functional.binary_cross_entropy_with_logits(
        after[frame_device.adapt_mask], frame_device.bits[frame_device.adapt_mask].float()
    )
    data_mask = frame_device.data_mask.bool()
    if bool(data_mask.any()):
        data_before_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            before[data_mask], frame_device.bits[data_mask].float()
        )
        data_after_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            after[data_mask], frame_device.bits[data_mask].float()
        )
        data_before_ber = torch.mean(
            (torch.sign(before[data_mask]) != torch.where(frame_device.bits[data_mask] > 0.5, 1.0, -1.0)).float()
        )
        data_after_ber = torch.mean(
            (torch.sign(after[data_mask]) != torch.where(frame_device.bits[data_mask] > 0.5, 1.0, -1.0)).float()
        )
    else:
        data_before_bce = data_after_bce = torch.zeros((), device=device)
        data_before_ber = data_after_ber = torch.zeros((), device=device)
    return {
        "candidate_name": candidate["name"],
        "objective": candidate["objective"],
        "updated_groups": sorted(candidate["groups"]),
        "adapt_bce_improvement": float((adapt_before - adapt_after).detach().cpu()),
        "pilot_reconstruction_improvement": float((reconstruction_before - reconstruction_after).detach().cpu()),
        "pilot_reconstruction_error_after": float(reconstruction_after.detach().cpu()),
        "reward_loss_improvement": float(result.get("reward_loss_improvement", 0.0)),
        "data_ber_improvement": float((data_before_ber - data_after_ber).detach().cpu()),
        "data_bce_improvement": float((data_before_bce - data_after_bce).detach().cpu()),
        "data_logits_abs_mean_delta": float(result.get("data_logits_abs_mean_delta", 0.0)),
        "data_soft_abs_mean_delta": float(result.get("data_soft_abs_mean_delta", 0.0)),
        "peft_delta_norm": float(result.get("peft_delta_norm", 0.0)),
        "data_labels_used_online": False,
        "diagnostic_uses_data_labels": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--snrs", nargs="+", type=float, default=[5.0, 10.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--delay", type=int, default=116)
    parser.add_argument("--pilot-total", type=int, default=256)
    parser.add_argument("--gap-seconds", type=float, default=120.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model_config = _load_model_config(config, Path(args.pretrained))
    rows = []
    for snr in args.snrs:
        for seed in args.seeds:
            env = CommunicationEnvironment(CommEnvConfig(
                level="B", max_delay=args.delay, snr_db=float(snr), total_pilot=args.pilot_total,
                layout="prefix", seed=90_000 + int(seed), acquisition_to_data_gap_seconds=float(args.gap_seconds),
            ))
            start = env.reset_episode()
            cir = _estimate_cir_from_known_frame(start.acquisition, args.delay).to(args.device)
            frames = [env.next_frame() for _ in range(args.frames)]
            for frame_index, frame in enumerate(frames):
                model = _build_equalizer(model_config, Path(args.pretrained), args.device)
                model.attach_online_physics_residual_adapter()
                model.attach_online_channel_residual_adapter()
                tail = start.initial_soft_tail.to(args.device)
                condition = condition_from_cir(cir, float(snr))
                state = _pilot_state(frame, cir, tail, condition)
                base = {
                    "snr_db": float(snr), "seed": int(seed), "frame_index": int(frame_index),
                    **state, "data_labels_used_online": False,
                }
                for candidate in _candidate_specs(args.lr, args.steps):
                    rows.append({**base, **_evaluate_candidate(model, frame, condition, tail, candidate)})
    summary = []
    for score_name in ("adapt_bce_improvement", "reward_loss_improvement", "pilot_reconstruction_improvement", "data_bce_improvement"):
        by_trial = {}
        for row in rows:
            key = (row["snr_db"], row["seed"], row["frame_index"])
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
    payload = {"metric": "pilot_candidate_ranking_replay", "rows": rows, "summary": summary, "data_labels_used_online": False, "diagnostic_uses_data_labels": True}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    print(f"saved {target}")


if __name__ == "__main__":
    main()
