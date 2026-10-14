# -*- coding: utf-8 -*-
"""连续 Pilot 状态下的 phase_trend 参数作用域 replay。"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.cir_estimator import condition_from_cir, pilot_sparse_cir_update
from agent.pilot_state import PilotStateEmbedding, build_pilot_state_summary
from env.comm_env import CommEnvConfig, CommunicationEnvironment
from evaluation.research_diagnostics import (
    _frame_to_device,
    _model_logits,
    _pilot_signal_reconstruction_loss_for_diagnostic,
    apply_adapt_only_peft_update,
)
from scripts.replay_pilot_candidate_ranking import _pilot_state
from training.meta_training import _estimate_cir_from_known_frame
from training.rl_modulated_online import _build_equalizer, _load_model_config


def _ber(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return 0.0
    decisions = torch.sign(logits[mask])
    labels = torch.where(targets[mask] > 0.5, 1.0, -1.0)
    return float(torch.mean((decisions != labels).float()).detach().cpu())


def _bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return 0.0
    return float(
        torch.nn.functional.binary_cross_entropy_with_logits(
            logits[mask], targets[mask].float()
        ).detach().cpu()
    )


def _make_model(model_config: dict, checkpoint: Path, device: str):
    model = _build_equalizer(model_config, checkpoint, device)
    model.attach_online_phase_trend_adapter()
    return model


def _evaluate_scope(
    *,
    frames: list,
    cir: torch.Tensor,
    soft_tail: torch.Tensor,
    snr: float,
    seed: int,
    pilot_total: int,
    gap_seconds: float,
    reward_pilot_total: int,
    model_config: dict,
    checkpoint: Path,
    device: str,
    lr: float,
    steps: int,
    scope: str,
    reward_gate: bool,
) -> list[dict]:
    model = _make_model(model_config, checkpoint, device)
    frozen_model = copy.deepcopy(model).eval()
    condition = condition_from_cir(cir, float(snr))
    tail_batch = soft_tail.to(device).reshape(1, -1).to(torch.complex64)
    tail = soft_tail.to(device)
    rows = []

    for frame_index, frame in enumerate(frames):
        frame_device = _frame_to_device(frame, device)
        state = _pilot_state(frame, cir, tail, condition)
        embedding_cir = pilot_sparse_cir_update(
            frame_device,
            cir.to(device),
            tail,
            max_paths=24,
            alpha=1.0,
            cfo_hint=float(state["cfo_cycles_per_symbol"]),
        ).to(device)
        state_summary = build_pilot_state_summary(
            cir=embedding_cir,
            reference_cir=cir.to(device),
            phase0=float(state["phase0"]),
            cfo_cycles_per_symbol=float(state["cfo_cycles_per_symbol"]),
            noise_variance=condition.noise_variance,
            confidence=float(state["confidence"]),
            reconstruction_error=float(state["reconstruction_error"]),
        )
        embedding = PilotStateEmbedding()(state_summary).to(device)

        before = _model_logits(
            model, frame_device, condition, tail_batch, state_embedding=embedding
        )
        pilot_reconstruction_before = _pilot_signal_reconstruction_loss_for_diagnostic(
            model,
            frame_device,
            condition,
            tail,
            state_embedding=embedding,
        )
        frozen = _model_logits(
            frozen_model, frame_device, condition, tail_batch, state_embedding=embedding
        )
        adapter = model.online_phase_trend_adapter
        direct_before = torch.stack(
            (adapter.raw_phase.detach().clone(), adapter.raw_cfo.detach().clone())
        )
        adapter_state_before = copy.deepcopy(adapter.state_dict())
        mapping_before = {
            name: parameter.detach().clone()
            for name, parameter in adapter.state_to_phase_cfo.named_parameters()
        }

        update = apply_adapt_only_peft_update(
            model=model,
            frame=frame,
            condition=condition,
            soft_tail=tail,
            groups={"phase_trend"},
            lr=float(lr),
            steps=int(steps),
            objective="pilot_signal_reconstruction",
            state_embedding=embedding,
            trainable_parameter_scope=scope,
        )
        data_mask = frame_device.data_mask.bool()
        proposed_after = _model_logits(
            model, frame_device, condition, tail_batch, state_embedding=embedding
        )
        proposed_direct = torch.stack(
            (adapter.raw_phase.detach(), adapter.raw_cfo.detach())
        )
        proposed_direct_delta_norm = float(
            torch.linalg.vector_norm(proposed_direct - direct_before).detach().cpu()
        )
        proposed_mapping_delta_sq = torch.zeros((), device=device)
        for name, parameter in adapter.state_to_phase_cfo.named_parameters():
            proposed_mapping_delta_sq += (
                parameter.detach() - mapping_before[name]
            ).float().pow(2).sum()
        proposed_mapping_delta_norm = float(
            torch.sqrt(proposed_mapping_delta_sq).detach().cpu()
        )
        reward_gate_accepted = bool(
            not reward_gate or update["reward_loss_improvement"] > 0.0
        )
        if not reward_gate_accepted:
            adapter.load_state_dict(adapter_state_before)
        after = _model_logits(
            model, frame_device, condition, tail_batch, state_embedding=embedding
        )
        pilot_reconstruction_after = _pilot_signal_reconstruction_loss_for_diagnostic(
            model,
            frame_device,
            condition,
            tail,
            state_embedding=embedding,
        )
        direct_after = torch.stack(
            (adapter.raw_phase.detach(), adapter.raw_cfo.detach())
        )
        direct_delta_norm = float(
            torch.linalg.vector_norm(direct_after - direct_before).detach().cpu()
        )
        mapping_delta_sq = torch.zeros((), device=device)
        for name, parameter in adapter.state_to_phase_cfo.named_parameters():
            mapping_delta_sq += (parameter.detach() - mapping_before[name]).float().pow(2).sum()
        mapping_delta_norm = float(torch.sqrt(mapping_delta_sq).detach().cpu())

        rows.append(
            {
                "scope": scope,
                "snr_db": float(snr),
                "seed": int(seed),
                "frame_index": int(frame_index),
                "pilot_total": int(pilot_total),
                "reward_pilot_total": int(reward_pilot_total),
                "gap_seconds": float(gap_seconds),
                "phase0": float(state["phase0"]),
                "cfo_cycles_per_symbol": float(state["cfo_cycles_per_symbol"]),
                "state_embedding_norm": float(torch.linalg.vector_norm(embedding).cpu()),
                "pilot_signal_reconstruction_improvement": float(
                    (pilot_reconstruction_before - pilot_reconstruction_after)
                    .detach()
                    .cpu()
                ),
                "reward_loss_improvement": float(update["reward_loss_improvement"]),
                "reward_gate_accepted": reward_gate_accepted,
                "reward_loss_improvement_after_gate": float(
                    update["reward_loss_improvement"] if reward_gate_accepted else 0.0
                ),
                "candidate_data_ber_improvement_vs_frozen": _ber(
                    frozen, frame_device.bits, data_mask
                )
                - _ber(proposed_after, frame_device.bits, data_mask),
                "candidate_data_bce_improvement_vs_frozen": _bce(
                    frozen, frame_device.bits, data_mask
                )
                - _bce(proposed_after, frame_device.bits, data_mask),
                "data_ber_before_update": _ber(before, frame_device.bits, data_mask),
                "data_ber_after_update": _ber(after, frame_device.bits, data_mask),
                "data_ber_frozen": _ber(frozen, frame_device.bits, data_mask),
                "data_ber_improvement_vs_frozen": _ber(
                    frozen, frame_device.bits, data_mask
                )
                - _ber(after, frame_device.bits, data_mask),
                "data_bce_before_update": _bce(before, frame_device.bits, data_mask),
                "data_bce_after_update": _bce(after, frame_device.bits, data_mask),
                "data_bce_frozen": _bce(frozen, frame_device.bits, data_mask),
                "data_bce_improvement_vs_frozen": _bce(
                    frozen, frame_device.bits, data_mask
                )
                - _bce(after, frame_device.bits, data_mask),
                "data_logits_abs_mean_delta_this_update": float(
                    torch.mean(torch.abs(after[data_mask] - before[data_mask]))
                    .detach()
                    .cpu()
                    if bool(data_mask.any())
                    else 0.0
                ),
                "data_logits_direction_change_fraction": float(
                    torch.mean(
                        (torch.sign(before[data_mask]) != torch.sign(after[data_mask])).float()
                    )
                    .detach()
                    .cpu()
                    if bool(data_mask.any())
                    else 0.0
                ),
                "data_soft_abs_mean_delta": float(
                    torch.mean(
                        torch.abs(torch.sigmoid(after[data_mask]) - torch.sigmoid(before[data_mask]))
                    )
                    .detach()
                    .cpu()
                    if bool(data_mask.any())
                    else 0.0
                ),
                "direct_phase_cfo_delta_norm_this_update": direct_delta_norm,
                "state_mapping_delta_norm_this_update": mapping_delta_norm,
                "proposed_direct_phase_cfo_delta_norm": proposed_direct_delta_norm,
                "proposed_state_mapping_delta_norm": proposed_mapping_delta_norm,
                "peft_delta_norm_this_update": float(update["peft_delta_norm"]),
                "diagnostic_uses_data_labels": True,
                "data_labels_used_online": False,
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--snrs", nargs="+", type=float, default=[5.0, 10.0, 15.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--delay", type=int, default=116)
    parser.add_argument("--pilot-total", type=int, default=256)
    parser.add_argument("--gap-seconds", type=float, default=600.0)
    parser.add_argument("--reward-pilot-total", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument(
        "--reward-gate",
        action="store_true",
        help="仅当 Reward Pilot BCE 严格改善时提交候选更新，否则回滚。",
    )
    parser.add_argument(
        "--scopes",
        nargs="+",
        choices=["all", "state_mapping"],
        default=["all", "state_mapping"],
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = Path(args.pretrained)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model_config = _load_model_config(config, checkpoint)
    rows = []
    for snr in args.snrs:
        for seed in args.seeds:
            env = CommunicationEnvironment(
                CommEnvConfig(
                    level="B",
                    max_delay=int(args.delay),
                    snr_db=float(snr),
                    total_pilot=int(args.pilot_total),
                    layout="prefix",
                    seed=90_000 + int(seed),
                    acquisition_to_data_gap_seconds=float(args.gap_seconds),
                    reward_pilot_total=int(args.reward_pilot_total),
                )
            )
            start = env.reset_episode()
            cir = _estimate_cir_from_known_frame(start.acquisition, int(args.delay)).to(args.device)
            frames = [env.next_frame() for _ in range(int(args.frames))]
            for scope in args.scopes:
                rows.extend(
                    _evaluate_scope(
                        frames=frames,
                        cir=cir,
                        soft_tail=start.initial_soft_tail.to(args.device),
                        snr=float(snr),
                        seed=int(seed),
                        pilot_total=int(args.pilot_total),
                        gap_seconds=float(args.gap_seconds),
                        reward_pilot_total=int(args.reward_pilot_total),
                        model_config=model_config,
                        checkpoint=checkpoint,
                        device=args.device,
                        lr=float(args.lr),
                        steps=int(args.steps),
                        scope=str(scope),
                        reward_gate=bool(args.reward_gate),
                    )
                )

    summaries = []
    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["scope"]), float(row["snr_db"]))].append(row)
    for (scope, snr), values in sorted(grouped.items()):
        summaries.append(
            {
                "scope": scope,
                "snr_db": snr,
                "n_seed_frames": len(values),
                "mean_data_ber_improvement_vs_frozen": sum(
                    row["data_ber_improvement_vs_frozen"] for row in values
                )
                / len(values),
                "positive_seed_count": sum(
                    sum(row["data_ber_improvement_vs_frozen"] for row in values if row["seed"] == seed)
                    > 0.0
                    for seed in sorted({row["seed"] for row in values})
                ),
                "seed_count": len({row["seed"] for row in values}),
                "mean_data_bce_improvement_vs_frozen": sum(
                    row["data_bce_improvement_vs_frozen"] for row in values
                )
                / len(values),
                "reward_acceptance_rate": sum(
                    bool(row["reward_gate_accepted"]) for row in values
                )
                / len(values),
                "mean_candidate_data_ber_improvement_before_gate": sum(
                    row["candidate_data_ber_improvement_vs_frozen"] for row in values
                )
                / len(values),
                "mean_candidate_data_bce_improvement_before_gate": sum(
                    row["candidate_data_bce_improvement_vs_frozen"] for row in values
                )
                / len(values),
                "mean_logits_delta_per_update": sum(
                    row["data_logits_abs_mean_delta_this_update"] for row in values
                )
                / len(values),
                "mean_direct_phase_cfo_delta_per_update": sum(
                    row["direct_phase_cfo_delta_norm_this_update"] for row in values
                )
                / len(values),
                "mean_state_mapping_delta_per_update": sum(
                    row["state_mapping_delta_norm_this_update"] for row in values
                )
                / len(values),
            }
        )

    payload = {
        "metric": "persistent_phase_state_mapping_sequence_replay",
        "checkpoint": str(checkpoint),
        "channel_level": "B",
        "delay": int(args.delay),
        "pilot_layout": "prefix",
        "pilot_total": int(args.pilot_total),
        "reward_pilot_total": int(args.reward_pilot_total),
        "gap_seconds": float(args.gap_seconds),
        "data_labels_used_online": False,
        "diagnostic_uses_data_labels": True,
        "reward_gate_enabled": bool(args.reward_gate),
        "rows": rows,
        "summary": summaries,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
