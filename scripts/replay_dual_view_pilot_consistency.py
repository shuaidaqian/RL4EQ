# -*- coding: utf-8 -*-
"""用 Adapt Pilot 的两个子段检查 PEFT 更新方向是否一致。

本脚本是离线诊断 replay，不改变在线实现和离线 checkpoint。每个候选更新只使用
Adapt Pilot 的观测和已知 Pilot 符号；Data 标签只在最后计算事后 BER/BCE，用于判断
一致性门控有没有减少错误方向的更新。
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.cir_estimator import condition_from_cir
from baseline.traditional_equalizers import estimate_phase_residual_features
from env.comm_env import CommEnvConfig, CommunicationEnvironment
from evaluation.research_diagnostics import (
    _condition_to_device,
    _frame_to_device,
    _model_logits,
    _retag_unfolded_peft_groups,
    apply_adapt_only_peft_update,
)
from training.meta_training import _estimate_cir_from_known_frame
from training.rl_modulated_online import _build_equalizer, _load_model_config


def _selected_parameter_names(model, groups: set[str]) -> list[str]:
    resolved = model.peft.resolve(set(groups))
    return [
        name
        for name, parameter in model.named_parameters()
        if getattr(parameter, "_peft_group", None) in resolved
    ]


def _parameter_vector(model, names: list[str]) -> torch.Tensor:
    named = dict(model.named_parameters())
    if not names:
        return torch.zeros(1, device=next(model.parameters()).device)
    return torch.cat([named[name].detach().float().reshape(-1) for name in names])


def _split_adapt_frame(frame):
    """把同一帧的 Adapt Pilot 拆成前后两个互不重叠的视图。"""

    indices = torch.nonzero(frame.adapt_mask, as_tuple=False).reshape(-1)
    if indices.numel() < 4:
        raise ValueError("Adapt Pilot 太短，无法拆成两个子段。")
    midpoint = indices.numel() // 2
    masks = []
    for selected in (indices[:midpoint], indices[midpoint:]):
        mask = torch.zeros_like(frame.adapt_mask, dtype=torch.bool)
        mask[selected] = True
        masks.append(mask)
    return [dataclasses.replace(frame, adapt_mask=mask) for mask in masks]


def _apply_preview(model, frame, condition, tail, *, lr: float, steps: int) -> torch.Tensor:
    _retag_unfolded_peft_groups(model)
    names = _selected_parameter_names(model, {"phase_trend"})
    before = _parameter_vector(model, names)
    apply_adapt_only_peft_update(
        model=model,
        frame=frame,
        condition=condition,
        soft_tail=tail,
        groups={"phase_trend"},
        lr=lr,
        steps=steps,
        objective="pilot_signal_reconstruction",
    )
    return _parameter_vector(model, names) - before


def _frame_metrics(model, frame, condition, tail) -> dict[str, float]:
    device = next(model.parameters()).device
    frame_device = _frame_to_device(frame, device)
    condition_device = _condition_to_device(condition, device)
    with torch.no_grad():
        logits = _model_logits(
            model,
            frame_device,
            condition_device,
            tail.to(device).unsqueeze(0).to(torch.complex64),
        )
    data_mask = frame_device.data_mask.bool()
    before_bits = frame_device.bits[data_mask].float()
    data_logits = logits[data_mask]
    return {
        "data_bce": float(F.binary_cross_entropy_with_logits(data_logits, before_bits).cpu()),
        "data_ber": float(
            ((data_logits >= 0.0) != (before_bits >= 0.5)).float().mean().cpu()
        ),
        "data_logits_abs_mean": float(data_logits.abs().mean().cpu()),
    }


def _set_phase_parameters_from_pilot(model, frame, cir, tail, *, smoothing: float = 1.0) -> dict[str, float]:
    """把 Adapt Pilot 估计的相位/CFO 写入 phase Adapter 参数。"""

    phase0, cfo = estimate_phase_residual_features(frame.receiver_view(), cir, tail)
    adapter = model.online_phase_trend_adapter
    if adapter is None:
        raise RuntimeError("phase_trend Adapter 尚未挂载。")
    max_phase = float(adapter.max_phase)
    max_cfo = float(adapter.max_cfo)
    phase0 = max(-0.999 * max_phase, min(0.999 * max_phase, float(phase0)))
    cfo = max(-0.999 * max_cfo, min(0.999 * max_cfo, float(cfo)))
    with torch.no_grad():
        adapter.raw_phase.copy_(
            torch.atanh(torch.tensor(phase0 / max_phase, device=adapter.raw_phase.device))
            * float(smoothing)
        )
        adapter.raw_cfo.copy_(
            torch.atanh(torch.tensor(cfo / max_cfo, device=adapter.raw_cfo.device))
            * float(smoothing)
        )
    return {"pilot_phase0": float(phase0), "pilot_cfo": float(cfo)}


def _run_trial(
    *,
    model_config,
    pretrained: Path,
    device: str,
    snr: float,
    seed: int,
    frames_count: int,
    delay: int,
    pilot_total: int,
    gap_seconds: float,
    lr: float,
    steps: int,
    cosine_threshold: float,
    analytic_smoothing: float,
) -> list[dict]:
    env = CommunicationEnvironment(
        CommEnvConfig(
            level="B",
            max_delay=delay,
            snr_db=float(snr),
            total_pilot=pilot_total,
            layout="prefix",
            seed=90_000 + int(seed),
            acquisition_to_data_gap_seconds=float(gap_seconds),
        )
    )
    acquisition = env.reset_episode()
    cir = _estimate_cir_from_known_frame(acquisition.acquisition, delay).to(device)
    condition = condition_from_cir(cir, float(snr))
    frames = [env.next_frame() for _ in range(frames_count)]
    rows: list[dict] = []
    for frame_index, frame in enumerate(frames):
        base_model = _build_equalizer(model_config, pretrained, device)
        base_model.attach_online_phase_trend_adapter()
        tail = acquisition.initial_soft_tail.to(device)
        base_metrics = _frame_metrics(base_model, frame, condition, tail)
        half_frames = _split_adapt_frame(frame)

        previews = []
        for half_index, half_frame in enumerate(half_frames):
            preview_model = copy.deepcopy(base_model)
            delta = _apply_preview(
                preview_model,
                half_frame,
                condition,
                tail,
                lr=lr,
                steps=steps,
            )
            previews.append(delta)
        first, second = previews
        cosine = float(
            torch.dot(first, second)
            / (torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)).clamp_min(1e-12)
        )
        norm_first = float(torch.linalg.vector_norm(first).cpu())
        norm_second = float(torch.linalg.vector_norm(second).cpu())
        consistent = bool(
            cosine >= float(cosine_threshold)
            and norm_first > 1e-8
            and norm_second > 1e-8
        )

        ungated_model = copy.deepcopy(base_model)
        _apply_preview(
            ungated_model,
            frame,
            condition,
            tail,
            lr=lr,
            steps=steps,
        )
        ungated = _frame_metrics(ungated_model, frame, condition, tail)

        analytic_model = copy.deepcopy(base_model)
        analytic_state = _set_phase_parameters_from_pilot(
            analytic_model,
            frame,
            cir,
            tail,
            smoothing=analytic_smoothing,
        )
        analytic = _frame_metrics(analytic_model, frame, condition, tail)

        gated_model = copy.deepcopy(base_model)
        if consistent:
            _apply_preview(
                gated_model,
                frame,
                condition,
                tail,
                lr=lr,
                steps=steps,
            )
        gated = _frame_metrics(gated_model, frame, condition, tail)
        rows.append(
            {
                "snr_db": float(snr),
                "seed": int(seed),
                "frame_index": int(frame_index),
                "gap_seconds": float(gap_seconds),
                "cosine_similarity": cosine,
                "first_half_delta_norm": norm_first,
                "second_half_delta_norm": norm_second,
                "consistency_threshold": float(cosine_threshold),
                "consistent": consistent,
                "ungated_update_applied": True,
                "gated_update_applied": consistent,
                "frozen_data_bce": base_metrics["data_bce"],
                "frozen_data_ber": base_metrics["data_ber"],
                "ungated_data_bce": ungated["data_bce"],
                "ungated_data_ber": ungated["data_ber"],
                "gated_data_bce": gated["data_bce"],
                "gated_data_ber": gated["data_ber"],
                "analytic_data_bce": analytic["data_bce"],
                "analytic_data_ber": analytic["data_ber"],
                "ungated_data_bce_improvement": base_metrics["data_bce"] - ungated["data_bce"],
                "ungated_data_ber_improvement": base_metrics["data_ber"] - ungated["data_ber"],
                "gated_data_bce_improvement": base_metrics["data_bce"] - gated["data_bce"],
                "gated_data_ber_improvement": base_metrics["data_ber"] - gated["data_ber"],
                "analytic_data_bce_improvement": base_metrics["data_bce"] - analytic["data_bce"],
                "analytic_data_ber_improvement": base_metrics["data_ber"] - analytic["data_ber"],
                **analytic_state,
                "analytic_smoothing": float(analytic_smoothing),
                "data_labels_used_online": False,
                "diagnostic_uses_data_labels": True,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--snrs", nargs="+", type=float, default=[5.0, 10.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--delay", type=int, default=116)
    parser.add_argument("--pilot-total", type=int, default=256)
    parser.add_argument("--gap-seconds", type=float, default=120.0)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--cosine-threshold", type=float, default=0.0)
    parser.add_argument("--analytic-smoothing", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model_config = _load_model_config(config, Path(args.pretrained))
    rows = []
    for snr in args.snrs:
        for seed in args.seeds:
            rows.extend(
                _run_trial(
                    model_config=model_config,
                    pretrained=Path(args.pretrained),
                    device=args.device,
                    snr=snr,
                    seed=seed,
                    frames_count=args.frames,
                    delay=args.delay,
                    pilot_total=args.pilot_total,
                    gap_seconds=args.gap_seconds,
                    lr=args.lr,
                    steps=args.steps,
                    cosine_threshold=args.cosine_threshold,
                    analytic_smoothing=args.analytic_smoothing,
                )
            )
    def mean(name: str, values: list[dict]) -> float:
        return float(sum(float(row[name]) for row in values) / len(values)) if values else math.nan

    summary = []
    for snr in args.snrs:
        values = [row for row in rows if float(row["snr_db"]) == float(snr)]
        summary.append(
            {
                "snr_db": float(snr),
                "rows": len(values),
                "consistent_fraction": mean("gated_update_applied", values),
                "ungated_mean_data_ber_improvement": mean("ungated_data_ber_improvement", values),
                "gated_mean_data_ber_improvement": mean("gated_data_ber_improvement", values),
                "analytic_mean_data_ber_improvement": mean("analytic_data_ber_improvement", values),
                "ungated_positive_fraction": mean(
                    "ungated_data_ber_improvement", [
                        {"ungated_data_ber_improvement": float(row["ungated_data_ber_improvement"] > 0.0)}
                        for row in values
                    ]
                ),
                "gated_positive_fraction": mean(
                    "gated_data_ber_improvement", [
                        {"gated_data_ber_improvement": float(row["gated_data_ber_improvement"] > 0.0)}
                        for row in values
                    ]
                ),
                "analytic_positive_fraction": mean(
                    "analytic_data_ber_improvement", [
                        {"analytic_data_ber_improvement": float(row["analytic_data_ber_improvement"] > 0.0)}
                        for row in values
                    ]
                ),
                "mean_cosine_similarity": mean("cosine_similarity", values),
            }
        )
    payload = {
        "schema_version": "dual-view-pilot-consistency-replay-v1",
        "method": "phase_trend_pilot_signal_reconstruction",
        "pilot_split": "adapt_prefix_first_half_vs_second_half",
        "rows": rows,
        "summary": summary,
        "data_labels_used_online": False,
        "diagnostic_uses_data_labels": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(rows), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
