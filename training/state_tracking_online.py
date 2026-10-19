# -*- coding: utf-8 -*-
"""冻结离线均衡器 + Pilot-only 物理状态后验跟踪在线入口。"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch

from agent.cir_estimator import PilotPhysicalState, condition_from_cir
from agent.pilot_risk import compute_pilot_risk
from agent.state_tracking import (
    commit_if_safe,
    initial_state,
    update_from_adapt_pilot,
    update_tail_from_adapt_pilot,
)
from baseline.traditional_equalizers import estimate_phase_residual_vector
from env.comm_env import CommunicationEnvironment
from env.experiment_config import build_comm_env_config, effective_channel_metadata
from training.meta_training import estimate_acquisition_cir_for_profile
from training.online_adaptation import _ber, _build_model, _frame_to_device, _load_model_config


def run_state_tracking_online(
    *,
    config_path: str | Path,
    pretrained: str | Path,
    frames: int = 300,
    num_seeds: int = 1,
    output_dir: str | Path = "logs/state_tracking",
    delays: list[int] | None = None,
    snrs: list[float] | None = None,
    pilot_total: int | None = None,
    pilot_layout: str | None = None,
    state_split: str | None = None,
    device: str = "cpu",
) -> dict:
    """运行冻结主干与物理状态跟踪；适配不访问 Data 发射符号或标签。"""

    config_path = Path(config_path)
    checkpoint_path = Path(pretrained)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"离线预训练 checkpoint 不存在：{checkpoint_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"实验配置不存在：{config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selected_delays = delays or [int(value) for value in config.get("main_delays", [116])]
    selected_snrs = snrs or [float(value) for value in config.get("main_snrs", [0, 5, 10, 15])]
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    pretrained_path = checkpoint_path
    model_config = _load_model_config(config, pretrained_path)
    rows: list[dict] = []
    channel_metadata = None

    for delay in selected_delays:
        for snr_db in selected_snrs:
            for seed in range(int(num_seeds)):
                model = _build_model(model_config, pretrained_path, device)
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                env_config = build_comm_env_config(
                    config,
                    level="B",
                    snr_db=float(snr_db),
                    seed=70_000 + int(seed),
                    max_delay=int(delay),
                    total_pilot=pilot_total,
                    pilot_layout=str(pilot_layout or config.get("pilot_layout", "prefix")),
                    state_split=state_split,
                )
                channel_metadata = channel_metadata or effective_channel_metadata(env_config)
                env = CommunicationEnvironment(env_config)
                start = env.reset_episode()
                profile = str(config.get("impairment_profile", "clean"))
                if profile == "clean":
                    acquisition_cir = estimate_acquisition_cir_for_profile(
                        start.acquisition, int(delay), profile
                    )
                    acquisition_cfo = 0.0
                else:
                    from baseline.traditional_equalizers import estimate_acquisition_cir_with_cfo

                    acquisition_cir, acquisition_cfo = estimate_acquisition_cir_with_cfo(
                        start.acquisition,
                        int(delay),
                        cfo_limit=float(config.get("acquisition_cfo_limit", 0.004)),
                    )
                acquisition_cir = acquisition_cir.to(device=device, dtype=torch.complex64)
                posterior = initial_state(
                    acquisition_cir,
                    start.initial_soft_tail.to(device=device, dtype=torch.complex64),
                )
                posterior = replace(
                    posterior,
                    phase=PilotPhysicalState(cfo_cycles_per_symbol=float(acquisition_cfo), confidence=0.5),
                )

                for frame_index in range(int(frames)):
                    frame = _frame_to_device(env.next_frame(), device)
                    prior_posterior = posterior
                    update = update_from_adapt_pilot(
                        frame,
                        posterior,
                        snr_db=float(snr_db),
                        cfo_limit=float(config.get("online_phase_tracking_cfo_limit", 0.0012)),
                        phase_smoothing=float(config.get("online_phase_tracking_smoothing", 0.5)),
                        min_confidence=float(config.get("online_phase_tracking_min_confidence", 0.15)),
                        cir_alpha=float(config.get("state_tracking_cir_alpha", 0.35)),
                        max_paths=int(config.get("state_tracking_max_paths", 24)),
                        frame_index=frame_index,
                    )
                    posterior = commit_if_safe(
                        update,
                        minimum_confidence=float(config.get("state_tracking_min_confidence", 0.15)),
                    )
                    posterior = replace(
                        posterior,
                        tail=update_tail_from_adapt_pilot(
                            frame,
                            posterior,
                            snr_db=float(snr_db),
                        ),
                        frame_index=frame_index,
                    )
                    phase_features = estimate_phase_residual_vector(
                        frame.receiver_view(),
                        posterior.cir,
                        posterior.tail.mean,
                        blocks=int(model_config.phase_correction_segments),
                        cfo_hint=float(posterior.phase.cfo_cycles_per_symbol),
                    ).to(device)
                    condition = condition_from_cir(
                        posterior.cir,
                        float(snr_db),
                        phase_features=phase_features,
                    )
                    tail = posterior.tail.mean.unsqueeze(0).to(device=device, dtype=torch.complex64)
                    rx_iq = torch.stack(
                        (frame.rx_symbols.real, frame.rx_symbols.imag), dim=-1
                    ).unsqueeze(0).float()
                    adapt_symbols = frame.receiver_view().adapt_symbols.unsqueeze(0).to(torch.complex64)
                    adapt_mask = frame.adapt_mask.unsqueeze(0).bool()
                    region_ids = frame.model_region_ids.unsqueeze(0).long()
                    risk = compute_pilot_risk(
                        frame,
                        posterior,
                        noise_variance=10.0 ** (-float(snr_db) / 10.0),
                    )
                    with torch.no_grad():
                        logits, _ = model(
                            rx_iq,
                            condition,
                            region_ids,
                            tail,
                            adapt_symbols=adapt_symbols,
                            adapt_mask=adapt_mask,
                        )
                    logits = logits.squeeze(0)
                    # Data 标签只在这里用于事后 BER 统计；在线状态、参数和动作均未读取标签。
                    ber_data = _ber(logits[frame.data_mask], frame.bits[frame.data_mask])
                    rows.append(
                        {
                            "method": "Pilot State Tracking Frozen NN",
                            "seed": int(seed),
                            "delay": int(delay),
                            "snr_db": float(snr_db),
                            "frame_index": int(frame_index),
                            "ber_data": float(ber_data),
                            "state_update_committed": bool(update.committed),
                            "state_update_reason": update.reason,
                            "adapt_pilot_count": int(update.adapt_pilot_count),
                            "cir_delta_norm": float(torch.linalg.vector_norm(posterior.cir - prior_posterior.cir).detach().cpu()),
                            "residual_cfo": float(posterior.phase.cfo_cycles_per_symbol),
                            "phase_confidence": float(posterior.phase.confidence),
                            "tail_variance_mean": float(posterior.tail.variance.mean().detach().cpu()),
                            "pilot_risk_score": float(risk.score),
                            "pilot_risk_abstain": bool(risk.abstain),
                            "pilot_risk_confidence": float(risk.confidence),
                            "pretrained_loaded": True,
                            "online_parameter_update": False,
                            "data_labels_used_online": False,
                            "online_waveform_visibility": "adapt_pilot_for_state; full_rx_for_frozen_detection; no_data_labels_for_update",
                        }
                    )

    if not rows:
        raise ValueError("在线运行没有生成任何帧。")
    payload = {
        "schema_version": "pilot-state-tracking-online-v1",
        "method": "Pilot State Tracking Frozen NN",
        "config_path": str(config_path),
        "pretrained_path": str(checkpoint_path),
        "pretrained_loaded": True,
        "data_labels_used_online": False,
        "online_parameter_update": False,
        "pilot_risk_is_validated_for_data_ber": False,
        "channel": channel_metadata,
        "frame_count": len(rows),
        "mean_ber_data": float(sum(row["ber_data"] for row in rows) / len(rows)),
        "mean_pilot_risk_score": float(sum(row["pilot_risk_score"] for row in rows) / len(rows)),
        "rows": rows,
    }
    (target / "frame_metrics.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (target / "online_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload
