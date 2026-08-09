# -*- coding: utf-8 -*-
"""长 episode 退化诊断。

本模块只做诊断实验，不进入正式 baseline。oracle tail / oracle CIR 会使用
仿真器暴露的真值，用于定位误差来源。
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import torch

from agent.cir_estimator import HybridCIREstimator, condition_from_cir, decision_directed_cir_update
from baseline.block_equalizers import bit_error_rate
from env.comm_env import CommEnvConfig, CommunicationEnvironment, ReceiverState
from training.meta_training import _estimate_cir_from_known_frame
from training.rl_modulated_online import _build_equalizer, _load_model_config


def run_long_episode_diagnostic(
    config_path: str | Path,
    pretrained: str | Path,
    output_dir: str | Path,
    delay: int = 40,
    snr_db: float = 10.0,
    frames: int = 64,
    seeds: Iterable[int] = (0, 1, 2),
    tail_modes: Iterable[str] = ("soft", "hard", "oracle", "zero"),
    cir_modes: Iterable[str] = ("fixed", "decision_directed", "oracle"),
    rhos: Iterable[float] = (0.99, 1.0),
    pilot_total: int = 128,
    pilot_layout: str = "prefix",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """运行 tail/CIR/drift 正交诊断并写出 JSONL 与 summary。"""

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    pretrained_path = Path(pretrained)
    model_config = _load_model_config(config, pretrained_path)
    rows: list[dict] = []
    for rho in rhos:
        for seed in seeds:
            env = CommunicationEnvironment(
                CommEnvConfig(
                    level="B",
                    max_delay=int(delay),
                    snr_db=float(snr_db),
                    rho=float(rho),
                    total_pilot=int(pilot_total),
                    layout=str(pilot_layout),
                    seed=70_000 + int(seed),
                )
            )
            start = env.reset_episode()
            frames_cache = [env.next_frame() for _ in range(int(frames))]
            acquisition_cir = _estimate_cir_from_known_frame(start.acquisition, int(delay))
            for tail_mode in tail_modes:
                for cir_mode in cir_modes:
                    model = _build_equalizer(model_config, pretrained_path, device)
                    receiver_state = ReceiverState(_initial_tail(str(tail_mode), start.initial_soft_tail, int(delay)).to(device))
                    cir_state = acquisition_cir.clone().to(device)
                    for frame_index, frame_cpu in enumerate(frames_cache, start=1):
                        frame = _frame_to_device(frame_cpu, device)
                        cir_for_frame = _select_cir(str(cir_mode), cir_state, frame, int(delay), device)
                        logits = _run_model(model, frame, cir_for_frame, receiver_state.soft_tail, float(snr_db))
                        rows.append(
                            {
                                "method": "Offline NN diagnostic",
                                "diagnostic_only": True,
                                "level": "B",
                                "delay": int(delay),
                                "snr_db": float(snr_db),
                                "rho": float(rho),
                                "seed": int(seed),
                                "frame": int(frame_index),
                                "frame_bin": _frame_bin(frame_index),
                                "tail_mode": str(tail_mode),
                                "cir_mode": str(cir_mode),
                                "ber_data": bit_error_rate(logits[frame.data_mask], frame.bits[frame.data_mask]),
                                "ber_reward_pilot": bit_error_rate(logits[frame.reward_mask], frame.bits[frame.reward_mask]),
                                "ber_adapt_pilot": bit_error_rate(logits[frame.adapt_mask], frame.bits[frame.adapt_mask]),
                                "uses_oracle_tail": str(tail_mode) == "oracle",
                                "uses_oracle_cir": str(cir_mode) == "oracle",
                                "cir_estimation_uses_adapt_pilot": str(cir_mode) == "adapt_pilot",
                            }
                        )
                        if str(cir_mode) == "decision_directed":
                            cir_state = decision_directed_cir_update(
                                frame,
                                logits.detach(),
                                int(delay),
                                cir_state,
                                alpha=0.2,
                            ).to(device)
                        receiver_state.update_tail(_next_tail(str(tail_mode), logits.detach(), frame, int(delay)))
    _write_rows(target / "frame_metrics.jsonl", rows)
    summary = _summarize(rows)
    payload = {
        "schema_version": "long-episode-diagnostic-v1",
        "diagnostic_only": True,
        "rows": rows,
        "summary": summary,
    }
    (target / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _run_model(model, frame, cir: torch.Tensor, tail: torch.Tensor, snr_db: float) -> torch.Tensor:
    rx_iq = torch.stack((frame.rx_symbols.real, frame.rx_symbols.imag), dim=-1).unsqueeze(0).float()
    region_ids = frame.model_region_ids.unsqueeze(0).long()
    condition = condition_from_cir(cir, snr_db)
    logits, _ = model(rx_iq, condition, region_ids, tail.unsqueeze(0).to(torch.complex64))
    return logits.squeeze(0)


def _initial_tail(mode: str, initial_soft_tail: torch.Tensor, delay: int) -> torch.Tensor:
    if mode == "zero":
        return torch.zeros(int(delay), dtype=torch.complex64)
    return initial_soft_tail.clone().to(torch.complex64)


def _next_tail(mode: str, logits: torch.Tensor, frame, delay: int) -> torch.Tensor:
    if mode == "oracle":
        return frame.tx_symbols[-int(delay) :].detach().clone().to(torch.complex64)
    if mode == "hard":
        hard = torch.where(logits[-int(delay) :] >= 0, torch.ones(int(delay), device=logits.device), -torch.ones(int(delay), device=logits.device))
        return torch.complex(hard, torch.zeros_like(hard))
    if mode == "zero":
        return torch.zeros(int(delay), dtype=torch.complex64, device=logits.device)
    soft = torch.tanh(logits[-int(delay) :] / 2.0)
    return torch.complex(soft, torch.zeros_like(soft))


def _select_cir(mode: str, cir_state: torch.Tensor, frame, delay: int, device: str) -> torch.Tensor:
    if mode == "oracle":
        return frame.true_cir.to(device)
    if mode == "adapt_pilot":
        return _estimate_cir_from_adapt_pilot(frame, int(delay), device, fallback=cir_state)
    return cir_state.to(device)


def _estimate_cir_from_adapt_pilot(frame, delay: int, device: str, fallback: torch.Tensor) -> torch.Tensor:
    view = frame.receiver_view()
    estimator = HybridCIREstimator(max_delay=int(delay)).to(device)
    rx = view.rx_symbols.to(device)
    rx_iq = torch.stack((rx.real, rx.imag), dim=-1).unsqueeze(0).float()
    condition = estimator(
        rx_iq,
        view.adapt_symbols.to(device).unsqueeze(0),
        view.adapt_mask.to(device).unsqueeze(0),
        view.model_region_ids.to(device).unsqueeze(0),
    )
    cir = condition.complex_cir.squeeze(0)
    power = torch.sum(torch.abs(cir) ** 2)
    if float(power.detach().cpu()) <= 1e-12:
        return fallback.to(device)
    return cir / torch.sqrt(power.clamp_min(1e-12))


def _frame_bin(frame_index: int) -> str:
    start = ((int(frame_index) - 1) // 16) * 16 + 1
    return f"{start}-{start + 15}"


def _summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["rho"], row["tail_mode"], row["cir_mode"], row["frame_bin"])].append(float(row["ber_data"]))
    return [
        {
            "rho": key[0],
            "tail_mode": key[1],
            "cir_mode": key[2],
            "frame_bin": key[3],
            "mean_ber_data": float(sum(values) / max(1, len(values))),
            "count": len(values),
        }
        for key, values in sorted(grouped.items())
    ]


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _frame_to_device(frame, device: str):
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
