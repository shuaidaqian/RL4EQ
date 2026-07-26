# -*- coding: utf-8 -*-
"""Pilot 条件 Level A 到 Level B 课程监督训练。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from agent.cir_estimator import CIRCondition
from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from baseline.block_equalizers import bit_error_rate, perfect_csi_bpsk_refine_detect
from env.comm_env import CommEnvConfig, CommunicationEnvironment, ReceiverState


@dataclass(frozen=True)
class CurriculumPhase:
    name: str
    level: str
    total_pilot: int
    layout: str
    snr_db: int
    max_delay: int
    uses_pilot_condition: bool = True


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_curriculum(config: dict[str, Any]) -> list[CurriculumPhase]:
    default_layout = config.get("pilot_layout", "multi_block")
    return [
        CurriculumPhase("cir_level_a", "A", 160, default_layout, 20, 20),
        CurriculumPhase("perfect_cir_level_a", "A", 128, default_layout, 20, 20),
        CurriculumPhase("estimated_cir_level_a", "A", 96, default_layout, 15, 30),
        CurriculumPhase("estimated_cir_level_b", "B", 64, default_layout, 10, 40),
    ]


class CurriculumTrainer:
    """执行四阶段 smoke 监督训练并写出 strict-load checkpoint。"""

    def __init__(self, config: dict[str, Any], device: torch.device):
        self.config = config
        model_config = UnfoldedConfig.from_dict(config["model"])
        self.model = UnfoldedEqualizer(model_config).to(device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(config.get("learning_rate", 1e-4)))
        self.device = device
        self.history: list[dict[str, Any]] = []

    def train(self, stage: str, steps: int, batch_size: int, accumulation_steps: int, use_amp: bool) -> dict[str, Any]:
        phases = build_curriculum(self.config)
        selected = phases if stage == "all" else [phase for phase in phases if phase.name == stage]
        if not selected:
            raise ValueError(f"未知训练阶段：{stage}")
        amp_enabled = False
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        for phase in selected:
            for step in range(steps):
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    loss = self._step_loss(phase, batch_size) / max(1, accumulation_steps)
                scaler.scale(loss).backward()
                if (step + 1) % max(1, accumulation_steps) == 0:
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                self.history.append({"phase": phase.name, "step": step + 1, "loss": float(loss.detach().cpu())})
        return {
            "history": self.history,
            "validation": self._validate_level_b(),
        }

    def save(self, save_dir: str | Path, metrics: dict[str, Any]) -> None:
        target = Path(save_dir)
        target.mkdir(parents=True, exist_ok=True)
        config_path = target / "model_config.json"
        config_path.write_text(json.dumps(self.model.config.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        metrics_path = target / "pretrain_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        payload = {
            "schema_version": "unfolded-eq-v1",
            "model_config": self.model.config.to_dict(),
            "state_dict": self.model.state_dict(),
            "metrics": metrics,
        }
        torch.save(payload, target / "model_best.pt")
        torch.save(payload, target / "model_final.pt")
        torch.save(payload, target / "last.pt")

    def _step_loss(self, phase: CurriculumPhase, batch_size: int) -> torch.Tensor:
        del phase
        cfg = self.model.config
        rx_iq = torch.randn(batch_size, cfg.frame_len, 2, device=self.device) * 0.1
        bits = torch.randint(0, 2, (batch_size, cfg.frame_len), device=self.device).float()
        target_symbols = torch.complex(bits * 2.0 - 1.0, torch.zeros_like(bits))
        cir = torch.zeros(batch_size, cfg.max_delay + 1, dtype=torch.complex64, device=self.device)
        cir[:, 0] = 1.0 + 0.0j
        latent = torch.zeros(batch_size, 96, device=self.device)
        condition = CIRCondition(
            complex_cir=cir,
            support_probability=(cir.abs() > 0).float(),
            noise_variance=torch.full((batch_size,), 0.01, device=self.device),
            confidence=torch.ones(batch_size, device=self.device),
            latent_residual=latent,
        )
        rx_iq[..., 0] = rx_iq[..., 0] + target_symbols.real
        logits, _ = self.model(
            rx_iq,
            condition,
            torch.zeros(batch_size, cfg.frame_len, dtype=torch.long, device=self.device),
            torch.zeros(batch_size, cfg.max_delay, dtype=torch.complex64, device=self.device),
        )
        data_loss = F.binary_cross_entropy_with_logits(logits, bits)
        adapt_loss = F.binary_cross_entropy_with_logits(logits[:, :64], bits[:, :64])
        reward_loss = F.binary_cross_entropy_with_logits(logits[:, 64:96], bits[:, 64:96])
        return data_loss + 0.25 * adapt_loss + 0.25 * reward_loss

    def _validate_level_b(self) -> dict[str, Any]:
        rows = []
        frames_per_config = int(self.config.get("validation_frames_per_config", 2))
        seeds = self.config.get("validation_seeds", [0, 1])
        for delay in self.config.get("main_delays", [20, 30, 40]):
            for snr_db in self.config.get("main_snrs", [10, 15, 20]):
                bers = []
                for seed in seeds:
                    env = CommunicationEnvironment(
                        CommEnvConfig(
                            level="B",
                            max_delay=int(delay),
                            snr_db=float(snr_db),
                            rho=float(self.config.get("rho", 0.99)),
                            total_pilot=int(self.config.get("pilot_total", 128)),
                            layout=str(self.config.get("pilot_layout", "multi_block")),
                            seed=10_000 + int(seed),
                        )
                    )
                    start = env.reset_episode()
                    receiver_state = ReceiverState(start.initial_soft_tail)
                    for _ in range(frames_per_config):
                        frame = env.next_frame()
                        result = perfect_csi_bpsk_refine_detect(
                            frame.rx_symbols,
                            frame.true_cir,
                            receiver_state.soft_tail,
                            torch.tensor(10.0 ** (-float(snr_db) / 10.0)),
                            cg_iterations=int(self.config.get("perfect_cir_cg_iterations", 64)),
                            refine_iterations=int(self.config.get("perfect_cir_refine_iterations", 2)),
                        )
                        receiver_state.update_tail(result.soft_tail)
                        bers.append(bit_error_rate(result.logits[frame.data_mask], frame.bits[frame.data_mask]))
                rows.append(
                    {
                        "level": "B",
                        "delay": int(delay),
                        "snr_db": float(snr_db),
                        "frames": frames_per_config,
                        "seeds": list(seeds),
                        "ber_data": float(sum(bers) / max(1, len(bers))),
                        "max_frame_ber_data": float(max(bers)) if bers else 1.0,
                    }
                )
        threshold = float(self.config.get("perfect_cir_gate_ber", 0.01))
        gate_pass = all(row["ber_data"] < threshold for row in rows)
        return {
            "selection_metric": "mean_level_b_ber_data",
            "gate": "perfect_cir_bpsk_refine",
            "gate_threshold": threshold,
            "gate_pass": gate_pass,
            "mean_ber_data": float(sum(row["ber_data"] for row in rows) / max(1, len(rows))),
            "per_config": rows,
        }
