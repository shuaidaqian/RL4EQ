# -*- coding: utf-8 -*-
"""让 Pilot replay 复用正式比较的信道、模型维度和 acquisition 协议。"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from agent.cir_estimator import CIRCondition, condition_from_cir, pilot_sparse_cir_update
from agent.pilot_state import PilotStateEmbedding, build_pilot_state_summary
from baseline.traditional_equalizers import (
    estimate_acquisition_cir_with_cfo,
    estimate_phase_residual_vector,
)
from compare import _neural_phase_features, _online_condition_source_from_config
from env.comm_env import CommunicationEnvironment, EpisodeStart
from env.experiment_config import build_comm_env_config, validate_model_dimensions
from env.linear_operator import LinearChannelOperator
from training.meta_training import _estimate_cir_from_known_frame


@dataclass
class PilotReplayEpisode:
    env: CommunicationEnvironment
    start: EpisodeStart
    cir: torch.Tensor
    acquisition_cfo: float
    acquisition_phase_features: torch.Tensor
    condition_source: str

    def condition(self, frame, tail: torch.Tensor) -> CIRCondition:
        """两组共享同一条件；诊断中不递推或写回主干 CIR。"""

        features = _neural_phase_features(
            condition_source=self.condition_source, receiver_view=frame.receiver_view(),
            cir=self.cir, soft_tail=tail, acquisition_cfo=self.acquisition_cfo,
            acquisition_phase_features=self.acquisition_phase_features,
        )
        return condition_from_cir(
            self.cir, self.env.config.snr_db,
            cfo_residual=self.acquisition_cfo,
            phase_features=features,
        )

    def metadata(self) -> dict:
        return {
            "replay_protocol_version": "configured-channel-phase-aligned-v3",
            "environment_config": asdict(self.env.config),
            "acquisition_cfo": self.acquisition_cfo,
            "condition_source": self.condition_source,
            "condition_role": (
                "acquisition_only_ablation" if self.condition_source == "acquisition"
                else "current_adapt_pilot_phase"
            ),
            "cir_update_mode": "fixed_acquisition_shared",
            "main_aggregation_allowed": False,
        }


def build_pilot_replay_episode(
    experiment: dict,
    model_config,
    *,
    snr: float,
    seed: int,
    delay: int,
    pilot_total: int,
    reward_pilot_total: int | None,
    gap_seconds: float | None,
    device: str,
    condition_source: str | None = None,
) -> PilotReplayEpisode:
    """加载完整配置，严格检查维度，禁止静默使用旧版默认环境。"""

    config = dict(experiment)
    if gap_seconds is not None:
        gap = float(gap_seconds)
        if not math.isfinite(gap) or gap < 0.0:
            raise ValueError("gap_seconds 必须为有限非负数。")
        config["acquisition_to_data_gap_seconds"] = gap
    source = _online_condition_source_from_config(config, condition_source)
    env_config = build_comm_env_config(
        config, level="B", snr_db=float(snr), seed=50_000 + int(seed),
        max_delay=int(delay), total_pilot=int(pilot_total),
        reward_pilot_total=reward_pilot_total, pilot_layout="prefix",
    )
    validate_model_dimensions(model_config, env_config)
    if env_config.acquisition_to_data_gap_seconds and env_config.profile_name == "legacy_sparse_v1":
        raise ValueError("legacy 信道不支持 acquisition 老化，不能使用非零 gap。")
    env = CommunicationEnvironment(env_config)
    start = env.reset_episode()
    if env_config.impairment_profile == "clean":
        cir = _estimate_cir_from_known_frame(start.acquisition, env_config.max_delay)
        cfo = 0.0
        features = torch.zeros(16, dtype=torch.float32)
    else:
        cir, cfo = estimate_acquisition_cir_with_cfo(
            start.acquisition, env_config.max_delay,
            cfo_limit=float(config.get("acquisition_cfo_limit", 0.004)),
        )
        features = estimate_phase_residual_vector(
            start.acquisition.receiver_view(), cir, start.initial_soft_tail,
            blocks=4, cfo_hint=float(cfo),
        )
    return PilotReplayEpisode(env, start, cir.to(device), float(cfo), features, source)


def pilot_state_embedding(frame, cir, tail, condition) -> torch.Tensor:
    """仅提取 Adapt Pilot 状态送入 Adapter，不改写主干物理条件。"""

    cir = cir.reshape(-1)
    view = frame.receiver_view()
    phase = estimate_phase_residual_vector(view, cir, tail, blocks=4)
    operator = LinearChannelOperator(frame_len=view.adapt_symbols.numel(), max_delay=cir.numel() - 1)
    predicted = operator.forward(view.adapt_symbols, cir, tail).reshape(-1)
    mask = view.adapt_mask.bool()
    rx = view.rx_symbols
    residual = (predicted[mask] - rx[mask]).abs().pow(2).mean()
    error = float((residual / rx[mask].abs().pow(2).mean().clamp_min(1e-8)).cpu())
    embedding_cir = pilot_sparse_cir_update(
        frame, cir, tail, max_paths=24, alpha=1.0, cfo_hint=float(phase[1]),
    )
    summary = build_pilot_state_summary(
        cir=embedding_cir, reference_cir=cir,
        phase0=float(phase[0]), cfo_cycles_per_symbol=float(phase[1]),
        noise_variance=condition.noise_variance, confidence=condition.confidence,
        reconstruction_error=error,
    )
    return PilotStateEmbedding()(summary).to(cir.device)
