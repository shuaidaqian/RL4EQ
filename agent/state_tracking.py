# -*- coding: utf-8 -*-
"""Pilot-only 物理状态后验与跨帧 soft-tail belief。

该模块是在线接收机的状态边界：更新只接受 Adapt Pilot 和上一帧的接收机
内部状态，不读取 Data 波形、Data 标签或 Reward Pilot 标签。神经均衡器主干
保持冻结，状态以后验均值和置信度的形式传给前向检测器。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from agent.cir_estimator import (
    PilotPhysicalState,
    pilot_sparse_cir_update,
    track_pilot_physical_state,
)


@dataclass(frozen=True)
class TailBelief:
    """跨帧尾状态的均值、方差和整体置信度。"""

    mean: torch.Tensor
    variance: torch.Tensor
    confidence: float

    def __post_init__(self) -> None:
        mean = torch.as_tensor(self.mean, dtype=torch.complex64).reshape(-1)
        variance = torch.as_tensor(self.variance, dtype=torch.float32).reshape(-1)
        if mean.shape != variance.shape:
            raise ValueError("TailBelief 的均值和方差形状必须一致。")
        if not torch.isfinite(mean.real).all() or not torch.isfinite(variance).all():
            raise ValueError("TailBelief 必须是有限数。")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "variance", variance.clamp(0.0, 1.0))
        object.__setattr__(self, "confidence", float(max(0.0, min(1.0, self.confidence))))

    @classmethod
    def from_mean(cls, mean: torch.Tensor, variance: float = 1.0) -> "TailBelief":
        mean = torch.as_tensor(mean, dtype=torch.complex64).reshape(-1)
        return cls(mean, torch.full_like(mean.real, float(variance)), 1.0 - float(variance))

    def clone(self) -> "TailBelief":
        return TailBelief(self.mean.clone(), self.variance.clone(), self.confidence)

    def as_soft_tail(self) -> torch.Tensor:
        """返回可直接传给均衡器的均值。"""

        return self.mean.clone()


def inflate_tail_uncertainty(
    belief: TailBelief,
    *,
    variance_increment: float = 0.1,
    mean_decay: float = 0.1,
) -> TailBelief:
    """在本帧没有可靠 Pilot tail 观测时传播不确定度。

    该传播只使用接收机已有的 tail belief，不读取当前帧 Data 波形或标签。
    未知 Data 会使跨帧 tail 均值逐渐失去可预测性，因此均值向零收缩、方差
    增大；它不是新的观测更新，也不应被解释为信道真值估计。
    """

    increment = max(0.0, float(variance_increment))
    decay = min(1.0, max(0.0, float(mean_decay)))
    variance = (belief.variance + increment).clamp(0.0, 1.0)
    mean = belief.mean * (1.0 - decay)
    confidence = float((1.0 - variance.mean()).clamp(0.0, 1.0).detach().cpu())
    return TailBelief(mean, variance, confidence)


@dataclass(frozen=True)
class PhysicalStatePosterior:
    """当前帧物理状态后验。

    ``cir`` 是稀疏 CIR 的后验均值；``cir_variance`` 是每个 tap 的保守方差；
    ``phase`` 保存 residual phase/CFO；``tail`` 是跨帧 soft-tail belief。
    """

    cir: torch.Tensor
    cir_variance: torch.Tensor
    phase: PilotPhysicalState
    tail: TailBelief
    innovation_variance: float
    confidence: float
    frame_index: int = -1

    def __post_init__(self) -> None:
        cir = torch.as_tensor(self.cir, dtype=torch.complex64).reshape(-1)
        variance = torch.as_tensor(self.cir_variance, dtype=torch.float32).reshape(-1)
        if cir.shape != variance.shape:
            raise ValueError("CIR 后验均值和方差形状必须一致。")
        if cir.numel() == 0:
            raise ValueError("CIR 后验不能为空。")
        object.__setattr__(self, "cir", cir)
        object.__setattr__(self, "cir_variance", variance.clamp_min(0.0))
        object.__setattr__(self, "innovation_variance", float(max(0.0, self.innovation_variance)))
        object.__setattr__(self, "confidence", float(max(0.0, min(1.0, self.confidence))))

    @property
    def data_labels_used_online(self) -> bool:
        return False

    def clone(self) -> "PhysicalStatePosterior":
        return PhysicalStatePosterior(
            self.cir.clone(),
            self.cir_variance.clone(),
            self.phase.clone(),
            self.tail.clone(),
            self.innovation_variance,
            self.confidence,
            self.frame_index,
        )

    def condition_confidence(self) -> float:
        return min(self.confidence, float(self.phase.confidence), self.tail.confidence)


@dataclass(frozen=True)
class PilotStateUpdate:
    """一次 Pilot 更新的可审计结果。"""

    prior: PhysicalStatePosterior
    candidate: PhysicalStatePosterior
    committed: bool
    reason: str
    adapt_pilot_count: int
    data_labels_used_online: bool = False
    online_waveform_visibility: str = "adapt_pilot_only"


def initial_state(cir: torch.Tensor, soft_tail: torch.Tensor, *, frame_index: int = -1) -> PhysicalStatePosterior:
    """用 acquisition 输出建立保守的初始后验。"""

    cir = torch.as_tensor(cir, dtype=torch.complex64).reshape(-1)
    tail = TailBelief.from_mean(soft_tail, variance=0.0)
    return PhysicalStatePosterior(
        cir=cir.clone(),
        cir_variance=torch.full_like(cir.real, 0.25),
        phase=PilotPhysicalState(),
        tail=tail,
        innovation_variance=1.0,
        confidence=0.25,
        frame_index=frame_index,
    )


def update_from_adapt_pilot(
    frame,
    prior: PhysicalStatePosterior,
    *,
    snr_db: float,
    cfo_limit: float = 0.0012,
    phase_smoothing: float = 0.5,
    min_confidence: float = 0.15,
    cir_alpha: float = 0.35,
    max_paths: int = 24,
    frame_index: int | None = None,
    track_cir: bool = True,
    track_phase: bool = True,
) -> PilotStateUpdate:
    """只用当前 Adapt Pilot 生成后验候选，并执行低置信度冻结。

    CIR 更新和 phase/CFO 更新都通过已有 Pilot-only 估计器完成。任何估计失败
    或置信度不足都会返回未提交候选，调用方可以安全保持上一帧状态。
    """

    if not hasattr(frame, "adapt_mask") or not hasattr(frame, "receiver_view"):
        raise TypeError("frame 必须提供 receiver_view 和 adapt_mask。")
    count = int(frame.adapt_mask.to(torch.bool).sum().item())
    if count == 0:
        return PilotStateUpdate(prior, prior, False, "no_adapt_pilot", 0)
    try:
        phase = track_pilot_physical_state(
            frame,
            prior.cir,
            prior.tail.mean,
            previous=prior.phase,
            cfo_limit=float(cfo_limit),
            smoothing=float(phase_smoothing),
            min_confidence=float(min_confidence),
        )
        updated_cir = (
            pilot_sparse_cir_update(
                frame,
                prior.cir,
                prior.tail.mean,
                max_paths=int(max_paths),
                alpha=float(cir_alpha),
                cfo_hint=float(phase.cfo_cycles_per_symbol),
            )
            if track_cir
            else prior.cir.clone()
        )
    except (RuntimeError, ValueError, torch.linalg.LinAlgError):
        return PilotStateUpdate(prior, prior, False, "pilot_estimation_failed", count)
    confidence = float(phase.confidence)
    if not torch.isfinite(updated_cir).all() or confidence < float(min_confidence):
        reason = "low_pilot_confidence" if confidence < float(min_confidence) else "nonfinite_cir"
        return PilotStateUpdate(prior, prior, False, reason, count)
    committed_phase = phase if track_phase else prior.phase.clone()
    innovation = float(torch.mean(torch.abs(updated_cir - prior.cir).square()).detach().cpu())
    candidate = PhysicalStatePosterior(
        cir=updated_cir,
        cir_variance=(0.75 * prior.cir_variance + 0.25 * innovation).clamp_min(1e-8),
        phase=committed_phase,
        tail=prior.tail.clone(),
        innovation_variance=innovation,
        confidence=(
            min(1.0, 0.5 * prior.confidence + 0.5 * confidence)
            if track_phase
            else prior.confidence
        ),
        frame_index=prior.frame_index if frame_index is None else int(frame_index),
    )
    return PilotStateUpdate(prior, candidate, True, "pilot_posterior_committed", count)


def update_tail_from_adapt_pilot(
    frame,
    posterior: PhysicalStatePosterior,
    *,
    snr_db: float,
    minimum_rows: int = 8,
) -> TailBelief:
    """用前缀 Adapt Pilot 对跨帧 soft-tail 做复数 MAP 更新。"""

    prior = posterior.tail
    tail_len = int(prior.mean.numel())
    cir = posterior.cir.reshape(-1).to(torch.complex64)
    if tail_len == 0 or cir.numel() == 0:
        return prior.clone()
    view = frame.receiver_view()
    rx = view.rx_symbols.flatten().to(torch.complex64)
    tx = view.adapt_symbols.flatten().to(torch.complex64)
    mask = view.adapt_mask.flatten().bool()
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    noise = float(10.0 ** (-float(snr_db) / 10.0))
    phase = posterior.phase.phase0 + 2.0 * torch.pi * posterior.phase.cfo_cycles_per_symbol * torch.arange(
        rx.numel(), dtype=torch.float32, device=rx.device
    )
    corrected_rx = rx * torch.exp(-1j * phase).to(torch.complex64)
    rows = []
    targets = []
    for position in indices.tolist():
        row = torch.zeros(tail_len, dtype=torch.complex64, device=rx.device)
        known_prediction = torch.zeros((), dtype=torch.complex64, device=rx.device)
        valid = True
        for delay, tap in enumerate(cir):
            source = int(position) - int(delay)
            if source >= 0:
                if not bool(mask[source]):
                    valid = False
                    break
                known_prediction = known_prediction + tap.to(rx.device) * tx[source]
            else:
                tail_index = tail_len + source
                if tail_index < 0:
                    valid = False
                    break
                row[tail_index] = row[tail_index] + tap.to(rx.device)
        if valid and bool(torch.any(row.abs() > 0)):
            rows.append(row)
            targets.append(corrected_rx[position] - known_prediction)
    if len(rows) < max(int(minimum_rows), 1):
        return prior.clone()
    design = torch.stack(rows)
    target = torch.stack(targets)
    prior_mean = prior.mean.to(rx.device)
    prior_var = prior.variance.to(rx.device).clamp_min(1e-3)
    precision = torch.diag(1.0 / prior_var).to(torch.complex64)
    normal = design.mH @ design / max(noise, 1e-6) + precision
    rhs = design.mH @ target / max(noise, 1e-6) + prior_mean / prior_var
    try:
        covariance = torch.linalg.inv(normal)
        estimate = covariance @ rhs
    except torch.linalg.LinAlgError:
        return prior.clone()
    mean = estimate.real.clamp(-1.0, 1.0).to(torch.complex64)
    variance = covariance.diagonal().real.clamp(0.05, 1.0)
    confidence = float((1.0 - variance.mean()).clamp(0.0, 1.0).detach().cpu())
    return TailBelief(mean, variance, confidence)


def is_safe_to_commit(update: PilotStateUpdate, *, minimum_confidence: float = 0.15) -> bool:
    """判断 CIR、phase/CFO 与 tail 是否可以作为一个整体提交。"""

    return bool(
        update.committed
        and update.candidate.condition_confidence() >= float(minimum_confidence)
    )


def commit_if_safe(update: PilotStateUpdate, *, minimum_confidence: float = 0.15) -> PhysicalStatePosterior:
    """以原子方式提交状态；未通过门槛时返回深拷贝的 prior。"""

    if not is_safe_to_commit(update, minimum_confidence=minimum_confidence):
        return update.prior.clone()
    return update.candidate.clone()
