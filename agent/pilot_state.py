# -*- coding: utf-8 -*-
"""Pilot-only 状态摘要、状态 embedding 与漂移检测工具。

本模块只处理接收端可见的 Adapt Pilot 统计量，不读取 Reward/Data 标签。
它先作为离线 replay 和在线更新门控的公共表示，后续才允许接入 PEFT 调制。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PilotStateSummary:
    """由当前 Adapt Pilot 构造的低维、可审计状态摘要。"""

    cir_residual_norm: float
    phase0: float
    cfo_cycles_per_symbol: float
    noise_variance: float
    confidence: float
    reconstruction_error: float

    def as_vector(self) -> torch.Tensor:
        """返回固定顺序的数值向量，便于状态条件化和漂移检测。"""

        return torch.tensor(
            [
                self.cir_residual_norm,
                self.phase0,
                self.cfo_cycles_per_symbol,
                self.noise_variance,
                self.confidence,
                self.reconstruction_error,
            ],
            dtype=torch.float32,
        )

    @property
    def data_labels_used_online(self) -> bool:
        """状态摘要只使用 Adapt Pilot，因此永远不读取 Data 标签。"""

        return False


def build_pilot_state_summary(
    *,
    cir: torch.Tensor,
    reference_cir: torch.Tensor,
    phase0: float = 0.0,
    cfo_cycles_per_symbol: float = 0.0,
    noise_variance: float | torch.Tensor = 0.0,
    confidence: float | torch.Tensor = 0.0,
    reconstruction_error: float | torch.Tensor = 0.0,
) -> PilotStateSummary:
    """把 Pilot-only 物理量归一为可比较的状态摘要。"""

    current = cir.reshape(-1).to(torch.complex64)
    reference = reference_cir.reshape(-1).to(current.device, dtype=current.dtype)
    if current.numel() != reference.numel():
        raise ValueError("cir 与 reference_cir 的长度必须一致。")
    denominator = torch.linalg.vector_norm(reference).clamp_min(1e-8)
    residual_norm = torch.linalg.vector_norm(current - reference) / denominator

    def _scalar(value: float | torch.Tensor) -> float:
        tensor = torch.as_tensor(value, dtype=torch.float32)
        return float(tensor.mean().detach().cpu())

    return PilotStateSummary(
        cir_residual_norm=float(residual_norm.detach().cpu()),
        phase0=float(phase0),
        cfo_cycles_per_symbol=float(cfo_cycles_per_symbol),
        noise_variance=_scalar(noise_variance),
        confidence=_scalar(confidence),
        reconstruction_error=_scalar(reconstruction_error),
    )


class PilotStateEmbedding:
    """固定、无参数的状态 embedding。

    采用参考尺度归一化和有界 tanh，避免把量纲差异直接传给 PEFT 调制器。
    它不是离线模型的一部分，不改变 checkpoint。
    """

    def __init__(self, scales: torch.Tensor | None = None) -> None:
        default = torch.tensor([1.0, 1.0, 0.0012, 1.0, 1.0, 1.0], dtype=torch.float32)
        self.scales = default if scales is None else torch.as_tensor(scales, dtype=torch.float32)
        if self.scales.numel() != 6 or bool(torch.any(self.scales <= 0)):
            raise ValueError("状态归一化 scales 必须包含 6 个正数。")

    def __call__(self, summary: PilotStateSummary) -> torch.Tensor:
        vector = summary.as_vector()
        normalized = vector / self.scales
        return torch.tanh(normalized)


class PilotDriftDetector:
    """基于 Pilot 状态 embedding 的无标签漂移检测器。"""

    def __init__(self, threshold: float = 0.25, min_confidence: float = 0.15) -> None:
        if threshold <= 0.0:
            raise ValueError("漂移阈值必须为正数。")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("最小置信度必须位于 [0, 1]。")
        self.threshold = float(threshold)
        self.min_confidence = float(min_confidence)
        self._previous: torch.Tensor | None = None

    def reset(self) -> None:
        self._previous = None

    def update(self, embedding: torch.Tensor, confidence: float) -> tuple[bool, float]:
        """返回 ``(是否漂移, 距离)``，首帧只建立基线，不触发更新。"""

        current = torch.as_tensor(embedding, dtype=torch.float32).reshape(-1).detach().cpu()
        if current.numel() != 6:
            raise ValueError("Pilot 状态 embedding 必须包含 6 个元素。")
        if not torch.isfinite(current).all():
            raise ValueError("Pilot 状态 embedding 必须为有限数。")
        if self._previous is None:
            self._previous = current
            return False, 0.0
        distance = float(torch.linalg.vector_norm(current - self._previous).item())
        self._previous = current
        drift = float(confidence) >= self.min_confidence and distance >= self.threshold
        return bool(drift), distance


class PilotTemporalConsistency:
    """检查连续 Adapt Pilot 状态漂移是否保持同一方向。"""

    def __init__(self, min_distance: float = 0.05, min_cosine: float = 0.25) -> None:
        if min_distance < 0.0:
            raise ValueError("min_distance 不能为负数。")
        if not -1.0 <= min_cosine <= 1.0:
            raise ValueError("min_cosine 必须位于 [-1, 1]。")
        self.min_distance = float(min_distance)
        self.min_cosine = float(min_cosine)
        self._previous: torch.Tensor | None = None
        self._last_delta: torch.Tensor | None = None

    def reset(self) -> None:
        self._previous = None
        self._last_delta = None

    def update(self, embedding: torch.Tensor) -> tuple[bool, float, float]:
        """返回 ``(是否连续一致, 当前距离, 方向余弦)``。"""

        current = torch.as_tensor(embedding, dtype=torch.float32).reshape(-1).detach().cpu()
        if current.numel() != 6 or not torch.isfinite(current).all():
            raise ValueError("Pilot 状态 embedding 必须是 6 维有限数。")
        if self._previous is None:
            self._previous = current
            return False, 0.0, 0.0
        delta = current - self._previous
        distance = float(torch.linalg.vector_norm(delta).item())
        self._previous = current
        if self._last_delta is None:
            self._last_delta = delta
            return False, distance, 0.0
        previous_norm = torch.linalg.vector_norm(self._last_delta)
        current_norm = torch.linalg.vector_norm(delta)
        if float(previous_norm) <= 1e-8 or float(current_norm) <= 1e-8:
            cosine = 0.0
        else:
            cosine = float(torch.dot(delta, self._last_delta).item() / (current_norm * previous_norm).item())
        self._last_delta = delta
        consistent = distance >= self.min_distance and cosine >= self.min_cosine
        return bool(consistent), distance, cosine
