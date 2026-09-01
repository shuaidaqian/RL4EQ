# -*- coding: utf-8 -*-
"""跨帧 soft tail 的均值、方差和置信度状态。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SoftTailState:
    """每个历史符号的复均值和不确定性。"""

    mean: torch.Tensor
    variance: torch.Tensor

    def __post_init__(self) -> None:
        self.mean = self.mean.to(torch.complex64)
        self.variance = self.variance.to(torch.float32)
        if self.mean.ndim != 1 or self.variance.ndim != 1:
            raise ValueError("soft tail 的 mean 和 variance 必须是一维张量。")
        if self.mean.shape != self.variance.shape:
            raise ValueError("soft tail 的 mean 和 variance 尺寸必须一致。")
        self.variance = self.variance.clamp(0.0, 1.0)

    @property
    def confidence(self) -> torch.Tensor:
        """返回逐符号置信度，1 表示确定，0 表示完全不确定。"""

        return (1.0 - self.variance).clamp(0.0, 1.0)

    @property
    def data_labels_used_online(self) -> bool:
        """该状态只由接收端输出构造，不允许读取 Data 标签。"""

        return False

    def clone(self) -> "SoftTailState":
        return SoftTailState(self.mean.clone(), self.variance.clone())


def update_soft_tail_state(
    logits: torch.Tensor,
    *,
    previous: SoftTailState | None = None,
    alpha: float = 0.5,
) -> SoftTailState:
    """根据当前帧尾部 logits 更新 soft tail，并按置信度抑制跳变。

    候选均值使用 BPSK 的 ``tanh(logit / 2)``。低置信度候选仍可缓慢修正旧
    状态，但更新权重只有高置信度候选的四分之一，避免错误 tail 在连续帧中放大。
    """

    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha 必须位于 [0, 1]。")
    flat_logits = logits.flatten().to(torch.float32)
    candidate_mean = torch.complex(
        torch.tanh(flat_logits / 2.0),
        torch.zeros_like(flat_logits),
    )
    candidate_variance = (1.0 - candidate_mean.real.square()).clamp(0.0, 1.0)
    candidate = SoftTailState(candidate_mean, candidate_variance)
    if previous is None:
        return candidate
    if previous.mean.shape != candidate.mean.shape:
        raise ValueError("previous soft tail 与当前 logits 尺寸必须一致。")
    confidence_weight = 0.25 + 0.75 * candidate.confidence
    weight = float(alpha) * confidence_weight
    mean = (1.0 - weight).to(torch.float32) * previous.mean + weight.to(torch.float32) * candidate.mean
    variance = (1.0 - weight) * previous.variance + weight * candidate.variance
    return SoftTailState(mean, variance)
