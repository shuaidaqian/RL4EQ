# -*- coding: utf-8 -*-
"""Pilot-only 物理一致性与未来风险代理。

风险代理用于记录和筛选状态后验，不读取 Data 标签，也不把 Reward Pilot BCE
作为默认控制目标。它在未知信道突变时允许 abstain；正相关只能作为独立轨迹
上的实证验收条件，不能对任意时变信道作理论保证。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from agent.state_tracking import PhysicalStatePosterior
from env.linear_operator import LinearChannelOperator


@dataclass(frozen=True)
class PilotRisk:
    score: float
    reconstruction_nll: float
    innovation_whiteness: float
    boundary_margin: float
    tail_propagation_risk: float
    uncertainty: float
    confidence: float
    abstain: bool
    data_labels_used_online: bool = False
    pilot_subset: str = "reward"
    online_waveform_visibility: str = "reward_pilot_waveform_only"


def _pilot_symbols(frame, pilot_subset: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    view = frame.receiver_view()
    if pilot_subset == "reward":
        mask = frame.reward_mask.to(torch.bool)
        symbols = view.reward_symbols
    elif pilot_subset == "adapt":
        mask = view.adapt_mask.to(torch.bool)
        symbols = view.adapt_symbols
    elif pilot_subset == "all":
        mask = (view.adapt_mask | frame.reward_mask).to(torch.bool)
        symbols = view.adapt_symbols + view.reward_symbols
    else:
        raise ValueError("pilot_subset 必须是 reward、adapt 或 all。")
    tx = torch.where(mask, symbols, torch.zeros_like(view.rx_symbols))
    return tx.to(torch.complex64), view.rx_symbols.to(torch.complex64), mask


def compute_pilot_risk(
    frame,
    posterior: PhysicalStatePosterior,
    *,
    noise_variance: float | None = None,
    pilot_subset: str = "reward",
    boundary_fraction: float = 0.25,
    uncertainty_threshold: float = 0.8,
) -> PilotRisk:
    """计算只依赖 Pilot 波形和已知 Pilot 符号的风险分数。"""

    tx, rx, mask = _pilot_symbols(frame, pilot_subset)
    cir = posterior.cir.to(rx.device)
    tail = posterior.tail.mean.to(rx.device)
    operator = LinearChannelOperator(frame_len=rx.numel(), max_delay=cir.numel() - 1)
    predicted = operator.forward(tx, cir, tail)
    selected = mask & torch.isfinite(predicted.real) & torch.isfinite(rx.real)
    if not bool(selected.any()):
        return PilotRisk(
            float("inf"), float("inf"), float("inf"), 0.0, 1.0, 1.0, 0.0, True,
            pilot_subset=pilot_subset,
        )
    selected_indices = torch.nonzero(selected, as_tuple=False).flatten()
    residual = rx[selected_indices] - predicted[selected_indices]
    sigma = float(noise_variance) if noise_variance is not None else float(posterior.innovation_variance)
    sigma = max(sigma, 1e-6)
    nll = float((torch.abs(residual).square() / sigma).mean().detach().cpu() + math.log(sigma))
    adjacent = selected_indices[1:] == selected_indices[:-1] + 1
    if bool(adjacent.any()):
        left = residual[:-1][adjacent]
        right = residual[1:][adjacent]
        power = torch.mean(torch.abs(residual).square()).clamp_min(1e-8)
        whiteness = float(torch.abs(torch.mean(right * torch.conj(left))) / power)
        whiteness = min(1.0, whiteness)
    else:
        whiteness = 0.0
    boundary_count = max(1, int(round(float(selected_indices.numel()) * float(boundary_fraction))))
    boundary_residual = residual[-boundary_count:]
    boundary_margin = float((1.0 / (1.0 + torch.mean(torch.abs(boundary_residual)))).detach().cpu())
    tail_risk = float(posterior.tail.variance.mean().detach().cpu())
    uncertainty = float((posterior.cir_variance.mean() + posterior.tail.variance.mean()).detach().cpu())
    confidence = float(max(0.0, min(1.0, posterior.condition_confidence())))
    score = float(nll + 0.25 * whiteness + 0.5 * tail_risk + 0.5 * uncertainty - 0.1 * boundary_margin)
    abstain = confidence < 0.15 or uncertainty > float(uncertainty_threshold) or not torch.isfinite(torch.tensor(score))
    return PilotRisk(
        score, nll, whiteness, boundary_margin, tail_risk, uncertainty,
        confidence, bool(abstain), pilot_subset=pilot_subset,
    )


def rank_pilot_candidates(risks: dict[str, PilotRisk]) -> tuple[str | None, bool]:
    """按风险排序；置信度不足或候选不可区分时 abstain。"""

    valid = [(name, risk) for name, risk in risks.items() if not risk.abstain and torch.isfinite(torch.tensor(risk.score))]
    if not valid:
        return None, True
    valid.sort(key=lambda item: item[1].score)
    if len(valid) > 1:
        gap = valid[1][1].score - valid[0][1].score
        scale = max(1.0, abs(valid[0][1].score))
        if gap <= 0.01 * scale:
            return None, True
    return valid[0][0], False
