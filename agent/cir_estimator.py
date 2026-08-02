# -*- coding: utf-8 -*-
"""Adapt Pilot 驱动的显式稀疏 CIR 估计器。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class CIRCondition:
    complex_cir: torch.Tensor
    support_probability: torch.Tensor
    noise_variance: torch.Tensor
    confidence: torch.Tensor
    latent_residual: torch.Tensor


class HybridCIREstimator(nn.Module):
    """Adapt Pilot/acquisition 驱动的显式稀疏 CIR 与隐式残差估计器。"""

    def __init__(self, max_delay: int = 40, latent_dim: int = 96, support_temperature: float = 12.0):
        super().__init__()
        self.max_delay = int(max_delay)
        self.latent_dim = int(latent_dim)
        self.support_temperature = float(support_temperature)
        if self.max_delay < 0:
            raise ValueError("max_delay 不能为负。")

    def forward(
        self,
        rx_iq: torch.Tensor,
        adapt_symbols: torch.Tensor,
        adapt_mask: torch.Tensor,
        region_ids: torch.Tensor,
    ) -> CIRCondition:
        del region_ids
        rx = torch.complex(rx_iq[..., 0], rx_iq[..., 1]).to(torch.complex64)
        tx = adapt_symbols.to(torch.complex64)
        mask = adapt_mask.to(torch.bool)
        cir = self._least_squares_cir(rx, tx, mask)
        prediction = self._predict(tx, cir)
        residual = torch.where(mask, rx - prediction, torch.zeros_like(rx))
        noise = self._estimate_noise(residual, mask)
        power = torch.abs(cir)
        reference = power.amax(dim=1, keepdim=True).clamp_min(1e-8)
        support_probability = torch.sigmoid(self.support_temperature * (power / reference - 0.15))
        confidence = (1.0 / (1.0 + noise)).to(torch.float32)
        latent_residual = self._pool_residual(residual, mask)
        return CIRCondition(
            complex_cir=cir,
            support_probability=support_probability,
            noise_variance=noise,
            confidence=confidence,
            latent_residual=latent_residual,
        )

    def _least_squares_cir(self, rx: torch.Tensor, tx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, frame_len = rx.shape
        cir = torch.zeros(batch, self.max_delay + 1, dtype=torch.complex64, device=rx.device)
        for batch_index in range(batch):
            rows = []
            targets = []
            valid_positions = torch.nonzero(mask[batch_index], as_tuple=False).flatten()
            for pos in valid_positions.tolist():
                row = torch.zeros(self.max_delay + 1, dtype=torch.complex64, device=rx.device)
                for delay in range(self.max_delay + 1):
                    source = pos - delay
                    if source >= 0:
                        row[delay] = tx[batch_index, source]
                if torch.any(row != 0):
                    rows.append(row)
                    targets.append(rx[batch_index, pos])
            if not rows:
                continue
            design = torch.stack(rows, dim=0)
            target = torch.stack(targets, dim=0)
            solution = torch.linalg.lstsq(design, target).solution
            cir[batch_index] = solution.to(torch.complex64)
        return cir

    def _predict(self, tx: torch.Tensor, cir: torch.Tensor) -> torch.Tensor:
        batch, frame_len = tx.shape
        prediction = torch.zeros(batch, frame_len, dtype=torch.complex64, device=tx.device)
        for delay in range(self.max_delay + 1):
            if delay == 0:
                shifted = tx
            else:
                shifted = torch.zeros_like(tx)
                shifted[:, delay:] = tx[:, :-delay]
            prediction = prediction + cir[:, delay].unsqueeze(1) * shifted
        return prediction

    def _estimate_noise(self, residual: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        squared = torch.abs(residual) ** 2
        count = mask.sum(dim=1).clamp_min(1).to(squared.dtype)
        return (squared.sum(dim=1) / count).to(torch.float32)

    def _pool_residual(self, residual: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        visible = torch.where(mask, residual.real, torch.zeros_like(residual.real))
        if visible.shape[1] >= self.latent_dim:
            return visible[:, : self.latent_dim].to(torch.float32)
        padding = torch.zeros(visible.shape[0], self.latent_dim - visible.shape[1], device=visible.device)
        return torch.cat([visible, padding], dim=1).to(torch.float32)


def decision_directed_cir_update(frame, logits: torch.Tensor, max_delay: int, previous_cir: torch.Tensor, alpha: float) -> torch.Tensor:
    """用当前帧硬判决和 Adapt Pilot 标签更新接收端 CIR。

    该更新只使用接收端在线可见信息：Data 区域使用模型硬判决，Adapt Pilot
    区域使用已知导频符号；Reward/Data 标签不会进入估计。
    """

    hard_symbols = torch.where(logits >= 0, torch.ones_like(logits), -torch.ones_like(logits)).to(torch.complex64)
    tx_estimate = hard_symbols.clone()
    tx_estimate[frame.adapt_mask] = frame.tx_symbols[frame.adapt_mask]
    rows = []
    targets = []
    for pos in range(max_delay, tx_estimate.numel()):
        row = torch.zeros(max_delay + 1, dtype=torch.complex64, device=tx_estimate.device)
        for delay in range(max_delay + 1):
            row[delay] = tx_estimate[pos - delay]
        rows.append(row)
        targets.append(frame.rx_symbols[pos])
    estimate = torch.linalg.lstsq(torch.stack(rows), torch.stack(targets)).solution.to(torch.complex64)
    estimate = estimate / torch.sqrt(torch.sum(torch.abs(estimate) ** 2).clamp_min(1e-12))
    blended = (1.0 - alpha) * previous_cir + alpha * estimate
    return blended / torch.sqrt(torch.sum(torch.abs(blended) ** 2).clamp_min(1e-12))
