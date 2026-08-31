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


def decision_directed_cir_update(
    frame,
    logits: torch.Tensor,
    max_delay: int,
    previous_cir: torch.Tensor,
    alpha: float,
    confidence_threshold: float | None = None,
) -> torch.Tensor:
    """用当前帧硬判决和 Adapt Pilot 标签更新接收端 CIR。

    该更新只使用接收端在线可见信息：Data 区域使用模型硬判决，Adapt Pilot
    区域使用已知导频符号；Reward/Data 标签不会进入估计。
    """

    hard_symbols = torch.where(logits >= 0, torch.ones_like(logits), -torch.ones_like(logits)).to(torch.complex64)
    tx_estimate = hard_symbols.clone()
    tx_estimate[frame.adapt_mask] = frame.tx_symbols[frame.adapt_mask]
    fit_mask = _decision_directed_fit_mask(frame.adapt_mask, logits, confidence_threshold)
    rows = []
    targets = []
    for pos in range(max_delay, tx_estimate.numel()):
        if not bool(fit_mask[pos].item()):
            continue
        row = torch.zeros(max_delay + 1, dtype=torch.complex64, device=tx_estimate.device)
        for delay in range(max_delay + 1):
            row[delay] = tx_estimate[pos - delay]
        rows.append(row)
        targets.append(frame.rx_symbols[pos])
    if len(rows) < max_delay + 1:
        return previous_cir / torch.sqrt(torch.sum(torch.abs(previous_cir) ** 2).clamp_min(1e-12))
    estimate = torch.linalg.lstsq(torch.stack(rows), torch.stack(targets)).solution.to(torch.complex64)
    estimate = estimate / torch.sqrt(torch.sum(torch.abs(estimate) ** 2).clamp_min(1e-12))
    blended = (1.0 - alpha) * previous_cir + alpha * estimate
    return blended / torch.sqrt(torch.sum(torch.abs(blended) ** 2).clamp_min(1e-12))


def pilot_sparse_cir_update(
    frame,
    previous_cir: torch.Tensor,
    soft_tail: torch.Tensor,
    *,
    max_paths: int = 24,
    alpha: float = 0.5,
    cfo_hint: float = 0.0,
) -> torch.Tensor:
    """只用当前 Adapt Pilot 更新 acquisition CIR 的主径增益。

    长时延 prefix Pilot 通常不足以估计完整 CIR，因此固定 acquisition 得到的
    稀疏 support，只对能量最大的若干 tap 做复增益最小二乘更新。未知 Data 和
    Reward Pilot 不参与行选择或目标构造；soft_tail 只作为跨帧已知历史输入。
    """

    previous = previous_cir.reshape(-1).to(torch.complex64)
    if previous.numel() == 0:
        raise ValueError("previous_cir 不能为空。")
    if int(max_paths) <= 0:
        raise ValueError("max_paths 必须为正数。")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha 必须位于 [0, 1]。")
    view = frame.receiver_view()
    rx = view.rx_symbols.reshape(-1).to(torch.complex64)
    adapt_symbols = view.adapt_symbols.reshape(-1).to(torch.complex64)
    adapt_mask = view.adapt_mask.reshape(-1).to(torch.bool)
    max_delay = previous.numel() - 1
    tail = soft_tail.reshape(-1).to(torch.complex64)
    if tail.numel() < max_delay:
        tail = torch.cat(
            (
                torch.zeros(
                    max_delay - tail.numel(),
                    dtype=torch.complex64,
                    device=tail.device,
                ),
                tail,
            )
        )
    tail = tail[-max_delay:] if max_delay > 0 else tail[:0]
    if rx.numel() != adapt_symbols.numel() or rx.numel() != adapt_mask.numel():
        raise ValueError("frame 的接收信号、Adapt Pilot 和 mask 长度必须一致。")
    support_count = min(int(max_paths), previous.numel())
    support = torch.topk(torch.abs(previous), support_count).indices.sort().values
    padded_symbols = torch.cat((tail.to(adapt_symbols.device), adapt_symbols))
    padded_known = torch.cat(
        (
            torch.ones(max_delay, dtype=torch.bool, device=adapt_symbols.device),
            adapt_mask,
        )
    )
    # 先用 acquisition CFO 先验和当前 Pilot 的公共相位估计去除慢旋转，
    # 否则长前缀上的相位斜率会被错误吸收到 tap 增益中。
    all_delays = torch.arange(max_delay + 1, device=adapt_symbols.device)
    phase_positions = torch.nonzero(adapt_mask, as_tuple=False).flatten()
    reference_values = []
    for position in phase_positions.tolist():
        source_indices = max_delay + int(position) - all_delays
        reference_values.append(torch.sum(padded_symbols[source_indices] * previous.to(padded_symbols.device)))
    reference = torch.stack(reference_values) if reference_values else torch.empty(0, dtype=torch.complex64)
    if reference.numel() > 0:
        positions = phase_positions.to(torch.float32)
        cfo_phase = 2.0 * torch.pi * float(cfo_hint) * positions
        phase_samples = torch.angle(
            rx[phase_positions] * torch.conj(reference)
            * torch.exp(-1j * cfo_phase).to(torch.complex64)
        )
        phase0 = torch.median(phase_samples)
    else:
        phase0 = torch.zeros((), dtype=torch.float32, device=adapt_symbols.device)
    positions_all = torch.arange(rx.numel(), device=rx.device, dtype=torch.float32)
    correction = torch.exp(
        -1j * (phase0.to(positions_all.device) + 2.0 * torch.pi * float(cfo_hint) * positions_all)
    ).to(torch.complex64)
    corrected_rx = rx * correction

    rows = []
    targets = []
    for position in torch.nonzero(adapt_mask, as_tuple=False).flatten().tolist():
        source_indices = max_delay + int(position) - support
        if not bool(torch.all(padded_known[source_indices]).item()):
            continue
        rows.append(padded_symbols[source_indices])
        targets.append(corrected_rx[position])
    if len(rows) < support_count:
        return previous / torch.sqrt(torch.sum(torch.abs(previous) ** 2).clamp_min(1e-12))
    estimate = torch.linalg.lstsq(torch.stack(rows), torch.stack(targets)).solution.to(torch.complex64)
    updated = previous.clone()
    updated[support] = (1.0 - float(alpha)) * previous[support] + float(alpha) * estimate
    return updated / torch.sqrt(torch.sum(torch.abs(updated) ** 2).clamp_min(1e-12))


def _decision_directed_fit_mask(
    adapt_mask: torch.Tensor,
    logits: torch.Tensor,
    confidence_threshold: float | None,
) -> torch.Tensor:
    """选择 DD-CIR LS 更新可使用的观测位置。"""

    if confidence_threshold is None:
        return torch.ones_like(adapt_mask, dtype=torch.bool)
    confidence = torch.sigmoid(torch.abs(logits.float()))
    fit_mask = confidence >= float(confidence_threshold)
    return fit_mask.to(torch.bool) | adapt_mask.to(torch.bool)


def condition_from_cir(
    cir: torch.Tensor,
    snr_db: float,
    phase_residual: float | torch.Tensor = 0.0,
    cfo_residual: float | torch.Tensor = 0.0,
    phase_features: torch.Tensor | None = None,
) -> CIRCondition:
    """从显式 CIR 构造神经均衡器需要的帧级条件。"""

    cir_b = cir.unsqueeze(0) if cir.ndim == 1 else cir
    cir_b = cir_b.to(torch.complex64)
    power = cir_b.abs()
    support = power / power.sum(dim=1, keepdim=True).clamp_min(1e-8)
    latent = torch.zeros(cir_b.shape[0], 96, dtype=torch.float32, device=cir_b.device)
    if phase_features is not None:
        features = torch.as_tensor(phase_features, dtype=torch.float32, device=cir_b.device)
        if features.ndim == 1:
            features = features.unsqueeze(0)
        features = features.reshape(features.shape[0], -1)
        if features.shape[0] == 1 and cir_b.shape[0] > 1:
            features = features.expand(cir_b.shape[0], -1)
        count = min(latent.shape[1], features.shape[1])
        latent[:, :count] = features[: cir_b.shape[0], :count]
    else:
        latent[:, 0] = torch.as_tensor(phase_residual, dtype=torch.float32, device=cir_b.device).reshape(-1)[0]
        latent[:, 1] = torch.as_tensor(cfo_residual, dtype=torch.float32, device=cir_b.device).reshape(-1)[0]
    return CIRCondition(
        complex_cir=cir_b,
        support_probability=support.to(torch.float32),
        noise_variance=torch.full((cir_b.shape[0],), 10.0 ** (-float(snr_db) / 10.0), dtype=torch.float32, device=cir_b.device),
        confidence=torch.ones(cir_b.shape[0], dtype=torch.float32, device=cir_b.device),
        latent_residual=latent,
    )
