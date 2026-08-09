# -*- coding: utf-8 -*-
"""传统非神经、非 RL 均衡器集合。

这些 baseline 只使用接收端在线可见信息：接收符号、Adapt Pilot 已知发送
符号、acquisition/Adapt 估计得到的 CIR、soft tail 和 SNR。实现不读取
Reward Pilot 标签或 Data 标签，也不包含神经网络或 RL 策略。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


TRADITIONAL_BASELINES = (
    "LMMSE-FIR",
    "LMS",
    "NLMS",
    "RLS Linear",
    "DFE-RLS",
    "SC-FDE-MMSE",
)


@dataclass(frozen=True)
class TraditionalResult:
    method: str
    logits: torch.Tensor
    soft_tail: torch.Tensor
    iterations: int
    extra: dict


def run_traditional_equalizer(
    method: str,
    receiver_view,
    cir: torch.Tensor,
    soft_tail: torch.Tensor,
    snr_db: float,
) -> TraditionalResult:
    """运行一个传统均衡器。"""

    if method not in TRADITIONAL_BASELINES:
        raise ValueError(f"未知传统均衡器：{method}")

    rx = receiver_view.rx_symbols.flatten().to(torch.complex64)
    adapt_mask = receiver_view.adapt_mask.flatten().bool()
    adapt_tx = receiver_view.adapt_symbols.flatten().to(torch.complex64)
    cir = cir.flatten().to(torch.complex64)
    noise_variance = torch.tensor(10.0 ** (-float(snr_db) / 10.0), dtype=torch.float32, device=rx.device)
    filter_len = _filter_length(cir)

    if method == "LMMSE-FIR":
        estimate, iterations = _lmmse_fir(rx, adapt_mask, adapt_tx, filter_len, noise_variance)
        algorithm = "time_domain_lmmse_fir"
    elif method == "LMS":
        estimate, iterations = _lms(rx, adapt_mask, adapt_tx, filter_len, step_size=0.015, normalized=False)
        algorithm = "pilot_lms_linear"
    elif method == "NLMS":
        estimate, iterations = _lms(rx, adapt_mask, adapt_tx, filter_len, step_size=0.35, normalized=True)
        algorithm = "pilot_nlms_linear"
    elif method == "RLS Linear":
        estimate, iterations = _rls_linear(rx, adapt_mask, adapt_tx, filter_len)
        algorithm = "pilot_rls_linear"
    elif method == "DFE-RLS":
        estimate, iterations = _dfe_rls(rx, adapt_mask, adapt_tx, filter_len)
        algorithm = "pilot_rls_dfe"
    elif method == "SC-FDE-MMSE":
        estimate, iterations = _sc_fde_mmse(rx, cir, noise_variance)
        algorithm = "single_carrier_fde_mmse"
    else:
        raise ValueError(f"未知传统均衡器：{method}")

    logits = _estimate_to_logits(estimate, noise_variance)
    return TraditionalResult(
        method=method,
        logits=logits,
        soft_tail=soft_tail.clone(),
        iterations=int(iterations),
        extra={
            "traditional": True,
            "uses_neural_network": False,
            "uses_rl": False,
            "algorithm": algorithm,
            "shared_placeholder_kernel": False,
        },
    )


def _filter_length(cir: torch.Tensor) -> int:
    return int(max(5, min(41, cir.numel() * 2 + 1)))


def _lag_matrix(rx: torch.Tensor, filter_len: int) -> torch.Tensor:
    frame_len = rx.numel()
    center = filter_len // 2
    padded = torch.cat(
        (
            torch.zeros(center, dtype=rx.dtype, device=rx.device),
            rx,
            torch.zeros(filter_len - center - 1, dtype=rx.dtype, device=rx.device),
        )
    )
    rows = [padded[offset : offset + frame_len] for offset in range(filter_len)]
    return torch.stack(rows, dim=1)


def _adapt_targets(adapt_mask: torch.Tensor, adapt_tx: torch.Tensor, frame_len: int) -> torch.Tensor:
    target = torch.zeros(frame_len, dtype=torch.complex64, device=adapt_tx.device)
    target[adapt_mask] = adapt_tx[adapt_mask]
    return target


def _lmmse_fir(
    rx: torch.Tensor,
    adapt_mask: torch.Tensor,
    adapt_tx: torch.Tensor,
    filter_len: int,
    noise_variance: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    x = _lag_matrix(rx, filter_len)
    d_full = _adapt_targets(adapt_mask, adapt_tx, rx.numel())
    xa = x[adapt_mask]
    da = d_full[adapt_mask]
    reg = noise_variance.real.clamp_min(1e-5) * torch.eye(filter_len, dtype=torch.complex64, device=rx.device)
    weights = torch.linalg.solve(xa.conj().T @ xa + reg, xa.conj().T @ da)
    return x @ weights, filter_len


def _lms(
    rx: torch.Tensor,
    adapt_mask: torch.Tensor,
    adapt_tx: torch.Tensor,
    filter_len: int,
    step_size: float,
    normalized: bool,
) -> tuple[torch.Tensor, int]:
    x = _lag_matrix(rx, filter_len)
    d_full = _adapt_targets(adapt_mask, adapt_tx, rx.numel())
    weights = torch.zeros(filter_len, dtype=torch.complex64, device=rx.device)
    weights[filter_len // 2] = 1.0 + 0.0j
    updates = 0
    for index in torch.nonzero(adapt_mask, as_tuple=False).flatten().tolist():
        vector = x[index]
        prediction = torch.dot(vector, weights)
        error = d_full[index] - prediction
        denom = vector.abs().square().sum().real.clamp_min(1e-4) if normalized else torch.tensor(1.0, device=rx.device)
        weights = weights + (step_size / denom) * torch.conj(vector) * error
        updates += 1
    return x @ weights, updates


def _rls_linear(
    rx: torch.Tensor,
    adapt_mask: torch.Tensor,
    adapt_tx: torch.Tensor,
    filter_len: int,
    forgetting: float = 0.995,
) -> tuple[torch.Tensor, int]:
    x = _lag_matrix(rx, filter_len)
    d_full = _adapt_targets(adapt_mask, adapt_tx, rx.numel())
    weights = torch.zeros(filter_len, dtype=torch.complex64, device=rx.device)
    weights[filter_len // 2] = 1.0 + 0.0j
    inverse_cov = torch.eye(filter_len, dtype=torch.complex64, device=rx.device) * 10.0
    updates = 0
    for index in torch.nonzero(adapt_mask, as_tuple=False).flatten().tolist():
        vector = x[index].unsqueeze(1)
        gain_den = forgetting + (vector.conj().T @ inverse_cov @ vector).squeeze().real
        gain = (inverse_cov @ vector / gain_den.clamp_min(1e-6)).squeeze(1)
        prediction = torch.dot(x[index], weights)
        error = d_full[index] - prediction
        weights = weights + torch.conj(gain) * error
        inverse_cov = (inverse_cov - gain.unsqueeze(1) @ vector.conj().T @ inverse_cov) / forgetting
        updates += 1
    return x @ weights, updates


def _dfe_rls(
    rx: torch.Tensor,
    adapt_mask: torch.Tensor,
    adapt_tx: torch.Tensor,
    filter_len: int,
    feedback_len: int = 5,
) -> tuple[torch.Tensor, int]:
    x = _lag_matrix(rx, filter_len)
    d_full = _adapt_targets(adapt_mask, adapt_tx, rx.numel())
    feature_len = filter_len + feedback_len
    weights = torch.zeros(feature_len, dtype=torch.complex64, device=rx.device)
    weights[filter_len // 2] = 1.0 + 0.0j
    inverse_cov = torch.eye(feature_len, dtype=torch.complex64, device=rx.device) * 10.0
    decisions = torch.zeros(rx.numel(), dtype=torch.complex64, device=rx.device)
    updates = 0
    for index in torch.nonzero(adapt_mask, as_tuple=False).flatten().tolist():
        feedback = _feedback_features(decisions, index, feedback_len)
        vector = torch.cat((x[index], feedback))
        prediction = torch.dot(vector, weights)
        error = d_full[index] - prediction
        col = vector.unsqueeze(1)
        gain_den = 0.995 + (col.conj().T @ inverse_cov @ col).squeeze().real
        gain = (inverse_cov @ col / gain_den.clamp_min(1e-6)).squeeze(1)
        weights = weights + torch.conj(gain) * error
        inverse_cov = (inverse_cov - gain.unsqueeze(1) @ col.conj().T @ inverse_cov) / 0.995
        decisions[index] = d_full[index]
        updates += 1

    output = torch.zeros_like(rx)
    for index in range(rx.numel()):
        hard = torch.where(output.real >= 0, torch.ones_like(output.real), -torch.ones_like(output.real)).to(torch.complex64)
        vector = torch.cat((x[index], _feedback_features(hard, index, feedback_len)))
        estimate = torch.dot(vector, weights)
        output[index] = 1.0 + 0.0j if estimate.real >= 0 else -1.0 + 0.0j
    return output, updates


def _feedback_features(decisions: torch.Tensor, index: int, feedback_len: int) -> torch.Tensor:
    values = []
    for delay in range(1, feedback_len + 1):
        pos = index - delay
        values.append(decisions[pos] if pos >= 0 else torch.zeros((), dtype=torch.complex64, device=decisions.device))
    return torch.stack(values)


def _sc_fde_mmse(rx: torch.Tensor, cir: torch.Tensor, noise_variance: torch.Tensor) -> tuple[torch.Tensor, int]:
    frame_len = rx.numel()
    padded_cir = torch.zeros(frame_len, dtype=torch.complex64, device=rx.device)
    padded_cir[: min(frame_len, cir.numel())] = cir[: min(frame_len, cir.numel())]
    h = torch.fft.fft(padded_cir)
    y = torch.fft.fft(rx)
    equalizer = torch.conj(h) / (h.abs().square() + noise_variance.real.clamp_min(1e-6))
    return torch.fft.ifft(y * equalizer), frame_len


def _estimate_to_logits(estimate: torch.Tensor, noise_variance: torch.Tensor) -> torch.Tensor:
    scale = 2.0 / noise_variance.real.clamp_min(1e-4)
    return (estimate.real * scale).to(torch.float32).clamp(-40.0, 40.0)
