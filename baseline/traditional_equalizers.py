# -*- coding: utf-8 -*-
"""传统非神经、非 RL 均衡器集合。

这些 baseline 只使用接收端在线可见信息：接收符号、Adapt Pilot 已知发送
符号、acquisition/Adapt 估计得到的 CIR、soft tail 和 SNR。实现不读取
Reward Pilot 标签或 Data 标签，也不包含神经网络或 RL 策略。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from baseline.synchronization_compensation import apply_phase_correction, estimate_pilot_phase_line, unwrap_phase as _unwrap_phase_local


TRADITIONAL_BASELINES = (
    "LMMSE-FIR",
    "LMS",
    "NLMS",
    "RLS Linear",
    "DFE-RLS",
    "SC-FDE-MMSE",
    "CFO-Corrected LMMSE-FIR",
    "CFO-Corrected DFE-RLS",
    "CFO+DD-Phase LMMSE-FIR",
    "CFO+DD-Phase DFE-RLS",
)


@dataclass(frozen=True)
class TraditionalResult:
    method: str
    logits: torch.Tensor
    soft_tail: torch.Tensor
    iterations: int
    extra: dict


@dataclass
class TraditionalPhaseState:
    phase0: float = 0.0
    cfo_cycles_per_symbol: float = 0.0
    smoothing: float = 0.35

    def clone(self) -> "TraditionalPhaseState":
        return TraditionalPhaseState(
            phase0=float(self.phase0),
            cfo_cycles_per_symbol=float(self.cfo_cycles_per_symbol),
            smoothing=float(self.smoothing),
        )


def run_traditional_equalizer(
    method: str,
    receiver_view,
    cir: torch.Tensor,
    soft_tail: torch.Tensor,
    snr_db: float,
    phase_state: TraditionalPhaseState | None = None,
) -> TraditionalResult:
    """运行一个传统均衡器。"""

    if method not in TRADITIONAL_BASELINES:
        raise ValueError(f"未知传统均衡器：{method}")

    residual_cfo_limit = 0.001
    compensation_extra: dict = {}
    base_method = method
    rx_source = receiver_view.rx_symbols
    use_dd_phase_tracker = method.startswith("CFO+DD-Phase ")
    if use_dd_phase_tracker:
        base_method = method.removeprefix("CFO+DD-Phase ")
        reference, reference_mask = _pilot_reference_from_cir(receiver_view, cir, soft_tail)
        phase_estimate = estimate_pilot_phase_line(
            receiver_view,
            reference_symbols=reference,
            reference_mask=reference_mask,
            max_abs_cfo_cycles_per_symbol=residual_cfo_limit,
        )
        if phase_state is None:
            phase_state = TraditionalPhaseState()
        phase0 = (1.0 - phase_state.smoothing) * phase_state.phase0 + phase_state.smoothing * phase_estimate.phase0
        cfo = (1.0 - phase_state.smoothing) * phase_state.cfo_cycles_per_symbol + phase_state.smoothing * phase_estimate.cfo_cycles_per_symbol
        rx_source = apply_phase_correction(receiver_view.rx_symbols, phase0, cfo)
        phase_state.phase0 = float(phase0)
        phase_state.cfo_cycles_per_symbol = float(cfo)
        compensation_extra = {
            "phase_compensation": "pilot_linear_fit+decision_directed_tracker",
            "estimated_cfo_cycles_per_symbol": float(cfo),
            "phase_pilot_count": phase_estimate.pilot_count,
        }
    elif method.startswith("CFO-Corrected "):
        base_method = method.removeprefix("CFO-Corrected ")
        reference, reference_mask = _pilot_reference_from_cir(receiver_view, cir, soft_tail)
        phase_estimate = estimate_pilot_phase_line(
            receiver_view,
            reference_symbols=reference,
            reference_mask=reference_mask,
            max_abs_cfo_cycles_per_symbol=residual_cfo_limit,
        )
        phase0 = _pilot_common_phase(receiver_view, reference, reference_mask)
        rx_source = apply_phase_correction(receiver_view.rx_symbols, phase0, 0.0)
        compensation_extra = {
            "phase_compensation": "pilot_common_phase",
            "estimated_cfo_cycles_per_symbol": phase_estimate.cfo_cycles_per_symbol,
            "estimated_phase0": phase0,
            "phase_pilot_count": phase_estimate.pilot_count,
        }
    rx = rx_source.flatten().to(torch.complex64)
    adapt_mask = receiver_view.adapt_mask.flatten().bool()
    adapt_tx = receiver_view.adapt_symbols.flatten().to(torch.complex64)
    cir = cir.flatten().to(torch.complex64)
    noise_variance = torch.tensor(10.0 ** (-float(snr_db) / 10.0), dtype=torch.float32, device=rx.device)
    filter_len = _filter_length(cir)

    if base_method == "LMMSE-FIR":
        estimate, iterations = _lmmse_fir(rx, adapt_mask, adapt_tx, filter_len, noise_variance)
        algorithm = "time_domain_lmmse_fir"
    elif base_method == "LMS":
        estimate, iterations = _lms(rx, adapt_mask, adapt_tx, filter_len, step_size=0.015, normalized=False)
        algorithm = "pilot_lms_linear"
    elif base_method == "NLMS":
        estimate, iterations = _lms(rx, adapt_mask, adapt_tx, filter_len, step_size=0.35, normalized=True)
        algorithm = "pilot_nlms_linear"
    elif base_method == "RLS Linear":
        estimate, iterations = _rls_linear(rx, adapt_mask, adapt_tx, filter_len)
        algorithm = "pilot_rls_linear"
    elif base_method == "DFE-RLS":
        estimate, iterations = _dfe_rls(rx, adapt_mask, adapt_tx, filter_len)
        algorithm = "pilot_rls_dfe"
    elif base_method == "SC-FDE-MMSE":
        estimate, iterations = _sc_fde_mmse(rx, cir, noise_variance)
        algorithm = "single_carrier_fde_mmse"
    else:
        raise ValueError(f"未知传统均衡器：{method}")

    logits = _estimate_to_logits(estimate, noise_variance)
    if use_dd_phase_tracker and phase_state is not None:
        residual = _decision_directed_phase_residual(receiver_view.rx_symbols, estimate, cir, next_tail=_tail_from_estimate(estimate, soft_tail.numel()))
        phase_state.phase0 = float((1.0 - phase_state.smoothing) * phase_state.phase0 + phase_state.smoothing * residual)
        compensation_extra["decision_directed_phase_residual"] = float(residual)
    next_tail = _tail_from_estimate(estimate, soft_tail.numel())
    return TraditionalResult(
        method=method,
        logits=logits,
        soft_tail=next_tail.to(device=soft_tail.device),
        iterations=int(iterations),
        extra={
            "traditional": True,
            "uses_neural_network": False,
            "uses_rl": False,
            "algorithm": f"dd_phase_tracked_{algorithm}" if use_dd_phase_tracker else f"cfo_corrected_{algorithm}" if compensation_extra else algorithm,
            "shared_placeholder_kernel": False,
            "uses_reward_or_data_labels": False,
            **compensation_extra,
        },
    )


def _filter_length(cir: torch.Tensor) -> int:
    return int(max(5, min(41, cir.numel() * 2 + 1)))


def estimate_acquisition_cir_with_cfo(frame, max_delay: int, cfo_limit: float = 0.004, grid_points: int = 41) -> tuple[torch.Tensor, float]:
    """用全已知 acquisition frame 估计 CFO 补偿后的 CIR。

    acquisition frame 的发送符号在接收端可知，因此这是传统同步/信道估计可用信息，
    不读取 Reward/Data 标签，也不使用真实 impairment 参数。
    """

    return _estimate_cir_with_cfo_grid(
        frame.tx_symbols.to(torch.complex64),
        frame.rx_symbols.to(torch.complex64),
        int(max_delay),
        cfo_limit=float(cfo_limit),
        grid_points=int(grid_points),
    )


def estimate_phase_residual_features(receiver_view, cir: torch.Tensor, soft_tail: torch.Tensor) -> tuple[float, float]:
    """从 Adapt Pilot 估计神经接收机可见的相位残差特征。

    该特征与传统补偿使用同一信息边界：rx、Adapt Pilot、CIR、soft tail。
    不读取 Reward/Data 标签，也不读取真实 impairment。
    """

    reference, reference_mask = _pilot_reference_from_cir(receiver_view, cir, soft_tail)
    estimate = estimate_pilot_phase_line(receiver_view, reference_symbols=reference, reference_mask=reference_mask)
    return float(estimate.phase0), float(estimate.cfo_cycles_per_symbol)


def estimate_phase_residual_vector(
    receiver_view,
    cir: torch.Tensor,
    soft_tail: torch.Tensor,
    blocks: int = 4,
) -> torch.Tensor:
    """提取局部 Pilot 相位统计，供神经 conditioner 使用。

    每个 block 输出四个传统可见统计：
    phase intercept、phase slope(CFO cycles/symbol)、phase residual variance、
    normalized residual energy。
    """

    reference, reference_mask = _pilot_reference_from_cir(receiver_view, cir, soft_tail)
    rx = receiver_view.rx_symbols.flatten().to(torch.complex64)
    adapt_mask = receiver_view.adapt_mask.flatten().bool()
    mask = adapt_mask & reference_mask.flatten().bool() & (reference.flatten().abs() > 1e-6)
    positions = torch.nonzero(mask, as_tuple=False).flatten()
    output = torch.zeros(int(blocks) * 4, dtype=torch.float32, device=rx.device)
    if positions.numel() < 2:
        return output.cpu()
    chunks = torch.chunk(positions, int(blocks))
    reference_flat = reference.flatten().to(torch.complex64)
    for block_index, chunk in enumerate(chunks):
        if chunk.numel() < 2:
            continue
        phase = torch.angle(rx[chunk] * torch.conj(reference_flat[chunk]))
        phase = _unwrap_phase_local(phase)
        t = chunk.to(torch.float32)
        centered_t = t - t.mean()
        centered_phase = phase - phase.mean()
        denom = centered_t.square().sum().clamp_min(1e-8)
        slope = (centered_t * centered_phase).sum() / denom
        intercept = phase.mean() - slope * t.mean()
        residual = phase - (intercept + slope * t)
        residual_energy = torch.mean(torch.abs(rx[chunk] - reference_flat[chunk]) ** 2) / torch.mean(torch.abs(reference_flat[chunk]) ** 2).clamp_min(1e-8)
        offset = block_index * 4
        output[offset + 0] = intercept.to(torch.float32)
        output[offset + 1] = (slope / (2.0 * torch.pi)).to(torch.float32)
        output[offset + 2] = torch.var(residual, unbiased=False).to(torch.float32)
        output[offset + 3] = residual_energy.real.to(torch.float32)
    return output.cpu()


def _pilot_common_phase(receiver_view, reference_symbols: torch.Tensor, reference_mask: torch.Tensor) -> float:
    """用可靠 Adapt Pilot 的中位公共相位做保守补偿。"""

    rx = receiver_view.rx_symbols.flatten().to(torch.complex64)
    reference = reference_symbols.flatten().to(torch.complex64)
    mask = receiver_view.adapt_mask.flatten().bool() & reference_mask.flatten().bool() & (reference.abs() > 1e-6)
    if int(mask.sum().item()) < 2:
        return 0.0
    phase = torch.angle(rx[mask] * torch.conj(reference[mask]))
    return float(torch.median(phase).item())


def _estimate_cir_with_cfo_grid(
    tx: torch.Tensor,
    rx: torch.Tensor,
    max_delay: int,
    *,
    cfo_limit: float = 0.004,
    grid_points: int = 41,
) -> tuple[torch.Tensor, float]:
    """传统网格搜索：试探 CFO，反旋 rx 后做 LS CIR，并选择残差最小者。"""

    tx = tx.flatten().to(torch.complex64)
    rx = rx.flatten().to(torch.complex64)
    frame_len = tx.numel()
    if frame_len <= max_delay:
        raise ValueError("用于 acquisition CIR 估计的帧长必须大于 max_delay。")
    design = _known_tx_design(tx, int(max_delay))
    positions = torch.arange(int(max_delay), frame_len, dtype=torch.float32, device=rx.device)
    targets = rx[int(max_delay) :]
    candidates = torch.linspace(-float(cfo_limit), float(cfo_limit), int(grid_points), dtype=torch.float32, device=rx.device)
    best_cir: torch.Tensor | None = None
    best_cfo = 0.0
    best_error = float("inf")
    for cfo in candidates:
        phase = torch.exp(-1j * (2.0 * torch.pi * cfo * positions)).to(torch.complex64)
        corrected = targets * phase
        cir = torch.linalg.lstsq(design, corrected).solution.to(torch.complex64)
        residual = design @ cir - corrected
        error = float(torch.mean(torch.abs(residual) ** 2).real.item())
        if error < best_error:
            best_error = error
            best_cfo = float(cfo.item())
            best_cir = cir
    if best_cir is None:
        best_cir = torch.zeros(int(max_delay) + 1, dtype=torch.complex64, device=rx.device)
        best_cir[0] = 1.0 + 0.0j
    norm = torch.sqrt(torch.sum(torch.abs(best_cir) ** 2).clamp_min(1e-12))
    return (best_cir / norm).to(torch.complex64), best_cfo


def _known_tx_design(tx: torch.Tensor, max_delay: int) -> torch.Tensor:
    rows = []
    for pos in range(max_delay, tx.numel()):
        row = torch.zeros(max_delay + 1, dtype=torch.complex64, device=tx.device)
        for delay in range(max_delay + 1):
            row[delay] = tx[pos - delay]
        rows.append(row)
    return torch.stack(rows, dim=0)


def _pilot_reference_from_cir(receiver_view, cir: torch.Tensor, soft_tail: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """用可见 Adapt Pilot、soft tail 和 acquisition CIR 构造 pilot 参考接收波形。

    对于前缀 Pilot，只有由 tail 与 Adapt Pilot 完全决定的样本才可靠；一旦卷积窗口
    触及未知 Reward/Data 区域，就不用于相位/CFO 拟合。
    """

    adapt_tx = receiver_view.adapt_symbols.flatten().to(torch.complex64)
    adapt_mask = receiver_view.adapt_mask.flatten().bool()
    cir = cir.flatten().to(torch.complex64)
    max_delay = max(0, cir.numel() - 1)
    tail = soft_tail.flatten().to(torch.complex64)
    if max_delay > 0:
        if tail.numel() < max_delay:
            tail = torch.cat((torch.zeros(max_delay - tail.numel(), dtype=torch.complex64, device=adapt_tx.device), tail))
        tail = tail[-max_delay:].to(device=adapt_tx.device)
    padded_symbols = torch.cat((tail, adapt_tx)) if max_delay > 0 else adapt_tx
    padded_known = torch.cat((torch.ones(max_delay, dtype=torch.bool, device=adapt_tx.device), adapt_mask)) if max_delay > 0 else adapt_mask
    frame_len = adapt_tx.numel()
    reference = torch.zeros(frame_len, dtype=torch.complex64, device=adapt_tx.device)
    reliable = torch.zeros(frame_len, dtype=torch.bool, device=adapt_tx.device)
    origin = max_delay
    indices = torch.arange(frame_len, device=adapt_tx.device)
    for delay, tap in enumerate(cir):
        if tap.abs() <= 1e-8:
            continue
        positions = origin + indices - int(delay)
        valid = (positions >= 0) & (positions < padded_symbols.numel())
        reference[valid] += tap.to(device=adapt_tx.device) * padded_symbols[positions[valid]]
        reliable = reliable | (~valid)
    known = torch.ones(frame_len, dtype=torch.bool, device=adapt_tx.device)
    for delay, tap in enumerate(cir):
        if tap.abs() <= 1e-8:
            continue
        positions = origin + indices - int(delay)
        valid = (positions >= 0) & (positions < padded_known.numel())
        known = known & valid
        known[valid] = known[valid] & padded_known[positions[valid]]
        known[~valid] = False
    reliable = adapt_mask & known & (reference.abs() > 1e-6)
    return reference, reliable


def _decision_directed_phase_residual(rx_symbols: torch.Tensor, estimate: torch.Tensor, cir: torch.Tensor, next_tail: torch.Tensor) -> float:
    """用硬判决重构接收波形，估计残余公共相位。

    这是传统 decision-directed 跟踪：只使用当前接收信号、当前均衡判决和 CIR，
    不读取 Reward/Data 标签。
    """

    hard = torch.where(estimate.real >= 0, torch.ones_like(estimate.real), -torch.ones_like(estimate.real))
    tx_hat = torch.complex(hard.to(torch.float32), torch.zeros_like(hard).to(torch.float32))
    max_delay = max(0, cir.numel() - 1)
    if max_delay <= 0:
        reference = tx_hat
    else:
        tail = next_tail.flatten().to(torch.complex64)
        if tail.numel() < max_delay:
            tail = torch.cat((torch.zeros(max_delay - tail.numel(), dtype=torch.complex64, device=tx_hat.device), tail))
        padded = torch.cat((tail[-max_delay:].to(device=tx_hat.device), tx_hat))
        reference = torch.zeros_like(tx_hat)
        origin = max_delay
        indices = torch.arange(tx_hat.numel(), device=tx_hat.device)
        for delay, tap in enumerate(cir.flatten().to(torch.complex64)):
            if tap.abs() <= 1e-8:
                continue
            reference += tap.to(device=tx_hat.device) * padded[origin + indices - int(delay)]
    mask = reference.abs() > 1e-5
    if int(mask.sum().item()) < 8:
        return 0.0
    residual = torch.angle(rx_symbols.flatten().to(torch.complex64)[mask] * torch.conj(reference[mask]))
    return float(torch.median(residual).item())


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


def _tail_from_estimate(estimate: torch.Tensor, tail_len: int) -> torch.Tensor:
    if int(tail_len) <= 0:
        return torch.zeros(0, dtype=torch.complex64, device=estimate.device)
    tail_source = estimate[-int(tail_len) :].real
    hard = torch.where(tail_source >= 0, torch.ones_like(tail_source), -torch.ones_like(tail_source))
    return torch.complex(hard.to(torch.float32), torch.zeros_like(hard).to(torch.float32))
