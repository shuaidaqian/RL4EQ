# -*- coding: utf-8 -*-
"""持续均衡开发接收机：合法估噪、未归一化 CIR 与前缀 Pilot 递归。

本模块不读取环境 SNR、Data/Reward 标签、真实状态或普通帧真实 tail。
线性高斯递推及 soft Data 更新仍是近似模型，校准与收益需独立验证。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from scipy.optimize import minimize_scalar

from agent.cir_estimator import CIRCondition
from baseline.traditional_equalizers import estimate_phase_residual_vector
from env.frame_structure import Frame
from training.pilot_transfer import TransferContext, context_logits, tail_from_logits


ARMS = ("acquisition", "current_pilot", "recursive_pilot", "recursive_soft")


def acquisition_transition_intervals(frame_duration_seconds, gap_seconds=0.0):
    """acquisition 帧末的一次演化，加上已知额外空档的等效帧数。"""

    duration, gap = float(frame_duration_seconds), float(gap_seconds)
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(gap) or gap < 0:
        raise ValueError("帧时长须为正有限数，acquisition 空档须为非负有限数。")
    intervals = 1.0 + gap / duration
    if not math.isfinite(intervals):
        raise ValueError("acquisition 到首帧的等效转移间隔必须有限。")
    return intervals


def predict_cir_state(mean, covariance, *, state_model, process_noise, state_rho=None,
                      stationary_power=None, transition_intervals=1.0):
    """应用冻结的随机游走或离线估计 AR(1) CIR 预测。"""

    intervals = float(transition_intervals)
    if not math.isfinite(intervals) or intervals < 0:
        raise ValueError("状态预测的转移间隔须为非负有限数。")
    if state_model == "random_walk":
        if process_noise < 0 or not math.isfinite(process_noise):
            raise ValueError("随机游走过程噪声必须非负且有限。")
        predicted_covariance = covariance + intervals * float(process_noise) * torch.eye(
            mean.numel(), device=mean.device, dtype=mean.dtype
        )
        return mean, predicted_covariance
    if state_model != "ar1":
        raise ValueError("state_model 必须为 random_walk 或 ar1。")
    if state_rho is None or not math.isfinite(float(state_rho)) or not 0 <= float(state_rho) < 1:
        raise ValueError("AR(1) rho 必须位于 [0,1) 且有限。")
    power = torch.as_tensor(stationary_power, device=mean.device, dtype=mean.real.dtype)
    if power.ndim != 1 or power.numel() != mean.numel() or not bool(torch.isfinite(power).all()) or bool((power < 0).any()):
        raise ValueError("AR(1) stationary_power 必须是 CIR 维度一致的非负有限向量。")
    rho = float(state_rho) ** intervals
    stationary_covariance = torch.diag(power.to(dtype=covariance.dtype))
    predicted_mean = rho * mean
    predicted_covariance = rho**2 * covariance + (1.0 - rho**2) * stationary_covariance
    predicted_covariance = (predicted_covariance + predicted_covariance.mH) * 0.5
    return predicted_mean, predicted_covariance


def receiver_frame(frame: Frame) -> Frame:
    """在接收边界移除全部隐藏字段；发送符号只保留 Adapt 部分。"""

    tx = torch.zeros_like(frame.rx_symbols)
    tx[frame.adapt_mask] = frame.tx_symbols[frame.adapt_mask]
    return Frame(
        frame_index=frame.frame_index, bits=torch.zeros_like(frame.adapt_mask),
        tx_symbols=tx, rx_symbols=frame.rx_symbols.detach().clone(),
        adapt_mask=frame.adapt_mask.clone(), reward_mask=frame.reward_mask.clone(),
        data_mask=frame.data_mask.clone(), model_region_ids=frame.model_region_ids.clone(),
        adapt_block_lengths=tuple(frame.adapt_block_lengths),
    )


def _design(symbols, positions, max_delay):
    return torch.stack([symbols[positions - delay] for delay in range(max_delay + 1)], 1)


def _hermitian(matrix):
    return (matrix + matrix.mH) * 0.5


@dataclass(frozen=True)
class AcquisitionEstimate:
    mean: torch.Tensor
    covariance: torch.Tensor
    cfo: float
    noise_variance: float
    residual_power: float
    rank: int
    observations: int
    noise_source: str


def estimate_acquisition(tx, rx, max_delay, *, cfo_limit=0.004, grid_points=41,
                         noise_reference=None):
    """从全已知 acquisition 估计 CFO/CIR 与残差噪声，无需预知 SNR。

    QR 投影复用所有 CFO 候选，局部细化粗网格最优值。残差自由度修正仅在
    固定正确同步与线性白噪声模型下无偏；这里记录为残差噪声近似估计。
    ``noise_reference`` 只供显式标注的理想参照实验使用。
    """

    if max_delay < 0 or tx.ndim != 1 or tx.shape != rx.shape or tx.numel() <= max_delay:
        raise ValueError("acquisition 维度或信道记忆非法。")
    if cfo_limit < 0 or grid_points < 3:
        raise ValueError("CFO 搜索范围和网格数非法。")
    destination = rx.device
    known = tx.detach().to(device="cpu", dtype=torch.complex128)
    received = rx.detach().to(device="cpu", dtype=torch.complex128)
    if not torch.isfinite(known).all() or not torch.isfinite(received).all():
        raise ValueError("acquisition 必须为有限观测。")
    positions = torch.arange(max_delay, known.numel())
    design = _design(known, positions, max_delay)
    rank = int(torch.linalg.matrix_rank(design))
    count, dimension = design.shape
    if rank != dimension or count <= rank:
        raise ValueError("acquisition 设计必须满列秩且有正残差自由度。")
    q, r = torch.linalg.qr(design, mode="reduced")
    times = positions.double()
    target = received[positions]

    def corrected(cfo):
        return target * torch.exp(-2j * torch.pi * float(cfo) * times)

    def objective(cfo):
        value = corrected(cfo)
        residual = value - q @ (q.mH @ value)
        return float(residual.abs().square().sum())

    candidates = torch.linspace(-cfo_limit, cfo_limit, grid_points, dtype=torch.float64)
    targets = target[:, None] * torch.exp(-2j * torch.pi * times[:, None] * candidates)
    residuals = targets - q @ (q.mH @ targets)
    index = int(residuals.abs().square().sum(0).argmin())
    best_cfo = float(candidates[index])
    if cfo_limit > 0:
        bounds = (float(candidates[max(0, index - 1)]),
                  float(candidates[min(grid_points - 1, index + 1)]))
        fit = minimize_scalar(objective, bounds=bounds, method="bounded",
                              options={"xatol": 1e-10})
        if fit.success and fit.fun < objective(best_cfo):
            best_cfo = float(fit.x)
    value = corrected(best_cfo)
    mean = torch.linalg.solve_triangular(r, (q.mH @ value)[:, None], upper=True)[:, 0]
    residual = value - design @ mean
    energy = float(residual.abs().square().sum())
    estimated_noise = max(energy / (count - rank), 1e-10)
    noise = estimated_noise if noise_reference is None else float(noise_reference)
    if not math.isfinite(noise) or noise <= 0:
        raise ValueError("噪声功率必须为正有限数。")
    covariance = noise * torch.linalg.inv(design.mH @ design)
    return AcquisitionEstimate(
        mean.to(destination, torch.complex64),
        _hermitian(covariance).to(destination, torch.complex64),
        best_cfo, noise, energy / count, rank, count,
        "acquisition_residual" if noise_reference is None else "oracle_noise_reference",
    )


def gaussian_update(mean, covariance, design, target, row_noise):
    """无单位范数投影的复高斯线性更新，保持有效 CIR 幅度。"""

    if not design.shape[0]:
        return mean, covariance
    identity = torch.eye(mean.numel(), dtype=mean.dtype, device=mean.device)
    prior_precision = torch.linalg.solve(_hermitian(covariance), identity)
    weights = row_noise.clamp_min(1e-10).reciprocal()
    precision = _hermitian(prior_precision + design.mH @ (weights[:, None] * design))
    rhs = prior_precision @ mean + design.mH @ (weights * target)
    factor = torch.linalg.cholesky(precision)
    posterior_mean = torch.cholesky_solve(rhs[:, None], factor)[:, 0]
    posterior_covariance = torch.cholesky_inverse(factor)
    return posterior_mean.detach(), _hermitian(posterior_covariance).detach()


def phase_features(frame, cir, tail, cfo):
    view = SimpleNamespace(
        rx_symbols=torch.where(frame.adapt_mask, frame.rx_symbols, 0),
        adapt_symbols=frame.tx_symbols, adapt_mask=frame.adapt_mask,
    )
    return estimate_phase_residual_vector(view, cir, tail, blocks=4, cfo_hint=cfo).to(cir.device)


def _corrected(frame, positions, features):
    phase = features[0] + 2 * torch.pi * features[1] * positions.float()
    return frame.rx_symbols[positions] * torch.exp(-1j * phase)


def _pilot_design(frame, mean):
    """仅返回卷积窗口中全部符号均为合法 Adapt 的观测行。"""

    max_delay = mean.numel() - 1
    positions = torch.arange(max_delay, frame.rx_symbols.numel(), device=mean.device)
    known = _design(frame.adapt_mask, positions, max_delay).all(1)
    positions = positions[known]
    design = _design(frame.tx_symbols, positions, max_delay)
    return positions, design


def pilot_innovation_statistics(frame, mean, covariance, noise, features):
    """记录更新前合法 Pilot 残差，不把同批 Pilot 拟合后的相位当已知真值。

    预测方差只包含 CIR 先验和白噪声，不含同步不确定性；这些均值统计
    不能作为已校准的卡方创新检验，也不回流接收状态或噪声估计。
    """

    positions, design = _pilot_design(frame, mean)
    if positions.numel() == 0:
        return {'pilot_observations': 0, 'pilot_innovation_mse': None,
                'pilot_innovation_variance_mean': None}
    residual = _corrected(frame, positions, features) - design @ mean
    predicted_variance = ((design @ covariance) * design.conj()).sum(1).real + noise
    return {
        'pilot_observations': int(positions.numel()),
        'pilot_innovation_mse': float(residual.abs().square().mean()),
        'pilot_innovation_variance_mean': float(predicted_variance.mean()),
    }


def pilot_update(frame, mean, covariance, noise, features):
    positions, design = _pilot_design(frame, mean)
    row_noise = torch.full((positions.numel(),), noise, device=mean.device)
    return gaussian_update(mean, covariance, design, _corrected(frame, positions, features), row_noise)


def soft_update(frame, logits, mean, covariance, noise, features):
    max_delay = mean.numel() - 1
    soft = torch.tanh(logits / 2).to(torch.complex64)
    soft = torch.where(frame.adapt_mask, frame.tx_symbols, soft)
    variance = 1 - soft.abs().square()
    positions = torch.nonzero(frame.data_mask, as_tuple=False).flatten()
    positions = positions[positions >= max_delay]
    design = _design(soft, positions, max_delay)
    row_noise = noise + _design(variance, positions, max_delay) @ mean.abs().square()
    return gaussian_update(mean, covariance, design, _corrected(frame, positions, features), row_noise)


def make_receiver_context(frame, mean, tail, noise, features):
    """神经条件直接使用估计噪声功率，不通过真实 SNR 构造。"""

    latent = torch.zeros((1, 96), device=mean.device)
    latent[0, :features.numel()] = features
    amplitude = mean.abs()[None]
    condition = CIRCondition(
        complex_cir=mean[None],
        support_probability=amplitude / amplitude.sum(1, keepdim=True).clamp_min(1e-8),
        noise_variance=torch.tensor([noise], device=mean.device),
        confidence=torch.ones(1, device=mean.device), latent_residual=latent,
    )
    return TransferContext(frame, condition, tail, cfo=float(features[1]))


class ContinualReceiver:
    """每帧一次前向；Pilot 只更新一次，Data 只影响后续帧。"""

    def __init__(self, acquisition, tail, *, arm, process_noise=1e-4, phase_estimator="unwrap_median",
                 phase_reference_backend="vectorized", state_model="random_walk", state_rho=None,
                 stationary_power=None, initial_prediction_intervals=1.0):
        if arm not in ARMS or not math.isfinite(process_noise) or process_noise < 0:
            raise ValueError("接收臂或过程噪声非法。")
        self.acquisition, self.arm = acquisition, arm
        self.mean = acquisition.mean.clone()
        self.covariance = acquisition.covariance.clone()
        self.tail = tail.detach().clone()
        self.process_noise = float(process_noise)
        self.state_model = state_model
        self.state_rho = state_rho
        self.stationary_power = stationary_power
        self.initial_prediction_intervals = float(initial_prediction_intervals)
        if not math.isfinite(self.initial_prediction_intervals) or self.initial_prediction_intervals < 0:
            raise ValueError("首帧预测的转移间隔须为非负有限数。")
        if state_model not in {"random_walk", "ar1"}:
            raise ValueError("未知 CIR 状态预测模型。")
        if state_model == "ar1":
            # 在初始化时先验证参数，避免运行中才发现 manifest 错误。
            predict_cir_state(self.mean, self.covariance, state_model=state_model,
                              process_noise=self.process_noise, state_rho=state_rho,
                              stationary_power=stationary_power)
        if phase_estimator not in {"unwrap_median", "circular_all", "circular_clean"}:
            raise ValueError("未知相位估计方法。")
        self.phase_estimator = phase_estimator
        if phase_reference_backend not in {"legacy", "vectorized"}:
            raise ValueError("未知相位参考计算实现。")
        self.phase_reference_backend = phase_reference_backend
        self.frames_seen = 0

    def _phase_features(self, frame, mean):
        if self.phase_estimator == "unwrap_median":
            return phase_features(frame, mean, self.tail, self.acquisition.cfo)
        from training.continual_phase import circular_phase_features
        return circular_phase_features(frame, mean, self.tail, self.acquisition.cfo,
            clean_rows=self.phase_estimator == "circular_clean", reference_backend=self.phase_reference_backend)

    def step(self, model, frame):
        if frame.true_cir is not None or frame.tail_symbols is not None or bool(frame.bits.any()):
            raise ValueError("接收机只接受剥离隐藏字段的接收视图。")
        if bool(frame.tx_symbols[~frame.adapt_mask].abs().any()) or frame.lookahead_rx_symbols is not None:
            raise ValueError("接收视图包含未授权符号或未来观测。")
        if self.arm in {"acquisition", "current_pilot"}:
            mean = self.acquisition.mean.clone()
            covariance = self.acquisition.covariance.clone()
        else:
            mean, covariance = self.mean, self.covariance
        if self.arm in {"recursive_pilot", "recursive_soft"}:
            # 首帧也须覆盖 acquisition 发送结束后的状态演化；0 仅供显式历史对照。
            intervals = self.initial_prediction_intervals if self.frames_seen == 0 else 1.0
            mean, covariance = predict_cir_state(
                mean, covariance, state_model=self.state_model, process_noise=self.process_noise,
                state_rho=self.state_rho, stationary_power=self.stationary_power,
                transition_intervals=intervals,
            )
        features = self._phase_features(frame, mean)
        pilot_statistics = pilot_innovation_statistics(
            frame, mean, covariance, self.acquisition.noise_variance, features)
        prior_covariance_trace = float(covariance.trace().real)
        if self.arm != "acquisition":
            mean, covariance = pilot_update(frame, mean, covariance,
                                            self.acquisition.noise_variance, features)
        decode_mean, decode_covariance = mean.clone(), covariance.clone()
        features = self._phase_features(frame, mean)
        context = make_receiver_context(frame, mean, self.tail,
                                        self.acquisition.noise_variance, features)
        with torch.no_grad():
            logits = context_logits(model, context, None, visible="data")
        self.tail = tail_from_logits(logits, self.tail)
        if self.arm == "recursive_soft":
            mean, covariance = soft_update(frame, logits, mean, covariance,
                                           self.acquisition.noise_variance, features)
        self.mean, self.covariance = mean, covariance
        self.frames_seen += 1
        return logits, {
            "decode_cir": decode_mean, "decode_covariance_trace": float(decode_covariance.trace().real),
            "prior_covariance_trace": prior_covariance_trace, **pilot_statistics,
            "posterior_covariance_trace": float(covariance.trace().real),
            "pilot_updates": int(self.arm != "acquisition"),
            "data_updates": int(self.arm == "recursive_soft"), "neural_forwards": 1,
        }
