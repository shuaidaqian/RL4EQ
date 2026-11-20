# -*- coding: utf-8 -*-
"""严格在接收完成后评分的状态尺度与 soft 可靠性诊断。"""

import torch
import torch.nn.functional as F

from agent.unfolded_equalizer import OnlinePhaseTrendAdapter
from env.linear_operator import LinearChannelOperator


RECEIVER_SCORE_FIELDS = (
    'aligned_nmse_diagnostic', 'phase_only_nmse_diagnostic', 'decode_cir_norm',
    'true_cir_norm_diagnostic', 'data_bce', 'data_brier', 'confident_data_count',
    'confident_wrong_data_count', 'confidence_threshold',
)

LONG_DIAGNOSTIC_FIELDS = (
    'rx_power', 'input_soft_tail_power', 'decode_phase0_estimate_rad', 'decode_cfo_estimate',
    'decoder_effective_cfo_estimate_diagnostic', 'decoder_cfo_error_diagnostic',
    'decoder_phase_data_rms_rad_diagnostic', 'decoder_phase_data_centered_rms_rad_diagnostic',
    'decoder_phase_data_mean_rad_diagnostic', 'true_cir_power_diagnostic',
    'true_frequency_flatness_diagnostic', 'true_data_signal_power_diagnostic',
    'realized_data_noise_power_diagnostic', 'true_effective_esn0_db_diagnostic',
    'decoder_signal_mismatch_nmse_diagnostic', 'decoder_observation_residual_mse_diagnostic',
)


def receiver_diagnostics(logits, labels, data_mask, estimate, truth, *, confidence_threshold=.95):
    """输入含离线真值，禁止从接收机或在线更新路径调用。"""
    selected = logits[data_mask].float()
    target = labels[data_mask].bool()
    probability = selected.sigmoid()
    confident = selected.abs().sigmoid() >= confidence_threshold
    wrong = (selected>=0) != target
    energy = estimate.abs().square().sum().clamp_min(1e-12)
    truth_energy = truth.abs().square().sum().clamp_min(1e-12)
    inner = torch.vdot(estimate.flatten(), truth.flatten())
    scale = inner/energy
    phase = inner/inner.abs().clamp_min(1e-12)
    return dict(
        aligned_nmse_diagnostic=float((truth-scale*estimate).abs().square().sum()/truth_energy),
        phase_only_nmse_diagnostic=float((truth-phase*estimate).abs().square().sum()/truth_energy),
        decode_cir_norm=float(estimate.norm()), true_cir_norm_diagnostic=float(truth.norm()),
        data_bce=float(F.binary_cross_entropy_with_logits(selected, target.float())),
        data_brier=float((probability-target.float()).square().mean()),
        confident_data_count=int(confident.sum()), confident_wrong_data_count=int((confident & wrong).sum()),
        confidence_threshold=confidence_threshold,
    )


@torch.no_grad()
def long_receiver_diagnostics(branch, actual_inputs, frame, true_phase, true_cfo, noise_truth):
    """接收完成后，基于实际输入快照重算解码预处理并评分，不调用完整解码器。

    CIR 与公共相位有规范自由度。相位误差先用复内积对齐 CIR，再计入
    解码器实际相位分支的权重和缩放；不能拿 Pilot 的 phase0 直接减真值。
    真值和 Data 符号仅用于本函数，返回标量禁止进入下一帧接收状态。
    """

    model = getattr(branch, 'model', branch)
    if model.training:
        raise ValueError('长程事后诊断要求冻结 eval 模型。')
    phase_adapter = model.online_phase_trend_adapter
    if phase_adapter is not None and type(phase_adapter) is not OnlinePhaseTrendAdapter:
        raise ValueError('长程评分只支持已审计的确定性相位趋势适配器。')
    if model.config.enable_channel_state_branch:
        raise ValueError('当前长程评分不支持解码器内部另行改变 CIR。')
    args, _ = actual_inputs
    rx_iq, condition, _, soft_tail = args
    estimate = condition.complex_cir.flatten()
    truth = frame.true_cir.flatten()
    device = estimate.device
    phase = true_phase.to(device=device).flatten()
    if phase.shape != frame.rx_symbols.shape or noise_truth <= 0:
        raise ValueError('长程评分需要当前帧实际相位及正噪声功率。')

    # 对单位复信号重算相同的确定性校正，包含已学习相位偏移与缩放。
    # 不调用 forward，不改调用计数、状态或 tail，也不向相位分支传入真值。
    unit_iq = torch.zeros_like(rx_iq)
    unit_iq[..., 0] = 1
    corrected_iq = model._apply_phase_correction(unit_iq, condition)
    if phase_adapter is not None:
        corrected_iq = phase_adapter(corrected_iq)
    rotation = torch.complex(corrected_iq[..., 0], corrected_iq[..., 1]).flatten()
    effective_cfo = -torch.angle((rotation[1:] * rotation[:-1].conj()).sum()) / (2 * torch.pi)
    alignment_phase = torch.angle(torch.vdot(estimate, truth))
    residual_rotation = torch.exp(1j * (phase + alignment_phase)) * rotation
    data_phase = torch.angle(residual_rotation[frame.data_mask])
    mean_phase = torch.angle(torch.exp(1j * data_phase).mean())
    centered_phase = torch.angle(torch.exp(1j * (data_phase - mean_phase)))

    operator = LinearChannelOperator(frame.rx_symbols.numel(), estimate.numel() - 1, backend='fft')
    unrotated_signal = operator.forward(frame.tx_symbols, truth, frame.tail_symbols)
    true_signal = unrotated_signal * torch.exp(1j * phase)
    predicted_signal = operator.forward(frame.tx_symbols, estimate, soft_tail.flatten())
    mask = frame.data_mask
    data_signal_power = true_signal[mask].abs().square().mean().clamp_min(1e-12)
    corrected_signal = true_signal * rotation
    corrected_rx = frame.rx_symbols * rotation
    spectrum = torch.fft.fft(truth, n=frame.rx_symbols.numel()).abs().square().clamp_min(1e-12)
    true_power = truth.abs().square().sum().clamp_min(1e-12)
    latent = condition.latent_residual.flatten()
    return dict(
        rx_power=float(frame.rx_symbols.abs().square().mean()),
        input_soft_tail_power=float(soft_tail.abs().square().mean()),
        decode_phase0_estimate_rad=float(latent[0]), decode_cfo_estimate=float(latent[1]),
        decoder_effective_cfo_estimate_diagnostic=float(effective_cfo),
        decoder_cfo_error_diagnostic=float(effective_cfo - true_cfo),
        decoder_phase_data_rms_rad_diagnostic=float(data_phase.square().mean().sqrt()),
        decoder_phase_data_centered_rms_rad_diagnostic=float(centered_phase.square().mean().sqrt()),
        decoder_phase_data_mean_rad_diagnostic=float(mean_phase),
        true_cir_power_diagnostic=float(true_power),
        true_frequency_flatness_diagnostic=float(spectrum.log().mean().exp() / spectrum.mean()),
        true_data_signal_power_diagnostic=float(data_signal_power),
        realized_data_noise_power_diagnostic=float((frame.rx_symbols[mask] - true_signal[mask]).abs().square().mean()),
        true_effective_esn0_db_diagnostic=float(10 * torch.log10(true_power / noise_truth)),
        decoder_signal_mismatch_nmse_diagnostic=float(
            (corrected_signal[mask] - predicted_signal[mask]).abs().square().mean() / data_signal_power),
        decoder_observation_residual_mse_diagnostic=float(
            (corrected_rx[mask] - predicted_signal[mask]).abs().square().mean()),
    )
