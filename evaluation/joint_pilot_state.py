# -*- coding: utf-8 -*-
"""共同物理坐标的 Pilot 后验诊断；不接入正式神经/RL 控制。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from agent.neural_channel_state import known_pilot_equations


@dataclass(frozen=True)
class PilotBlock:
    """仅保存完全已知 Adapt 卷积行，不携带 Data、Reward 或生成真值。"""

    design: torch.Tensor
    target: torch.Tensor
    noise_variance: float
    frame_index: int

    def information(self):
        if self.design.ndim != 2 or self.target.shape != (self.design.shape[0],):
            raise ValueError("Pilot 方程的矩阵和目标维度不匹配。")
        if not self.design.shape[0] or not 0 < self.noise_variance < float("inf"):
            raise ValueError("Pilot 方程需要非空行和正有限噪声方差。")
        if not torch.isfinite(self.design).all() or not torch.isfinite(self.target).all():
            raise ValueError("Pilot 方程必须有限。")
        design, target = self.design.to(torch.complex128), self.target.to(torch.complex128)
        return design.mH @ design / self.noise_variance, design.mH @ target / self.noise_variance


def joint_posterior(mean, covariance, blocks, *, prior_copies=1):
    """复高斯一次先验加独立似然；prior_copies>1 只用于反例诊断。"""
    if not blocks or type(prior_copies) is not int or prior_copies < 1:
        raise ValueError("联合后验需要非空 Pilot 列表和正整数先验计数。")
    mean, covariance = mean.to(torch.complex128), covariance.to(torch.complex128)
    if mean.ndim != 1 or covariance.shape != (mean.numel(), mean.numel()):
        raise ValueError("先验均值和协方差维度不匹配。")
    if not torch.isfinite(mean).all() or not torch.isfinite(covariance).all():
        raise ValueError("先验必须有限。")
    torch.testing.assert_close(covariance, covariance.mH)
    torch.linalg.cholesky(covariance)
    identity = torch.eye(mean.numel(), dtype=mean.dtype, device=mean.device)
    precision = torch.linalg.solve(covariance, identity) * prior_copies
    information = precision @ mean
    for block in blocks:
        gram, target = block.information()
        precision, information = precision + gram, information + target
    posterior = torch.linalg.solve(precision, information)
    uncertainty = torch.linalg.solve(precision, identity)
    return posterior, (uncertainty + uncertainty.mH) / 2


def recursive_posterior(mean, covariance, blocks, *, rho=1., process_covariance=None):
    """第一帧先验已到位；后续按实际帧间距作 AR(1) 预测，再消费新似然。"""
    if not blocks or not 0 < rho <= 1:
        raise ValueError("递推需要非空 Pilot 列表和合法相关系数。")
    if any(right.frame_index <= left.frame_index for left, right in zip(blocks, blocks[1:])):
        raise ValueError("Pilot 帧号必须严格递增，禁止重复消费同一观测。")
    mean, covariance = mean.to(torch.complex128), covariance.to(torch.complex128)
    process = torch.zeros_like(covariance) if process_covariance is None else process_covariance.to(covariance)
    torch.testing.assert_close(process, process.mH)
    if float(torch.linalg.eigvalsh(process).min()) < -1e-12:
        raise ValueError("过程协方差必须半正定。")
    for index, block in enumerate(blocks):
        if index:
            gap = block.frame_index - blocks[index - 1].frame_index
            transition = rho ** gap
            factor = float(gap) if rho == 1 else (1 - rho ** (2 * gap)) / (1 - rho ** 2)
            mean = transition * mean
            covariance = transition ** 2 * covariance + factor * process
        mean, covariance = joint_posterior(mean, covariance, [block])
    return mean, covariance


def transport_block(block, phase):
    """变换观测与设计两边，避免把重新选坐标误当成新增物理信息。"""
    phase = torch.as_tensor(phase, dtype=torch.float64, device=block.target.device)
    rotation = torch.exp(-1j * phase)
    return PilotBlock(block.design * rotation[..., None], block.target * rotation,
                      block.noise_variance, block.frame_index)


def extract_adapt_blocks(model, context):
    """只从接收波形、Adapt 符号和既有条件生成三个离线诊断坐标。"""
    frame = context.frame
    view = frame.receiver_view()
    rx = torch.where(view.adapt_mask, view.rx_symbols, torch.zeros_like(view.rx_symbols))
    iq = torch.stack((rx.real, rx.imag), -1).unsqueeze(0).float()
    with torch.no_grad():
        corrected = model._apply_phase_correction(iq, context.condition)
        latent = context.condition.latent_residual[0]
        # 这里只复现现有 Pilot 估计，不使用仿真器真实 CFO/相位。
        phase0 = torch.median(latent[0:16:4])
        cfo = torch.median(latent[1:16:4])
        phase = phase0 + 2 * torch.pi * cfo * torch.arange(rx.numel(), device=rx.device)
        pilot_corrected = rx * torch.exp(-1j * phase)
        mapping = []
        if model.channel_state is not None and model.channel_state.memory_scale is not None:
            hook = model.channel_state.memory_scale.register_forward_hook(
                lambda module, args, output: mapping.append(torch.exp(torch.tanh(output)).detach()))
            try:
                model._channel_state_condition(corrected, context.condition,
                    view.adapt_symbols.unsqueeze(0), view.adapt_mask.unsqueeze(0), frame.frame_index)
            finally:
                hook.remove()
    blocks = {}
    for name, target in (("raw", rx), ("pilot_phase", pilot_corrected),
                         ("neural_phase", torch.complex(corrected[0, :, 0], corrected[0, :, 1]))):
        design, values = known_pilot_equations(target, view.adapt_symbols,
            view.adapt_mask, view.adapt_mask, context.condition.complex_cir.shape[-1] - 1)
        blocks[name] = PilotBlock(design, values, float(context.condition.noise_variance.flatten()[0]), frame.frame_index)
    return blocks, mapping[0].squeeze(0) if mapping else None


def prediction_loss(block, mean):
    """以复噪声方差归一化的留出 Pilot 前向误差，不读取 Data。"""
    return float((block.design.to(mean) @ mean - block.target.to(mean)).abs().square().mean() / block.noise_variance)


class SharedNeuralPilotObjective(torch.nn.Module):
    """历史方程共同约束当前神经 CIR，先验只归属这个当前状态；仅供诊断。"""

    def __init__(self, model, contexts, *, prior_copies=1):
        super().__init__()
        if not contexts or prior_copies not in {1, len(contexts)}:
            raise ValueError("共同状态诊断需要非空上下文及一次或逐帧重复先验。")
        indices = [context.frame.frame_index for context in contexts]
        if any(right != left + 1 for left, right in zip(indices, indices[1:])):
            raise ValueError("共同状态诊断要求相邻历史帧。")
        prior = contexts[-1].condition.complex_cir
        if any(not torch.equal(context.condition.complex_cir, prior) for context in contexts):
            raise ValueError("历史必须共享同一次 acquisition 的 CIR 坐标。")
        self.model, self.context = model, contexts[-1]
        self.prior_copies = prior_copies
        self.blocks = [extract_adapt_blocks(model, context)[0]["neural_phase"] for context in contexts]
        view = self.context.frame.receiver_view()
        rx = torch.where(view.adapt_mask, view.rx_symbols, torch.zeros_like(view.rx_symbols))
        iq = torch.stack((rx.real, rx.imag), -1).unsqueeze(0).float()
        with torch.no_grad():
            self.corrected = model._apply_phase_correction(iq, self.context.condition).detach()
        self.adapt_symbols, self.adapt_mask = view.adapt_symbols.unsqueeze(0), view.adapt_mask.unsqueeze(0)

    def forward(self):
        context = self.context
        state = self.model._channel_state_condition(self.corrected, context.condition,
            self.adapt_symbols, self.adapt_mask, context.frame.frame_index).complex_cir[0]
        # 方程在缓存时固定，梯度只通过当前网络的受限复数记忆。
        likelihood = sum((block.design @ state - block.target).abs().square().sum() / block.noise_variance
                         for block in self.blocks)
        prior = context.condition.complex_cir[0]
        noise = context.condition.noise_variance[0].reshape(()).clamp_min(1e-6)
        variance = self.model.channel_state.prior_variance(prior, noise, context.frame.frame_index).clamp_min(1e-10)
        penalty = ((state - prior).abs().square() / variance).sum()
        return (likelihood + self.prior_copies * penalty) / sum(block.design.shape[0] for block in self.blocks)
