# -*- coding: utf-8 -*-
"""合法历史 Adapt 方程约束当前神经 CIR 的共同状态目标。"""

import torch

from agent.neural_channel_state import known_pilot_equations


def adapt_only_history_context(frame, condition, tail):
    """缓存历史时丢弃所有标签、非 Adapt 波形、后缀及仿真真值。"""
    from dataclasses import replace
    from training.pilot_transfer import TransferContext

    view = frame.receiver_view()
    clean = replace(frame, bits=torch.zeros_like(frame.bits), tx_symbols=view.adapt_symbols.detach().clone(),
        rx_symbols=torch.where(view.adapt_mask, view.rx_symbols, torch.zeros_like(view.rx_symbols)).detach().clone(),
        true_cir=None, tail_symbols=None, lookahead_rx_symbols=None, lookahead_adapt_symbols=None,
        lookahead_frame_index=None)
    detached = replace(condition, **{name: getattr(condition, name).detach().clone() for name in
        ("complex_cir", "support_probability", "noise_variance", "confidence", "latent_residual")})
    return TransferContext(clean, detached, tail.detach().clone())


def _corrected_adapt_equation(model, context):
    """只提取完全已知 Adapt 行，不保存 Reward/Data 标签或生成真值。"""
    view = context.frame.receiver_view()
    rx = torch.where(view.adapt_mask, view.rx_symbols, torch.zeros_like(view.rx_symbols))
    iq = torch.stack((rx.real, rx.imag), -1).unsqueeze(0).float()
    with torch.no_grad():
        corrected = model._apply_phase_correction(iq, context.condition)
        target = torch.complex(corrected[0, :, 0], corrected[0, :, 1])
        design, values = known_pilot_equations(target, view.adapt_symbols, view.adapt_mask,
            view.adapt_mask, context.condition.complex_cir.shape[-1] - 1)
    return design.detach(), values.detach(), float(context.condition.noise_variance.flatten()[0])


class SharedNeuralPilotObjective(torch.nn.Module):
    """历史方程共同约束当前神经 CIR，先验只归属这个当前状态。"""

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
        self.blocks = [_corrected_adapt_equation(model, context) for context in contexts]
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
        likelihood = sum((design @ state - target).abs().square().sum() / noise
                         for design, target, noise in self.blocks)
        prior = context.condition.complex_cir[0]
        noise = context.condition.noise_variance[0].reshape(()).clamp_min(1e-6)
        variance = self.model.channel_state.prior_variance(prior, noise, context.frame.frame_index).clamp_min(1e-10)
        penalty = ((state - prior).abs().square() / variance).sum()
        return (likelihood + self.prior_copies * penalty) / sum(design.shape[0] for design, _, _ in self.blocks)
