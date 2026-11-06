# -*- coding: utf-8 -*-
"""Data 位置的无标签软边缘化模型似然（ELBO）。

该目标只使用：
- 已接收波形 ``frame.rx_symbols``；
- 模型自身 soft 输出 ``mu = tanh(logit/2)`` 与方差 ``1 - mu^2``；
- Pilot 驱动估计的 CIR/soft-tail（``context.condition`` 与 ``context.tail``）。

它不读取 Data 发送符号、Data 比特或仿真真实 CIR。目标是给高 SNR 下
Pilot BCE 梯度消失时提供仍然合法的 Data 侧信号；不能把 ELBO 下降直接
当作 BER 改善，必须另行做动作后 Data 评分。
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from env.linear_operator import LinearChannelOperator
from training.pilot_transfer import context_logits


def soft_symbol_moments(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BPSK 均值场后验：mu = tanh(z/2)，var = 1 - mu^2。"""

    mu = torch.tanh(logits / 2.0)
    return mu, (1.0 - mu.square()).clamp_min(0.0)


def _shifted_variance_term(variance: torch.Tensor, cir_energy: torch.Tensor) -> torch.Tensor:
    """计算 sum_l |h_l|^2 Var_{t-l}；帧首负索引视为确定性 tail（方差 0）。"""

    length = variance.numel()
    term = torch.zeros_like(variance)
    for delay in range(cir_energy.numel()):
        shifted = F.pad(variance, (delay, 0))[:length]
        term = term + cir_energy[delay].to(term.dtype) * shifted
    return term


def data_aided_elbo(
    model,
    context,
    parameters=None,
    *,
    variance_weight: float = 1.0,
    anchor_weight: float = 0.0,
    anchor_logits: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """返回 Data 位置软边缘化模型似然及其诊断分量。

    ``L = E[||y - H s||^2] / sigma^2
        = ||y - H mu||^2 / sigma^2 + sum_l |h_l|^2 Var_{t-l} / sigma^2``

    ``anchor_weight>0`` 时额外加入 ``mean((mu - mu_anchor)^2)``，其中
    ``mu_anchor`` 默认是同一模型在 ``parameters=None``（Frozen）下的输出。
    anchor 使用 stop-gradient，避免把目标变成简单的自蒸馏。
    """

    if variance_weight < 0 or anchor_weight < 0:
        raise ValueError("ELBO 权重必须非负。")
    frame = context.frame
    logits = context_logits(model, context, parameters, visible="data")
    mu, variance = soft_symbol_moments(logits)
    mask = frame.data_mask.bool()
    if not bool(mask.any()):
        raise ValueError("ELBO 需要至少一个 Data 位置。")

    cir = context.condition.complex_cir.to(device=logits.device, dtype=torch.complex64)
    if cir.ndim == 1:
        cir = cir.unsqueeze(0)
    tail = context.tail.to(device=logits.device, dtype=torch.complex64)
    if tail.ndim == 1:
        tail = tail.unsqueeze(0)
    operator = LinearChannelOperator(frame_len=mu.numel(), max_delay=cir.shape[-1] - 1)
    reconstructed = operator.forward(
        torch.complex(mu, torch.zeros_like(mu)).unsqueeze(0), cir, tail,
    ).squeeze(0)
    residual = frame.rx_symbols.to(reconstructed.device).to(torch.complex64) - reconstructed
    noise_variance = context.condition.noise_variance.to(
        device=logits.device, dtype=residual.real.dtype,
    ).reshape(-1)[0].clamp_min(1e-6)
    reconstruction = (residual.real.square() + residual.imag.square())[mask].mean() / noise_variance
    cir_energy = cir.abs().square().sum(dim=0)
    variance_term = _shifted_variance_term(variance, cir_energy)[mask].mean() / noise_variance

    diagnostics = {
        "reconstruction": reconstruction,
        "variance": variance_term,
        "elbo": reconstruction + variance_weight * variance_term,
        "noise_variance": noise_variance.detach(),
        "mean_variance": variance[mask].mean().detach(),
        "mean_abs_mu": mu[mask].abs().mean().detach(),
    }
    if anchor_weight > 0.0:
        if anchor_logits is None:
            with torch.no_grad():
                anchor_logits = context_logits(model, context, None, visible="data")
        anchor_mu = torch.tanh(anchor_logits.detach() / 2.0)
        anchor = (mu[mask] - anchor_mu[mask]).square().mean()
        diagnostics["anchor"] = anchor
        diagnostics["elbo"] = diagnostics["elbo"] + anchor_weight * anchor
    return diagnostics
