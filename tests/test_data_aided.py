# -*- coding: utf-8 -*-
"""验证 Data 位置软边缘化 ELBO 的数值、标签隔离与梯度可达性。"""

from dataclasses import replace

import torch

from agent.cir_estimator import condition_from_cir
from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from env.frame_structure import FrameConfig, FrameGenerator
from training.data_aided import data_aided_elbo, soft_symbol_moments
from training.pilot_transfer import TransferContext


def _fixture():
    torch.manual_seed(93)
    model = UnfoldedEqualizer(UnfoldedConfig(
        frame_len=128, max_delay=4, iterations=1, d_model=16, num_heads=4,
        pilot_conditioned=True, enable_phase_correction_branch=True,
        phase_correction_initial_scale=1.0,
    )).eval()
    # 基础模型默认冻结；ELBO 诊断只对 head 反传，显式打开 head 梯度。
    for parameter in model.head.parameters():
        parameter.requires_grad_(True)
    frame = FrameGenerator(FrameConfig(frame_len=128, total_pilot=64, max_delay=4), seed=7).generate(0)
    positions = torch.arange(128)
    frame = replace(
        frame,
        rx_symbols=frame.tx_symbols * torch.exp(1j * (0.3 + 2 * torch.pi * 0.001 * positions)),
    )
    cir = torch.tensor([1, 0, 0, 0, 0], dtype=torch.complex64)
    phase = torch.tensor([0.2, 0.0008, 0, 0] * 4)
    condition = condition_from_cir(cir, 10, phase_features=phase)
    return model, TransferContext(frame, condition, torch.zeros(4, dtype=torch.complex64))


def test_soft_symbol_moments_match_bpsk_posterior():
    logits = torch.tensor([-2.0, 0.0, 2.0])
    mu, variance = soft_symbol_moments(logits)
    torch.testing.assert_close(mu, torch.tanh(logits / 2.0))
    torch.testing.assert_close(variance, 1.0 - mu.square())


def test_elbo_is_finite_nonnegative_and_label_free():
    model, context = _fixture()
    result = data_aided_elbo(model, context)
    assert torch.isfinite(result["elbo"])
    assert float(result["elbo"].detach()) >= 0.0
    bits = context.frame.bits.clone()
    bits[context.frame.data_mask] = ~bits[context.frame.data_mask]
    poisoned = replace(context, frame=replace(context.frame, bits=bits))
    result_poisoned = data_aided_elbo(model, poisoned)
    torch.testing.assert_close(result["elbo"], result_poisoned["elbo"], rtol=0, atol=0)


def test_elbo_does_not_read_true_cir():
    model, context = _fixture()
    baseline = data_aided_elbo(model, context)
    poisoned_frame = replace(
        context.frame,
        true_cir=torch.full((1, 5), 123.0 + 456.0j, dtype=torch.complex64),
    )
    poisoned = replace(context, frame=poisoned_frame)
    poisoned_result = data_aided_elbo(model, poisoned)
    torch.testing.assert_close(baseline["elbo"], poisoned_result["elbo"], rtol=0, atol=0)


def test_elbo_gradient_reaches_head():
    model, context = _fixture()
    result = data_aided_elbo(model, context)
    grads = torch.autograd.grad(
        result["elbo"], [model.head.weight, model.head.bias], allow_unused=True,
    )
    assert any(g is not None and float(g.abs().sum()) > 0.0 for g in grads)


def test_elbo_anchor_uses_stop_gradient():
    model, context = _fixture()
    result = data_aided_elbo(model, context, anchor_weight=1.0)
    assert "anchor" in result
    assert torch.isfinite(result["elbo"])
