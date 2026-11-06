# -*- coding: utf-8 -*-
"""验证数据辅助 CIR 精化的数值、标签隔离与回退路径。"""

from dataclasses import replace

import torch

from env.frame_structure import FrameConfig, FrameGenerator
from env.linear_operator import LinearChannelOperator
from training.data_aided_cir import data_aided_cir_update


def _fixture():
    frame = FrameGenerator(FrameConfig(frame_len=128, total_pilot=64, max_delay=4), seed=7).generate(0)
    cir = torch.tensor([0.8, 0.0, 0.3, 0.0, 0.6], dtype=torch.complex64)
    tail = torch.zeros(4, dtype=torch.complex64)
    operator = LinearChannelOperator(frame_len=128, max_delay=4)
    rx = operator.forward(
        frame.tx_symbols.to(torch.complex64).unsqueeze(0), cir.unsqueeze(0), tail.unsqueeze(0),
    ).squeeze(0)
    frame = replace(frame, rx_symbols=rx)
    logits = frame.tx_symbols.real.float() * 10.0
    return frame, cir, logits


def test_data_aided_cir_recovers_known_channel():
    frame, cir, logits = _fixture()
    estimate = data_aided_cir_update(
        frame, logits, 4, cir, alpha=1.0, confidence_threshold=0.5, ridge=0.0,
    )
    torch.testing.assert_close(estimate, cir / cir.norm(), rtol=1e-4, atol=1e-5)


def test_data_aided_cir_is_label_free():
    frame, cir, logits = _fixture()
    baseline = data_aided_cir_update(frame, logits, 4, cir, alpha=1.0)
    bits = frame.bits.clone()
    bits[frame.data_mask] = ~bits[frame.data_mask]
    poisoned = replace(frame, bits=bits)
    poisoned_estimate = data_aided_cir_update(poisoned, logits, 4, cir, alpha=1.0)
    # lstsq 在 float32 下可能有 1e-7 级数值差异；语义上不得依赖 Data 标签。
    torch.testing.assert_close(baseline, poisoned_estimate, rtol=1e-6, atol=1e-6)


def test_data_aided_cir_falls_back_when_no_confident_symbols():
    frame, cir, logits = _fixture()
    estimate = data_aided_cir_update(
        frame, torch.zeros_like(logits), 4, cir, alpha=1.0, confidence_threshold=0.5,
    )
    torch.testing.assert_close(estimate, cir / cir.norm(), rtol=0, atol=1e-6)
