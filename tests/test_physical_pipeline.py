# -*- coding: utf-8 -*-
"""物理管线核心测试：dd_soft tail 递推、协议缺省与 Data 标签隔离。"""

from dataclasses import replace

import pytest
import torch

from agent.cir_estimator import condition_from_cir
from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from env.frame_structure import FrameConfig, FrameGenerator
from training.pilot_transfer import TransferContext, context_logits, next_pilot_tail, tail_update_mode


def _fixture():
    torch.manual_seed(91)
    model = UnfoldedEqualizer(UnfoldedConfig(
        frame_len=128, max_delay=4, iterations=1, d_model=16, num_heads=4,
        pilot_conditioned=True, enable_phase_correction_branch=True,
        phase_correction_initial_scale=1.0,
    )).eval()
    model.attach_online_phase_trend_adapter()
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


def test_prefix_pilot_does_not_update_unobserved_tail_positions():
    """prefix 布局帧尾全是 Data 时，pilot_only 严格保留历史 tail。"""

    model, context = _fixture()
    context = replace(context, tail=torch.tensor([0.2, -0.3, 0.5, -0.8], dtype=torch.complex64))
    torch.testing.assert_close(
        next_pilot_tail(model, context, alpha=0.5), context.tail, rtol=0, atol=0,
    )


def test_dd_soft_tail_uses_frame_tail_without_data_labels():
    """dd_soft 用帧尾 soft 判决递推，翻转 Data 标签不改变结果。"""

    model, context = _fixture()
    tail_len = context.tail.numel()
    with torch.no_grad():
        logits = context_logits(model, context, None, visible="data")
    expected = torch.complex(
        torch.tanh(logits[-tail_len:] / 2.0),
        torch.zeros(tail_len, dtype=logits.dtype),
    )
    frame = context.frame
    poisoned_bits = frame.bits.clone()
    poisoned_bits[frame.data_mask] = ~poisoned_bits[frame.data_mask]
    poisoned = replace(context, frame=replace(frame, bits=poisoned_bits))
    got = next_pilot_tail(model, poisoned, alpha=1.0, mode="dd_soft")
    torch.testing.assert_close(got, expected)
    assert not torch.equal(got, context.tail)


def test_tail_update_mode_defaults_to_legacy_pilot_only():
    assert tail_update_mode({}) == "pilot_only"
    assert tail_update_mode({"tail_update_mode": "dd_soft"}) == "dd_soft"
    with pytest.raises(ValueError):
        tail_update_mode({"tail_update_mode": "unknown"})
