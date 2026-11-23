# -*- coding: utf-8 -*-
"""批量前向与 CUDA Graph 的等价性和输入合并契约。"""

import pytest
import torch

from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from env.frame_structure import FrameConfig, FrameGenerator
from training.continual_accel import BatchedForward, merge_calls
from training.continual_receiver import (
    ContinualReceiver, estimate_acquisition, receiver_frame,
)
from training.pilot_transfer import context_logits, context_model_inputs


def _fixture(arm):
    rng = torch.Generator().manual_seed(7)
    tx = (2 * torch.randint(0, 2, (128,), generator=rng) - 1).to(torch.complex64)
    cir = torch.tensor([0.9 + .2j, .25 - .1j, -.1j])
    rx = sum(cir[k] * torch.roll(tx, k) for k in range(3))
    noise = .05 * torch.complex(torch.randn(128, generator=rng), torch.randn(128, generator=rng))
    acq = estimate_acquisition(tx, rx + noise, 2, cfo_limit=0)
    frame = FrameGenerator(FrameConfig(frame_len=128, total_pilot=64, max_delay=2), seed=3).generate(0)
    receiver = ContinualReceiver(acq, torch.zeros(2, dtype=torch.complex64), arm=arm)
    return receiver, receiver_frame(frame)


def _model():
    model = UnfoldedEqualizer(UnfoldedConfig(
        frame_len=128, max_delay=2, iterations=1, d_model=8, num_heads=2,
        physics_warm_start_iterations=1))
    return model.eval()


def test_merge_calls_stacks_one_dimensional_tail():
    args = (torch.zeros(1, 128, 2), 'condition', torch.zeros(1, 128, dtype=torch.long),
            torch.zeros(2, dtype=torch.complex64))
    merged_args, _ = merge_calls([(args, {}), (args, {})])
    assert merged_args[3].shape == (2, 2)
    assert merged_args[0].shape == (2, 128, 2)


@pytest.mark.parametrize('arm', ['recursive_pilot', 'recursive_soft'])
def test_prepare_finish_matches_step(arm):
    model = _model()
    baseline, frame = _fixture(arm)
    split, frame2 = _fixture(arm)
    expected, info_expected = baseline.step(model, frame)
    prepared = split.prepare(frame2)
    with torch.no_grad():
        logits = context_logits(model, prepared.context, None, visible='data')
    actual, info_actual = split.finish(prepared, logits)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(split.mean, baseline.mean, rtol=0, atol=0)
    torch.testing.assert_close(split.tail, baseline.tail, rtol=0, atol=0)
    assert info_actual['neural_forwards'] == info_expected['neural_forwards'] == 1
    assert info_actual['data_updates'] == info_expected['data_updates']


def test_batched_forward_matches_individual_calls():
    model = _model()
    first, frame = _fixture('recursive_pilot')
    second, _ = _fixture('recursive_soft')
    calls, singles = [], []
    for receiver in (first, second):
        prepared = receiver.prepare(frame)
        calls.append(context_model_inputs(model, prepared.context, visible='data'))
        with torch.no_grad():
            singles.append(context_logits(model, prepared.context, None, visible='data'))
    batched = BatchedForward(model)
    with torch.no_grad():
        merged = batched([calls[0], (calls[1][0], calls[1][1])])
    torch.testing.assert_close(merged[0], singles[0], rtol=0, atol=1e-6)
    torch.testing.assert_close(merged[1], singles[1], rtol=0, atol=1e-6)
    assert batched.calls == 1
