# -*- coding: utf-8 -*-
"""验证 FFT 加速仍表示同一个带历史尾部的线性信道及其伴随。"""

import pytest
import torch

from env.linear_operator import CrossFrameLinearOperator, LinearChannelOperator


@pytest.mark.parametrize("length,delay,batch", [(1, 0, 1), (7, 12, 2), (64, 7, 2), (1024, 116, 1)])
def test_fft_and_direct_match_with_nonzero_tail(length, delay, batch):
    torch.manual_seed(417)
    x = torch.randn(batch, length, dtype=torch.complex64)
    h = torch.randn(batch, delay + 1, dtype=torch.complex64)
    tail = torch.randn(batch, delay, dtype=torch.complex64)
    y = torch.randn_like(x)
    direct = LinearChannelOperator(length, delay, backend="direct")
    fast = LinearChannelOperator(length, delay, backend="fft")
    torch.testing.assert_close(fast.forward(x, h, tail), direct.forward(x, h, tail), atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(fast.adjoint(y, h), direct.adjoint(y, h), atol=2e-5, rtol=2e-5)


def test_fft_operator_adjoint_and_parameter_gradients_match():
    torch.manual_seed(513)
    x = torch.randn(2, 17, dtype=torch.complex128, requires_grad=True)
    h = torch.randn(2, 8, dtype=torch.complex128, requires_grad=True)
    tail = torch.randn(2, 7, dtype=torch.complex128, requires_grad=True)
    y = torch.randn_like(x)
    fast = LinearChannelOperator(17, 7, backend="fft")
    direct = LinearChannelOperator(17, 7, backend="direct")
    losses = [operator.forward(x, h, tail).abs().square().mean() for operator in (fast, direct)]
    gradients = [torch.autograd.grad(loss, (x, h, tail), create_graph=True) for loss in losses]
    for actual, expected in zip(*gradients):
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    second = [torch.autograd.grad(gradient[0].abs().square().sum(), h)[0] for gradient in gradients]
    torch.testing.assert_close(second[0], second[1], atol=1e-9, rtol=1e-9)
    zero_tail = torch.zeros_like(tail)
    lhs = (fast.forward(x, h, zero_tail).conj() * y).sum()
    rhs = (x.conj() * fast.adjoint(y, h)).sum()
    torch.testing.assert_close(lhs, rhs, atol=1e-10, rtol=1e-10)


def test_fft_backend_keeps_unbatched_shape_and_handles_tail_boundary():
    x = torch.tensor([1, -1, 1, 1, -1, -1, 1, -1], dtype=torch.complex64)
    h = torch.tensor([0.8, 0, 0, 0.6], dtype=torch.complex64)
    tail = torch.tensor([-1, 1, -1], dtype=torch.complex64)
    fast = LinearChannelOperator(8, 3, backend="fft")
    direct = LinearChannelOperator(8, 3, backend="direct")
    actual, expected = fast.forward(x, h, tail), direct.forward(x, h, tail)
    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert float(actual[0].real) == pytest.approx(0.2, abs=1e-6)


def test_invalid_operator_backend_is_rejected():
    with pytest.raises(ValueError, match="backend"):
        LinearChannelOperator(64, 7, backend="circular")


def test_fft_backend_preserves_real_dtype():
    torch.manual_seed(17)
    x = torch.randn(2, 64, dtype=torch.float64)
    h = torch.randn(2, 8, dtype=torch.float64)
    tail = torch.randn(2, 7, dtype=torch.float64)
    direct = LinearChannelOperator(64, 7, backend="direct")
    fast = LinearChannelOperator(64, 7, backend="fft")
    for actual, expected in ((fast.forward(x, h, tail), direct.forward(x, h, tail)),
                             (fast.adjoint(x, h), direct.adjoint(x, h))):
        assert actual.dtype == expected.dtype == torch.float64
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("backend", ["direct", "fft"])
def test_cross_frame_operator_matches_one_contiguous_linear_convolution(backend):
    torch.manual_seed(922)
    frame_len, delay, lookahead = 31, 9, 9
    current = torch.randn(frame_len, dtype=torch.complex64)
    future_prefix = torch.randn(lookahead, dtype=torch.complex64)
    cir = torch.randn(delay + 1, dtype=torch.complex64)
    previous_tail = torch.randn(delay, dtype=torch.complex64)
    operator = CrossFrameLinearOperator(frame_len, delay, lookahead, backend=backend)
    actual = operator.forward(current, future_prefix, cir, previous_tail)
    expected = LinearChannelOperator(frame_len + lookahead, delay, backend="direct").forward(
        torch.cat((current, future_prefix)), cir, previous_tail
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("backend", ["direct", "fft"])
def test_cross_frame_adjoint_splits_current_and_lookahead_gradients(backend):
    torch.manual_seed(931)
    frame_len, delay, lookahead = 23, 7, 7
    current = torch.randn(2, frame_len, dtype=torch.complex64)
    future_prefix = torch.randn(2, lookahead, dtype=torch.complex64)
    cir = torch.randn(2, delay + 1, dtype=torch.complex64)
    previous_tail = torch.zeros(2, delay, dtype=torch.complex64)
    observations = torch.randn(2, frame_len + lookahead, dtype=torch.complex64)
    operator = CrossFrameLinearOperator(frame_len, delay, lookahead, backend=backend)
    predicted = operator.forward(current, future_prefix, cir, previous_tail)
    grad_current, grad_future = operator.adjoint(observations, cir)
    lhs = (predicted.conj() * observations).sum()
    rhs = (current.conj() * grad_current).sum() + (future_prefix.conj() * grad_future).sum()
    torch.testing.assert_close(lhs, rhs, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("backend", ["direct", "fft"])
def test_cross_frame_known_next_prefix_can_be_removed_without_data_labels(backend):
    torch.manual_seed(947)
    frame_len, delay, lookahead = 29, 8, 8
    current = torch.randn(frame_len, dtype=torch.complex64)
    next_adapt_pilot = torch.randn(lookahead, dtype=torch.complex64)
    cir = torch.randn(delay + 1, dtype=torch.complex64)
    previous_tail = torch.randn(delay, dtype=torch.complex64)
    zeros_current = torch.zeros_like(current)
    zeros_future = torch.zeros_like(next_adapt_pilot)
    operator = CrossFrameLinearOperator(frame_len, delay, lookahead, backend=backend)

    joint = operator.forward(current, next_adapt_pilot, cir, previous_tail)
    current_response = operator.forward(current, zeros_future, cir, previous_tail)
    known_pilot_response = operator.forward(zeros_current, next_adapt_pilot, cir, torch.zeros_like(previous_tail))

    torch.testing.assert_close(joint, current_response + known_pilot_response, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("lookahead", [0, 8])
def test_cross_frame_operator_rejects_lookahead_outside_channel_memory(lookahead):
    with pytest.raises(ValueError, match="lookahead_len"):
        CrossFrameLinearOperator(frame_len=32, max_delay=7, lookahead_len=lookahead)
