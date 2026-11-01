# -*- coding: utf-8 -*-
"""验证 FFT 加速仍表示同一个带历史尾部的线性信道及其伴随。"""

import pytest
import torch

from env.linear_operator import LinearChannelOperator


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
