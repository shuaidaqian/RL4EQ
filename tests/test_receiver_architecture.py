import torch

from baseline.block_equalizers import bit_error_rate, perfect_csi_cg_detect
from env.linear_operator import LinearChannelOperator


def test_linear_operator_adjoint_identity():
    operator = LinearChannelOperator(frame_len=64, max_delay=7)
    x = torch.randn(2, 64, dtype=torch.complex64)
    y = torch.randn(2, 64, dtype=torch.complex64)
    cir = torch.randn(2, 8, dtype=torch.complex64)
    tail = torch.zeros(2, 7, dtype=torch.complex64)
    lhs = (operator.forward(x, cir, tail).conj() * y).sum()
    rhs = (x.conj() * operator.adjoint(y, cir)).sum()
    assert torch.allclose(lhs, rhs, atol=1e-4, rtol=1e-4)


def test_perfect_csi_detector_recovers_noiseless_bpsk():
    bits = torch.tensor([[0, 1, 1, 0, 1, 0, 0, 1]], dtype=torch.bool)
    symbols = torch.complex(bits.to(torch.float32) * 2.0 - 1.0, torch.zeros_like(bits, dtype=torch.float32))
    cir = torch.zeros(1, 4, dtype=torch.complex64)
    cir[:, 0] = 1.0 + 0.0j
    tail = torch.zeros(1, 3, dtype=torch.complex64)
    rx = LinearChannelOperator(frame_len=8, max_delay=3).forward(symbols, cir, tail)
    result = perfect_csi_cg_detect(rx, cir, tail, torch.tensor(1e-6), iterations=16)
    assert result.logits.shape == bits.shape
    assert result.probabilities.shape == bits.shape
    assert bit_error_rate(result.logits, bits) == 0.0
