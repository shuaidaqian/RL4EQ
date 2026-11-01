# -*- coding: utf-8 -*-
"""用独立的稠密最优解核验无 CP 线性 MMSE 校准参考。"""

import pytest
import torch

from baseline.linear_mmse import linear_block_lmmse


@pytest.mark.parametrize("backend", ["direct", "fft"])
@pytest.mark.parametrize("real_bpsk", [False, True])
def test_linear_mmse_matches_dense_constrained_solution(monkeypatch, backend, real_bpsk):
    monkeypatch.setenv("RL4EQ_LINEAR_OPERATOR_BACKEND", backend)
    torch.manual_seed(271)
    n, memory = 48, 7
    cir = torch.randn(memory + 1, dtype=torch.complex64) / 3
    tail = torch.randn(memory, dtype=torch.complex64)
    rx = torch.randn(n, dtype=torch.complex64)
    known = torch.zeros(n, dtype=torch.bool)
    known[:12] = True
    symbols = (2 * torch.randint(0, 2, (n,)) - 1).to(torch.complex64)
    # 独立按下标构造 Toeplitz 矩阵和历史贡献，不调用待测信道算子。
    h = torch.zeros(n, n, dtype=torch.complex64)
    history = torch.zeros(n, dtype=torch.complex64)
    for row in range(n):
        for delay in range(memory + 1):
            if row >= delay:
                h[row, row - delay] = cir[delay]
            else:
                history[row] += cir[delay] * tail[memory + row - delay]
    hu = h[:, ~known]
    target = rx - history - h[:, known] @ symbols[known]
    gram, rhs = hu.conj().T @ hu, hu.conj().T @ target
    variance = 0.17
    if real_bpsk:
        gram, rhs = gram.real, rhs.real
    expected = torch.linalg.solve(gram + torch.eye(gram.shape[0]) * variance / (2 if real_bpsk else 1), rhs)
    estimate, extra = linear_block_lmmse(rx, cir, tail, variance, known_symbols=symbols,
                                        known_mask=known, real_bpsk=real_bpsk, rtol=1e-6)
    torch.testing.assert_close(estimate[~known], expected.to(torch.complex64), atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(estimate[known], symbols[known], atol=0, rtol=0)
    assert extra["converged"]
    changed = symbols.clone()
    changed[~known] = complex(float('nan'), float('nan'))
    poisoned, _ = linear_block_lmmse(rx, cir, tail, variance, known_symbols=changed,
                                    known_mask=known, real_bpsk=real_bpsk)
    torch.testing.assert_close(estimate, poisoned, atol=0, rtol=0)


def test_linear_mmse_handles_long_echo_beyond_legacy_filter_limit(monkeypatch):
    monkeypatch.setenv("RL4EQ_LINEAR_OPERATOR_BACKEND", "fft")
    torch.manual_seed(273)
    n, memory = 384, 116
    symbols = (2 * torch.randint(0, 2, (n,)) - 1).to(torch.complex64)
    cir = torch.zeros(memory + 1, dtype=torch.complex64)
    cir[0], cir[116] = 1, 0.8j
    tail = torch.zeros(memory, dtype=torch.complex64)
    rx = symbols.clone()
    rx[116:] += 0.8j * symbols[:-116]
    estimate, extra = linear_block_lmmse(rx, cir, tail, 0.001, real_bpsk=True)
    assert torch.equal(estimate.real >= 0, symbols.real >= 0)
    assert extra["relative_normal_residual"] < 3e-6
    assert extra["uses_decision_feedback"] is False


def test_heldout_calibration_requires_frozen_matching_settings(tmp_path):
    """留出核验禁止重新扫网格或复用校准 seed。"""
    import hashlib
    import json
    from scripts.calibrate_linear_baselines import run_calibration

    config = {"max_delay": 116, "main_snrs": [0, 5, 10, 15]}
    output = tmp_path / "should_not_exist.json"
    with pytest.raises(ValueError, match="--selection"):
        run_calibration(config, [1], 1, output, state_split="heldout_edge")
    with pytest.raises(ValueError, match="不匹配"):
        run_calibration(config, [1], 1, output, state_split="heldout_edge", selection={})
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
    with pytest.raises(ValueError, match="重叠"):
        run_calibration(config, [1], 1, output, state_split="heldout_edge",
                        selection={"config_sha256": signature, "calibration_seeds": [1]})
    assert not output.exists()
