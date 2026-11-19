# -*- coding: utf-8 -*-
"""防止 NMSE 的幅度对齐和高置信错误掩盖接收失配。"""

import pytest
import torch

from evaluation.receiver_diagnostics import receiver_diagnostics


def test_scale_aligned_nmse_does_not_hide_reported_gain_error():
    truth = torch.tensor([1.+0j, 0j])
    result = receiver_diagnostics(torch.zeros(2), torch.zeros(2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool), 2*truth, truth)
    assert result['aligned_nmse_diagnostic'] == 0
    assert result['phase_only_nmse_diagnostic'] == pytest.approx(1.)
    assert result['decode_cir_norm'] == 2*result['true_cir_norm_diagnostic']


def test_high_confidence_errors_are_reported_separately_from_confidence():
    result = receiver_diagnostics(torch.tensor([40., -40.]), torch.tensor([False, True]),
        torch.ones(2, dtype=torch.bool), torch.ones(1, dtype=torch.complex64), torch.ones(1, dtype=torch.complex64))
    assert result['confident_data_count'] == result['confident_wrong_data_count'] == 2
    assert result['data_bce'] == pytest.approx(40.)
    assert result['data_brier'] == pytest.approx(1.)
