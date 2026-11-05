# -*- coding: utf-8 -*-
"""共享均衡器输出头的信息审计应只依赖Adapt设计矩阵与logit。"""

import pytest
import torch

from scripts.diagnose_head_fisher import head_fisher, summarize_fisher


def test_head_fisher_is_symmetric_psd_and_uses_only_adapt_rows():
    torch.manual_seed(881)
    jacobian = torch.randn(12, 5, dtype=torch.float64)
    logits = torch.linspace(-2, 2, 12, dtype=torch.float64)
    mask = torch.tensor([True, True, True, True, False, False, False, False, False, False, False, False])
    fisher = head_fisher(jacobian, logits, mask)
    torch.testing.assert_close(fisher, fisher.T, atol=0, rtol=0)
    assert torch.linalg.eigvalsh(fisher).min() >= -1e-12
    changed = logits.clone()
    changed[~mask] = 1000
    torch.testing.assert_close(head_fisher(jacobian, changed, mask), fisher)
    assert fisher.shape == (5, 5)


def test_fisher_summary_reports_rank_and_rejects_invalid_geometry():
    fisher = torch.diag(torch.tensor([4.0, 1.0, 1e-10], dtype=torch.float64))
    summary = summarize_fisher(fisher, relative_tolerance=1e-6)
    assert summary["numerical_rank"] == 2
    assert summary["active_condition_number"] == pytest.approx(4.0)
    with pytest.raises(ValueError):
        head_fisher(torch.randn(4, 3), torch.randn(4), torch.zeros(4, dtype=torch.bool))
