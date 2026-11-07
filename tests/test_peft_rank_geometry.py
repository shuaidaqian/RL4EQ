# -*- coding: utf-8 -*-
"""秩几何只读诊断的纯函数契约。

诊断依据：
- 秩度量需要同形状噪声基线（arXiv:2610.22438）；
- LoRA 内在维度由秩-r 子空间饱和点估计（arXiv:2604.08749）；
- 响应谱集中度用多帧响应主方向捕获比例（arXiv:2609.33765）。
"""

import pytest
import torch

from scripts.diagnose_peft_rank_geometry import (
    gaussian_baseline,
    rank_measures,
    subspace_capture,
)


def test_rank_measures_on_rank_two_diagonal():
    matrix = torch.diag(torch.tensor([4.0, 1.0, 1e-10], dtype=torch.float64))
    measures = rank_measures(matrix)
    # 阈值计数：4 与 1 均不低于 0.01*4，1e-10 低于阈值。
    assert measures["threshold_count"] == pytest.approx(2.0 / 3.0)
    # 能量秩：第一个方向已占 16/17 > 90% 能量。
    assert measures["energy_rank_90"] == pytest.approx(1.0 / 3.0)
    # 全等奇异值矩阵的各向同性极限：所有度量趋近 1。
    assert measures["stable_rank"] < 1.0
    assert measures["entropy_rank"] < 1.0


def test_rank_measures_isotropic_matrix_is_near_full_rank():
    torch.manual_seed(7)
    q, _ = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64))
    measures = rank_measures(q)
    for key in ("threshold_count", "energy_rank_90", "entropy_rank",
                "participation_ratio", "stable_rank"):
        assert measures[key] == pytest.approx(1.0, abs=1e-6), key


def test_rank_measures_rejects_non_matrix():
    with pytest.raises(ValueError):
        rank_measures(torch.randn(8))


def test_subspace_capture_hits_one_for_matching_basis_and_zero_for_orthogonal():
    basis = torch.eye(6, dtype=torch.float64)[:, :2]
    target = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 2.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    assert subspace_capture(basis, target) == pytest.approx(1.0)

    orthogonal = torch.tensor([[0.0, 0.0, 3.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    assert subspace_capture(basis, orthogonal) == pytest.approx(0.0, abs=1e-12)


def test_gaussian_baseline_is_deterministic_and_shaped():
    first = gaussian_baseline((12, 5), trials=16, seed=3)
    second = gaussian_baseline((12, 5), trials=16, seed=3)
    assert first == second
    assert set(first) == {"threshold_count", "energy_rank_90", "entropy_rank",
                          "participation_ratio", "stable_rank"}
    assert all(0.0 <= value <= 1.0 for value in first.values())
