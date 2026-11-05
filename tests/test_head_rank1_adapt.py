# -*- coding: utf-8 -*-
"""rank-1共享head更新必须留在Pilot Fisher子空间和累计信赖域内。"""

import torch

from scripts.prototype_head_rank1_adapt import _project_candidate


def test_projected_head_candidate_stays_in_rank_one_basis_and_radius():
    names = ["head.weight", "head.bias"]
    origin = {"head.weight": torch.zeros(1, 4), "head.bias": torch.zeros(1)}
    before = {"head.weight": torch.tensor([[0.01, 0.0, 0.0, 0.0]]),
              "head.bias": torch.tensor([0.0])}
    candidate = {"head.weight": torch.tensor([[0.03, 0.04, -0.02, 0.01]]),
                 "head.bias": torch.tensor([0.03])}
    basis = torch.zeros(5)
    basis[0] = 1.0
    projected, raw_norm, projected_norm = _project_candidate(
        candidate, before, origin, names, basis, radius=0.02)
    flat = torch.cat((projected["head.weight"].reshape(-1), projected["head.bias"].reshape(-1)))
    assert flat[0] == torch.tensor(0.02)
    assert torch.count_nonzero(flat[1:]) == 0
    assert torch.linalg.vector_norm(flat) <= 0.0200001
    assert raw_norm > projected_norm > 0
