import torch

from scripts.diagnose_subspace_gradient_alignment import _cosine


def test_gradient_cosine_reports_alignment_and_zero_signal():
    assert _cosine(torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0])) == 1.0
    assert _cosine(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])) == 0.0
    assert _cosine(torch.zeros(2), torch.ones(2)) is None
