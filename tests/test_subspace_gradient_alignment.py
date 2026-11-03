import torch

import pytest

from scripts.diagnose_subspace_gradient_alignment import _cosine, parse_temperatures


def test_gradient_cosine_reports_alignment_and_zero_signal():
    assert _cosine(torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0])) == 1.0
    assert _cosine(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])) == 0.0
    assert _cosine(torch.zeros(2), torch.ones(2)) is None


def test_adapt_temperature_grid_requires_unique_positive_finite_values():
    assert parse_temperatures("0.25, 0.5, 1, 2, 4") == [0.25, 0.5, 1.0, 2.0, 4.0]
    for invalid in ("", "0,1", "1,-1", "1,nan", "1,1"):
        with pytest.raises(ValueError):
            parse_temperatures(invalid)
