import torch
import pytest

from scripts.diagnose_subspace_paired_validation import moving_block_interval, paired_improvement
from scripts.diagnose_subspace_data_reachability import optimize_data_coefficients
from scripts.diagnose_subspace_delayed_validation import _pilot_stats


def test_paired_validation_improvement_uses_symbolwise_loss_differences():
    before = torch.tensor([2.0, -1.0, 1.0])
    after = torch.tensor([3.0, -2.0, 0.5])
    labels = torch.tensor([1.0, 0.0, 1.0])
    mask = torch.tensor([True, True, False])
    values = paired_improvement(before, after, labels, mask)
    assert values.shape == (2,)
    assert bool((values > 0).all())


def test_moving_block_interval_is_deterministic_and_rejects_invalid_inputs():
    values = torch.tensor([1.0, 0.8, 0.9, 0.7, 1.1, 0.6, 0.8, 0.9])
    first = moving_block_interval(values, block_length=2, seed=42)
    second = moving_block_interval(values, block_length=2, seed=42)
    assert first == second
    assert first["lower95"] <= first["mean"] <= first["upper95"]
    with pytest.raises(ValueError):
        moving_block_interval(values, block_length=9)


def test_data_reachability_optimizer_changes_only_the_supplied_coefficient_copy():
    class LinearBasis:
        dimension = 1

        @staticmethod
        def basis(_observation):
            return torch.ones(4, 1)

    class Observation:
        base_logits = torch.zeros(4)

    initial = torch.zeros(1)
    result = optimize_data_coefficients(LinearBasis(), Observation(),
        labels=torch.ones(4), mask=torch.ones(4, dtype=torch.bool), initial=initial,
        steps=8, learning_rate=0.1)
    assert result["bce_gain_oracle"] > 0
    assert result["coefficient_delta_norm"] > 0
    assert initial.item() == 0


def test_delayed_pilot_statistics_pool_symbols_without_reading_data():
    class LinearBasis:
        @staticmethod
        def basis(_observation):
            return torch.ones(3, 1)

    class Observation:
        base_logits = torch.zeros(3)
        validation_mask = torch.ones(3, dtype=torch.bool)
        validation_labels = torch.ones(3)

    stats = _pilot_stats(LinearBasis(), torch.ones(1), torch.zeros(1), [Observation()])
    assert stats["symbols"] == 3
    assert stats["mean_bce_gain"] > 0
    assert stats["hard_error_nonincrease"]
