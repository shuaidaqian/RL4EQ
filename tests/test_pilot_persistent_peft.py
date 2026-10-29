"""验证连续 Pilot PEFT 的参数累计与单步限幅。"""

import pytest
import torch

from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from scripts.replay_pilot_persistent_peft import (
    _aggregate_reward_history,
    _clip_candidate_delta,
    _clip_candidate_to_reference,
    _parameter_delta_norm,
    _group_parameters,
    _reward_history_is_consistent,
)


def test_persistent_adapter_delta_is_clipped_without_changing_offline_parameters():
    torch.manual_seed(32)
    model = UnfoldedEqualizer(UnfoldedConfig(
        frame_len=128,
        max_delay=4,
        iterations=1,
        d_model=16,
        num_heads=4,
        enable_phase_correction_branch=True,
        phase_correction_initial_scale=1.0,
    ))
    model.attach_online_phase_trend_adapter()
    reference = UnfoldedEqualizer(model.config)
    reference.attach_online_phase_trend_adapter()
    reference.load_state_dict(model.state_dict(), strict=True)
    before = {
        name: parameter.detach().clone()
        for name, parameter in _group_parameters(model, "phase_trend").items()
    }
    with torch.no_grad():
        for parameter in _group_parameters(model, "phase_trend").values():
            parameter.add_(0.25)

    clipped_norm = _clip_candidate_delta(model, before, "phase_trend", 0.02)

    assert clipped_norm == 0.02
    assert _parameter_delta_norm(model, reference, "phase_trend") <= 0.020001
    for name, parameter in reference.named_parameters():
        if getattr(parameter, "_peft_group", None) != "phase_trend":
            assert torch.equal(parameter, dict(model.named_parameters())[name])


def test_channel_residual_adapter_delta_uses_the_selected_parameter_group():
    torch.manual_seed(33)
    model = UnfoldedEqualizer(UnfoldedConfig(
        frame_len=128,
        max_delay=4,
        iterations=1,
        d_model=16,
        num_heads=4,
    ))
    model.attach_online_channel_residual_adapter()
    reference = UnfoldedEqualizer(model.config)
    reference.attach_online_channel_residual_adapter()
    reference.load_state_dict(model.state_dict(), strict=True)
    before = {
        name: parameter.detach().clone()
        for name, parameter in _group_parameters(model, "channel_residual").items()
    }
    with torch.no_grad():
        for parameter in _group_parameters(model, "channel_residual").values():
            parameter.add_(0.25)

    clipped_norm = _clip_candidate_delta(model, before, "channel_residual", 0.02)

    assert clipped_norm == 0.02
    assert _parameter_delta_norm(model, reference, "channel_residual") <= 0.020001
    for name, parameter in reference.named_parameters():
        if getattr(parameter, "_peft_group", None) != "channel_residual":
            assert torch.equal(parameter, dict(model.named_parameters())[name])


def test_cumulative_trust_region_caps_channel_residual_against_frozen_reference():
    torch.manual_seed(34)
    model = UnfoldedEqualizer(UnfoldedConfig(
        frame_len=128,
        max_delay=4,
        iterations=1,
        d_model=16,
        num_heads=4,
    ))
    model.attach_online_channel_residual_adapter()
    reference = UnfoldedEqualizer(model.config)
    reference.attach_online_channel_residual_adapter()
    reference.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        for parameter in _group_parameters(model, "channel_residual").values():
            parameter.add_(0.1)

    clipped_norm = _clip_candidate_to_reference(model, reference, "channel_residual", 0.03)

    assert clipped_norm == pytest.approx(0.03)
    assert _parameter_delta_norm(model, reference, "channel_residual") <= 0.030001


def test_reward_history_averages_only_pilot_scores_and_checks_majority_consistency():
    aggregate = _aggregate_reward_history([
        {
            "bce_improvement": 0.02,
            "signal_improvement": 0.03,
            "errors_before": 4,
            "errors_after": 3,
        },
        {
            "bce_improvement": 0.01,
            "signal_improvement": -0.01,
            "errors_before": 3,
            "errors_after": 3,
        },
        {
            "bce_improvement": -0.01,
            "signal_improvement": 0.02,
            "errors_before": 2,
            "errors_after": 3,
        },
    ])

    assert aggregate["frame_count"] == 3
    assert aggregate["bce_improvement"] == pytest.approx(0.02 / 3)
    assert aggregate["signal_improvement"] == pytest.approx(0.04 / 3)
    assert aggregate["signal_positive_fraction"] == pytest.approx(2 / 3)
    assert aggregate["error_nonincrease_fraction"] == pytest.approx(2 / 3)
    assert _reward_history_is_consistent(aggregate)


def test_reward_history_rejects_candidate_when_majority_consistency_is_not_met():
    aggregate = _aggregate_reward_history([
        {
            "bce_improvement": 0.1,
            "signal_improvement": 0.02,
            "errors_before": 1,
            "errors_after": 1,
        },
        {
            "bce_improvement": 0.1,
            "signal_improvement": 0.01,
            "errors_before": 2,
            "errors_after": 3,
        },
        {
            "bce_improvement": 0.1,
            "signal_improvement": -0.01,
            "errors_before": 3,
            "errors_after": 4,
        },
    ])

    assert not _reward_history_is_consistent(aggregate)
