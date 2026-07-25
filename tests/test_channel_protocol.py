import numpy as np
import pytest

from env.channel_profiles import (
    ChannelLevel,
    ChannelProfileConfig,
    ProfileSamplingError,
    classify_for_aggregation,
    sample_profile,
)


def test_level_b_profile_obeys_power_constraints_and_seed():
    cfg = ChannelProfileConfig(level=ChannelLevel.B, max_delay=40, seed=17)
    first = sample_profile(cfg)
    second = sample_profile(cfg)
    assert first.delays == second.delays
    assert np.allclose(first.taps, second.taps)
    assert first.delays[0] == 0 and first.delays[-1] == 40
    assert 3 <= len(first.delays) <= 7
    assert 0.0 <= first.strongest_gap_db <= 6.0
    assert first.max_delay_relative_db >= -10.0
    assert 0.35 <= first.delayed_energy_ratio <= 0.75
    assert np.isclose(np.sum(np.abs(first.taps) ** 2), 1.0, atol=1e-6)


def test_level_a_profile_constraints_are_milder_main_path_dominant():
    profile = sample_profile(ChannelProfileConfig(level=ChannelLevel.A, max_delay=20, seed=19))
    assert profile.delays[0] == 0 and profile.delays[-1] == 20
    assert 3 <= len(profile.delays) <= 5
    assert 6.0 <= profile.strongest_gap_db <= 15.0
    assert -20.0 <= profile.max_delay_relative_db <= -10.0
    assert 0.10 <= profile.delayed_energy_ratio <= 0.35
    assert np.isfinite(profile.notch_depth_db)
    assert np.isfinite(profile.condition_proxy)


def test_level_c_profile_is_pressure_only_and_not_mixed_into_level_b():
    profile = sample_profile(ChannelProfileConfig(level=ChannelLevel.C, max_delay=50, seed=23))
    assert profile.delays[0] == 0 and profile.delays[-1] == 50
    assert 3 <= len(profile.delays) <= 10
    assert 0.50 <= profile.delayed_energy_ratio <= 0.90
    assert classify_for_aggregation(profile.level) == "pressure"
    assert classify_for_aggregation(ChannelLevel.B) == "main"


def test_invalid_profile_configurations_raise_clear_errors():
    with pytest.raises(ValueError):
        ChannelProfileConfig(level="D", max_delay=40, seed=1)
    with pytest.raises(ValueError):
        ChannelProfileConfig(level=ChannelLevel.B, max_delay=0, seed=1)
    with pytest.raises(ProfileSamplingError):
        sample_profile(ChannelProfileConfig(level=ChannelLevel.A, max_delay=2, seed=1, max_attempts=1))
