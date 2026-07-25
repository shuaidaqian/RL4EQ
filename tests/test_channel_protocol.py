import numpy as np
import pytest
import torch

from env.channel_profiles import (
    ChannelLevel,
    ChannelProfileConfig,
    ProfileSamplingError,
    classify_for_aggregation,
    sample_profile,
)
from env.extreme_delay_channel import ExtremeDelayChannel, ExtremeDelayChannelConfig


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


def test_channel_uses_known_history_and_preserves_support():
    channel = ExtremeDelayChannel(
        ExtremeDelayChannelConfig(level="B", max_delay=20, snr_db=10.0, rho=0.99, seed=3)
    )
    channel.reset_episode(torch.ones(20, dtype=torch.complex64))
    first_delays = channel.delays
    channel.transmit(torch.ones(64, dtype=torch.complex64), add_noise=False)
    second = channel.transmit(torch.zeros(64, dtype=torch.complex64), add_noise=False)
    assert torch.any(second[:20].abs() > 0)
    assert channel.delays == first_delays
    assert torch.allclose(channel.tap_power.sum(), torch.tensor(1.0), atol=1e-5)
    assert channel.true_cir().shape == (21,)


def test_channel_noise_uses_fixed_esn0_without_rx_power_rescaling():
    channel = ExtremeDelayChannel(
        ExtremeDelayChannelConfig(level="A", max_delay=12, snr_db=10.0, rho=1.0, seed=5)
    )
    channel.reset_episode(torch.ones(12, dtype=torch.complex64))
    clean = channel.transmit(torch.ones(4096, dtype=torch.complex64), add_noise=False)
    channel.reset_episode(torch.ones(12, dtype=torch.complex64))
    noisy = channel.transmit(torch.ones(4096, dtype=torch.complex64), add_noise=True)
    measured = torch.mean(torch.abs(noisy - clean) ** 2).item()
    assert measured == pytest.approx(0.1, rel=0.25)


def test_channel_support_fixed_while_taps_evolve_slowly():
    channel = ExtremeDelayChannel(
        ExtremeDelayChannelConfig(level="C", max_delay=30, snr_db=20.0, rho=0.995, seed=7)
    )
    channel.reset_episode(torch.ones(30, dtype=torch.complex64))
    delays = channel.delays
    first = channel.true_cir()
    channel.transmit(torch.ones(128, dtype=torch.complex64), add_noise=False)
    second = channel.true_cir()
    assert channel.delays == delays
    assert torch.linalg.norm(first - second).item() > 0.0
    assert torch.linalg.norm(first - second).item() < 0.5
