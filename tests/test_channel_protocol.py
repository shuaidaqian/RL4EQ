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
from env.comm_env import CommEnvConfig, CommunicationEnvironment, ReceiverState
from baseline.synchronization_compensation import unwrap_phase
from env.frame_structure import FrameConfig, FrameGenerator


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


def test_level_a_supports_three_path_boundary_case():
    profile = sample_profile(ChannelProfileConfig(level=ChannelLevel.A, max_delay=2, seed=31))
    assert profile.delays == (0, 1, 2)
    assert len(profile.delays) == 3
    assert 6.0 <= profile.strongest_gap_db <= 15.0
    assert -20.0 <= profile.max_delay_relative_db <= -10.0
    assert 0.10 <= profile.delayed_energy_ratio <= 0.35


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
        sample_profile(ChannelProfileConfig(level=ChannelLevel.A, max_delay=20, seed=18, max_attempts=1))


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


def test_cfo_impairment_rotates_phase_continuously_across_frames():
    """残余 CFO 必须按 episode 全局符号时钟连续旋转，不能每帧重置。"""

    channel = ExtremeDelayChannel(
        ExtremeDelayChannelConfig(
            level="A",
            max_delay=12,
            snr_db=80.0,
            rho=1.0,
            seed=8100,
            cfo_cycles_per_symbol=0.01,
        )
    )
    warmup = torch.ones(12, dtype=torch.complex64)
    symbols = torch.ones(32, dtype=torch.complex64)
    channel.reset_episode(warmup)
    first = channel.transmit(symbols, add_noise=False)
    second = channel.transmit(symbols, add_noise=False)

    combined = torch.cat((first, second))
    phase = unwrap_phase(torch.angle(combined))
    increments = phase[1:] - phase[:-1]
    expected = torch.full_like(increments, 2.0 * torch.pi * 0.01)
    assert torch.allclose(increments, expected, atol=2e-3)


def test_phase_impairment_is_seed_reproducible_and_clean_default_unchanged():
    """慢相位扰动必须可复现；默认 clean 配置不能改变既有线性信道行为。"""

    symbols = torch.ones(64, dtype=torch.complex64)
    warmup = torch.ones(12, dtype=torch.complex64)
    clean_a = ExtremeDelayChannel(ExtremeDelayChannelConfig(level="A", max_delay=12, snr_db=80.0, rho=1.0, seed=8200))
    clean_b = ExtremeDelayChannel(ExtremeDelayChannelConfig(level="A", max_delay=12, snr_db=80.0, rho=1.0, seed=8200))
    clean_a.reset_episode(warmup)
    clean_b.reset_episode(warmup)
    assert torch.allclose(clean_a.transmit(symbols, add_noise=False), clean_b.transmit(symbols, add_noise=False))

    impaired_a = ExtremeDelayChannel(
        ExtremeDelayChannelConfig(
            level="A",
            max_delay=12,
            snr_db=80.0,
            rho=1.0,
            seed=8300,
            phase_noise_std=0.003,
        )
    )
    impaired_b = ExtremeDelayChannel(
        ExtremeDelayChannelConfig(
            level="A",
            max_delay=12,
            snr_db=80.0,
            rho=1.0,
            seed=8300,
            phase_noise_std=0.003,
        )
    )
    impaired_c = ExtremeDelayChannel(
        ExtremeDelayChannelConfig(
            level="A",
            max_delay=12,
            snr_db=80.0,
            rho=1.0,
            seed=8301,
            phase_noise_std=0.003,
        )
    )
    for channel in (impaired_a, impaired_b, impaired_c):
        channel.reset_episode(warmup)
    out_a = impaired_a.transmit(symbols, add_noise=False)
    out_b = impaired_b.transmit(symbols, add_noise=False)
    out_c = impaired_c.transmit(symbols, add_noise=False)
    assert torch.allclose(out_a, out_b)
    assert not torch.allclose(out_a, out_c)


def test_tiny_impairment_profiles_are_weaker_than_light_profiles():
    from env.impairments import settings_from_profile

    import numpy as np

    tiny = settings_from_profile("cfo_phase_tiny", np.random.default_rng(9000))
    light = settings_from_profile("cfo_phase_light", np.random.default_rng(9000))

    assert abs(tiny.cfo_cycles_per_symbol) < abs(light.cfo_cycles_per_symbol)
    assert tiny.phase_noise_std < light.phase_noise_std


def test_eme_slow_drift_v1_profile_freezes_calibrated_impairment_budget():
    from env.impairments import settings_from_profile

    import numpy as np

    profile = settings_from_profile("eme_slow_drift_v1", np.random.default_rng(9001))

    assert abs(profile.cfo_cycles_per_symbol) == 0.0008
    assert profile.phase_noise_std == 0.003


@pytest.mark.parametrize("total", [64, 96, 128, 160])
@pytest.mark.parametrize("layout", ["prefix", "two_block", "multi_block"])
def test_frame_masks_and_unknown_regions(total, layout):
    frame = FrameGenerator(FrameConfig(total_pilot=total, layout=layout, max_delay=40), seed=9).generate(2)
    assert int(frame.adapt_mask.sum()) == 3 * total // 4
    assert int(frame.reward_mask.sum()) == total // 4
    assert int(frame.data_mask.sum()) == 512 - total
    assert not torch.any(frame.adapt_mask & frame.reward_mask)
    assert not torch.any(frame.adapt_mask & frame.data_mask)
    assert not torch.any(frame.reward_mask & frame.data_mask)
    assert torch.equal(frame.unknown_region_mask, frame.reward_mask | frame.data_mask)
    assert torch.all(frame.model_region_ids[frame.reward_mask] == frame.model_region_ids[frame.data_mask][0])
    assert max(frame.adapt_block_lengths) >= 48


def test_receiver_view_hides_reward_data_labels_positions_and_true_cir():
    frame = FrameGenerator(FrameConfig(total_pilot=128, layout="multi_block", max_delay=40), seed=11).generate(3)
    view_a = frame.receiver_view()
    assert not torch.equal(view_a.rx_symbols[frame.unknown_region_mask], frame.tx_symbols[frame.unknown_region_mask])
    changed = frame.with_replaced_hidden_labels(
        reward_bits=torch.logical_not(frame.bits.bool()),
        data_bits=torch.logical_not(frame.bits.bool()),
    )
    view_b = changed.receiver_view()
    assert torch.equal(view_a.rx_symbols, view_b.rx_symbols)
    assert torch.equal(view_a.adapt_symbols, view_b.adapt_symbols)
    assert torch.equal(view_a.adapt_mask, view_b.adapt_mask)
    assert not hasattr(view_a, "reward_mask")
    assert not hasattr(view_a, "reward_bits")
    assert not hasattr(view_a, "data_bits")
    assert not hasattr(view_a, "true_cir")


def test_episode_acquisition_tail_and_receiver_state_isolation():
    env = CommunicationEnvironment(
        CommEnvConfig(level="B", max_delay=20, snr_db=15.0, total_pilot=96, layout="two_block", seed=13)
    )
    start = env.reset_episode()
    first = env.next_frame()
    assert torch.equal(first.tail_symbols, start.acquisition.tx_symbols[-20:])
    assert torch.equal(first.true_cir, env.channel.last_cir_used())
    state_a = ReceiverState(start.initial_soft_tail.clone())
    state_b = state_a.clone()
    state_a.update_tail(torch.zeros(20, dtype=torch.complex64))
    assert not torch.equal(state_a.soft_tail, state_b.soft_tail)


def _eme_env_config(**changes):
    payload = {
        "profile_name": "eme_measurement_v1",
        "level": "B",
        "sample_rate_hz": 2_000.0,
        "symbol_rate_hz": 2_000.0,
        "frame_len": 256,
        "coherence_time_seconds": 120.0,
        "strong_path_count": (3, 7),
        "diffuse_energy_ratio": (0.05, 0.15),
        "total_pilot": 64,
        "layout": "prefix",
        "seed": 31,
    }
    payload.update(changes)
    return CommEnvConfig(**payload)


def test_eme_environment_forwards_frame_length_and_uses_physical_tail():
    env = CommunicationEnvironment(_eme_env_config())

    assert env.channel.max_delay == 24
    assert env.channel.config.frame_len == 256
    assert env.frame_generator.config.frame_len == 256
    assert env.frame_generator.config.max_delay == 24

    start = env.reset_episode()
    frame = env.next_frame()

    assert start.warmup_symbols.shape == (24,)
    assert start.acquisition.tx_symbols.shape == (256,)
    assert start.initial_soft_tail.shape == (24,)
    assert frame.tx_symbols.shape == (256,)
    assert frame.tail_symbols.shape == (24,)
    assert torch.equal(frame.tail_symbols, start.acquisition.tx_symbols[-24:])


def test_eme_environment_requires_prefix_but_legacy_keeps_diagnostic_layouts():
    with pytest.raises(ValueError, match="eme_measurement_v1.*prefix"):
        _eme_env_config(layout="two_block")

    legacy = CommEnvConfig(layout="multi_block")
    assert legacy.layout == "multi_block"


def test_legacy_environment_config_keeps_existing_positional_argument_order():
    config = CommEnvConfig("B", 20, 15.0, 0.99, 96, "two_block", 13)

    assert config.profile_name == "legacy_sparse_v1"
    assert config.level == "B"
    assert config.max_delay == 20
    assert config.total_pilot == 96
    assert config.layout == "two_block"
    assert config.seed == 13


def test_eme_receiver_view_hides_truth_impairments_metadata_and_hidden_labels():
    env = CommunicationEnvironment(
        _eme_env_config(
            impairment_profile="eme_slow_drift_v1",
            cfo_cycles_per_symbol=0.0008,
            phase_noise_std=0.003,
        )
    )
    start = env.reset_episode()
    frame = env.next_frame()

    for view in (start.acquisition.receiver_view(), frame.receiver_view()):
        for hidden_name in (
            "true_cir",
            "cfo_cycles_per_symbol",
            "phase_noise_std",
            "phase_state",
            "profile_metadata",
            "diffuse_mask",
            "reward_mask",
            "reward_bits",
            "data_mask",
            "data_bits",
        ):
            assert not hasattr(view, hidden_name)

    changed = frame.with_replaced_hidden_labels(
        reward_bits=torch.logical_not(frame.bits),
        data_bits=torch.logical_not(frame.bits),
    )
    assert torch.equal(frame.receiver_view().rx_symbols, changed.receiver_view().rx_symbols)
    assert torch.equal(frame.receiver_view().adapt_symbols, changed.receiver_view().adapt_symbols)


def test_pilot_sequence_reproducible_per_frame_but_changes_across_frames():
    generator_a = FrameGenerator(FrameConfig(total_pilot=64, layout="prefix", max_delay=20), seed=21)
    generator_b = FrameGenerator(FrameConfig(total_pilot=64, layout="prefix", max_delay=20), seed=21)
    same_a = generator_a.generate(5)
    same_b = generator_b.generate(5)
    different = generator_a.generate(6)
    assert torch.equal(same_a.bits, same_b.bits)
    assert not torch.equal(same_a.bits[same_a.adapt_mask], different.bits[different.adapt_mask])
