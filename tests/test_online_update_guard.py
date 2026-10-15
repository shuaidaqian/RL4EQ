"""验证在线 PEFT 更新不会因数值噪声造成无意义接受。"""

import pytest


def test_peft_guard_requires_configured_reward_improvement():
    import compare

    assert compare._accept_online_peft_update(0.40, 0.39, 0.005) is True
    assert compare._accept_online_peft_update(0.40, 0.396, 0.005) is False
    assert compare._accept_online_peft_update(0.40, 0.40, 0.005) is False


def test_windowed_reward_guard_rejects_one_bad_reward_subwindow():
    import torch
    import compare

    labels = torch.zeros(8)
    mask = torch.ones(8, dtype=torch.bool)
    before = torch.zeros(8)
    after = torch.tensor([-2.0, -2.0, -2.0, -2.0, 0.2, 0.2, 0.2, 0.2])

    accepted, gains = compare._accept_windowed_reward_update(
        before,
        after,
        labels,
        mask,
        min_improvement=0.01,
        windows=2,
    )

    assert accepted is False
    assert gains[0] > 0.01
    assert gains[1] < 0.0


def test_windowed_reward_guard_accepts_consistent_reward_improvement():
    import torch
    import compare

    labels = torch.zeros(8)
    mask = torch.ones(8, dtype=torch.bool)
    before = torch.zeros(8)
    after = torch.full((8,), -2.0)

    accepted, gains = compare._accept_windowed_reward_update(
        before,
        after,
        labels,
        mask,
        min_improvement=0.01,
        windows=2,
    )

    assert accepted is True
    assert all(gain > 0.01 for gain in gains)


def test_windowed_reward_guard_supports_relative_improvement_at_high_snr():
    import torch
    import compare

    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    mask = torch.ones(4, dtype=torch.bool)
    before = torch.tensor([-4.0, -4.0, 4.0, 4.0])
    after = torch.tensor([-5.0, -5.0, 5.0, 5.0])

    accepted, gains = compare._accept_windowed_reward_update(
        before,
        after,
        labels,
        mask,
        min_improvement=0.0005,
        windows=2,
        relative_min_improvement=0.01,
    )

    assert accepted is True
    assert len(gains) == 2


def test_pilot_reward_robust_ranking_penalizes_inconsistent_subwindows():
    import compare

    stable_score, stable_stats = compare._pilot_reward_ranking_score(
        [0.010, 0.011, 0.010, 0.011],
        0.0105,
        mode="robust_lcb",
        dispersion=0.5,
    )
    unstable_score, unstable_stats = compare._pilot_reward_ranking_score(
        [0.002, 0.002, 0.002, 0.050],
        0.014,
        mode="robust_lcb",
        dispersion=0.5,
    )

    assert stable_stats["mad_gain"] == pytest.approx(0.0005)
    assert stable_score > unstable_score


def test_pilot_reward_ranking_rejects_unknown_mode():
    import compare

    with pytest.raises(ValueError, match="online_reward_ranking"):
        compare._pilot_reward_ranking_score([0.1, 0.1], 0.1, mode="data_oracle")


def test_pilot_consensus_ranking_requires_bce_and_signal_improvement():
    import compare

    signal_supported, supported_stats = compare._pilot_reward_ranking_score(
        [0.08, 0.09, 0.08, 0.09],
        0.085,
        mode="pilot_consensus",
        dispersion=0.5,
        baseline_loss=1.0,
        signal_gain_relative=0.04,
    )
    signal_regressed, regressed_stats = compare._pilot_reward_ranking_score(
        [0.12, 0.13, 0.12, 0.13],
        0.125,
        mode="pilot_consensus",
        dispersion=0.5,
        baseline_loss=1.0,
        signal_gain_relative=-0.01,
    )

    assert supported_stats["bce_lcb"] > 0.0
    assert supported_stats["signal_gain_relative"] == pytest.approx(0.04)
    assert signal_supported == pytest.approx(0.04)
    assert regressed_stats["signal_gain_relative"] == pytest.approx(-0.01)
    assert signal_regressed == pytest.approx(-0.01)
    assert signal_supported > signal_regressed


def test_pilot_balanced_ranking_rewards_two_consistent_pilot_views():
    import compare

    balanced, stats = compare._pilot_reward_ranking_score(
        [0.08, 0.09, 0.08, 0.09],
        0.085,
        mode="pilot_balanced",
        dispersion=0.5,
        baseline_loss=1.0,
        signal_gain_relative=0.04,
    )
    bce_only = compare._pilot_reward_ranking_score(
        [0.08, 0.09, 0.08, 0.09],
        0.085,
        mode="pilot_balanced",
        dispersion=0.5,
        baseline_loss=1.0,
        signal_gain_relative=0.0025,
    )[0]

    assert stats["bce_lcb"] > 0.0
    assert balanced == pytest.approx((stats["bce_lcb"] * 0.04) ** 0.5)
    assert balanced > bce_only


def test_pilot_stable_consensus_penalizes_inconsistent_reward_windows():
    import compare

    stable, stable_stats = compare._pilot_reward_ranking_score(
        [0.08, 0.09, 0.08, 0.09],
        0.085,
        mode="pilot_stable_consensus",
        dispersion=0.5,
        baseline_loss=1.0,
        signal_gain_relative=0.04,
    )
    inconsistent, inconsistent_stats = compare._pilot_reward_ranking_score(
        [0.08, 0.08, 0.08, -0.02],
        0.055,
        mode="pilot_stable_consensus",
        dispersion=0.5,
        baseline_loss=1.0,
        signal_gain_relative=0.04,
    )

    assert stable_stats["window_stability"] == pytest.approx(
        1.0 - 1.4826 * 0.005 / 0.085
    )
    assert 0.0 < inconsistent_stats["window_stability"] < stable_stats["window_stability"]
    assert stable > inconsistent


def test_pilot_stable_consensus_rejects_missing_or_regressed_signal_view():
    import compare

    score, stats = compare._pilot_reward_ranking_score(
        [0.08, 0.09, 0.08, 0.09],
        0.085,
        mode="pilot_stable_consensus",
        baseline_loss=1.0,
        signal_gain_relative=-0.01,
    )

    assert stats["window_stability"] > 0.0
    assert score < 0.0


def test_reward_pilot_hard_ber_guard_rejects_more_bit_errors():
    import torch
    import compare

    assert compare._accept_reward_pilot_hard_ber(
        logits_before=torch.tensor([-2.0, 2.0, -2.0, 2.0]),
        logits_after=torch.tensor([-2.0, -2.0, -2.0, 2.0]),
        reward_mask=torch.ones(4, dtype=torch.bool),
        labels=torch.tensor([0.0, 1.0, 0.0, 1.0]),
    ) is False

    assert compare._accept_reward_pilot_hard_ber(
        logits_before=torch.tensor([-2.0, 2.0, -2.0, 2.0]),
        logits_after=torch.tensor([-3.0, 3.0, -2.0, 2.0]),
        reward_mask=torch.ones(4, dtype=torch.bool),
        labels=torch.tensor([0.0, 1.0, 0.0, 1.0]),
    ) is True


def test_reward_pilot_hard_guard_uses_pilot_symbols_instead_of_data_labels():
    import torch
    import compare

    before = torch.tensor([-2.0, 2.0])
    after = torch.tensor([-3.0, 1.0])
    mask = torch.ones(2, dtype=torch.bool)
    reward_symbols = torch.tensor([1.0, -1.0], dtype=torch.complex64)
    assert compare._accept_reward_pilot_hard_ber(
        before,
        after,
        reward_mask=mask,
        reward_symbols=reward_symbols,
    ) is True
    assert compare._accept_reward_pilot_hard_ber(
        before,
        after,
        reward_mask=mask,
        reward_symbols=reward_symbols,
        labels=torch.tensor([0.0, 1.0]),
    ) is True


def test_previous_online_update_guard_detects_cross_frame_reward_regression():
    import compare

    assert compare._previous_update_is_harmful(0.51, 0.50, 1e-5) is True
    assert compare._previous_update_is_harmful(0.500005, 0.50, 1e-5) is False


def test_online_cli_overrides_include_physical_state_confidence_gate():
    import compare

    config = {"online_phase_tracking_min_confidence": 0.5}
    compare._apply_online_cli_overrides(
        config,
        {
            "online_phase_tracking_min_confidence": 0.15,
            "online_phase_tracking_smoothing": 0.4,
        },
    )

    assert config["online_phase_tracking_min_confidence"] == 0.15
    assert config["online_phase_tracking_smoothing"] == 0.4


def test_online_cli_overrides_can_select_sgd_reconstruction_objective():
    import compare

    config = {
        "online_adaptation_algorithm": "rls",
        "online_adaptation_objective": "bce",
    }
    compare._apply_online_cli_overrides(
        config,
        {
            "online_adaptation_algorithm": "sgd",
            "online_adaptation_objective": "pilot_reconstruction",
        },
    )

    assert config["online_adaptation_algorithm"] == "sgd"
    assert config["online_adaptation_objective"] == "pilot_reconstruction"


def test_online_cli_override_can_defer_peft_commit():
    import compare

    config = {}
    compare._apply_online_cli_overrides(
        config,
        {"online_defer_peft_commit": True},
    )

    assert config["online_defer_peft_commit"] is True


def test_pilot_margin_consensus_uses_soft_margin_as_third_pilot_view():
    import compare

    supported, stats = compare._pilot_reward_ranking_score(
        [0.08, 0.09, 0.08, 0.09],
        0.085,
        mode="pilot_margin_consensus",
        dispersion=0.5,
        baseline_loss=1.0,
        signal_gain_relative=0.04,
        margin_window_gains=[0.20, 0.22, 0.20, 0.22],
        margin_gain_relative=0.20,
    )
    rejected, rejected_stats = compare._pilot_reward_ranking_score(
        [0.08, 0.09, 0.08, 0.09],
        0.085,
        mode="pilot_margin_consensus",
        dispersion=0.5,
        baseline_loss=1.0,
        signal_gain_relative=0.04,
        margin_window_gains=[-0.20, -0.18, -0.20, -0.18],
        margin_gain_relative=-0.19,
    )

    assert stats["margin_gain_relative"] == pytest.approx(0.20)
    assert rejected_stats["margin_gain_relative"] == pytest.approx(-0.19)
    assert supported > 0.0
    assert rejected < supported


def test_unknown_data_waveform_is_removed_before_online_cir_update():
    import torch
    from types import SimpleNamespace
    from agent.cir_estimator import decision_directed_cir_update

    frame = SimpleNamespace(
        adapt_mask=torch.tensor([True, True, True, True, False, False]),
        tx_symbols=torch.tensor([1, -1, 1, -1, 1, -1], dtype=torch.complex64),
        rx_symbols=torch.tensor([1, -1, 1, -1, 9, -9], dtype=torch.complex64),
    )
    changed = SimpleNamespace(
        adapt_mask=frame.adapt_mask,
        tx_symbols=frame.tx_symbols,
        rx_symbols=torch.tensor([1, -1, 1, -1, -900, 900], dtype=torch.complex64),
    )
    logits = torch.tensor([2.0, -2.0, 2.0, -2.0, 100.0, -100.0])
    previous = torch.tensor([1.0 + 0j, 0.1 + 0j], dtype=torch.complex64)
    result_a = decision_directed_cir_update(frame, logits, 1, previous, alpha=0.5)
    result_b = decision_directed_cir_update(changed, logits, 1, previous, alpha=0.5)

    assert torch.allclose(result_a, result_b, atol=1e-6, rtol=1e-6)
