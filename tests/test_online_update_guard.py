"""验证在线 PEFT 更新不会因数值噪声造成无意义接受。"""


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
