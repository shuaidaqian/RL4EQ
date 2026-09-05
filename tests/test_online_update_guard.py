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
