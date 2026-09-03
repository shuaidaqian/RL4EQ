"""验证在线 PEFT 更新不会因数值噪声造成无意义接受。"""


def test_peft_guard_requires_configured_reward_improvement():
    import compare

    assert compare._accept_online_peft_update(0.40, 0.39, 0.005) is True
    assert compare._accept_online_peft_update(0.40, 0.396, 0.005) is False
    assert compare._accept_online_peft_update(0.40, 0.40, 0.005) is False


def test_previous_online_update_guard_detects_cross_frame_reward_regression():
    import compare

    assert compare._previous_update_is_harmful(0.51, 0.50, 1e-5) is True
    assert compare._previous_update_is_harmful(0.500005, 0.50, 1e-5) is False
