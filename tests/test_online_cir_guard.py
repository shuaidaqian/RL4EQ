"""验证在线 Pilot CIR 更新也受留出 Reward Pilot 保护。"""


def test_pilot_cir_guard_accepts_only_non_degrading_reward_loss():
    import compare

    assert compare._accept_pilot_cir_update(0.40, 0.39) is True
    assert compare._accept_pilot_cir_update(0.40, 0.40) is True
    assert compare._accept_pilot_cir_update(0.40, 0.41) is False
