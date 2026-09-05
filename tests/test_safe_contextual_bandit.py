import math

import pytest

from agent.safe_contextual_bandit import SafeContextualBandit


def test_bandit_exposes_only_discrete_safe_update_actions():
    bandit = SafeContextualBandit(seed=7)

    names = {action.name for action in bandit.actions}
    assert names == {
        "skip",
        "phase_weak",
        "head_weak",
        "head_nominal",
        "film_nominal",
        "joint_nominal",
    }
    assert all(action.hold_frames >= 1 for action in bandit.actions)
    assert all(action.learning_rate_scale >= 0.0 for action in bandit.actions)


def test_bandit_uses_safe_shield_when_pilot_confidence_is_low():
    bandit = SafeContextualBandit(seed=11)

    action = bandit.select({"adapt_loss": 1.0, "pilot_confidence": 0.05})

    assert action.name in {"skip", "phase_weak"}
    assert action.name != "joint_nominal"


def test_bandit_update_changes_action_statistics_without_data_labels():
    bandit = SafeContextualBandit(seed=13)
    context = {"adapt_loss": 0.7, "pilot_confidence": 0.9}
    before = bandit.statistics("head_weak")

    bandit.update("head_weak", context, reward=0.4, accepted=True)
    after = bandit.statistics("head_weak")

    assert after["count"] == before["count"] + 1
    assert after["reward_sum"] == pytest.approx(before["reward_sum"] + 0.4)
    assert after["accepted_count"] == before["accepted_count"] + 1
    assert bandit.data_labels_used_online is False
    assert all(math.isfinite(value) for value in after.values())


def test_bandit_rejects_unknown_or_nonfinite_feedback():
    bandit = SafeContextualBandit(seed=17)
    context = {"adapt_loss": 0.7, "pilot_confidence": 0.9}

    with pytest.raises(ValueError):
        bandit.update("unknown", context, reward=0.1, accepted=True)
    with pytest.raises(ValueError):
        bandit.update("head_weak", context, reward=float("nan"), accepted=True)


def test_bandit_default_context_covers_channel_and_update_state():
    """Bandit 上下文必须能区分相位、CIR 漂移和历史拒绝状态。"""

    bandit = SafeContextualBandit(seed=19)

    assert set(bandit.feature_names) >= {
        "adapt_loss",
        "pilot_confidence",
        "residual_cfo",
        "phase_slope",
        "cir_drift",
        "snr_db",
        "consecutive_rejections",
        "parameter_delta_norm",
    }


def test_bandit_rejects_data_label_feedback_argument():
    """Bandit 更新接口不能接受 Data 标签作为动作反馈。"""

    bandit = SafeContextualBandit(seed=23)

    with pytest.raises(TypeError):
        bandit.update(
            "skip",
            {"pilot_confidence": 1.0},
            reward=0.0,
            accepted=True,
            data_ber=0.0,
        )
