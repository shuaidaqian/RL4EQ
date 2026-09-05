from scripts.diagnose_action_delay import summarize_action_delay


def test_action_delay_summary_reports_multiple_horizons_and_delayed_effect():
    """动作延迟诊断应按 seed/配置对齐未来帧，而不是读取 Data 标签。"""

    rows = []
    for seed in (0, 1):
        for snr_db in (10.0, 15.0):
            for frame in range(1, 7):
                rows.append(
                    {
                        "action": "phase_weak" if frame == 1 else "skip",
                        "seed": seed,
                        "delay": 116,
                        "snr_db": snr_db,
                        "frame": frame,
                        "bandit_reward": 0.1 if frame == 1 else (0.3 if frame == 2 else 0.0),
                        "data_labels_used_online": False,
                    }
                )

    report = summarize_action_delay(rows, horizons=(0, 1, 2, 4))

    assert report["horizons"] == [0, 1, 2, 4]
    assert report["by_action"]["phase_weak"]["0"]["count"] == 4
    assert report["by_action"]["phase_weak"]["1"]["mean_reward"] == 0.3
    assert report["delayed_effect_detected"] is True
    assert report["diagnostic_uses_data_labels"] is False


def test_action_delay_summary_does_not_upgrade_from_one_seed_and_snr():
    """单条轨迹上的延迟峰值不能触发循环 Double DQN。"""

    rows = [
        {
            "action": "head_weak" if frame == 1 else "skip",
            "seed": 0,
            "delay": 116,
            "snr_db": 10.0,
            "frame": frame,
            "bandit_reward": 0.1 if frame == 1 else (0.5 if frame == 2 else 0.0),
            "data_labels_used_online": False,
        }
        for frame in range(1, 5)
    ]

    report = summarize_action_delay(rows)

    assert report["delayed_effect_detected"] is False
    assert report["recommended_controller"] == "contextual_bandit"


def test_action_delay_summary_ignores_non_online_comparison_rows():
    """主矩阵混有传统方法行时，诊断只应读取在线 Bandit 行。"""

    rows = [
        {
            "method": "CFO+DD-Phase DFE-RLS",
            "seed": 0,
            "delay": 116,
            "snr_db": 10.0,
            "frame": 1,
            "data_labels_used_online": False,
        },
        {
            "method": "Pilot-Driven Online Adaptation",
            "action": "phase_weak",
            "seed": 0,
            "delay": 116,
            "snr_db": 10.0,
            "pilot_total": 128,
            "pilot_layout": "prefix",
            "state_split": None,
            "frame": 1,
            "bandit_reward": 0.1,
            "update_applied": True,
            "adaptation_accepted": True,
            "data_labels_used_online": False,
        },
    ]

    report = summarize_action_delay(rows)

    assert report["by_action"]["phase_weak"]["0"]["count"] == 1
    assert report["diagnostic_uses_data_labels"] is False
