from scripts.diagnose_action_delay import summarize_action_delay


def test_action_delay_summary_reports_multiple_horizons_and_delayed_effect():
    """动作延迟诊断应按 seed/配置对齐未来帧，而不是读取 Data 标签。"""

    rows = []
    for frame in range(1, 7):
        rows.append(
            {
                "action": "phase_weak" if frame == 1 else "skip",
                "seed": 0,
                "delay": 116,
                "snr_db": 10.0,
                "frame": frame,
                "bandit_reward": 0.1 if frame == 1 else (0.3 if frame == 2 else 0.0),
                "data_labels_used_online": False,
            }
        )

    report = summarize_action_delay(rows, horizons=(0, 1, 2, 4))

    assert report["horizons"] == [0, 1, 2, 4]
    assert report["by_action"]["phase_weak"]["0"]["count"] == 1
    assert report["by_action"]["phase_weak"]["1"]["mean_reward"] == 0.3
    assert report["delayed_effect_detected"] is True
    assert report["diagnostic_uses_data_labels"] is False
