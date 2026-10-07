# -*- coding: utf-8 -*-

from scripts.diagnose_pilot_objective_ranking import _event_rows, _summarize


def test_objective_ranking_is_pilot_only_and_keeps_future_data_diagnostic_only():
    rows = []
    for frame in range(1, 5):
        rows.append(
            {
                "method": "Pilot-Driven Online Adaptation",
                "delay": 116,
                "snr_db": 10.0,
                "pilot_total": 256,
                "reward_pilot_total": 32,
                "pilot_layout": "prefix",
                "seed": 0,
                "frame": frame,
                "online_update_scheduled": True,
                "reward_pilot_loss_before": 0.5,
                "reward_pilot_loss_after": 0.4,
                "reward_pilot_window_gains": [0.1],
                "pilot_phase_confidence": 1.0,
                "ber_data": 0.4 - 0.05 * frame,
            }
        )
    events = _event_rows(rows, 2)
    assert events
    assert all(event["data_labels_used_online"] is False for event in events)
    assert all(event["diagnostic_uses_data_labels"] is True for event in events)
    summary = _summarize(events, "single_frame_bce")
    assert summary["events"] == len(events)

