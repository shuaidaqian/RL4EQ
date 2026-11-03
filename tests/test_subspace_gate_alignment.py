import pytest

from scripts.diagnose_subspace_gate_alignment import analyze


def _row(*, snr=0, seed=1, frame=0, accepted=True, gain=0.1, data_gain=0.2):
    return {
        "role": "evaluation",
        "training": "B_post_adapt",
        "snr_db": snr,
        "seed": seed,
        "frame": frame,
        "candidate_accepted": accepted,
        "reset_to_origin": False,
        "validation_loss_before": 1.0,
        "validation_loss_candidate": 1.0 - gain,
        "validation_loss_retained": 1.0 - gain,
        "frozen_data_loss": 1.0,
        "online_data_loss": 1.0 - data_gain,
        "ber_frozen": 0.2,
        "ber_online": 0.2 - data_gain,
        "data_labels_used_for_control": False,
    }


def test_gate_alignment_groups_only_post_adapt_evaluation_by_seed():
    rows = [
        _row(frame=0, gain=0.1, data_gain=0.2),
        _row(frame=1, gain=0.2, data_gain=0.1),
        {**_row(), "training": "A_supervised"},
    ]

    result = analyze(rows)

    assert result["snrs_db"] == [0]
    assert result["by_snr"][0]["independent_seed_sessions"] == 1
    assert result["sessions"][0]["frames"] == 2
    assert result["sessions"][0]["candidate_accepted"] == 2
    assert result["online_control_uses_data_labels"] is False


def test_gate_alignment_rejects_data_label_control_leakage():
    with pytest.raises(ValueError, match="Data标签"):
        analyze([{**_row(), "data_labels_used_for_control": True}])
