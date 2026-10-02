"""验证 A/B/C 统一小样本 replay 的方法边界。"""

from copy import deepcopy


def test_replay_specs_define_three_candidates_and_shared_protocol():
    from scripts.replay_abc import build_replay_specs

    base_config = {
        "model": {"neural_residual_scale": 0.1},
        "online_adaptation_algorithm": "rls",
    }
    specs = build_replay_specs(base_config)

    assert tuple(spec.name for spec in specs) == (
        "ABC-A Pilot State Gate",
        "ABC-B Pilot RLS Residual",
        "ABC-C Physics Unfolded",
    )
    assert specs[0].source_method == "Pilot CIR only"
    assert specs[1].source_method == "Pilot-Driven Online Adaptation"
    assert specs[2].source_method == "Pilot CIR only"
    assert specs[2].cir_update == "pilot_sparse"
    assert specs[1].config["online_adaptation_algorithm"] == "rls"
    assert specs[2].config["model"]["neural_residual_scale"] == 0.0
    assert all(spec.scheduler == "fixed" for spec in specs)


def test_replay_specs_can_select_only_a_and_c():
    from scripts.replay_abc import build_replay_specs

    specs = build_replay_specs({}, candidates=("A", "C"))

    assert tuple(spec.name for spec in specs) == (
        "ABC-A Pilot State Gate",
        "ABC-C Physics Unfolded",
    )
    assert all("ABC-B" not in spec.name for spec in specs)


def test_relabel_rows_preserves_pair_key_and_data_label_boundary():
    from scripts.replay_abc import replay_pair_key, relabel_rows

    row = {
        "method": "Pilot CIR only",
        "level": "B",
        "profile_name": "eme_long_memory_v2",
        "impairment_profile": "cfo_phase_tiny",
        "delay": 116,
        "snr_db": 10.0,
        "seed": 0,
        "frame": 1,
        "pilot_total": 256,
        "reward_pilot_total": 32,
        "pilot_layout": "prefix",
        "ber_data": 0.1,
        "data_labels_used_online": False,
    }
    original_key = replay_pair_key(row)
    relabeled = relabel_rows([deepcopy(row)], "ABC-A Pilot State Gate", "Pilot CIR only")

    assert relabeled[0]["method"] == "ABC-A Pilot State Gate"
    assert relabeled[0]["source_method"] == "Pilot CIR only"
    assert relabeled[0]["data_labels_used_online"] is False
    assert replay_pair_key(relabeled[0]) == original_key
