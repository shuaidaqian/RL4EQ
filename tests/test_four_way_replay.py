"""验证四路在线消融的定义边界。"""

import json
from pathlib import Path


def test_four_way_replay_definitions_are_disjoint_and_label_free():
    from scripts.replay_four_way import build_four_way_specs

    config = json.loads(Path("configs/eme_long_memory_v2.json").read_text(encoding="utf-8"))
    specs = build_four_way_specs(config)

    assert tuple(spec.name for spec in specs) == (
        "Frozen Offline NN",
        "Pilot State Only",
        "PEFT Only",
        "Joint State + PEFT",
    )
    assert specs[0].source_method == "Frozen Offline NN"
    assert specs[1].source_method == "Pilot CIR only"
    assert specs[2].source_method == "Pilot-Driven Online Adaptation"
    assert specs[3].source_method == "Pilot-Driven Online Adaptation"
    assert specs[0].cir_update == "fixed"
    assert specs[1].cir_update == "pilot_sparse"
    assert specs[2].cir_update == "fixed"
    assert specs[3].cir_update == "pilot_sparse"
    assert specs[0].config["online_adaptation_algorithm"] == "sgd"
    assert specs[2].config["online_adaptation_groups"]
    assert specs[3].config["online_adaptation_groups"]
    assert specs[0].config["online_adaptation_groups"] == []
    assert all(spec.scheduler == "fixed" for spec in specs)


def test_four_way_replay_keeps_data_label_boundary():
    from scripts.replay_four_way import build_four_way_specs

    specs = build_four_way_specs({"online_adaptation_algorithm": "rls"})
    assert all(spec.config["online_adaptation_algorithm"] == "sgd" for spec in specs)
    assert all(spec.config["data_labels_used_online"] is False for spec in specs)
