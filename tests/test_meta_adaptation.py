import json
import subprocess
import sys

import torch

from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from training.curriculum import build_curriculum, load_config


def test_curriculum_is_always_pilot_conditioned_and_level_ordered():
    schedule = build_curriculum(load_config("configs/continual_ppo.json"))
    assert [phase.name for phase in schedule] == [
        "cir_level_a",
        "perfect_cir_level_a",
        "estimated_cir_level_a",
        "estimated_cir_level_b",
    ]
    assert all(phase.total_pilot in {64, 96, 128, 160} for phase in schedule)
    assert all(phase.layout in {"prefix", "two_block", "multi_block"} for phase in schedule)
    assert all(phase.uses_pilot_condition for phase in schedule)


def test_pretrain_smoke_writes_strict_loadable_checkpoint(tmp_path):
    save_dir = tmp_path / "pretrained"
    subprocess.run(
        [
            sys.executable,
            "pretrain.py",
            "--config",
            "configs/continual_ppo.json",
            "--stage",
            "all",
            "--steps",
            "2",
            "--batch-size",
            "1",
            "--accumulation-steps",
            "1",
            "--save-dir",
            str(save_dir),
        ],
        check=True,
    )
    payload = torch.load(save_dir / "model_best.pt", map_location="cpu", weights_only=False)
    config = UnfoldedConfig.from_dict(json.loads((save_dir / "model_config.json").read_text(encoding="utf-8")))
    model = UnfoldedEqualizer(config)
    model.load_state_dict(payload["state_dict"], strict=True)
    assert (save_dir / "pretrain_metrics.json").exists()
    assert payload["schema_version"] == "unfolded-eq-v1"
