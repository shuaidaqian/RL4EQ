# -*- coding: utf-8 -*-
"""实验配置到物理信道配置的端到端契约测试。"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from env.experiment_config import (
    build_comm_env_config,
    effective_channel_metadata,
    validate_model_dimensions,
)


CONFIG_PATH = Path("configs/continual_ppo_eme_measurement_v1.json")


def _eme_experiment() -> dict:
    experiment = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    experiment["eme_physical_fields_passthrough"] = "implemented"
    return experiment


def _write_eme_experiment(tmp_path, name="eme.json"):
    path = tmp_path / name
    path.write_text(json.dumps(_eme_experiment(), ensure_ascii=False), encoding="utf-8")
    return path


def test_eme_experiment_transmits_all_frozen_physical_fields():
    experiment = _eme_experiment()

    config = build_comm_env_config(
        experiment,
        level="B",
        snr_db=10.0,
        seed=123,
        max_delay=24,
        total_pilot=128,
        pilot_layout="prefix",
    )

    assert config.profile_name == "eme_measurement_v1"
    assert config.level == "B"
    assert config.max_delay == 24
    assert config.sample_rate_hz == pytest.approx(2000.0)
    assert config.symbol_rate_hz == pytest.approx(2000.0)
    assert config.frame_len == 512
    assert config.max_delay_seconds == pytest.approx(0.0116)
    assert config.coherence_time_seconds == pytest.approx(120.0)
    assert config.rho == pytest.approx(0.997868940604912)
    assert config.strong_path_count == (3, 7)
    assert config.diffuse_energy_ratio == pytest.approx((0.05, 0.15))
    assert config.include_anomalous_scatterer is False


def test_checkpoint_keeps_dimensions_but_accepts_runtime_physics_override(tmp_path):
    """旧权重可复用新的 CG 推理迭代，不应被 checkpoint 的旧运行参数锁死。"""

    from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
    from compare import _load_model_config

    model = UnfoldedEqualizer(
        UnfoldedConfig(
            frame_len=32,
            max_delay=4,
            iterations=1,
            d_model=16,
            num_heads=4,
            adapter_rank=4,
            lora_rank=4,
            physics_warm_start_iterations=2,
        )
    )
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model_config.json").write_text(
        json.dumps(model.config.to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )

    config = {
        "model": {
            "frame_len": 32,
            "max_delay": 4,
            "physics_warm_start_iterations": 8,
            "physics_warm_start_scale": 1.0,
            "analytic_logit_skip_scale": 0.0,
            "neural_residual_scale": 0.1,
        }
    }

    loaded = _load_model_config(config, checkpoint_dir / "model_best.pt")

    assert loaded.frame_len == 32
    assert loaded.max_delay == 4
    assert loaded.physics_warm_start_iterations == 8


def test_eme_config_forwards_explicit_reward_pilot_total():
    experiment = _eme_experiment()
    experiment["reward_pilot_total"] = 48

    config = build_comm_env_config(
        experiment,
        level="B",
        snr_db=10.0,
        seed=7,
    )
    metadata = effective_channel_metadata(config)

    assert config.reward_pilot_total == 48
    assert metadata["reward_pilot_total"] == 48
    assert metadata["adapt_pilot_total"] == 80


def test_eme_config_reward_pilot_override_takes_precedence():
    experiment = _eme_experiment()
    experiment["reward_pilot_total"] = 32

    config = build_comm_env_config(
        experiment,
        level="B",
        snr_db=10.0,
        seed=7,
        reward_pilot_total=64,
    )

    assert config.reward_pilot_total == 64
    assert config.layout == "prefix"
    assert config.total_pilot == 128
    assert config.impairment_profile == "cfo_phase_tiny"


def test_eme_experiment_rejects_unimplemented_passthrough_marker():
    experiment = _eme_experiment()
    experiment["eme_physical_fields_passthrough"] = "not_implemented"

    with pytest.raises(ValueError, match="eme_physical_fields_passthrough.*implemented"):
        build_comm_env_config(experiment, level="B", snr_db=10.0, seed=1)


@pytest.mark.parametrize(
    "field",
    [
        "profile_name",
        "sample_rate_hz",
        "symbol_rate_hz",
        "frame_len",
        "max_delay_seconds",
        "coherence_time_seconds",
        "strong_path_count",
        "diffuse_energy_ratio",
        "include_anomalous_scatterer",
    ],
)
def test_eme_experiment_rejects_missing_frozen_field(field):
    experiment = _eme_experiment()
    del experiment[field]

    with pytest.raises(ValueError, match=field):
        build_comm_env_config(experiment, level="B", snr_db=10.0, seed=1)


def test_eme_experiment_rejects_delay_override_conflicting_with_physical_depth():
    experiment = _eme_experiment()

    with pytest.raises(ValueError, match="max_delay.*24.*40"):
        build_comm_env_config(
            experiment,
            level="B",
            snr_db=10.0,
            seed=1,
            max_delay=40,
        )


def test_eme_experiment_rejects_non_prefix_pilot_override():
    experiment = _eme_experiment()

    with pytest.raises(ValueError, match="prefix"):
        build_comm_env_config(
            experiment,
            level="B",
            snr_db=10.0,
            seed=1,
            pilot_layout="two_block",
        )


def test_eme_experiment_accepts_only_frozen_pilot_sweep_candidates():
    experiment = _eme_experiment()

    config = build_comm_env_config(
        experiment,
        level="B",
        snr_db=10.0,
        seed=1,
        total_pilot=64,
    )
    assert config.total_pilot == 64

    with pytest.raises(ValueError, match="total_pilot.*32.*64.*96.*128.*160"):
        build_comm_env_config(
            experiment,
            level="B",
            snr_db=10.0,
            seed=1,
            total_pilot=32,
        )


def test_eme_experiment_rejects_model_dimensions_conflicting_with_channel():
    experiment = _eme_experiment()
    experiment["model"]["max_delay"] = 40

    with pytest.raises(ValueError, match="model.max_delay.*40.*24"):
        build_comm_env_config(experiment, level="B", snr_db=10.0, seed=1)


def test_eme_experiment_rejects_level_c_in_main_frozen_config():
    experiment = _eme_experiment()

    with pytest.raises(ValueError, match="level.*B.*C"):
        build_comm_env_config(experiment, level="C", snr_db=10.0, seed=1)


def test_eme_experiment_rejects_snr_outside_main_matrix():
    experiment = _eme_experiment()

    with pytest.raises(ValueError, match="snr_db.*20.*0.*5.*10.*15"):
        build_comm_env_config(experiment, level="B", snr_db=20.0, seed=1)


def test_eme_experiment_rejects_impairment_override_of_frozen_main_profile():
    experiment = _eme_experiment()

    with pytest.raises(ValueError, match="impairment_profile.*cfo_phase_tiny.*clean"):
        build_comm_env_config(
            experiment,
            level="B",
            snr_db=10.0,
            seed=1,
            impairment_profile="clean",
        )


def test_explicit_eme_profile_name_cannot_fall_back_when_channel_profile_disagrees():
    experiment = _eme_experiment()
    experiment["channel_profile"] = "legacy_sparse_v1"

    with pytest.raises(ValueError, match="channel_profile.*profile_name"):
        build_comm_env_config(experiment, level="B", snr_db=10.0, seed=1)


def test_explicit_unknown_channel_profile_is_rejected():
    experiment = {"channel_profile": "unknown_profile", "profile_name": "unknown_profile"}

    with pytest.raises(ValueError, match="unknown_profile.*不支持"):
        build_comm_env_config(experiment, level="B", snr_db=10.0, seed=1)


def test_effective_metadata_is_derived_from_built_config():
    config = build_comm_env_config(
        _eme_experiment(),
        level="B",
        snr_db=5.0,
        seed=9,
    )

    metadata = effective_channel_metadata(config)

    assert metadata == {
        "profile_name": "eme_measurement_v1",
        "level": "B",
        "max_delay": 24,
        "sample_rate_hz": 2000.0,
        "symbol_rate_hz": 2000.0,
        "frame_len": 512,
        "max_delay_seconds": 0.0116,
        "coherence_time_seconds": 120.0,
        "acquisition_to_data_gap_seconds": 0.0,
        "rho_frame": pytest.approx(0.997868940604912),
        "strong_path_count": [3, 7],
        "diffuse_energy_ratio": pytest.approx([0.05, 0.15]),
        "include_anomalous_scatterer": False,
        "impairment_profile": "cfo_phase_tiny",
            "pilot_layout": "prefix",
            "pilot_total": 128,
            "reward_pilot_total": 32,
            "adapt_pilot_total": 96,
            "state_split": None,
        }


def test_legacy_experiment_keeps_existing_defaults_and_overrides():
    experiment = {
        "rho": 0.99,
        "pilot_total": 96,
        "pilot_layout": "multi_block",
        "impairment_profile": "clean",
    }

    config = build_comm_env_config(
        experiment,
        level="A",
        snr_db=15.0,
        seed=7,
        max_delay=30,
        pilot_layout="two_block",
    )

    assert config.profile_name == "legacy_sparse_v1"
    assert config.max_delay == 30
    assert config.rho == pytest.approx(0.99)
    assert config.total_pilot == 96
    assert config.layout == "two_block"
    assert config.frame_len == 512


def test_validation_does_not_mutate_experiment_mapping():
    experiment = _eme_experiment()
    before = copy.deepcopy(experiment)

    build_comm_env_config(experiment, level="B", snr_db=10.0, seed=1)

    assert experiment == before


@pytest.mark.parametrize(
    ("model_config", "message"),
    [
        ({"frame_len": 512, "max_delay": 40}, "max_delay.*40.*24"),
        ({"frame_len": 256, "max_delay": 24}, "frame_len.*256.*512"),
    ],
)
def test_actual_checkpoint_model_dimensions_must_match_built_environment(model_config, message):
    env_config = build_comm_env_config(
        _eme_experiment(),
        level="B",
        snr_db=10.0,
        seed=1,
    )

    with pytest.raises(ValueError, match=message):
        validate_model_dimensions(model_config, env_config)


def test_actual_checkpoint_model_dimensions_accept_frozen_environment():
    env_config = build_comm_env_config(
        _eme_experiment(),
        level="B",
        snr_db=10.0,
        seed=1,
    )

    validate_model_dimensions({"frame_len": 512, "max_delay": 24}, env_config)


def test_online_runner_rejects_conflicting_checkpoint_model_before_loading_weights(tmp_path):
    from training.windowed_discrete_ppo import run_windowed_discrete_online

    experiment_path = tmp_path / "eme.json"
    experiment_path.write_text(
        json.dumps(_eme_experiment(), ensure_ascii=False),
        encoding="utf-8",
    )
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model_config.json").write_text(
        json.dumps({"frame_len": 512, "max_delay": 40}),
        encoding="utf-8",
    )
    checkpoint = checkpoint_dir / "model_best.pt"
    checkpoint.write_bytes(b"must not be loaded")

    with pytest.raises(ValueError, match="max_delay.*40.*24"):
        run_windowed_discrete_online(
            config_path=experiment_path,
            frames=1,
            num_seeds=1,
            output_dir=tmp_path / "online",
            delays=[24],
            snrs=[10],
            pilot_total=128,
            pilot_layout="prefix",
            pretrained=checkpoint,
            device="cpu",
        )


def test_pretrain_meta_stage_fails_fast_for_frozen_eme_until_physically_connected(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "pretrain.py",
            "--config",
            str(CONFIG_PATH),
            "--stage",
            "meta",
            "--steps",
            "1",
            "--batch-size",
            "1",
            "--save-dir",
            str(tmp_path / "meta"),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "meta" in result.stderr
    assert "eme_measurement_v1" in result.stderr


def test_pretrain_online_meta_stage_is_removed_from_main_route(tmp_path):
    """主研究路线不再保留独立的 online_meta 第二层入口。"""

    result = subprocess.run(
        [
            sys.executable,
            "pretrain.py",
            "--config",
            "configs/continual_ppo.json",
            "--stage",
            "online_meta",
            "--steps",
            "1",
            "--batch-size",
            "1",
            "--save-dir",
            str(tmp_path / "removed_online_meta"),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "online_meta" in result.stderr
    assert "已移除" in result.stderr


def test_compare_profile_prior_uses_frozen_top_level_residual_cfo_limit():
    from compare import _profile_prior_from_config

    prior = _profile_prior_from_config(_eme_experiment(), "cfo_phase_tiny")

    assert prior["residual_cfo_limit"] == pytest.approx(0.0012)


def test_frozen_eme_curriculum_contains_only_level_b_physical_grid():
    from training.curriculum import build_curriculum

    phases = build_curriculum(_eme_experiment())

    assert len(phases) == 1
    phase = phases[0]
    assert phase.name == "estimated_cir_level_b"
    assert phase.level == "B"
    assert phase.delay_grid == (24,)
    assert phase.snr_grid == (0.0, 5.0, 10.0, 15.0)
    assert phase.layout == "prefix"
    assert phase.total_pilot == 128


def test_curriculum_trainer_has_no_direct_environment_config_bypass():
    import inspect

    from training.curriculum import CurriculumTrainer

    source = inspect.getsource(CurriculumTrainer)
    assert "CommEnvConfig(" not in source
    assert source.count("build_comm_env_config(") >= 3


def test_curriculum_training_metrics_record_effective_eme_channel(monkeypatch):
    from training.curriculum import CurriculumTrainer

    trainer = CurriculumTrainer(_eme_experiment(), torch.device("cpu"))
    parameter = next(trainer.model.parameters())
    monkeypatch.setattr(
        trainer,
        "_step_loss",
        lambda phase, batch_size: parameter.sum() * 0.0,
    )
    validation = {"mean_ber_data": 0.1, "gate_pass": True, "per_config": []}
    offline = {"mean_ber_data": 0.1, "gate_pass": True, "rows": []}
    monkeypatch.setattr(trainer, "_validate_level_b", lambda: validation)
    monkeypatch.setattr(trainer, "_validate_offline_nn", lambda: offline)

    metrics = trainer.train(
        stage="all",
        steps=1,
        batch_size=1,
        accumulation_steps=1,
        use_amp=False,
    )

    assert metrics["effective_channel"]["profile_name"] == "eme_measurement_v1"
    assert metrics["effective_channel"]["max_delay"] == 24
    assert metrics["effective_channel"]["coherence_time_seconds"] == pytest.approx(120.0)


def test_windowed_online_records_effective_eme_channel(tmp_path):
    from training.windowed_discrete_ppo import run_windowed_discrete_online

    result = run_windowed_discrete_online(
        config_path=_write_eme_experiment(tmp_path),
        frames=1,
        num_seeds=1,
        output_dir=tmp_path / "online",
        delays=[24],
        snrs=[10],
        pilot_total=128,
        pilot_layout="prefix",
        window_size=1,
        update_interval=1,
        device="cpu",
    )

    assert result["effective_channel"]["profile_name"] == "eme_measurement_v1"
    assert result["effective_channel"]["max_delay"] == 24
    assert {row["profile_name"] for row in result["rows"]} == {"eme_measurement_v1"}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"delays": [40], "pilot_layout": "prefix", "pilot_total": 128}, "max_delay"),
        ({"delays": [24], "pilot_layout": "two_block", "pilot_total": 128}, "prefix"),
        ({"delays": [24], "pilot_layout": "prefix", "pilot_total": 32}, "total_pilot"),
    ],
)
def test_windowed_online_rejects_frozen_eme_conflicts(tmp_path, kwargs, message):
    from training.windowed_discrete_ppo import run_windowed_discrete_online

    with pytest.raises(ValueError, match=message):
        run_windowed_discrete_online(
            config_path=_write_eme_experiment(tmp_path),
            frames=1,
            num_seeds=1,
            output_dir=tmp_path / "online_conflict",
            snrs=[10],
            device="cpu",
            **kwargs,
        )


def _compare_command(config_path, output_dir):
    return [
        sys.executable,
        "compare.py",
        "--config",
        str(config_path),
        "--methods",
        "LMMSE-FIR",
        "--delays",
        "24",
        "--snrs",
        "10",
        "--num-seeds",
        "1",
        "--frames",
        "1",
        "--pilot-total",
        "128",
        "--pilot-layout",
        "prefix",
        "--output-dir",
        str(output_dir),
        "--device",
        "cpu",
    ]


def test_compare_records_effective_eme_channel_from_built_environment(tmp_path):
    output_dir = tmp_path / "compare_eme"
    subprocess.run(
        _compare_command(_write_eme_experiment(tmp_path), output_dir),
        check=True,
        text=True,
        capture_output=True,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (output_dir / "frame_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert summary["effective_channel"]["profile_name"] == "eme_measurement_v1"
    assert summary["effective_channel"]["max_delay"] == 24
    assert summary["profile_prior"]["residual_cfo_limit"] == pytest.approx(0.0012)
    assert {row["profile_name"] for row in rows} == {"eme_measurement_v1"}


def test_compare_records_reward_pilot_override(tmp_path):
    output_dir = tmp_path / "compare_reward_split"
    command = _compare_command(_write_eme_experiment(tmp_path), output_dir)
    command.extend(["--reward-pilot-total", "48"])

    subprocess.run(command, check=True, text=True, capture_output=True)

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["effective_channel"]["reward_pilot_total"] == 48
    assert summary["effective_channel"]["adapt_pilot_total"] == 80


def test_compare_resume_key_distinguishes_reward_pilot_split():
    import compare

    base = {
        "method": "Pilot-Driven Online Adaptation",
        "delay": 116,
        "snr_db": 10.0,
        "seed": 0,
        "frame": 1,
        "pilot_total": 128,
        "pilot_layout": "prefix",
        "impairment_profile": "cfo_phase_tiny",
    }

    split_32 = {**base, "reward_pilot_total": 32}
    split_64 = {**base, "reward_pilot_total": 64}

    assert compare._row_key(split_32) != compare._row_key(split_64)


def test_compare_rejects_conflicting_checkpoint_model_before_loading_weights(tmp_path):
    checkpoint_dir = tmp_path / "compare_checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model_config.json").write_text(
        json.dumps({"frame_len": 512, "max_delay": 40}),
        encoding="utf-8",
    )
    checkpoint = checkpoint_dir / "model_best.pt"
    checkpoint.write_bytes(b"must not be loaded")
    command = _compare_command(
        _write_eme_experiment(tmp_path),
        tmp_path / "compare_checkpoint_conflict",
    )
    command[command.index("LMMSE-FIR")] = "Frozen Offline NN"
    command.extend(["--pretrained", str(checkpoint)])

    result = subprocess.run(command, check=False, text=True, capture_output=True)

    assert result.returncode != 0
    assert "max_delay" in result.stderr
    assert "40" in result.stderr
    assert "24" in result.stderr


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"--delays": "40"}, "max_delay"),
        ({"--pilot-layout": "two_block"}, "prefix"),
        ({"--pilot-total": "32"}, "total_pilot"),
        ({"--snrs": "20"}, "snr_db"),
        ({"--impairment-profile": "clean"}, "impairment_profile"),
    ],
)
def test_compare_rejects_frozen_eme_cli_conflicts(tmp_path, replacement, message):
    command = _compare_command(
        _write_eme_experiment(tmp_path),
        tmp_path / "compare_conflict",
    )
    for option, value in replacement.items():
        if option in command:
            command[command.index(option) + 1] = value
        else:
            command.extend([option, value])
    result = subprocess.run(command, check=False, text=True, capture_output=True)

    assert result.returncode != 0
    assert message in result.stderr


def test_compare_resume_rejects_rows_from_different_channel_profile(tmp_path):
    output_dir = tmp_path / "mixed_profile"
    legacy_command = _compare_command("configs/continual_ppo.json", output_dir)
    subprocess.run(legacy_command, check=True, text=True, capture_output=True)

    eme_command = _compare_command(
        _write_eme_experiment(tmp_path, "eme_resume.json"),
        output_dir,
    )
    eme_command.append("--resume")
    result = subprocess.run(eme_command, check=False, text=True, capture_output=True)

    assert result.returncode != 0
    assert "profile_name" in result.stderr
    assert "legacy_sparse_v1" in result.stderr
    assert "eme_measurement_v1" in result.stderr
