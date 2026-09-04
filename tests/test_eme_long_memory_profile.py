# -*- coding: utf-8 -*-
"""EME 长记忆主信道 profile 的契约测试。"""

import json
from pathlib import Path

import pytest

from env.experiment_config import build_comm_env_config
from env.experiment_config import validate_model_dimensions


def _load_config() -> dict:
    path = Path("configs/eme_long_memory_v2.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_eme_long_memory_profile_has_symbol_scale_long_memory():
    config = _load_config()
    env_config = build_comm_env_config(
        config,
        level="B",
        snr_db=10.0,
        seed=0,
        max_delay=116,
        total_pilot=128,
        pilot_layout="prefix",
    )

    assert env_config.profile_name == "eme_long_memory_v2"
    assert env_config.sample_rate_hz == 10_000.0
    assert env_config.symbol_rate_hz == 10_000.0
    assert env_config.max_delay == 116
    assert env_config.frame_len == 1024


def test_eme_long_memory_profile_is_not_the_24_tap_measurement_profile():
    config = _load_config()
    env_config = build_comm_env_config(
        config,
        level="B",
        snr_db=0.0,
        seed=1,
        max_delay=116,
        total_pilot=128,
        pilot_layout="prefix",
    )
    assert env_config.profile_name != "eme_measurement_v1"
    assert env_config.max_delay > 24


def test_eme_model_phase_conditioner_matches_four_block_features():
    config = _load_config()
    env_config = build_comm_env_config(
        config,
        level="B",
        snr_db=10.0,
        seed=2,
        max_delay=116,
        total_pilot=128,
        pilot_layout="prefix",
    )
    validate_model_dimensions(config["model"], env_config)

    invalid_model = dict(config["model"])
    invalid_model["phase_correction_segments"] = 8
    with pytest.raises(ValueError, match="4 个 block"):
        validate_model_dimensions(invalid_model, env_config)


def test_eme_main_configuration_matches_slow_variation_assumption():
    """主场景不能用远大于相干时间的 acquisition 空档制造强失配。"""

    config = _load_config()
    assert config["acquisition_to_data_gap_seconds"] == 0.0
    assert config["acquisition_to_data_gap_seconds"] < config["frame_duration_seconds"]
    assert config["coherence_time_seconds"] >= 600.0


def test_eme_main_configuration_uses_converged_physics_warm_start():
    """长记忆主场景应给展开式检测器足够的 CG 初值求解迭代。"""

    config = _load_config()
    assert config["model"]["physics_warm_start_iterations"] >= 8
