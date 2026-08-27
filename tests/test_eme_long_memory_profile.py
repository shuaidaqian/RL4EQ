# -*- coding: utf-8 -*-
"""EME 长记忆主信道 profile 的契约测试。"""

import json
from pathlib import Path

from env.experiment_config import build_comm_env_config


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
