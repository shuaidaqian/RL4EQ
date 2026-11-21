# -*- coding: utf-8 -*-
"""冻结 AR(1) 动态统计的来源与参数校验。"""

import hashlib
import json

import pytest

from evaluation.channel_dynamics import load_ar1_dynamics


def _write(tmp_path, payload):
    path = tmp_path / 'dynamics.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    return path


def test_load_ar1_dynamics_verifies_hash_and_tap_count(tmp_path):
    payload = {'rho_estimate': 0.9, 'stationary_power_by_tap': [1.0, 0.25]}
    path = _write(tmp_path, payload)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = load_ar1_dynamics(path, expected_sha256=digest, tap_count=2)
    assert loaded['rho_estimate'] == 0.9
    assert loaded['stationary_power_by_tap'] == [1.0, 0.25]
    assert loaded['sha256'] == digest
    with pytest.raises(ValueError, match='指纹'):
        load_ar1_dynamics(path, expected_sha256='0' * 64)
    with pytest.raises(ValueError, match='维度'):
        load_ar1_dynamics(path, tap_count=3)


@pytest.mark.parametrize('payload', [
    {'rho_estimate': 1.0, 'stationary_power_by_tap': [1.0]},
    {'rho_estimate': 0.9, 'stationary_power_by_tap': [-1.0]},
    {'rho_estimate': float('nan'), 'stationary_power_by_tap': [1.0]},
    {'rho_estimate': 0.9, 'stationary_power_by_tap': []},
])
def test_load_ar1_dynamics_rejects_invalid_parameters(tmp_path, payload):
    path = _write(tmp_path, payload)
    with pytest.raises(ValueError):
        load_ar1_dynamics(path)
