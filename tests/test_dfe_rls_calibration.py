# -*- coding: utf-8 -*-
"""DFE-RLS 的可调先验与遗忘因子应符合独立加权岭回归解。"""

import pytest
import torch
import hashlib
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

from baseline.traditional_equalizers import _dfe_rls
from scripts import calibrate_dfe_rls as calibration


@pytest.mark.parametrize("forgetting", [0.98, 0.995, 1.0])
@pytest.mark.parametrize("covariance", [0.001, 0.1, 100.0])
def test_dfe_rls_matches_independent_weighted_batch_solution(forgetting, covariance):
    generator = torch.Generator().manual_seed(414)
    rx = torch.complex(torch.randn(128, generator=generator), torch.randn(128, generator=generator))
    tx = (2 * torch.randint(0, 2, (128,), generator=generator) - 1).to(torch.complex64)
    mask = torch.arange(128) < 80
    estimate, updates = _dfe_rls(rx, mask, torch.where(mask, tx, 0), 5, feedback_len=3,
                                forgetting_factor=forgetting, initial_inverse_covariance=covariance)
    # 独立构造居中前馈与训练时的已知反馈，不调用实现中的特征函数。
    features = []
    for index in range(128):
        features.append(torch.stack([rx[index + offset] if 0 <= index + offset < 128 else rx.new_zeros(())
                                     for offset in range(-2, 3)]))
    design = torch.stack([torch.cat((features[index], torch.stack([
        tx[index - delay] if index >= delay else tx.new_zeros(()) for delay in range(1, 4)
    ]))) for index in range(80)]).to(torch.complex128)
    weights = forgetting ** torch.arange(79, -1, -1, dtype=torch.float64)
    precision = forgetting ** 80 / covariance
    prior = torch.zeros(8, dtype=torch.complex128)
    prior[2] = 1
    normal = design.conj().T @ (weights[:, None] * design) + precision * torch.eye(8, dtype=torch.complex128)
    rhs = design.conj().T @ (weights * tx[:80].to(torch.complex128)) + precision * prior
    coefficients = torch.linalg.solve(normal, rhs)
    expected = torch.zeros(128, dtype=torch.complex128)
    for index in range(128):
        feedback = torch.stack([expected[index - delay] if index >= delay else expected.new_zeros(())
                                for delay in range(1, 4)])
        prediction = torch.dot(torch.cat((features[index].to(torch.complex128), feedback)), coefficients)
        expected[index] = 1 if prediction.real >= 0 else -1
    assert updates == 80
    torch.testing.assert_close(estimate, expected.to(torch.complex64), rtol=0, atol=0)


@pytest.mark.parametrize("settings", [
    {"forgetting_factor": 0}, {"forgetting_factor": 1.1}, {"forgetting_factor": float("nan")},
    {"initial_inverse_covariance": 0}, {"initial_inverse_covariance": -1},
    {"initial_inverse_covariance": float("inf")},
])
def test_dfe_rls_rejects_invalid_priors_before_processing(settings):
    with pytest.raises(ValueError):
        _dfe_rls(torch.ones(8, dtype=torch.complex64), torch.ones(8, dtype=torch.bool),
                 torch.ones(8, dtype=torch.complex64), 3, **settings)


@pytest.mark.parametrize("violation", ["config", "overlap", "budget", "grid"])
def test_dfe_frozen_selection_guards_confirmation_scope(violation):
    config = {"main_snrs": [10], "max_delay": 116}
    selection = {"schema_version": "dfe-rls-frozen-selection-v1",
                 "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                 "calibration_seeds": [7], "confirmation_seeds": [8], "confirmation_frames": 2,
                 "by_snr": {"10": asdict(calibration.REFERENCE)}}
    calibration.validate_selection(selection, config, [8], 2)
    if violation == "config":
        config["max_delay"] = 20
    elif violation == "overlap":
        selection["calibration_seeds"] = [8]
    elif violation == "budget":
        selection["confirmation_frames"] = 3
    else:
        selection["by_snr"]["10"]["initial_inverse_covariance"] = 1000
    with pytest.raises(ValueError):
        calibration.validate_selection(selection, config, [8], 2)


def test_dfe_calibration_freezes_training_choice_and_only_confirms_fixed_settings(monkeypatch, tmp_path):
    from env.frame_structure import FrameConfig, FrameGenerator

    generated = FrameGenerator(FrameConfig(frame_len=128, total_pilot=64, max_delay=4), seed=21).generate(0)
    frame = replace(generated, rx_symbols=generated.tx_symbols)
    calls = []

    def estimate(rx, mask, known, length, *, feedback_len, forgetting_factor, initial_inverse_covariance):
        assert not bool(known[~mask].any())
        calls.append(calibration.DFESetting(length, feedback_len, forgetting_factor, initial_inverse_covariance))
        return rx.clone(), int(mask.sum())

    environment = SimpleNamespace(
        reset_episode=lambda: SimpleNamespace(acquisition=None, initial_soft_tail=torch.zeros(4, dtype=torch.complex64)),
        next_frame=lambda: frame,
    )
    monkeypatch.setattr(calibration, "CommunicationEnvironment", lambda *args: environment)
    monkeypatch.setattr(calibration, "build_comm_env_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(calibration, "estimate_acquisition_cir_with_cfo", lambda *args, **kwargs: (torch.ones(5, dtype=torch.complex64), 0))
    monkeypatch.setattr(calibration, "estimate_phase_residual_vector", lambda *args, **kwargs: torch.zeros(16))
    monkeypatch.setattr(calibration, "_dfe_rls", estimate)
    config = {"main_level": "B", "pilot_layout": "prefix", "max_delay": 4, "pilot_total": 64,
              "reward_pilot_total": 16, "main_snrs": [10]}
    frozen_path = tmp_path / "frozen.json"
    result = calibration.run_calibration(config, [7], 1, tmp_path / "calibration.json", freeze_output=frozen_path)
    assert len(result["rows"]) == 36
    assert set(calls) == set(calibration.calibration_grid(4))
    selection = json.loads(frozen_path.read_text(encoding="utf-8"))
    chosen = calibration.DFESetting(5, 4, 1.0, 0.001)
    assert selection["by_snr"]["10"] == asdict(chosen)
    calls.clear()
    confirmation = calibration.run_calibration(
        config, calibration.CONFIRMATION_SEEDS, calibration.CONFIRMATION_FRAMES,
        tmp_path / "confirmation.json", state_split="heldout_edge", selection=selection,
    )
    assert set(calls) == {calibration.REFERENCE, chosen}
    assert confirmation["frozen_selection"] == selection
    assert confirmation["per_snr"][0]["selected_vs_reference_ber_improvement"]["mean"] == 0
    assert all(not row["data_labels_used_for_receiver"] and not row["uses_neural_network"]
               and not row["uses_rl"] for row in confirmation["rows"])
