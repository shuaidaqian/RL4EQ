import json
import subprocess
import sys
from pathlib import Path

import torch


def test_eme_slow_drift_grid_builds_48_candidates_and_profile_aware_cfo_limits():
    from scripts.calibrate_eme_slow_drift_grid import (
        DEFAULT_CFO_ABS_VALUES,
        DEFAULT_PHASE_NOISE_STDS,
        DEFAULT_RHOS,
        build_candidates,
        profile_cfo_limit,
    )

    candidates = build_candidates(DEFAULT_CFO_ABS_VALUES, DEFAULT_PHASE_NOISE_STDS, DEFAULT_RHOS)

    assert len(candidates) == 48
    assert candidates[0]["candidate_id"] == "cfo0p0008_phase0p0008_rho0p995"
    assert profile_cfo_limit(0.0008) == 0.0012
    assert profile_cfo_limit(0.0030) == 0.0045
    assert profile_cfo_limit(0.0050) == 0.0075
    assert {candidate["shared_profile_prior"] for candidate in candidates} == {True}


def test_snr_layered_score_uses_best_traditional_by_snr_layer():
    from scripts.calibrate_eme_slow_drift_grid import score_candidate_summary

    summary = [
        {"method": "LMMSE-FIR", "snr_db": 0.0, "mean_ber_data": 0.30},
        {"method": "RLS Linear", "snr_db": 0.0, "mean_ber_data": 0.25},
        {"method": "LMMSE-FIR", "snr_db": 5.0, "mean_ber_data": 0.20},
        {"method": "RLS Linear", "snr_db": 5.0, "mean_ber_data": 0.18},
        {"method": "LMMSE-FIR", "snr_db": 10.0, "mean_ber_data": 0.12},
        {"method": "RLS Linear", "snr_db": 10.0, "mean_ber_data": 0.16},
        {"method": "LMMSE-FIR", "snr_db": 15.0, "mean_ber_data": 0.07},
        {"method": "RLS Linear", "snr_db": 15.0, "mean_ber_data": 0.09},
    ]

    score = score_candidate_summary(summary)

    assert score["passes_all_snr_layers"] is True
    assert score["best_by_snr"]["0.0"]["method"] == "RLS Linear"
    assert score["best_by_snr"]["5.0"]["mean_ber_data"] == 0.18
    assert score["best_by_snr"]["10.0"]["method"] == "LMMSE-FIR"
    assert score["best_by_snr"]["15.0"]["in_target_band"] is True
    assert score["score"] == 0.0


def test_eme_slow_drift_calibration_cli_smoke_writes_candidate_summary(tmp_path):
    output_dir = tmp_path / "eme_slow_drift"
    subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_eme_slow_drift_grid.py",
            "--cfo-abs-values",
            "0.0008",
            "--phase-noise-stds",
            "0.0008",
            "--rhos",
            "0.995",
            "--delays",
            "20",
            "--snrs",
            "15",
            "--seeds",
            "0",
            "--frames",
            "1",
            "--methods",
            "LMMSE-FIR",
            "CFO-Corrected LMMSE-FIR",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads((output_dir / "eme_slow_drift_grid.json").read_text(encoding="utf-8"))

    assert payload["schema_version"] == "eme-slow-drift-traditional-grid-v1"
    assert payload["candidate_count"] == 1
    assert payload["traditional_only"] is True
    assert payload["proposed_methods_included"] is False
    assert payload["shared_profile_prior"] is True
    assert payload["candidates"][0]["residual_cfo_limit"] == 0.0012
    assert payload["candidate_summaries"][0]["candidate_id"] == "cfo0p0008_phase0p0008_rho0p995"
    assert payload["candidate_summaries"][0]["score"]["best_by_snr"]["15.0"]["method"] in {
        "LMMSE-FIR",
        "CFO-Corrected LMMSE-FIR",
    }
    assert payload["selected_candidate"]["candidate_id"] == "cfo0p0008_phase0p0008_rho0p995"
    assert all(row["uses_neural_network"] is False and row["uses_rl"] is False for row in payload["rows"])
    assert all(row["cfo_abs_cycles_per_symbol"] == 0.0008 for row in payload["rows"])


def test_eme_slow_drift_v1_config_freezes_near_band_candidate_contract():
    config = json.loads(Path("configs/continual_ppo_eme_slow_drift_v1.json").read_text(encoding="utf-8"))

    assert config["main_level"] == "B"
    assert config["pilot_layout"] == "prefix"
    assert config["pilot_total"] == 128
    assert config["impairment_profile"] == "eme_slow_drift_v1"
    assert config["rho"] == 0.995
    assert config["snrs_main"] == [0, 5, 10, 15]
    assert config["main_snrs"] == [0, 5, 10, 15]
    assert config["main_delays"] == [20, 30, 40]
    assert config["profile_prior"]["cfo_abs_cycles_per_symbol"] == 0.0008
    assert config["profile_prior"]["phase_noise_std"] == 0.003
    assert config["profile_prior"]["residual_cfo_limit"] == 0.0012
    assert config["profile_prior"]["acquisition_cfo_limit"] == 0.004
    assert "0 dB slightly above target" in config["profile_prior"]["selection_note"]
    assert config["model"]["enable_phase_correction_branch"] is True


def test_compare_passes_profile_aware_residual_cfo_limit_to_traditional_baseline(monkeypatch):
    import compare
    from compare import BaselineMethodState
    from baseline.traditional_equalizers import TraditionalResult
    from env.comm_env import CommEnvConfig, CommunicationEnvironment, ReceiverState

    captured = {}

    def fake_run_traditional_equalizer(
        method,
        receiver_view,
        cir,
        soft_tail,
        snr_db,
        phase_state=None,
        residual_cfo_limit=0.001,
    ):
        captured["residual_cfo_limit"] = residual_cfo_limit
        logits = torch.zeros_like(receiver_view.rx_symbols.real, dtype=torch.float32)
        return TraditionalResult(
            method=method,
            logits=logits,
            soft_tail=soft_tail.clone(),
            iterations=1,
            extra={"residual_cfo_limit": residual_cfo_limit},
        )

    monkeypatch.setattr(compare, "run_traditional_equalizer", fake_run_traditional_equalizer)
    env = CommunicationEnvironment(
        CommEnvConfig(
            level="B",
            max_delay=20,
            snr_db=10.0,
            rho=0.995,
            total_pilot=128,
            layout="prefix",
            seed=9100,
            impairment_profile="eme_slow_drift_v1",
        )
    )
    start = env.reset_episode()
    frame = env.next_frame()
    state = BaselineMethodState(
        cir=torch.zeros(21, dtype=torch.complex64),
        receiver_state=ReceiverState(start.initial_soft_tail),
        residual_cfo_limit=0.0012,
    )

    result = compare._run_new_or_single_method("CFO+DD-Phase LMMSE-FIR", frame, 10.0, state, 20, 1, 4)

    assert captured["residual_cfo_limit"] == 0.0012
    assert result.extra["residual_cfo_limit"] == 0.0012


def test_compare_routes_dfe_rls_traditional_group_to_traditional_equalizer(monkeypatch):
    import compare
    from compare import BaselineMethodState
    from baseline.traditional_equalizers import TraditionalResult
    from env.comm_env import CommEnvConfig, CommunicationEnvironment, ReceiverState

    captured = {}

    def fake_run_traditional_equalizer(
        method,
        receiver_view,
        cir,
        soft_tail,
        snr_db,
        phase_state=None,
        residual_cfo_limit=0.001,
    ):
        captured["method"] = method
        captured["residual_cfo_limit"] = residual_cfo_limit
        logits = torch.zeros_like(receiver_view.rx_symbols.real, dtype=torch.float32)
        return TraditionalResult(
            method=method,
            logits=logits,
            soft_tail=soft_tail.clone(),
            iterations=1,
            extra={"residual_cfo_limit": residual_cfo_limit, "traditional": True},
        )

    monkeypatch.setattr(compare, "run_traditional_equalizer", fake_run_traditional_equalizer)
    env = CommunicationEnvironment(
        CommEnvConfig(level="B", max_delay=20, snr_db=10.0, rho=0.995, total_pilot=128, layout="prefix", seed=9101)
    )
    start = env.reset_episode()
    frame = env.next_frame()
    states = {
        "DFE-RLS": BaselineMethodState(
            cir=torch.zeros(21, dtype=torch.complex64),
            receiver_state=ReceiverState(start.initial_soft_tail),
            residual_cfo_limit=0.0012,
        )
    }

    results = compare._run_methods_for_frame(("DFE-RLS",), frame, 10.0, states, delay=20, frame_index=1, update_interval=4)

    assert captured == {"method": "DFE-RLS", "residual_cfo_limit": 0.0012}
    assert results["DFE-RLS"].extra["traditional"] is True
