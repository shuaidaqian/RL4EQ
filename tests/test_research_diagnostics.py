import copy

import pytest
import torch

from agent.cir_estimator import CIRCondition
from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from env.frame_structure import FrameConfig, FrameGenerator


def _tiny_identity_frame(frame_index: int = 0):
    frame = FrameGenerator(
        FrameConfig(frame_len=96, total_pilot=64, layout="multi_block", max_delay=4),
        seed=321,
    ).generate(frame_index)
    cir = torch.zeros(5, dtype=torch.complex64)
    cir[0] = 1.0 + 0.0j
    tail = torch.zeros(4, dtype=torch.complex64)
    return frame.with_channel_output(frame.tx_symbols.clone(), tail, cir)


def _tiny_condition() -> CIRCondition:
    cir = torch.zeros(1, 5, dtype=torch.complex64)
    cir[0, 0] = 1.0 + 0.0j
    return CIRCondition(
        complex_cir=cir,
        support_probability=torch.ones(1, 5),
        noise_variance=torch.full((1,), 0.1),
        confidence=torch.ones(1),
        latent_residual=torch.zeros(1, 96),
    )


def _tiny_model() -> UnfoldedEqualizer:
    return UnfoldedEqualizer(
        UnfoldedConfig(
            frame_len=96,
            max_delay=4,
            iterations=1,
            d_model=24,
            num_heads=4,
            adapter_rank=4,
            lora_rank=4,
        )
    )


def test_reward_data_correlation_report_uses_paired_improvements():
    from evaluation.research_diagnostics import (
        summarize_reward_data_correlation,
        summarize_reward_selected_actions,
        summarize_reward_surrogates,
    )


    rows = [
        {"action_name": "a", "seed": 0, "frame": 1, "reward_loss_improvement": 0.3, "reward_ber_improvement": 0.2, "reward_margin_improvement": 0.1, "data_ber_improvement": 0.03},
        {"action_name": "b", "seed": 0, "frame": 1, "reward_loss_improvement": 0.2, "reward_ber_improvement": 0.1, "reward_margin_improvement": 0.2, "data_ber_improvement": 0.02},
        {"action_name": "c", "seed": 1, "frame": 1, "reward_loss_improvement": 0.1, "reward_ber_improvement": 0.0, "reward_margin_improvement": 0.3, "data_ber_improvement": 0.01},
    ]

    summary = summarize_reward_data_correlation(rows, threshold=0.6)
    surrogates = summarize_reward_surrogates(rows, threshold=0.6)
    selected = summarize_reward_selected_actions(rows, surrogate_name="reward_ber_delta")

    assert summary["metric"] == "reward_loss_delta_vs_data_ber_delta"
    assert summary["spearman"] == 1.0
    assert summary["num_pairs"] == 3
    assert summary["gate_pass"] is True
    assert surrogates["metric"] == "reward_surrogate_vs_data_ber_delta"
    assert surrogates["best"]["name"] == "reward_loss_delta"
    assert "action_level_surrogates" in surrogates
    assert all("spearman" in item for item in surrogates["surrogates"])
    assert selected["metric"] == "reward_selected_discrete_action"
    assert selected["selection_uses_data_labels"] is False
    assert selected["diagnostic_uses_data_labels"] is True
    assert selected["selected_frames"] == 2


def test_pilot_replay_events_attribute_future_data_to_scheduled_update():
    from evaluation.research_diagnostics import build_pilot_replay_events

    rows = []
    for frame in range(1, 5):
        rows.extend(
            [
                {
                    "method": "Pilot-conditioned frozen NN",
                    "delay": 116,
                    "snr_db": 10.0,
                    "pilot_total": 128,
                    "pilot_layout": "prefix",
                    "seed": 0,
                    "frame": frame,
                    "ber_data": 0.20,
                },
                {
                    "method": "Pilot-Driven Online Adaptation",
                    "delay": 116,
                    "snr_db": 10.0,
                    "pilot_total": 128,
                    "pilot_layout": "prefix",
                    "seed": 0,
                    "frame": frame,
                    "ber_data": 0.15 if frame in {1, 2} else 0.25,
                    "online_update_scheduled": frame in {1, 3},
                    "online_update_candidate": "head_light" if frame == 1 else "skip",
                    "peft_update_applied": frame == 1,
                    "reward_pilot_loss_before": 0.40 if frame == 1 else 0.30,
                    "reward_pilot_loss_after": 0.35 if frame == 1 else 0.30,
                },
            ]
        )

    events = build_pilot_replay_events(rows, hold_frames=2)

    assert len(events) == 2
    assert events[0]["frames"] == [1, 2]
    assert events[0]["reward_loss_improvement"] == pytest.approx(0.05)
    assert events[0]["data_ber_improvement"] == pytest.approx(0.05)
    assert events[0]["future_data_ber_improvement"] == pytest.approx(0.05)
    assert events[1]["frames"] == [3, 4]
    assert events[1]["data_ber_improvement"] == pytest.approx(-0.05)
    assert events[1]["future_data_ber_improvement"] == pytest.approx(-0.05)
    assert events[0]["diagnostic_uses_data_labels"] is True
    assert events[0]["online_policy_uses_data_labels"] is False


def test_pilot_replay_events_requires_online_and_frozen_pairs():
    from evaluation.research_diagnostics import build_pilot_replay_events

    with pytest.raises(ValueError, match="配对"):
        build_pilot_replay_events(
            [
                {
                    "method": "Pilot-Driven Online Adaptation",
                    "delay": 116,
                    "snr_db": 10.0,
                    "pilot_total": 128,
                    "pilot_layout": "prefix",
                    "seed": 0,
                    "frame": 1,
                    "ber_data": 0.1,
                    "online_update_scheduled": True,
                }
            ],
            hold_frames=2,
        )


def test_reward_data_correlation_report_accepts_reward_ber_surrogate():
    from evaluation.research_diagnostics import summarize_reward_data_correlation

    rows = [
        {"action_name": "a", "reward_loss_improvement": 0.0, "reward_ber_improvement": 0.1, "reward_margin_improvement": 0.0, "data_ber_improvement": 0.01},
        {"action_name": "b", "reward_loss_improvement": 0.0, "reward_ber_improvement": 0.2, "reward_margin_improvement": 0.0, "data_ber_improvement": 0.02},
        {"action_name": "c", "reward_loss_improvement": 0.0, "reward_ber_improvement": 0.3, "reward_margin_improvement": 0.0, "data_ber_improvement": 0.03},
    ]

    summary = summarize_reward_data_correlation(rows, threshold=0.6, surrogate_name="reward_ber_delta")

    assert summary["metric"] == "reward_ber_delta_vs_data_ber_delta"
    assert summary["spearman"] == 1.0
    assert summary["gate_pass"] is True


def test_reward_surrogate_can_penalize_peft_delta_norm():
    from evaluation.research_diagnostics import summarize_reward_selected_actions

    rows = [
        {
            "action_name": "identity",
            "seed": 0,
            "frame": 1,
            "reward_loss_improvement": 0.0,
            "reward_ber_improvement": 0.0,
            "reward_margin_improvement": 0.0,
            "data_ber_improvement": 0.0,
            "peft_delta_norm": 0.0,
        },
        {
            "action_name": "large_update",
            "seed": 0,
            "frame": 1,
            "reward_loss_improvement": 0.00011,
            "reward_ber_improvement": 0.0,
            "reward_margin_improvement": 0.0,
            "data_ber_improvement": -0.01,
            "peft_delta_norm": 0.02,
        },
        {
            "action_name": "small_update",
            "seed": 0,
            "frame": 1,
            "reward_loss_improvement": 0.00010,
            "reward_ber_improvement": 0.0,
            "reward_margin_improvement": 0.0,
            "data_ber_improvement": 0.01,
            "peft_delta_norm": 0.001,
        },
    ]

    selected = summarize_reward_selected_actions(rows, surrogate_name="loss_minus_0.005_delta")

    assert selected["selected_rows"][0]["action_name"] == "small_update"
    assert selected["mean_data_ber_improvement"] > 0.0


def test_focused_peft_candidates_are_lightweight_and_non_redundant():
    from scripts.diagnose_research_assumptions import _focused_peft_candidates

    candidates = _focused_peft_candidates(base_lr=1e-4, base_steps=1)

    assert [item["name"] for item in candidates] == [
        "peft_head_light",
        "peft_head_fast",
        "peft_adapter_lora_conservative",
        "peft_adapter_lora_light",
        "peft_adapter_lora_head_light",
    ]
    assert candidates[1]["groups"] == {"head"}
    assert candidates[1]["lr"] == 5e-4
    assert candidates[2]["groups"] == {"adapter", "attention_lora", "ffn_lora"}
    assert candidates[2]["lr"] == 5e-5
    assert candidates[3]["groups"] == {"adapter", "attention_lora", "ffn_lora"}
    assert candidates[4]["groups"] == {"adapter_lora"}


def test_reward_selected_actions_fall_back_to_identity_without_positive_reward():
    from evaluation.research_diagnostics import summarize_reward_selected_actions

    rows = [
        {"action_name": "identity", "seed": 0, "frame": 1, "reward_loss_improvement": 0.0, "reward_ber_improvement": 0.0, "reward_margin_improvement": 0.0, "data_ber_improvement": 0.0, "action_delta_norm": 0.0},
        {"action_name": "bad", "seed": 0, "frame": 1, "reward_loss_improvement": -0.1, "reward_ber_improvement": 0.0, "reward_margin_improvement": 0.0, "data_ber_improvement": -0.2, "action_delta_norm": 1.0},
    ]

    selected = summarize_reward_selected_actions(rows, surrogate_name="reward_ber_delta")

    assert selected["selected_rows"][0]["action_name"] == "identity"
    assert selected["mean_data_ber_improvement"] == 0.0


def test_windowed_reward_summary_groups_frames_by_action_and_window():
    from evaluation.research_diagnostics import summarize_windowed_reward_data_correlation, summarize_windowed_selected_actions

    rows = []
    for frame in range(1, 5):
        rows.append(
            {
                "action_name": "identity",
                "delay": 20,
                "snr_db": 10.0,
                "pilot_total": 64,
                "pilot_layout": "prefix",
                "seed": 0,
                "frame": frame,
                "reward_loss_improvement": 0.0,
                "reward_ber_improvement": 0.0,
                "reward_margin_improvement": 0.0,
                "data_ber_improvement": 0.0,
                "peft_delta_norm": 0.0,
            }
        )
        rows.append(
            {
                "action_name": "peft_head_light",
                "delay": 20,
                "snr_db": 10.0,
                "pilot_total": 64,
                "pilot_layout": "prefix",
                "seed": 0,
                "frame": frame,
                "reward_loss_improvement": 0.1 * frame,
                "reward_ber_improvement": 0.0,
                "reward_margin_improvement": 0.01 * frame,
                "data_ber_improvement": 0.01 * frame,
                "peft_delta_norm": 0.001,
            }
        )

    summary = summarize_windowed_reward_data_correlation(rows, window_size=2, surrogate_name="reward_loss_delta")
    selected = summarize_windowed_selected_actions(rows, window_size=2, surrogate_name="reward_loss_delta")

    assert summary["metric"] == "windowed_reward_loss_delta_vs_data_ber_delta"
    assert summary["window_size"] == 2
    assert summary["num_pairs"] == 4
    assert summary["spearman"] == 1.0
    assert summary["gate_pass"] is True
    assert summary["diagnostic_uses_data_labels"] is True
    assert selected["metric"] == "windowed_reward_selected_discrete_action"
    assert selected["selected_windows"] == 2
    assert selected["mean_data_ber_improvement"] > 0.0
    assert selected["selection_uses_data_labels"] is False


def test_modulation_action_scan_reports_effective_action_candidates():
    from evaluation.research_diagnostics import structured_modulation_candidates, evaluate_modulation_candidates

    model = _tiny_model()
    frame = _tiny_identity_frame()
    condition = _tiny_condition()
    candidates = structured_modulation_candidates(num_blocks=len(model.blocks))

    rows = evaluate_modulation_candidates(
        model=model,
        frame=frame,
        condition=condition,
        soft_tail=torch.zeros(4, dtype=torch.complex64),
        candidates=candidates,
    )

    assert rows
    assert rows[0]["action_name"] == "identity"
    assert any(row["action_name"] != "identity" for row in rows)
    assert all("reward_loss_improvement" in row for row in rows)
    assert all("reward_ber_improvement" in row for row in rows)
    assert all("reward_margin_improvement" in row for row in rows)
    assert all("action_delta_norm" in row for row in rows)
    assert all("data_ber_improvement" in row for row in rows)
    assert all(row["diagnostic_uses_data_labels"] is True for row in rows)
    assert all(row["online_policy_uses_data_labels"] is False for row in rows)


def test_peft_adapt_update_does_not_depend_on_reward_or_data_labels():
    from evaluation.research_diagnostics import apply_adapt_only_peft_update

    frame = _tiny_identity_frame()
    flipped = frame.with_replaced_hidden_labels(~frame.bits, ~frame.bits)
    condition = _tiny_condition()
    base = _tiny_model()
    first = copy.deepcopy(base)
    second = copy.deepcopy(base)

    result_a = apply_adapt_only_peft_update(
        model=first,
        frame=frame,
        condition=condition,
        soft_tail=torch.zeros(4, dtype=torch.complex64),
        groups={"head"},
        lr=1e-3,
        steps=1,
    )
    result_b = apply_adapt_only_peft_update(
        model=second,
        frame=flipped,
        condition=condition,
        soft_tail=torch.zeros(4, dtype=torch.complex64),
        groups={"head"},
        lr=1e-3,
        steps=1,
    )

    assert result_a["updated_groups"] == ["head"]
    assert result_a["adapt_steps"] == 1
    assert result_a["diagnostic_uses_data_labels"] is True
    assert result_a["online_update_uses_data_labels"] is False
    for (name_a, param_a), (name_b, param_b) in zip(first.named_parameters(), second.named_parameters()):
        assert name_a == name_b
        assert torch.allclose(param_a, param_b, atol=1e-6)


def test_peft_adapt_update_reports_parameter_delta_norm():
    from evaluation.research_diagnostics import apply_adapt_only_peft_update

    frame = _tiny_identity_frame()
    condition = _tiny_condition()
    model = _tiny_model()

    result = apply_adapt_only_peft_update(
        model=model,
        frame=frame,
        condition=condition,
        soft_tail=torch.zeros(4, dtype=torch.complex64),
        groups={"head"},
        lr=1e-3,
        steps=1,
    )

    assert "peft_delta_norm" in result
    assert result["peft_delta_norm"] >= 0.0


def test_peft_window_candidate_scan_keeps_data_as_diagnostic_only():
    from evaluation.research_diagnostics import evaluate_peft_window_candidates

    frames = [_tiny_identity_frame(frame_index=idx) for idx in range(2)]
    model = _tiny_model()
    rows = evaluate_peft_window_candidates(
        model=model,
        frames=frames,
        condition=_tiny_condition(),
        soft_tail=torch.zeros(4, dtype=torch.complex64),
        candidates=[
            {"name": "identity", "groups": set(), "lr": 0.0, "steps": 0},
            {"name": "peft_head_light", "groups": {"head"}, "lr": 1e-3, "steps": 1},
        ],
        window_index=0,
    )

    assert [row["action_name"] for row in rows] == ["identity", "peft_head_light"]
    assert all(row["window_index"] == 0 for row in rows)
    assert all(row["window_size"] == 2 for row in rows)
    assert all("reward_loss_improvement" in row for row in rows)
    assert all("data_ber_improvement" in row for row in rows)
    assert all("peft_delta_norm" in row for row in rows)
    assert all(row["diagnostic_uses_data_labels"] is True for row in rows)
    assert all(row["online_update_uses_data_labels"] is False for row in rows)


def test_level_b_difficulty_scan_reports_only_traditional_baselines(tmp_path):
    from evaluation.research_diagnostics import run_level_b_difficulty_scan

    result = run_level_b_difficulty_scan(
        delays=[20],
        snrs=[10.0],
        pilot_totals=[64],
        pilot_layouts=["two_block"],
        seeds=[0],
        frames=1,
        output_dir=tmp_path,
    )

    assert result["metric"] == "level_b_traditional_difficulty"
    assert result["rows"]
    assert all(row["level"] == "B" for row in result["rows"])
    assert all(row["uses_neural_network"] is False for row in result["rows"])
    assert all(row["uses_rl"] is False for row in result["rows"])
    assert (tmp_path / "level_b_difficulty.json").exists()


def test_traditional_difficulty_calibration_cli_uses_prefix_and_low_snr_defaults(tmp_path):
    import json
    import subprocess
    import sys

    output_dir = tmp_path / "traditional_difficulty"
    subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_traditional_difficulty.py",
            "--delays",
            "20",
            "--pilot-totals",
            "64",
            "--seeds",
            "0",
            "--frames",
            "1",
            "--methods",
            "DFE-RLS",
            "SC-FDE-MMSE",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads((output_dir / "traditional_difficulty_calibration.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "traditional-difficulty-calibration-v1"
    assert payload["pilot_layouts"] == ["prefix"]
    assert payload["snrs"] == [0.0, 5.0, 10.0, 15.0]
    assert payload["optional_impairments"] == {
        "cfo": False,
        "phase_perturbation": False,
        "nonlinearity": False,
        "coding": False,
        "higher_order_modulation": False,
    }
    assert payload["level_b_difficulty"]["metric"] == "level_b_traditional_difficulty"
    assert {row["pilot_layout"] for row in payload["level_b_difficulty"]["rows"]} == {"prefix"}
    assert {row["snr_db"] for row in payload["level_b_difficulty"]["rows"]} == {0.0, 5.0, 10.0, 15.0}
    assert all(row["uses_neural_network"] is False and row["uses_rl"] is False for row in payload["level_b_difficulty"]["rows"])


def test_traditional_difficulty_calibration_cli_can_enable_cfo_phase_impairments(tmp_path):
    import json
    import subprocess
    import sys

    output_dir = tmp_path / "traditional_difficulty_cfo_phase"
    subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_traditional_difficulty.py",
            "--delays",
            "20",
            "--snrs",
            "0",
            "--pilot-totals",
            "64",
            "--seeds",
            "0",
            "--frames",
            "1",
            "--methods",
            "DFE-RLS",
            "CFO-Corrected DFE-RLS",
            "--enable-cfo",
            "--enable-phase-perturbation",
            "--impairment-profile",
            "cfo_phase_light",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads((output_dir / "traditional_difficulty_calibration.json").read_text(encoding="utf-8"))
    assert payload["optional_impairments"]["cfo"] is True
    assert payload["optional_impairments"]["phase_perturbation"] is True
    assert payload["impairment_profile"] == "cfo_phase_light"
    rows = payload["level_b_difficulty"]["rows"]
    assert rows
    assert {row["impairment_profile"] for row in rows} == {"cfo_phase_light"}
    assert all(row["uses_neural_network"] is False and row["uses_rl"] is False for row in rows)


def test_research_diagnostics_cli_smoke_writes_report(tmp_path):
    import json
    import subprocess
    import sys

    pretrained_dir = tmp_path / "pretrained"
    pretrained_dir.mkdir()
    model = UnfoldedEqualizer(
        UnfoldedConfig(
            frame_len=512,
            max_delay=20,
            iterations=1,
            d_model=24,
            num_heads=4,
            adapter_rank=4,
            lora_rank=4,
        )
    )
    (pretrained_dir / "model_config.json").write_text(json.dumps(model.config.to_dict(), ensure_ascii=False), encoding="utf-8")
    torch.save({"schema_version": "unfolded-eq-v1", "state_dict": model.state_dict()}, pretrained_dir / "model_best.pt")

    output_dir = tmp_path / "diagnostics"
    subprocess.run(
        [
            sys.executable,
            "scripts/diagnose_research_assumptions.py",
            "--config",
            "configs/continual_ppo.json",
            "--pretrained",
            str(pretrained_dir / "model_best.pt"),
            "--delays",
            "20",
            "--snrs",
            "10",
            "--pilot-totals",
            "64",
            "--pilot-layouts",
            "two_block",
            "--seeds",
            "0",
            "--frames",
            "1",
            "--window-size",
            "1",
            "--device",
            "cpu",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads((output_dir / "research_diagnostics.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "research-diagnostics-v1"
    assert payload["config"]["action_space"] == "focused_peft"
    assert payload["config"]["alignment_surrogate"] == "reward_ber_delta"
    assert payload["reward_data_alignment"]["metric"] == "reward_ber_delta_vs_data_ber_delta"
    assert payload["reward_data_alignment"]["surrogate_name"] == "reward_ber_delta"
    assert payload["reward_surrogates"]["metric"] == "reward_surrogate_vs_data_ber_delta"
    assert payload["reward_selected_actions"]["metric"] == "reward_selected_discrete_action"
    assert payload["reward_selected_actions"]["surrogate_name"] == "reward_ber_delta"
    assert payload["windowed_reward_data_alignment"]["metric"] == "windowed_reward_ber_delta_vs_data_ber_delta"
    assert payload["windowed_reward_selected_actions"]["metric"] == "windowed_reward_selected_discrete_action"
    assert payload["modulation_action_space"]["metric"] == "low_dim_modulation_action_effectiveness"
    assert payload["offline_nn_reference"]["metric"] == "offline_nn_reference"
    assert payload["peft_vs_modulation"]["metric"] == "adapt_only_peft_vs_low_dim_modulation"
    assert payload["peft_vs_modulation"]["peft_groups"]
    assert payload["level_b_difficulty"]["rows_count"] > 0
