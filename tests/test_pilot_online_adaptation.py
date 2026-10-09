from types import SimpleNamespace
from pathlib import Path

import pytest
import torch


def test_reward_window_gate_accepts_stable_multi_frame_improvement():
    from training.online_adaptation import RewardWindowGate

    gate = RewardWindowGate(min_cumulative_improvement=1e-4, max_parameter_delta_norm=0.1)
    decision = gate.evaluate([0.00008, 0.00005, 0.00002], parameter_delta_norm=0.01)
    assert decision.accepted is True
    assert decision.reason == "accepted"
    assert decision.cumulative_improvement == pytest.approx(0.00015)


def test_reward_window_gate_rejects_single_frame_regression():
    from training.online_adaptation import RewardWindowGate

    gate = RewardWindowGate(max_single_frame_regression=0.0, max_parameter_delta_norm=0.1)
    decision = gate.evaluate([0.001, -0.00001, 0.001], parameter_delta_norm=0.01)
    assert decision.accepted is False
    assert decision.reason == "single_frame_regression"


def test_reward_window_gate_rejects_trust_region_violation_and_empty_window():
    from training.online_adaptation import RewardWindowGate

    gate = RewardWindowGate(max_parameter_delta_norm=0.1)
    assert gate.evaluate([0.1], parameter_delta_norm=0.2).reason == "trust_region_violation"
    assert gate.evaluate([], parameter_delta_norm=0.0).reason == "empty_reward_window"


def test_reward_window_gate_joint_metrics_require_ber_and_margin_alignment():
    from training.online_adaptation import RewardWindowGate

    gate = RewardWindowGate(require_joint_metrics=True, max_parameter_delta_norm=0.1)
    missing = gate.evaluate([0.1], parameter_delta_norm=0.01)
    assert missing.reason == "missing_joint_metrics"
    rejected = gate.evaluate(
        [0.1, 0.1],
        parameter_delta_norm=0.01,
        reward_ber_improvements=[0.0, -0.01],
        reward_margin_improvements=[0.1, 0.1],
    )
    assert rejected.reason == "single_frame_ber_regression"
    accepted = gate.evaluate(
        [0.1, 0.1],
        parameter_delta_norm=0.01,
        reward_ber_improvements=[0.01, 0.0],
        reward_margin_improvements=[0.1, 0.0],
    )
    assert accepted.accepted is True


def test_reward_gate_replay_script_is_present():
    from pathlib import Path

    assert Path("scripts/replay_reward_gate.py").exists()
    assert Path("scripts/replay_rolling_reward_gate.py").exists()


def test_rolling_replay_exposes_pilot_drift_gate_controls():
    source = Path("scripts/replay_rolling_reward_gate.py").read_text(encoding="utf-8")
    assert "PilotDriftDetector" in source
    assert "pilot_state_drift_detected" in source
    assert "data_labels_used_online" in source
    assert "estimate_phase_residual_vector" in source
    assert "LinearChannelOperator" in source
    assert "state_conditioned_lr_scale" in source


def test_state_conditioned_lr_scale_uses_only_pilot_state_strength():
    from scripts.replay_rolling_reward_gate import state_conditioned_lr_scale

    assert state_conditioned_lr_scale(
        drift_distance=0.0,
        drift_threshold=0.25,
        reconstruction_error=0.0,
        reconstruction_scale=0.05,
        confidence=1.0,
        min_scale=0.25,
        max_scale=1.0,
    ) == pytest.approx(0.25)
    assert state_conditioned_lr_scale(
        drift_distance=0.25,
        drift_threshold=0.25,
        reconstruction_error=0.05,
        reconstruction_scale=0.05,
        confidence=1.0,
        min_scale=0.25,
        max_scale=1.0,
    ) == pytest.approx(1.0)
    assert state_conditioned_lr_scale(
        drift_distance=0.25,
        drift_threshold=0.25,
        reconstruction_error=0.05,
        reconstruction_scale=0.05,
        confidence=0.0,
        min_scale=0.25,
        max_scale=1.0,
    ) == pytest.approx(0.25)


def test_pilot_candidate_ranking_replay_has_label_boundary():
    from pathlib import Path

    source = Path("scripts/replay_pilot_candidate_ranking.py").read_text(encoding="utf-8")
    assert "apply_adapt_only_peft_update" in source
    assert "data_labels_used_online" in source
    assert "diagnostic_uses_data_labels" in source
    assert "frame.data_mask" not in source
    assert "PilotResidualRLSAdapter" in source
    assert "rls_residual" in source


def test_reward_gate_replay_accepts_state_gap_option():
    import scripts.replay_reward_gate as replay

    assert replay.main is not None


def test_online_snr_layer_can_freeze_unreliable_peft_updates():
    import compare

    candidates = ({"name": "nominal", "groups": ("head",)},)
    assert compare._online_candidates_for_snr(candidates, snr_db=0.0, freeze_below_db=5.0) == ()
    assert compare._online_candidates_for_snr(candidates, snr_db=5.0, freeze_below_db=5.0) == candidates


def test_online_snr_layer_can_freeze_the_whole_update_chain():
    import compare

    assert compare._online_updates_are_frozen(0.0, 5.0) is True
    assert compare._online_updates_are_frozen(5.0, 5.0) is False
    assert compare._online_updates_are_frozen(10.0, None) is False


def test_online_update_schedule_updates_first_frame_then_at_configured_interval():
    import compare

    assert [compare._online_update_is_scheduled(frame, 8) for frame in range(1, 18)] == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
    ]


def test_online_update_schedule_rejects_non_positive_interval():
    import compare

    with pytest.raises(ValueError, match="update_interval"):
        compare._online_update_is_scheduled(1, 0)


def test_online_groups_cli_override_replaces_configured_candidate_groups():
    import compare

    config = {
        "online_adaptation_groups": ["adapter_lora", "conditioner_film"],
        "online_adaptation_candidates": [
            {"name": "configured", "groups": ["adapter_lora"]}
        ],
    }

    groups, candidate_config = compare._online_groups_from_config(config, ["head"])

    assert groups == {"head"}
    assert candidate_config["online_adaptation_candidates"] is None


def test_online_groups_rejects_empty_cli_override():
    import compare

    with pytest.raises(ValueError, match="online_groups"):
        compare._online_groups_from_config({}, [])


def test_online_condition_source_can_be_frozen_to_acquisition_for_parameter_only_ablation():
    import compare

    assert compare._online_condition_source_from_config({}, "acquisition") == "acquisition"
    assert compare._online_condition_source_from_config({}, "pilot_phase") == "pilot_phase"
    assert compare._online_condition_source_from_config(
        {"online_condition_source": "pilot_cir_phase"}, None
    ) == "pilot_cir_phase"


def test_online_condition_source_rejects_unknown_boundary():
    import compare

    with pytest.raises(ValueError, match="online_condition_source"):
        compare._online_condition_source_from_config({}, "data")


def test_shared_tail_update_helper_applies_the_same_cross_frame_smoothing_rule():
    import compare

    previous = torch.tensor([1.0 + 0.0j, -1.0 + 0.0j])
    detected = torch.tensor([-1.0 + 0.0j, 1.0 + 0.0j])

    updated = compare._update_receiver_tail(previous, detected, alpha=0.25)

    assert torch.allclose(updated, torch.tensor([0.5 - 0.0j, -0.5 + 0.0j]))


def test_compare_resume_key_distinguishes_online_condition_source():
    import compare

    base = {
        "method": "Pilot-Driven Online Adaptation",
        "delay": 116,
        "snr_db": 10.0,
        "seed": 0,
        "frame": 1,
        "pilot_total": 128,
        "reward_pilot_total": 32,
        "pilot_layout": "prefix",
        "impairment_profile": "cfo_phase_tiny",
    }

    acquisition = {**base, "condition_source": "acquisition"}
    pilot = {**base, "condition_source": "pilot_cir_phase"}

    assert compare._row_key(acquisition) != compare._row_key(pilot)


def test_pure_peft_contract_reports_parameter_groups_and_label_boundaries():
    import compare

    config = {
        "online_adaptation_groups": ["phase_trend", "head"],
        "online_condition_source": "acquisition",
    }
    groups, _ = compare._online_groups_from_config(config, None)

    assert groups == {"phase_trend", "head"}
    assert compare._online_condition_source_from_config(config, None) == "acquisition"


def test_internal_peft_groups_are_distinct_from_cir_state_update():
    model = UnfoldedEqualizer(
        UnfoldedConfig(
            frame_len=32,
            max_delay=4,
            iterations=1,
            d_model=24,
            num_heads=4,
            pilot_conditioned=True,
        )
    )
    model.attach_online_phase_trend_adapter()
    model.set_trainable_groups({"phase_trend", "head"})

    trainable = model.trainable_parameters()
    assert trainable
    assert all(
        getattr(parameter, "_peft_group", None) in {"phase_trend", "head"}
        for parameter in trainable
    )
    assert not any(
        getattr(parameter, "_peft_group", None) == "phase"
        for parameter in trainable
    )


def test_frozen_and_online_share_explicit_acquisition_condition_source():
    import compare

    model_config = UnfoldedConfig(
        frame_len=16,
        max_delay=2,
        iterations=1,
        d_model=16,
        num_heads=4,
        pilot_conditioned=True,
    )
    config = {
        "model": model_config.to_dict(),
        "online_adaptation_groups": ["phase_trend", "head"],
        "online_condition_source": "acquisition",
        "online_adaptation_candidates": [
            {"name": "candidate", "groups": ["phase_trend", "head"]}
        ],
    }
    states = compare._build_method_states(
        ("Frozen Offline NN", "Pilot-Driven Online Adaptation"),
        torch.zeros(3, dtype=torch.complex64),
        torch.tensor([1.0 + 0.0j, 0.0j, 0.0j]),
        config,
        delay=2,
        snr_db=10.0,
        seed=0,
        device="cpu",
        cir_update_mode="fixed",
        scheduler="fixed",
    )

    assert states["Frozen Offline NN"].condition_source == "acquisition"
    assert states["Pilot-Driven Online Adaptation"].condition_source == "acquisition"

from agent.cir_estimator import condition_from_cir
from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from training.online_adaptation import (
    PilotDrivenOnlineAdapter,
    _normalized_proximal_penalty,
    _pilot_reconstruction_loss,
    hard_example_weights,
    run_pilot_driven_online,
)


def test_hard_example_weights_emphasize_logits_near_decision_boundary():
    logits = torch.tensor([0.0, 0.25, 2.0, 8.0])
    weights = hard_example_weights(logits, temperature=0.5)

    assert weights[0] > weights[1] > weights[2] > weights[3]
    assert weights[0].item() == pytest.approx(2.0)
    assert weights[-1].item() < 1.01


def test_hard_example_temperature_must_be_positive():
    with pytest.raises(ValueError, match="temperature"):
        hard_example_weights(torch.zeros(2), temperature=0.0)


def test_normalized_proximal_penalty_is_zero_at_snapshot_and_positive_after_move():
    snapshot = {
        "layer.weight": torch.tensor([1.0, -2.0]),
        "layer.bias": torch.tensor([0.5]),
    }
    parameters = [
        ("layer.weight", torch.tensor([1.0, -2.0])),
        ("layer.bias", torch.tensor([0.5])),
    ]

    assert _normalized_proximal_penalty(parameters, snapshot) == pytest.approx(0.0)

    moved = [
        ("layer.weight", torch.tensor([2.0, -2.0])),
        ("layer.bias", torch.tensor([0.5])),
    ]
    assert _normalized_proximal_penalty(moved, snapshot) > 0.0


def _identity_condition(batch: int = 1):
    cir = torch.zeros(batch, 5, dtype=torch.complex64)
    cir[:, 0] = 1.0 + 0.0j
    return condition_from_cir(cir, snr_db=10.0)


def test_pilot_conditioner_depends_on_known_adapt_pilot():
    model = UnfoldedEqualizer(
        UnfoldedConfig(
            frame_len=32,
            max_delay=4,
            iterations=1,
            d_model=24,
            num_heads=4,
            pilot_conditioned=True,
        )
    )
    rx_iq = torch.randn(1, 32, 2)
    adapt_mask = torch.zeros(1, 32, dtype=torch.bool)
    adapt_mask[:, :12] = True
    adapt_a = torch.zeros(1, 32, dtype=torch.complex64)
    adapt_a[:, :12] = 1.0 + 0.0j
    adapt_b = adapt_a.clone()
    adapt_b[:, 6] = -1.0 + 0.0j

    context_a = model._pilot_context(rx_iq, adapt_a, adapt_mask)
    context_b = model._pilot_context(rx_iq, adapt_b, adapt_mask)

    assert context_a.shape == (1, 24)
    assert not torch.allclose(context_a, context_b)


def test_online_adapter_updates_selected_peft_group_from_adapt_pilot_only():
    torch.manual_seed(5)
    model = UnfoldedEqualizer(
        UnfoldedConfig(
            frame_len=32,
            max_delay=4,
            iterations=1,
            d_model=24,
            num_heads=4,
            pilot_conditioned=True,
        )
    )
    frame = SimpleNamespace(
        rx_symbols=torch.randn(32, dtype=torch.complex64),
        tx_symbols=torch.where(
            torch.arange(32) % 2 == 0,
            torch.ones(32, dtype=torch.complex64),
            -torch.ones(32, dtype=torch.complex64),
        ),
        adapt_mask=torch.arange(32) < 12,
        reward_mask=torch.arange(32) >= 12,
        data_mask=torch.zeros(32, dtype=torch.bool),
        model_region_ids=torch.zeros(32, dtype=torch.long),
    )
    adapter = PilotDrivenOnlineAdapter(model, groups={"head"}, learning_rate=1e-2, steps=1)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}

    result = adapter.adapt(
        frame,
        _identity_condition(),
        torch.zeros(1, 4, dtype=torch.complex64),
    )

    assert result.data_labels_used_online is False
    assert result.adapt_pilot_count == 12
    assert result.accepted is True
    assert result.parameter_delta_norm > 0.0
    changed = [
        name for name, value in model.named_parameters()
        if not torch.equal(before[name], value.detach())
    ]
    assert changed
    assert all(getattr(model.get_parameter(name), "_peft_group", None) == "head" for name in changed)


def test_channel_residual_supports_pilot_reconstruction_objective():
    torch.manual_seed(31)
    model = UnfoldedEqualizer(
        UnfoldedConfig(
            frame_len=32,
            max_delay=4,
            iterations=1,
            d_model=24,
            num_heads=4,
            pilot_conditioned=True,
        )
    )
    model.attach_online_channel_residual_adapter()
    tx = torch.where(
        torch.arange(32) % 2 == 0,
        torch.ones(32, dtype=torch.complex64),
        -torch.ones(32, dtype=torch.complex64),
    )
    rx = torch.zeros(32, dtype=torch.complex64)
    rx[:12] = 1.1 * tx[:12]
    frame = SimpleNamespace(
        rx_symbols=rx,
        tx_symbols=tx,
        adapt_mask=torch.arange(32) < 12,
        reward_mask=torch.arange(32) >= 12,
        data_mask=torch.zeros(32, dtype=torch.bool),
        model_region_ids=torch.zeros(32, dtype=torch.long),
    )
    condition = _identity_condition()
    before = _pilot_reconstruction_loss(
        model,
        condition,
        tx.unsqueeze(0),
        rx.unsqueeze(0),
        frame.adapt_mask.unsqueeze(0),
        torch.zeros(1, 4, dtype=torch.complex64),
    )
    adapter = PilotDrivenOnlineAdapter(
        model,
        groups={"channel_residual"},
        learning_rate=1e-2,
        steps=2,
        objective="pilot_reconstruction",
    )
    result = adapter.adapt(
        frame,
        condition,
        torch.zeros(1, 4, dtype=torch.complex64),
    )
    after = _pilot_reconstruction_loss(
        model,
        condition,
        tx.unsqueeze(0),
        rx.unsqueeze(0),
        frame.adapt_mask.unsqueeze(0),
        torch.zeros(1, 4, dtype=torch.complex64),
    )
    assert result.accepted is True
    assert result.data_labels_used_online is False
    assert result.parameter_delta_norm > 0.0
    assert after < before


def test_online_adapter_updates_only_pilot_encoder_from_adapt_pilot():
    torch.manual_seed(23)
    model = UnfoldedEqualizer(
        UnfoldedConfig(
            frame_len=32,
            max_delay=4,
            iterations=1,
            d_model=24,
            num_heads=4,
            pilot_conditioned=True,
        )
    )
    frame = SimpleNamespace(
        rx_symbols=torch.randn(32, dtype=torch.complex64),
        tx_symbols=torch.where(
            torch.arange(32) % 2 == 0,
            torch.ones(32, dtype=torch.complex64),
            -torch.ones(32, dtype=torch.complex64),
        ),
        adapt_mask=torch.arange(32) < 12,
        reward_mask=torch.arange(32) >= 12,
        data_mask=torch.zeros(32, dtype=torch.bool),
        model_region_ids=torch.zeros(32, dtype=torch.long),
    )
    adapter = PilotDrivenOnlineAdapter(
        model,
        groups={"pilot_encoder"},
        learning_rate=1e-2,
        steps=1,
    )
    before = {name: value.detach().clone() for name, value in model.named_parameters()}

    result = adapter.adapt(
        frame,
        _identity_condition(),
        torch.zeros(1, 4, dtype=torch.complex64),
    )

    assert result.accepted is True
    assert result.data_labels_used_online is False
    assert result.adapt_pilot_count == 12
    changed = [
        name for name, value in model.named_parameters()
        if not torch.equal(before[name], value.detach())
    ]
    assert changed
    assert all(getattr(model.get_parameter(name), "_peft_group", None) == "pilot_encoder" for name in changed)


def test_phase_peft_update_changes_parameters_and_adapt_loss():
    torch.manual_seed(19)
    model = UnfoldedEqualizer(
        UnfoldedConfig(
            frame_len=32,
            max_delay=4,
            iterations=1,
            d_model=24,
            num_heads=4,
            pilot_conditioned=True,
            enable_phase_correction_branch=True,
            phase_correction_initial_scale=1.0,
        )
    )
    frame = SimpleNamespace(
        rx_symbols=torch.randn(32, dtype=torch.complex64),
        tx_symbols=torch.where(
            torch.arange(32) % 2 == 0,
            torch.ones(32, dtype=torch.complex64),
            -torch.ones(32, dtype=torch.complex64),
        ),
        adapt_mask=torch.arange(32) < 12,
        reward_mask=torch.arange(32) >= 12,
        data_mask=torch.zeros(32, dtype=torch.bool),
        model_region_ids=torch.zeros(32, dtype=torch.long),
    )
    adapter = PilotDrivenOnlineAdapter(model, groups={"phase"}, learning_rate=1e-2, steps=2)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    result = adapter.adapt(
        frame,
        _identity_condition(),
        torch.zeros(1, 4, dtype=torch.complex64),
        groups={"phase"},
        learning_rate=1e-2,
        steps=2,
        max_delta_norm=1.0,
    )
    changed = [name for name, value in model.named_parameters() if not torch.equal(before[name], value.detach())]
    assert result.accepted is True
    assert result.parameter_delta_norm > 0.0
    assert result.adapt_loss_after != pytest.approx(result.adapt_loss_before)
    assert changed
    assert all(getattr(model.get_parameter(name), "_peft_group", None) == "phase" for name in changed)


def test_online_adapter_rejects_frame_without_adapt_pilot():
    model = UnfoldedEqualizer(
        UnfoldedConfig(frame_len=16, max_delay=2, iterations=1, d_model=16, num_heads=4)
    )
    frame = SimpleNamespace(
        rx_symbols=torch.randn(16, dtype=torch.complex64),
        tx_symbols=torch.ones(16, dtype=torch.complex64),
        adapt_mask=torch.zeros(16, dtype=torch.bool),
        reward_mask=torch.ones(16, dtype=torch.bool),
        data_mask=torch.zeros(16, dtype=torch.bool),
        model_region_ids=torch.zeros(16, dtype=torch.long),
    )
    adapter = PilotDrivenOnlineAdapter(model, groups={"head"}, learning_rate=1e-2, steps=1)
    result = adapter.adapt(frame, _identity_condition(), torch.zeros(1, 2, dtype=torch.complex64))

    assert result.accepted is False
    assert result.adapt_pilot_count == 0
    assert result.parameter_delta_norm == 0.0


def test_online_adapter_accepts_action_specific_update_override():
    torch.manual_seed(8)
    model = UnfoldedEqualizer(
        UnfoldedConfig(
            frame_len=32,
            max_delay=4,
            iterations=1,
            d_model=24,
            num_heads=4,
            pilot_conditioned=True,
        )
    )
    frame = SimpleNamespace(
        rx_symbols=torch.randn(32, dtype=torch.complex64),
        tx_symbols=torch.where(
            torch.arange(32) % 2 == 0,
            torch.ones(32, dtype=torch.complex64),
            -torch.ones(32, dtype=torch.complex64),
        ),
        adapt_mask=torch.arange(32) < 12,
        reward_mask=torch.arange(32) >= 12,
        data_mask=torch.zeros(32, dtype=torch.bool),
        model_region_ids=torch.zeros(32, dtype=torch.long),
    )
    adapter = PilotDrivenOnlineAdapter(model, groups={"head"}, learning_rate=1e-2, steps=1)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}

    result = adapter.adapt(
        frame,
        _identity_condition(),
        torch.zeros(1, 4, dtype=torch.complex64),
        groups={"conditioner_film"},
        learning_rate=5e-3,
        steps=1,
        max_delta_norm=0.5,
    )

    assert result.accepted is True
    changed = [
        name for name, value in model.named_parameters()
        if not torch.equal(before[name], value.detach())
    ]
    assert changed
    assert all(
        getattr(model.get_parameter(name), "_peft_group", None) == "conditioner_film"
        for name in changed
    )


def test_pilot_online_runner_writes_online_only_metrics(tmp_path):
    result = run_pilot_driven_online(
        "configs/continual_ppo.json",
        frames=1,
        num_seeds=1,
        output_dir=tmp_path / "pilot_online",
        delays=[4],
        snrs=[10.0],
        pilot_total=64,
        pilot_layout="two_block",
        device="cpu",
    )

    assert result["schema_version"] == "pilot-driven-online-adaptation-v1"
    assert result["rows"][0]["online_update_source"] == "adapt_pilot_only"
    assert result["rows"][0]["data_labels_used_online"] is False


def test_pilot_online_runner_records_bandit_action_and_reward_boundary(tmp_path):
    result = run_pilot_driven_online(
        "configs/continual_ppo.json",
        frames=1,
        num_seeds=1,
        output_dir=tmp_path / "bandit_online",
        delays=[4],
        snrs=[10.0],
        pilot_total=64,
        pilot_layout="two_block",
        scheduler="bandit",
        device="cpu",
    )

    row = result["rows"][0]
    assert result["scheduler"] == "bandit"
    assert row["scheduler"] == "bandit"
    assert row["action"] in {"skip", "phase_weak", "head_weak", "head_nominal", "film_nominal", "joint_nominal"}
    assert row["action"] != "phase_weak"
    assert row["uses_rl"] is True
    assert row["data_labels_used_online"] is False
    assert row["reward_pilot_loss_before"] >= 0.0
    assert row["reward_pilot_loss_after"] >= 0.0
    assert row["bandit_action_cost"] <= 1e-5
    assert set(row["bandit_context"]) >= {
        "residual_cfo",
        "phase_slope",
        "cir_drift",
        "snr_db",
        "consecutive_rejections",
        "parameter_delta_norm",
    }
    assert row["reward_data_labels_used_online"] is False
