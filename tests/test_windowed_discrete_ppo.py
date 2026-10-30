import json

import torch


def test_removed_legacy_frozen_name_is_not_a_current_method():
    import compare

    assert "Offline NN only" not in compare.PROPOSED_METHODS
    assert "Strict Offline NN only" not in compare.PROPOSED_METHODS
    assert "Offline NN only" not in compare.method_group("main")


def test_neural_condition_update_mode_is_explicit_per_method(tmp_path):
    import compare
    from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer

    model = UnfoldedEqualizer(UnfoldedConfig(frame_len=32, max_delay=4, iterations=1, d_model=8, num_heads=2))
    pretrained_dir = tmp_path / "pretrained"
    pretrained_dir.mkdir()
    (pretrained_dir / "model_config.json").write_text(
        json.dumps(model.config.to_dict(), ensure_ascii=False), encoding="utf-8"
    )
    torch.save({"state_dict": model.state_dict()}, pretrained_dir / "model_best.pt")
    states = compare._build_method_states(
        ("Frozen Offline NN", "NN + Fixed Modulation"),
        torch.zeros(4, dtype=torch.complex64),
        torch.ones(5, dtype=torch.complex64),
        {"model": model.config.to_dict()},
        delay=4,
        snr_db=10.0,
        seed=0,
        pretrained_path=pretrained_dir / "model_best.pt",
        device="cpu",
        cir_update_mode="pilot_sparse",
    )
    assert states["Frozen Offline NN"].condition_update_mode == "fixed"
    assert states["Frozen Offline NN"].condition_source == "pilot_cir_phase"
    assert states["NN + Fixed Modulation"].condition_update_mode == "pilot_sparse"


def test_discrete_safe_action_catalog_is_bounded_and_has_identity():
    from agent.discrete_safe_policy import safe_modulation_actions

    actions = safe_modulation_actions(num_blocks=3, device="cpu")
    names = [action.name for action in actions]

    assert names == [
        "identity",
        "tail_alpha_slow",
        "tail_alpha_nominal",
        "tail_alpha_fast",
        "cir_alpha_slow",
        "cir_alpha_nominal",
        "cir_alpha_fast",
        "peft_head_light",
        "peft_head_fast",
        "peft_adapter_lora_conservative",
        "peft_adapter_lora_light",
        "peft_adapter_lora_head_light",
        "peft_phase_signal_conservative",
        "peft_phase_signal_moderate",
        "rollback_identity",
    ]
    assert actions[0].delta_norm == 0.0
    assert all(torch.isfinite(action.modulation.to_vector()).all() for action in actions)
    assert all(action.delta_norm >= 0.0 for action in actions)
    assert actions[names.index("cir_alpha_slow")].cir_alpha == 0.3
    assert actions[names.index("cir_alpha_nominal")].cir_alpha == 0.6
    assert actions[names.index("cir_alpha_fast")].cir_alpha == 0.8
    assert actions[names.index("cir_alpha_fast")].peft_groups is None
    assert actions[names.index("tail_alpha_slow")].tail_alpha == 0.25
    assert actions[names.index("tail_alpha_nominal")].tail_alpha == 0.5
    assert actions[names.index("tail_alpha_fast")].tail_alpha == 0.8
    assert actions[names.index("peft_head_light")].peft_groups == {"head"}
    assert actions[names.index("peft_head_fast")].peft_lr == 5e-4
    assert actions[names.index("peft_adapter_lora_conservative")].peft_lr == 5e-5
    assert actions[names.index("peft_adapter_lora_light")].peft_groups == {"adapter", "attention_lora", "ffn_lora"}
    assert actions[names.index("peft_phase_signal_conservative")].peft_groups == {"phase_trend"}
    assert actions[names.index("peft_phase_signal_conservative")].peft_objective == "pilot_signal_reconstruction"
    assert actions[names.index("peft_phase_signal_conservative")].peft_max_delta_norm == 0.05
    moderate = actions[names.index("peft_phase_signal_moderate")]
    assert moderate.peft_groups == {"phase_trend"}
    assert moderate.peft_lr == 2.5e-3
    assert moderate.peft_steps == 5
    assert moderate.peft_max_delta_norm == 0.05
    assert moderate.peft_objective == "pilot_signal_reconstruction"


def test_discrete_safe_action_catalog_can_focus_on_pilot_peft_actions():
    import pytest

    from agent.discrete_safe_policy import safe_modulation_actions

    actions = safe_modulation_actions(
        num_blocks=3,
        device="cpu",
        action_names=["identity", "peft_phase_signal_conservative", "rollback_identity"],
    )
    assert [action.name for action in actions] == [
        "identity",
        "peft_phase_signal_conservative",
        "rollback_identity",
    ]
    assert [action.index for action in actions] == [0, 1, 2]
    with pytest.raises(ValueError, match="identity"):
        safe_modulation_actions(
            num_blocks=3, device="cpu", action_names=["rollback_identity"]
        )
    with pytest.raises(ValueError, match="未知名称"):
        safe_modulation_actions(
            num_blocks=3, device="cpu", action_names=["identity", "unknown"]
        )


def test_discrete_policy_prior_starts_conservatively_from_identity():
    from agent.discrete_safe_policy import DiscreteSafePolicy, initialize_safe_discrete_policy_prior, safe_modulation_actions

    actions = safe_modulation_actions(num_blocks=3, device="cpu")
    names = [action.name for action in actions]
    policy = DiscreteSafePolicy(observation_dim=4, action_count=len(actions), hidden_size=8)

    initialize_safe_discrete_policy_prior(policy, actions)

    probabilities = torch.softmax(policy.logits.bias, dim=0)
    assert policy.logits.bias[names.index("identity")] > policy.logits.bias[names.index("peft_head_fast")]
    assert probabilities[names.index("identity")] > 0.6
    assert probabilities[names.index("peft_head_fast")] < 0.1
    assert policy.logits.bias[names.index("rollback_identity")] < policy.logits.bias[names.index("identity")]
    peft_probability = sum(
        probabilities[index]
        for index, action in enumerate(actions)
        if action.peft_groups
    )
    assert peft_probability > 0.15
    assert probabilities[names.index("peft_phase_signal_conservative")] > 0.1
    assert probabilities[names.index("peft_adapter_lora_conservative")] > 0.08
    assert probabilities[names.index("identity")] > 0.55


def test_focused_peft_candidate_keeps_identity_prior_and_gets_exploration_probability():
    from agent.discrete_safe_policy import (
        DiscreteSafePolicy,
        initialize_safe_discrete_policy_prior,
        safe_modulation_actions,
    )

    actions = safe_modulation_actions(
        num_blocks=3,
        device="cpu",
        action_names=["identity", "peft_adapter_lora_light", "rollback_identity"],
    )
    policy = DiscreteSafePolicy(observation_dim=4, action_count=len(actions), hidden_size=8)
    initialize_safe_discrete_policy_prior(policy, actions)
    probabilities = torch.softmax(policy.logits.bias, dim=0)
    names = [action.name for action in actions]

    assert probabilities[names.index("identity")] > 0.6
    assert probabilities[names.index("peft_adapter_lora_light")] > 0.1


def test_focused_moderate_phase_action_is_bounded_and_explored():
    from agent.discrete_safe_policy import (
        DiscreteSafePolicy,
        initialize_safe_discrete_policy_prior,
        safe_modulation_actions,
    )

    actions = safe_modulation_actions(
        num_blocks=3,
        device="cpu",
        action_names=["identity", "peft_phase_signal_moderate", "rollback_identity"],
    )
    policy = DiscreteSafePolicy(observation_dim=4, action_count=len(actions), hidden_size=8)
    initialize_safe_discrete_policy_prior(policy, actions)
    probabilities = torch.softmax(policy.logits.bias, dim=0)
    names = [action.name for action in actions]

    assert probabilities[names.index("identity")] > 0.6
    assert probabilities[names.index("peft_phase_signal_moderate")] > 0.1
    assert actions[names.index("peft_phase_signal_moderate")].peft_max_delta_norm == 0.05


def test_discrete_policy_observation_is_independent_of_reward_pilot_waveform():
    from types import SimpleNamespace

    from agent.cir_estimator import condition_from_cir
    from agent.discrete_safe_policy import DiscreteSafePolicy, safe_modulation_actions
    from agent.rl_modulator import ModulationObservationEncoder
    from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
    from baseline.traditional_equalizers import estimate_phase_residual_vector
    from env.comm_env import ReceiverState
    from training.windowed_discrete_ppo import (
        WindowedDiscreteOnlineState,
        _adapt_receiver_view,
        run_windowed_discrete_frame,
    )

    tx = torch.where(torch.arange(32) % 2 == 0, 1.0, -1.0).to(torch.complex64)
    adapt_mask = torch.arange(32) < 8
    reward_mask = (torch.arange(32) >= 8) & (torch.arange(32) < 12)
    data_mask = torch.arange(32) >= 12
    common_rx = tx * torch.exp(torch.tensor(0.1j))
    first_rx = common_rx.clone()
    second_rx = common_rx.clone()
    first_rx[reward_mask] = 0.2 + 0.1j
    second_rx[reward_mask] = -12.0 + 8.0j
    common = {
        "tx_symbols": tx,
        "bits": (tx.real > 0.0),
        "adapt_mask": adapt_mask,
        "reward_mask": reward_mask,
        "data_mask": data_mask,
        "model_region_ids": torch.zeros(32, dtype=torch.long),
    }
    cir = torch.zeros(5, dtype=torch.complex64)
    cir[0] = 1.0 + 0.0j
    actions = safe_modulation_actions(num_blocks=1, action_names=["identity"])
    encoder = ModulationObservationEncoder(include_pilot_state=True)

    def observe(reward_waveform):
        torch.manual_seed(147)
        model = UnfoldedEqualizer(
            UnfoldedConfig(frame_len=32, max_delay=4, iterations=1, d_model=16, num_heads=4)
        )
        policy = DiscreteSafePolicy(len(encoder.FIELDS), len(actions), hidden_size=8)
        state = WindowedDiscreteOnlineState(
            cir=cir.clone(),
            receiver_state=ReceiverState(torch.zeros(4, dtype=torch.complex64)),
            model=model,
            policy=policy,
            optimizer=torch.optim.AdamW(policy.parameters(), lr=1e-4),
            encoder=encoder,
            actions=actions,
            hidden=policy.initial_hidden(),
            window_size=4,
            previous_window_reward=0.0,
            last_action_delta_norm=0.0,
            rollout=[],
        )
        frame = SimpleNamespace(rx_symbols=reward_waveform.clone(), **common)
        phase_features = estimate_phase_residual_vector(
            _adapt_receiver_view(frame),
            cir,
            torch.zeros(4, dtype=torch.complex64),
            blocks=4,
            cfo_hint=0.0,
        )
        condition = condition_from_cir(cir, 10.0, phase_features=phase_features)
        row = run_windowed_discrete_frame(
            state,
            frame,
            condition,
            snr_db=10.0,
            frame_index=1,
            update_interval=8,
        )
        return state.current_observation, condition.latent_residual, row

    first_observation, first_condition, first_row = observe(first_rx)
    second_observation, second_condition, second_row = observe(second_rx)

    # 完整套件运行时观察到约 1e-7 的浮点差异；这里验证语义独立，不要求逐位相同。
    # Reward 波形使用大幅不同的值，真实信息泄漏仍会超过该容差。
    assert torch.allclose(first_observation, second_observation, rtol=1e-6, atol=1e-6)
    assert torch.allclose(first_condition, second_condition, rtol=1e-6, atol=1e-6)
    assert first_observation.shape[-1] == len(encoder.BASE_FIELDS) + 6
    assert torch.allclose(
        first_observation[..., -6:], second_observation[..., -6:], rtol=1e-6, atol=1e-6
    )
    assert first_row["policy_observation_source"] == "adapt_pilot_only"
    assert first_row["reward_pilot_waveform_used_for_policy_observation"] is False
    assert second_row["reward_pilot_waveform_used_for_policy_observation"] is False
    assert abs(sum(first_row["policy_action_probabilities"].values()) - 1.0) < 1e-6
    assert first_row["selected_action_probability"] == first_row[
        "policy_action_probabilities"
    ][first_row["policy_action_name"]]


def test_old_discrete_policy_checkpoint_zero_pads_new_pilot_state_inputs(tmp_path):
    from agent.discrete_safe_policy import DiscreteSafePolicy, safe_modulation_actions
    from training.windowed_discrete_ppo import _load_discrete_policy_if_available

    old_policy = DiscreteSafePolicy(observation_dim=16, action_count=3, hidden_size=8)
    new_policy = DiscreteSafePolicy(observation_dim=22, action_count=3, hidden_size=8)
    checkpoint_path = tmp_path / "old_policy.pt"
    torch.save(
        {
            "state_dict": old_policy.state_dict(),
            "action_names": ["identity", "moderate", "rollback"],
        },
        checkpoint_path,
    )

    loaded = _load_discrete_policy_if_available(
        new_policy,
        checkpoint_path,
        "cpu",
        required=True,
        expected_action_names=["identity", "moderate", "rollback"],
    )

    assert loaded
    assert torch.equal(
        new_policy.encoder[0].weight[:, :16], old_policy.encoder[0].weight
    )
    assert torch.count_nonzero(new_policy.encoder[0].weight[:, 16:]) == 0
    actions = safe_modulation_actions(
        num_blocks=1,
        action_names=["identity", "peft_phase_signal_moderate", "rollback_identity"],
    )
    old_observation = torch.randn(1, 16)
    new_observation = torch.cat((old_observation, torch.zeros(1, 6)), dim=-1)
    old_generator = torch.Generator().manual_seed(913)
    new_generator = torch.Generator().manual_seed(913)
    old_sample = old_policy.sample(
        old_observation, old_policy.initial_hidden(), actions, generator=old_generator
    )
    new_sample = new_policy.sample(
        new_observation, new_policy.initial_hidden(), actions, generator=new_generator
    )

    assert old_sample.action.name == new_sample.action.name
    assert torch.equal(old_sample.probabilities, new_sample.probabilities)
    assert torch.equal(old_sample.log_prob, new_sample.log_prob)


def test_window_peft_delta_projection_caps_cumulative_action_change():
    import pytest

    from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
    from training.windowed_discrete_ppo import (
        _clip_peft_delta_to_snapshot,
        _snapshot_delta_norm,
        _snapshot_peft_groups,
    )

    model = UnfoldedEqualizer(UnfoldedConfig(
        frame_len=32,
        max_delay=4,
        iterations=1,
        d_model=16,
        num_heads=4,
    ))
    model.attach_online_phase_trend_adapter()
    snapshot = _snapshot_peft_groups(model, {"phase_trend"})
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in snapshot:
                parameter.add_(0.25)

    projected_norm = _clip_peft_delta_to_snapshot(model, snapshot, 0.05)

    assert projected_norm == pytest.approx(0.05, abs=1e-6)
    assert _snapshot_delta_norm(model, snapshot) <= 0.050001


def test_trajectory_peft_delta_projection_caps_repeated_window_updates():
    import pytest

    from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
    from training.windowed_discrete_ppo import (
        _clip_peft_delta_to_snapshot,
        _snapshot_delta_norm,
        _snapshot_peft_groups,
    )

    model = UnfoldedEqualizer(UnfoldedConfig(
        frame_len=32,
        max_delay=4,
        iterations=1,
        d_model=16,
        num_heads=4,
    ))
    model.attach_online_phase_trend_adapter()
    trajectory_snapshot = _snapshot_peft_groups(model, {"phase_trend"})
    parameters = dict(model.named_parameters())
    parameter_name = next(iter(trajectory_snapshot))
    parameter = parameters[parameter_name]

    with torch.no_grad():
        parameter.view(-1)[0].add_(0.03)
    first_window_norm = _clip_peft_delta_to_snapshot(model, trajectory_snapshot, 0.05)
    with torch.no_grad():
        parameter.view(-1)[0].add_(0.03)
    second_window_norm = _clip_peft_delta_to_snapshot(model, trajectory_snapshot, 0.05)

    assert first_window_norm == pytest.approx(0.03, abs=1e-6)
    assert second_window_norm == pytest.approx(0.05, abs=1e-6)
    assert _snapshot_delta_norm(model, trajectory_snapshot) <= 0.050001


def test_peft_episode_delta_cap_is_optional_and_validated():
    import pytest

    from agent.discrete_safe_policy import safe_modulation_actions

    names = ["identity", "peft_phase_signal_moderate", "rollback_identity"]
    uncapped = safe_modulation_actions(num_blocks=3, action_names=names)
    capped = safe_modulation_actions(
        num_blocks=3,
        action_names=names,
        peft_max_episode_delta_norm=0.05,
    )

    assert uncapped[1].peft_max_episode_delta_norm is None
    assert capped[1].peft_max_episode_delta_norm == 0.05
    assert capped[0].peft_max_episode_delta_norm is None
    with pytest.raises(ValueError, match="正有限数"):
        safe_modulation_actions(
            num_blocks=3,
            action_names=names,
            peft_max_episode_delta_norm=0.0,
        )


def test_safe_policy_can_learn_different_actions_for_different_pilot_observations():
    from agent.discrete_safe_policy import (
        DiscreteSafePolicy,
        initialize_safe_discrete_policy_prior,
        safe_modulation_actions,
    )
    from training.windowed_discrete_ppo import _ppo_update

    torch.manual_seed(54)
    actions = safe_modulation_actions(
        num_blocks=1, action_names=["identity", "peft_phase_signal_conservative"]
    )
    policy = DiscreteSafePolicy(observation_dim=4, action_count=len(actions), hidden_size=8)
    initialize_safe_discrete_policy_prior(policy, actions)
    hidden = policy.initial_hidden()
    observations = [torch.ones(1, 4), -torch.ones(1, 4)]
    index = torch.tensor([1])
    rollout = []
    with torch.no_grad():
        for observation, reward in zip(observations, [1.0, -1.0]):
            log_prob, value, _ = policy.evaluate_action(observation, hidden, index)
            rollout.append({
                "observation": observation, "hidden": hidden.clone(), "action_index": index,
                "old_log_prob": log_prob, "value": value, "reward": reward,
            })
    assert torch.equal(rollout[0]["old_log_prob"], rollout[1]["old_log_prob"])

    _ppo_update(policy, torch.optim.AdamW(policy.parameters(), lr=0.01), rollout)

    with torch.no_grad():
        positive_log_prob, _, _ = policy.evaluate_action(observations[0], hidden, index)
        negative_log_prob, _, _ = policy.evaluate_action(observations[1], hidden, index)
    assert positive_log_prob.item() > negative_log_prob.item() + 1e-6


def test_rl_phase_signal_action_updates_only_from_adapt_pilot():
    from types import SimpleNamespace

    from agent.cir_estimator import condition_from_cir
    from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
    from training.windowed_discrete_ppo import _apply_adapt_peft_update

    torch.manual_seed(4)
    config = UnfoldedConfig(
        frame_len=32, max_delay=4, iterations=1, d_model=24, num_heads=4
    )
    first = UnfoldedEqualizer(config)
    second = UnfoldedEqualizer(config)
    first.attach_online_phase_trend_adapter()
    second.attach_online_phase_trend_adapter()
    second.load_state_dict(first.state_dict())
    cir = torch.zeros(5, dtype=torch.complex64)
    cir[0] = 1.0 + 0j
    cir.requires_grad_()
    condition = condition_from_cir(cir, 5.0)
    tx = torch.where(torch.arange(32) % 2 == 0, 1.0, -1.0).to(torch.complex64)
    rx = tx * torch.exp(torch.tensor(0.2j))
    adapt_mask = torch.arange(32) < 16
    common = {
        "tx_symbols": tx,
        "bits": (tx.real > 0).to(torch.bool),
        "adapt_mask": adapt_mask,
        "reward_mask": (torch.arange(32) >= 16) & (torch.arange(32) < 20),
        "model_region_ids": torch.zeros(32, dtype=torch.long),
    }
    first_frame = SimpleNamespace(rx_symbols=rx.clone(), **common)
    second_rx = rx.clone()
    second_rx[~adapt_mask] = torch.complex(
        torch.full((16,), 1000.0), torch.full((16,), -1000.0)
    )
    second_frame = SimpleNamespace(rx_symbols=second_rx, **common)

    tail = torch.zeros(1, 4, dtype=torch.complex64, requires_grad=True)
    first_delta = _apply_adapt_peft_update(
        first,
        first_frame,
        condition,
        tail,
        groups={"phase_trend"},
        lr=5e-4,
        steps=1,
        objective="pilot_signal_reconstruction",
    )
    second_delta = _apply_adapt_peft_update(
        second,
        second_frame,
        condition,
        tail,
        groups={"phase_trend"},
        lr=5e-4,
        steps=1,
        objective="pilot_signal_reconstruction",
    )

    assert first_delta > 0.0
    assert second_delta == first_delta
    assert cir.grad is None
    assert tail.grad is None
    assert all(
        torch.equal(left, right)
        for left, right in zip(first.state_dict().values(), second.state_dict().values())
    )


def test_rejected_peft_update_is_restored_before_full_data_evaluation(monkeypatch):
    from types import SimpleNamespace

    from agent.cir_estimator import condition_from_cir
    from agent.discrete_safe_policy import DiscreteSafePolicy, safe_modulation_actions
    from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
    from env.comm_env import ReceiverState
    from training import windowed_discrete_ppo
    from training.windowed_discrete_ppo import (
        WindowRewardAccumulator,
        WindowedDiscreteOnlineState,
        refine_logits_with_tail as original_refine_logits_with_tail,
        run_windowed_discrete_frame,
    )

    torch.manual_seed(23)
    model = UnfoldedEqualizer(
        UnfoldedConfig(frame_len=32, max_delay=4, iterations=1, d_model=24, num_heads=4)
    )
    model.attach_online_phase_trend_adapter()
    actions = safe_modulation_actions(
        num_blocks=len(model.blocks),
        device="cpu",
        action_names=["identity", "peft_phase_signal_conservative"],
    )
    candidate = actions[1]
    frame = SimpleNamespace(
        rx_symbols=torch.randn(32, dtype=torch.complex64),
        tx_symbols=torch.where(torch.arange(32) % 2 == 0, 1.0, -1.0).to(torch.complex64),
        bits=torch.randint(0, 2, (32,), dtype=torch.bool),
        adapt_mask=torch.arange(32) < 8,
        reward_mask=(torch.arange(32) >= 8) & (torch.arange(32) < 12),
        data_mask=torch.arange(32) >= 12,
        model_region_ids=torch.zeros(32, dtype=torch.long),
    )
    initial_phase = model.online_phase_trend_adapter.raw_phase.detach().clone()
    captured_data_eval_phases = []

    def force_candidate_update(*args, **kwargs):
        del args, kwargs
        with torch.no_grad():
            model.online_phase_trend_adapter.raw_phase.fill_(0.25)
        return 0.25

    def reject_candidate(**kwargs):
        del kwargs
        return False

    def capture_full_data_evaluation(model_arg, rx_iq, *args, **kwargs):
        if bool(frame.data_mask.any()) and torch.count_nonzero(rx_iq[0, frame.data_mask]).item() > 0:
            captured_data_eval_phases.append(
                model_arg.online_phase_trend_adapter.raw_phase.detach().clone()
            )
        return original_refine_logits_with_tail(model_arg, rx_iq, *args, **kwargs)

    monkeypatch.setattr(windowed_discrete_ppo, "_apply_adapt_peft_update", force_candidate_update)
    monkeypatch.setattr(windowed_discrete_ppo, "should_accept_reward_guard", reject_candidate)
    monkeypatch.setattr(windowed_discrete_ppo, "refine_logits_with_tail", capture_full_data_evaluation)

    cir = torch.zeros(5, dtype=torch.complex64)
    cir[0] = 1.0 + 0.0j
    state = WindowedDiscreteOnlineState(
        cir=cir,
        receiver_state=ReceiverState(torch.zeros(4, dtype=torch.complex64)),
        model=model,
        policy=DiscreteSafePolicy(observation_dim=4, action_count=len(actions), hidden_size=8),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
        encoder=SimpleNamespace(),
        actions=actions,
        hidden=torch.zeros(1, 1, 8),
        window_size=4,
        previous_window_reward=0.0,
        last_action_delta_norm=0.0,
        rollout=[],
        current_action=candidate,
        current_accumulator=WindowRewardAccumulator(action_delta_norm=0.0),
        frames_in_window=1,
    )

    row = run_windowed_discrete_frame(
        state,
        frame,
        condition_from_cir(cir, 10.0),
        snr_db=10.0,
        frame_index=1,
        update_interval=16,
    )

    assert row["peft_frame_accepted"] is False
    assert row["safe_action_accepted"] is False
    assert torch.equal(model.online_phase_trend_adapter.raw_phase, initial_phase)
    assert len(captured_data_eval_phases) == 2
    assert all(torch.equal(value, initial_phase) for value in captured_data_eval_phases)
    assert row["pre_update_ber_data"] == row["ber_data"]
    assert row["data_bce_improvement_eval_only"] == 0.0
    assert row["data_logits_abs_delta_mean_eval_only"] == 0.0
    assert row["data_logits_direction_change_rate_eval_only"] == 0.0
    assert row["data_soft_output_delta_mean_eval_only"] == 0.0


def test_window_reward_uses_reward_pilot_metrics_not_data_labels():
    from training.windowed_discrete_ppo import WindowRewardAccumulator

    accumulator = WindowRewardAccumulator(action_delta_norm=2.0)
    accumulator.add_frame(
        reward_loss_before=0.7,
        reward_loss_after=0.6,
        reward_ber_before=0.25,
        reward_ber_after=0.125,
        data_ber_before=1.0,
        data_ber_after=0.0,
    )
    first = accumulator.finalize()
    accumulator_with_bad_data = WindowRewardAccumulator(action_delta_norm=2.0)
    accumulator_with_bad_data.add_frame(
        reward_loss_before=0.7,
        reward_loss_after=0.6,
        reward_ber_before=0.25,
        reward_ber_after=0.125,
        data_ber_before=0.0,
        data_ber_after=1.0,
    )
    second = accumulator_with_bad_data.finalize()

    assert first.reward == second.reward
    assert first.data_labels_used_online is False
    assert first.reward > 0.0


def test_window_reward_penalizes_real_parameter_delta_norm():
    from training.windowed_discrete_ppo import WindowRewardAccumulator

    small = WindowRewardAccumulator(action_delta_norm=0.001)
    large = WindowRewardAccumulator(action_delta_norm=0.1)
    for accumulator in (small, large):
        accumulator.add_frame(
            reward_loss_before=0.2,
            reward_loss_after=0.1,
            reward_ber_before=0.0,
            reward_ber_after=0.0,
            data_ber_before=1.0,
            data_ber_after=0.0,
        )

    assert small.finalize().reward == 0.1 - 0.005 * 0.001
    assert small.finalize().reward > large.finalize().reward


def test_reward_pilot_guard_rejects_actions_that_worsen_reward_ber():
    from training.windowed_discrete_ppo import should_accept_reward_guard

    assert should_accept_reward_guard(
        reward_loss_before=0.5,
        reward_loss_after=0.4,
        reward_ber_before=0.0,
        reward_ber_after=0.125,
    ) is False
    assert should_accept_reward_guard(
        reward_loss_before=0.5,
        reward_loss_after=0.49,
        reward_ber_before=0.125,
        reward_ber_after=0.125,
    ) is False
    assert should_accept_reward_guard(
        reward_loss_before=0.5,
        reward_loss_after=0.49,
        reward_ber_before=0.125,
        reward_ber_after=0.125,
        allow_loss_only=True,
        peft_delta_norm=0.0,
    ) is True
    assert should_accept_reward_guard(
        reward_loss_before=0.50009,
        reward_loss_after=0.5,
        reward_ber_before=0.125,
        reward_ber_after=0.125,
        allow_loss_only=True,
        peft_delta_norm=0.02,
    ) is False
    assert should_accept_reward_guard(
        reward_loss_before=0.5,
        reward_loss_after=0.55,
        reward_ber_before=0.125,
        reward_ber_after=0.0,
    ) is True


def test_windowed_discrete_runner_reports_policy_without_data_reward(tmp_path):
    from training.windowed_discrete_ppo import run_windowed_discrete_online

    result = run_windowed_discrete_online(
        config_path="configs/continual_ppo.json",
        frames=2,
        num_seeds=1,
        output_dir=tmp_path,
        delays=[20],
        snrs=[10],
        pilot_total=64,
        pilot_layout="two_block",
        window_size=2,
    )

    assert result["schema_version"] == "windowed-discrete-ppo-v1"
    assert len(result["rows"]) == 2
    assert all(row["policy_learning"] == "windowed_discrete_safe_ppo" for row in result["rows"])
    assert all(row["data_labels_used_online"] is False for row in result["rows"])
    assert all("window_reward_rollback" in row for row in result["rows"])
    payload = torch.load(tmp_path / "policy.pt", map_location="cpu", weights_only=False)
    assert payload["schema_version"] == "windowed-discrete-policy-v1"
    assert payload["state_dict"]
    assert payload["action_names"][0] == "identity"


def test_windowed_discrete_runner_can_enable_decision_directed_cir_update(tmp_path):
    from training.windowed_discrete_ppo import run_windowed_discrete_online

    result = run_windowed_discrete_online(
        config_path="configs/continual_ppo.json",
        frames=1,
        num_seeds=1,
        output_dir=tmp_path,
        delays=[20],
        snrs=[10],
        pilot_total=64,
        pilot_layout="two_block",
        window_size=1,
        cir_update_mode="decision_directed",
        cir_update_alpha=0.35,
    )

    assert result["rows"][0]["cir_update_mode"] == "decision_directed"
    assert result["rows"][0]["cir_update_alpha"] == 0.35
    assert result["rows"][0]["cir_update_uses_data_labels"] is False


def test_rollback_alias_keeps_sampled_action_index_for_ppo(monkeypatch, tmp_path):
    from agent.discrete_safe_policy import DiscretePolicySample, DiscreteSafePolicy
    from training import windowed_discrete_ppo

    captured = []

    def sample_rollback(self, observation, hidden, actions, generator=None):
        del generator
        action = next(action for action in actions if action.name == "rollback_identity")
        index = torch.tensor([action.index], device=observation.device)
        dist, state, next_hidden = self._dist(observation, hidden)
        return DiscretePolicySample(
            action=action,
            probabilities=dist.probs,
            log_prob=dist.log_prob(index),
            value=self.value_head(state).squeeze(-1),
            entropy=dist.entropy(),
            hidden=next_hidden,
        )

    def capture_rollout(policy, optimizer, rollout):
        del optimizer
        for row in rollout:
            captured.append(int(row["action_index"].item()))
            log_prob, _, _ = policy.evaluate_action(
                row["observation"], row["hidden"], row["action_index"]
            )
            assert torch.allclose(log_prob, row["old_log_prob"])
        return 0.0

    monkeypatch.setattr(DiscreteSafePolicy, "sample", sample_rollback)
    monkeypatch.setattr(windowed_discrete_ppo, "_ppo_update", capture_rollout)
    result = windowed_discrete_ppo.run_windowed_discrete_online(
        config_path="configs/continual_ppo.json", frames=1, num_seeds=1,
        output_dir=tmp_path, delays=[20], snrs=[10], pilot_total=64,
        pilot_layout="prefix", window_size=1, update_interval=1, device="cpu",
    )
    row = result["rows"][0]
    assert row["policy_action_name"] == "rollback_identity"
    assert row["applied_action_name"] == "identity"
    assert row["window_reward_rollback"] is False
    assert len(captured) == 1
    assert captured[0] > 0


def test_windowed_frame_can_use_receiver_level_cir_alpha_action(monkeypatch):
    from agent.cir_estimator import condition_from_cir
    from agent.discrete_safe_policy import DiscreteSafePolicy, safe_modulation_actions
    from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
    from env.comm_env import ReceiverState
    from training import windowed_discrete_ppo
    from training.windowed_discrete_ppo import WindowRewardAccumulator, WindowedDiscreteOnlineState, run_windowed_discrete_frame
    from types import SimpleNamespace

    model = UnfoldedEqualizer(UnfoldedConfig(frame_len=32, max_delay=4, iterations=1, d_model=24, num_heads=4))
    actions = safe_modulation_actions(num_blocks=len(model.blocks), device="cpu")
    names = [action.name for action in actions]
    captured = {}

    def fake_update(frame, logits, max_delay, previous_cir, alpha, confidence_threshold=None):
        captured["alpha"] = alpha
        captured["confidence_threshold"] = confidence_threshold
        return previous_cir

    monkeypatch.setattr(windowed_discrete_ppo, "decision_directed_cir_update", fake_update)
    frame = SimpleNamespace(
        rx_symbols=torch.randn(32, dtype=torch.complex64),
        tx_symbols=torch.sign(torch.randn(32)).to(torch.complex64),
        bits=torch.randint(0, 2, (32,), dtype=torch.bool),
        adapt_mask=torch.arange(32) < 8,
        reward_mask=(torch.arange(32) >= 8) & (torch.arange(32) < 12),
        data_mask=torch.arange(32) >= 12,
        model_region_ids=torch.zeros(32, dtype=torch.long),
    )
    state = WindowedDiscreteOnlineState(
        cir=torch.zeros(5, dtype=torch.complex64),
        receiver_state=ReceiverState(torch.zeros(4, dtype=torch.complex64)),
        model=model,
        policy=DiscreteSafePolicy(observation_dim=4, action_count=len(actions), hidden_size=8),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
        encoder=SimpleNamespace(),
        actions=actions,
        hidden=torch.zeros(1, 1, 8),
        window_size=4,
        previous_window_reward=0.0,
        last_action_delta_norm=0.0,
        rollout=[],
        cir_update_mode="decision_directed",
        cir_update_alpha=0.6,
        current_action=actions[names.index("cir_alpha_fast")],
        current_accumulator=WindowRewardAccumulator(action_delta_norm=0.0),
        frames_in_window=1,
    )

    row = run_windowed_discrete_frame(
        state,
        frame,
        condition_from_cir(state.cir, 10.0),
        snr_db=10.0,
        frame_index=1,
        update_interval=16,
    )

    assert captured["alpha"] == 0.8
    assert row["selected_cir_update_alpha"] == 0.8
    assert row["cir_update_uses_data_labels"] is False


def test_window_action_rollback_preserves_pilot_observed_tail_state(monkeypatch):
    from agent.cir_estimator import condition_from_cir
    from agent.discrete_safe_policy import DiscreteSafeAction, DiscreteSafePolicy, safe_modulation_actions
    from agent.modulation import ModulationConfig, ModulationState
    from env.comm_env import ReceiverState
    from training import windowed_discrete_ppo
    from training.windowed_discrete_ppo import WindowRewardAccumulator, WindowedDiscreteOnlineState, run_windowed_discrete_frame
    from types import SimpleNamespace

    class DummyModel:
        def __init__(self):
            self.calls = 0
            self.tails = []

        def __call__(self, rx_iq, condition, region_ids, tail, modulation=None):
            del condition, region_ids, modulation
            self.calls += 1
            self.tails.append(tail.detach().clone())
            value = 10.0 if self.calls == 1 else -10.0
            return torch.full(rx_iq.shape[:2], value), torch.empty(rx_iq.shape[:2])

    model = DummyModel()
    actions = safe_modulation_actions(num_blocks=1, device="cpu")
    identity = actions[0]
    config = ModulationConfig(num_adapter_gates=1, num_lora_scales=1)
    risky = DiscreteSafeAction(
        index=99,
        name="risky_window_action",
        modulation=ModulationState.identity(config),
        delta_norm=0.0,
        cir_alpha=0.8,
        tail_alpha=0.5,
    )
    captured = {"alphas": []}

    def fake_update(frame, logits, max_delay, previous_cir, alpha, confidence_threshold=None):
        del frame, logits, max_delay, previous_cir, confidence_threshold
        captured["alphas"].append(float(alpha))
        return torch.ones(5, dtype=torch.complex64) / torch.sqrt(torch.tensor(5.0))

    monkeypatch.setattr(windowed_discrete_ppo, "decision_directed_cir_update", fake_update)
    frame = SimpleNamespace(
        rx_symbols=torch.randn(32, dtype=torch.complex64),
        tx_symbols=torch.ones(32, dtype=torch.complex64),
        bits=torch.ones(32, dtype=torch.bool),
        adapt_mask=(torch.arange(32) < 8) | (torch.arange(32) >= 28),
        reward_mask=(torch.arange(32) >= 8) & (torch.arange(32) < 12),
        data_mask=(torch.arange(32) >= 12) & (torch.arange(32) < 28),
        model_region_ids=torch.zeros(32, dtype=torch.long),
    )
    initial_cir = torch.zeros(5, dtype=torch.complex64)
    initial_tail = torch.zeros(4, dtype=torch.complex64)
    state = WindowedDiscreteOnlineState(
        cir=initial_cir.clone(),
        receiver_state=ReceiverState(initial_tail.clone()),
        model=model,
        policy=DiscreteSafePolicy(observation_dim=4, action_count=len(actions), hidden_size=8),
        optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.zeros(()))], lr=1e-4),
        encoder=SimpleNamespace(),
        actions=[identity, risky],
        hidden=torch.zeros(1, 1, 8),
        window_size=1,
        previous_window_reward=0.0,
        last_action_delta_norm=0.0,
        rollout=[],
        cir_update_mode="decision_directed",
        cir_update_alpha=0.6,
        current_action=risky,
        current_accumulator=WindowRewardAccumulator(action_delta_norm=0.0),
        current_window_cir_snapshot=initial_cir.clone(),
        current_window_tail_snapshot=initial_tail.clone(),
        frames_in_window=1,
    )

    row = run_windowed_discrete_frame(
        state,
        frame,
        condition_from_cir(initial_cir, 10.0),
        snr_db=10.0,
        frame_index=1,
        update_interval=16,
    )

    assert row["reward_pilot_ber_before"] == 0.0
    assert row["reward_pilot_ber_after"] == 1.0
    assert row["window_reward_rollback"] is True
    assert row["safe_action_accepted"] is False
    assert row["selected_tail_update_alpha"] == 0.5
    assert torch.equal(state.cir, initial_cir)
    expected_latest_tail = torch.complex(
        0.5 * torch.tanh(torch.full_like(initial_tail.real, -10.0) / 2.0),
        torch.zeros_like(initial_tail.real),
    )
    assert torch.allclose(state.receiver_state.soft_tail, expected_latest_tail)
    assert all(
        torch.allclose(tail, expected_latest_tail.unsqueeze(0))
        for tail in model.tails[-2:]
    )
    assert captured["alphas"] == [0.8]


def test_prefix_pilots_do_not_update_unobserved_rl_tail():
    from types import SimpleNamespace

    from agent.cir_estimator import condition_from_cir
    from agent.discrete_safe_policy import DiscreteSafePolicy, safe_modulation_actions
    from env.comm_env import ReceiverState
    from training.windowed_discrete_ppo import (
        WindowRewardAccumulator,
        WindowedDiscreteOnlineState,
        run_windowed_discrete_frame,
    )

    class DummyModel:
        def __init__(self):
            self.tails = []

        def __call__(self, rx_iq, condition, region_ids, tail, modulation=None):
            del condition, region_ids, modulation
            self.tails.append(tail.detach().clone())
            logits = torch.full(rx_iq.shape[:2], 10.0)
            return logits, torch.sigmoid(logits)

    model = DummyModel()
    actions = safe_modulation_actions(num_blocks=1, device="cpu", action_names=["identity"])
    identity = actions[0]
    initial_tail = torch.full((4,), 0.2 + 0.1j, dtype=torch.complex64)
    frame = SimpleNamespace(
        rx_symbols=torch.randn(32, dtype=torch.complex64),
        tx_symbols=torch.ones(32, dtype=torch.complex64),
        bits=torch.ones(32, dtype=torch.bool),
        adapt_mask=torch.arange(32) < 8,
        reward_mask=(torch.arange(32) >= 8) & (torch.arange(32) < 12),
        data_mask=torch.arange(32) >= 12,
        model_region_ids=torch.zeros(32, dtype=torch.long),
    )
    cir = torch.zeros(5, dtype=torch.complex64)
    cir[0] = 1.0 + 0.0j
    state = WindowedDiscreteOnlineState(
        cir=cir,
        receiver_state=ReceiverState(initial_tail.clone()),
        model=model,
        policy=DiscreteSafePolicy(observation_dim=4, action_count=1, hidden_size=8),
        optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.zeros(()))], lr=1e-4),
        encoder=SimpleNamespace(),
        actions=actions,
        hidden=torch.zeros(1, 1, 8),
        window_size=4,
        previous_window_reward=0.0,
        last_action_delta_norm=0.0,
        rollout=[],
        current_action=identity,
        current_accumulator=WindowRewardAccumulator(action_delta_norm=0.0),
        frames_in_window=1,
    )

    run_windowed_discrete_frame(
        state,
        frame,
        condition_from_cir(cir, 10.0),
        snr_db=10.0,
        frame_index=1,
        update_interval=16,
    )

    assert torch.equal(state.receiver_state.soft_tail, initial_tail)
    assert all(torch.equal(tail, initial_tail.unsqueeze(0)) for tail in model.tails[-2:])


def test_identity_rl_matches_frozen_over_prefix_pilot_frames(tmp_path):
    import compare
    from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
    from env.comm_env import CommEnvConfig, CommunicationEnvironment

    torch.manual_seed(53)
    model = UnfoldedEqualizer(
        UnfoldedConfig(
            frame_len=128, max_delay=4, iterations=1, d_model=8, num_heads=2,
            pilot_conditioned=True, enable_phase_correction_branch=True,
        )
    )
    (tmp_path / "model_config.json").write_text(
        json.dumps(model.config.to_dict(), ensure_ascii=False), encoding="utf-8"
    )
    torch.save({"state_dict": model.state_dict()}, tmp_path / "model_best.pt")
    env = CommunicationEnvironment(
        CommEnvConfig(frame_len=128, max_delay=4, total_pilot=64, layout="prefix", seed=53)
    )
    start = env.reset_episode()
    methods = ("Frozen Offline NN", "RL-Modulated Neural Block Equalizer")
    states = compare._build_method_states(
        methods, start.initial_soft_tail, env.channel.last_cir_used(),
        {"model": model.config.to_dict(), "rl_action_names": ["identity"], "tail_update_alpha": 0.5},
        delay=4, snr_db=10.0, seed=53,
        pretrained_path=tmp_path / "model_best.pt", device="cpu", cir_update_mode="fixed",
    )

    for frame_index in range(1, 4):
        results = compare._run_methods_for_frame(
            methods, env.next_frame(), 10.0, states,
            delay=4, frame_index=frame_index, update_interval=8,
        )
        frozen = results[methods[0]]
        online = results[methods[1]]
        assert online.ber_data == frozen.ber_data
        assert online.extra["peft_delta_norm"] == 0.0
        assert online.extra["data_logits_abs_delta_mean_eval_only"] == 0.0
        assert torch.equal(
            states[methods[1]].online_state.receiver_state.soft_tail,
            states[methods[0]].receiver_state.soft_tail,
        )


def test_tail_refinement_reuses_current_frame_soft_tail_without_labels():
    from training.windowed_discrete_ppo import refine_logits_with_tail

    class DummyModel:
        def __init__(self):
            self.tails = []

        def __call__(self, rx_iq, condition, region_ids, tail, modulation=None):
            del condition, region_ids, modulation
            self.tails.append(tail.detach().clone())
            logits = torch.full(rx_iq.shape[:2], 2.0 + len(self.tails))
            return logits, torch.sigmoid(logits)

    model = DummyModel()
    rx_iq = torch.zeros(1, 8, 2)
    condition = object()
    region_ids = torch.zeros(1, 8, dtype=torch.long)
    initial_tail = torch.zeros(1, 3, dtype=torch.complex64)

    logits = refine_logits_with_tail(
        model,
        rx_iq,
        condition,
        region_ids,
        initial_tail,
        modulation=None,
        passes=1,
    )

    assert len(model.tails) == 2
    expected_tail = torch.complex(
        torch.tanh(torch.full((1, 3), 3.0) / 2.0),
        torch.zeros(1, 3),
    )
    assert torch.allclose(model.tails[1], expected_tail)
    assert torch.equal(logits, torch.full((1, 8), 4.0))


def test_compare_rl_modulated_uses_windowed_discrete_policy(tmp_path):
    import subprocess
    import sys
    from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer

    pretrained_dir = tmp_path / "pretrained"
    pretrained_dir.mkdir()
    model = UnfoldedEqualizer(UnfoldedConfig(frame_len=512, max_delay=40, iterations=1, d_model=24, num_heads=4))
    (pretrained_dir / "model_config.json").write_text(json.dumps(model.config.to_dict(), ensure_ascii=False), encoding="utf-8")
    torch.save({"schema_version": "unfolded-eq-v1", "state_dict": model.state_dict()}, pretrained_dir / "model_best.pt")

    subprocess.run(
        [
            sys.executable,
            "compare.py",
            "--config",
            "configs/continual_ppo.json",
            "--methods",
            "RL-Modulated Neural Block Equalizer",
            "--pretrained",
            str(pretrained_dir / "model_best.pt"),
            "--delays",
            "20",
            "--snrs",
            "10",
            "--num-seeds",
            "1",
            "--frames",
            "2",
            "--pilot-total",
            "64",
            "--pilot-layout",
            "two_block",
            "--output-dir",
            str(tmp_path / "compare"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = [json.loads(line) for line in (tmp_path / "compare" / "frame_metrics.jsonl").read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 2
    assert all(row["policy_learning"] == "windowed_discrete_safe_ppo" for row in rows)
    assert all(row["data_labels_used_online"] is False for row in rows)


def test_compare_can_enable_decision_directed_cir_update_for_proposed(tmp_path):
    import subprocess
    import sys
    from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer

    pretrained_dir = tmp_path / "pretrained"
    pretrained_dir.mkdir()
    model = UnfoldedEqualizer(UnfoldedConfig(frame_len=512, max_delay=40, iterations=1, d_model=24, num_heads=4))
    (pretrained_dir / "model_config.json").write_text(json.dumps(model.config.to_dict(), ensure_ascii=False), encoding="utf-8")
    torch.save({"schema_version": "unfolded-eq-v1", "state_dict": model.state_dict()}, pretrained_dir / "model_best.pt")

    subprocess.run(
        [
            sys.executable,
            "compare.py",
            "--config",
            "configs/continual_ppo.json",
            "--methods",
            "Frozen Offline NN",
            "RL-Modulated Neural Block Equalizer",
            "--pretrained",
            str(pretrained_dir / "model_best.pt"),
            "--delays",
            "20",
            "--snrs",
            "10",
            "--num-seeds",
            "1",
            "--frames",
            "1",
            "--pilot-total",
            "64",
            "--pilot-layout",
            "two_block",
            "--cir-update",
            "decision_directed",
            "--cir-alpha",
            "0.6",
            "--output-dir",
            str(tmp_path / "compare_cir_update"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "compare_cir_update" / "frame_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert {row["method"] for row in rows} == {"Frozen Offline NN", "RL-Modulated Neural Block Equalizer"}
    assert all(row["cir_update_mode"] == "decision_directed" for row in rows)
    assert all(row["cir_update_alpha"] == 0.6 for row in rows)
    assert all(row["cir_update_uses_data_labels"] is False for row in rows)


def test_discrete_policy_loader_skips_incompatible_action_catalog_when_optional(tmp_path):
    from agent.discrete_safe_policy import DiscreteSafePolicy, safe_modulation_actions
    from training.windowed_discrete_ppo import _load_discrete_policy_if_available

    actions = safe_modulation_actions(num_blocks=3, device="cpu")
    policy = DiscreteSafePolicy(observation_dim=4, action_count=len(actions), hidden_size=8)
    old_policy = DiscreteSafePolicy(observation_dim=4, action_count=7, hidden_size=8)
    checkpoint = tmp_path / "old_policy.pt"
    torch.save(
        {
            "schema_version": "windowed-discrete-policy-v1",
            "state_dict": old_policy.state_dict(),
            "action_names": ["identity", "old_peft"],
        },
        checkpoint,
    )

    assert _load_discrete_policy_if_available(
        policy,
        checkpoint,
        "cpu",
        required=False,
        expected_action_names=[action.name for action in actions],
    ) is False

    try:
        _load_discrete_policy_if_available(
            policy,
            checkpoint,
            "cpu",
            required=True,
            expected_action_names=[action.name for action in actions],
        )
    except ValueError as exc:
        assert "动作表不兼容" in str(exc)
    else:
        raise AssertionError("required=True 时旧 policy 必须明确报错。")
