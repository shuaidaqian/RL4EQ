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
    assert states["Frozen Offline NN"].condition_source == "acquisition"
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


def test_window_action_rollback_does_not_rewind_dynamic_tail_state(monkeypatch):
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

        def __call__(self, rx_iq, condition, region_ids, tail, modulation=None):
            del condition, region_ids, tail, modulation
            self.calls += 1
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
        adapt_mask=torch.arange(32) < 8,
        reward_mask=(torch.arange(32) >= 8) & (torch.arange(32) < 12),
        data_mask=torch.arange(32) >= 12,
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
    assert captured["alphas"] == [0.8]


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
