import json

import torch


def test_discrete_safe_action_catalog_is_bounded_and_has_identity():
    from agent.discrete_safe_policy import safe_modulation_actions

    actions = safe_modulation_actions(num_blocks=3, device="cpu")
    names = [action.name for action in actions]

    assert names == [
        "identity",
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
    assert actions[names.index("peft_head_light")].peft_groups == {"head"}
    assert actions[names.index("peft_head_fast")].peft_lr == 5e-4
    assert actions[names.index("peft_adapter_lora_conservative")].peft_lr == 5e-5
    assert actions[names.index("peft_adapter_lora_light")].peft_groups == {"adapter", "attention_lora", "ffn_lora"}


def test_discrete_policy_prior_prefers_safe_peft_exploration_over_rollback():
    from agent.discrete_safe_policy import DiscreteSafePolicy, initialize_safe_discrete_policy_prior, safe_modulation_actions

    actions = safe_modulation_actions(num_blocks=3, device="cpu")
    names = [action.name for action in actions]
    policy = DiscreteSafePolicy(observation_dim=4, action_count=len(actions), hidden_size=8)

    initialize_safe_discrete_policy_prior(policy, actions)

    assert policy.logits.bias[names.index("peft_head_fast")] > policy.logits.bias[names.index("identity")]
    assert policy.logits.bias[names.index("peft_adapter_lora_conservative")] >= policy.logits.bias[names.index("identity")]
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
            "Offline NN only",
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

    assert {row["method"] for row in rows} == {"Offline NN only", "RL-Modulated Neural Block Equalizer"}
    assert all(row["cir_update_mode"] == "decision_directed" for row in rows)
    assert all(row["cir_update_uses_data_labels"] is False for row in rows)
