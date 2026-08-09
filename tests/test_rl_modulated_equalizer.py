import torch


def test_modulation_state_round_trip_and_bounds():
    from agent.modulation import ModulationConfig, ModulationState

    config = ModulationConfig(num_adapter_gates=3, num_lora_scales=2)
    raw = torch.tensor([10.0, -10.0, 0.0, 0.5, -0.5, 2.0, -2.0, 1.0, -1.0])
    state = ModulationState.from_raw(raw, config)

    assert state.adapter_gates.shape == (3,)
    assert state.lora_scales.shape == (2,)
    assert torch.all(state.adapter_gates >= 0.0)
    assert torch.all(state.adapter_gates <= 2.0)
    assert -0.5 <= float(state.film_residual_scale) <= 0.5
    assert 0.5 <= float(state.head_temperature) <= 2.0
    assert -1.0 <= float(state.head_bias) <= 1.0
    assert torch.allclose(ModulationState.from_vector(state.to_vector(), config).to_vector(), state.to_vector())


def test_bpsk_logit_and_bce_convention_are_consistent():
    import torch.nn.functional as F
    from baseline.block_equalizers import bit_error_rate
    from env.frame_structure import _bpsk

    bits = torch.tensor([False, True, True, False])
    symbols = _bpsk(bits)
    logits = torch.tensor([-8.0, 8.0, 8.0, -8.0])
    wrong_logits = -logits

    assert torch.equal(symbols.real, torch.tensor([-1.0, 1.0, 1.0, -1.0]))
    assert bit_error_rate(logits, bits) == 0.0
    assert bit_error_rate(wrong_logits, bits) == 1.0
    assert F.binary_cross_entropy_with_logits(logits, bits.float()) < F.binary_cross_entropy_with_logits(wrong_logits, bits.float())


def test_unfolded_equalizer_accepts_identity_modulation_without_shape_change():
    from agent.cir_estimator import CIRCondition
    from agent.modulation import ModulationConfig, ModulationState
    from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer

    model = UnfoldedEqualizer(
        UnfoldedConfig(frame_len=64, max_delay=4, iterations=1, d_model=24, num_heads=4, adapter_rank=4, lora_rank=4)
    )
    rx_iq = torch.zeros(1, 64, 2)
    region_ids = torch.zeros(1, 64, dtype=torch.long)
    soft_tail = torch.zeros(1, 4, dtype=torch.complex64)
    cir = torch.zeros(1, 5, dtype=torch.complex64)
    cir[:, 0] = 1.0 + 0.0j
    condition = CIRCondition(
        complex_cir=cir,
        support_probability=torch.ones(1, 5),
        noise_variance=torch.ones(1) * 0.01,
        confidence=torch.ones(1),
        latent_residual=torch.zeros(1, 96),
    )
    modulation = ModulationState.identity(ModulationConfig(num_adapter_gates=3, num_lora_scales=3))

    logits, probabilities = model(rx_iq, condition, region_ids, soft_tail, modulation=modulation)

    assert logits.shape == (1, 64)
    assert probabilities.shape == (1, 64)
    assert torch.isfinite(logits).all()


def test_head_temperature_and_bias_change_logits():
    from agent.cir_estimator import CIRCondition
    from agent.modulation import ModulationConfig, ModulationState
    from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer

    torch.manual_seed(7)
    model = UnfoldedEqualizer(
        UnfoldedConfig(frame_len=32, max_delay=3, iterations=1, d_model=24, num_heads=4, adapter_rank=4, lora_rank=4)
    )
    rx_iq = torch.randn(1, 32, 2) * 0.1
    region_ids = torch.zeros(1, 32, dtype=torch.long)
    soft_tail = torch.zeros(1, 3, dtype=torch.complex64)
    cir = torch.zeros(1, 4, dtype=torch.complex64)
    cir[:, 0] = 1.0 + 0.0j
    condition = CIRCondition(cir, torch.ones(1, 4), torch.ones(1) * 0.01, torch.ones(1), torch.zeros(1, 96))
    config = ModulationConfig(num_adapter_gates=3, num_lora_scales=3)
    base = ModulationState.identity(config)
    shifted = ModulationState.from_vector(torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.0]), config)

    logits_base, _ = model(rx_iq, condition, region_ids, soft_tail, modulation=base)
    logits_shifted, _ = model(rx_iq, condition, region_ids, soft_tail, modulation=shifted)

    assert not torch.equal(logits_base, logits_shifted)


def test_fixed_modulation_and_policy_prior_start_near_identity():
    from agent.modulation import ModulationConfig, ModulationState
    from agent.rl_modulator import ContinuousModulationPolicy, ModulationObservationEncoder, initialize_identity_policy_prior
    from compare import _fixed_modulation_for

    config = ModulationConfig(num_adapter_gates=3, num_lora_scales=3)
    fixed = _fixed_modulation_for("NN + Fixed Modulation", config, "cpu")
    identity = ModulationState.identity(config)
    assert torch.allclose(fixed.to_vector(), identity.to_vector())

    encoder = ModulationObservationEncoder()
    policy = ContinuousModulationPolicy(len(encoder.FIELDS), config)
    initialize_identity_policy_prior(policy)
    observation = torch.zeros(1, len(encoder.FIELDS))
    action, _log_prob, _value, _hidden = policy.sample(observation, policy.initial_hidden(batch_size=1))

    assert torch.allclose(action.state.to_vector(), identity.to_vector(), atol=0.15)


def test_rl_modulated_online_runner_reports_metrics_without_data_reward(tmp_path):
    from training.rl_modulated_online import run_rl_modulated_online

    result = run_rl_modulated_online(
        config_path="configs/continual_ppo.json",
        frames=1,
        num_seeds=1,
        output_dir=tmp_path,
        delays=[20],
        snrs=[10],
        pilot_total=64,
        pilot_layout="multi_block",
    )

    assert result["schema_version"] == "rl-modulated-online-v1"
    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert row["method"] == "RL-Modulated Neural Block Equalizer"
    assert row["policy_learning"] == "continuous_modulation_ppo"
    assert row["data_labels_used_online"] is False
    assert "data_loss_used_for_reward" not in row
    policy_payload = torch.load(tmp_path / "policy.pt", map_location="cpu", weights_only=False)
    assert policy_payload["schema_version"] == "continuous-modulation-policy-v1"
    assert policy_payload["state_dict"]


def test_compare_proposed_group_uses_pretrained_and_resume_without_duplicates(tmp_path):
    import json
    import subprocess
    import sys
    from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer

    pretrained_dir = tmp_path / "pretrained"
    pretrained_dir.mkdir()
    model = UnfoldedEqualizer(UnfoldedConfig(frame_len=512, max_delay=40, iterations=1, d_model=24, num_heads=4))
    (pretrained_dir / "model_config.json").write_text(json.dumps(model.config.to_dict(), ensure_ascii=False), encoding="utf-8")
    torch.save({"schema_version": "unfolded-eq-v1", "state_dict": model.state_dict()}, pretrained_dir / "model_best.pt")

    command = [
        sys.executable,
        "compare.py",
        "--config",
        "configs/continual_ppo.json",
        "--method-group",
        "proposed",
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
        "multi_block",
        "--pretrained",
        str(pretrained_dir / "model_best.pt"),
        "--resume",
        "--output-dir",
        str(tmp_path / "compare"),
    ]
    subprocess.run(command, check=True, text=True, capture_output=True)
    subprocess.run(command, check=True, text=True, capture_output=True)

    rows = [
        json.loads(line)
        for line in (tmp_path / "compare" / "frame_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    keys = [(row["method"], row["delay"], row["snr_db"], row["seed"], row["frame"], row["pilot_total"], row["pilot_layout"]) for row in rows]

    assert len(rows) == 5
    assert len(keys) == len(set(keys))
    assert all(row["pretrained_loaded"] is True for row in rows)
    assert any(row["method"] == "RL-Modulated Neural Block Equalizer" for row in rows)


def test_compare_rl_modulated_loads_policy_checkpoint(tmp_path):
    import json
    import subprocess
    import sys
    from agent.discrete_safe_policy import DiscreteSafePolicy, initialize_safe_discrete_policy_prior, safe_modulation_actions
    from agent.rl_modulator import ModulationObservationEncoder
    from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer

    pretrained_dir = tmp_path / "pretrained"
    pretrained_dir.mkdir()
    model = UnfoldedEqualizer(UnfoldedConfig(frame_len=512, max_delay=40, iterations=1, d_model=24, num_heads=4))
    (pretrained_dir / "model_config.json").write_text(json.dumps(model.config.to_dict(), ensure_ascii=False), encoding="utf-8")
    torch.save({"schema_version": "unfolded-eq-v1", "state_dict": model.state_dict()}, pretrained_dir / "model_best.pt")

    actions = safe_modulation_actions(len(model.blocks), device="cpu")
    encoder = ModulationObservationEncoder()
    policy = DiscreteSafePolicy(len(encoder.FIELDS), len(actions))
    initialize_safe_discrete_policy_prior(policy)
    policy_path = tmp_path / "policy.pt"
    torch.save(
        {
            "schema_version": "windowed-discrete-policy-v1",
            "state_dict": policy.state_dict(),
            "action_names": [action.name for action in actions],
        },
        policy_path,
    )

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
            "--policy",
            str(policy_path),
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
            "multi_block",
            "--output-dir",
            str(tmp_path / "compare_policy"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "compare_policy" / "frame_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert len(rows) == 1
    assert rows[0]["policy_loaded"] is True


def test_compare_rejects_empty_policy_checkpoint_when_policy_is_explicit(tmp_path):
    import json
    import subprocess
    import sys
    from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer

    pretrained_dir = tmp_path / "pretrained"
    pretrained_dir.mkdir()
    model = UnfoldedEqualizer(UnfoldedConfig(frame_len=512, max_delay=40, iterations=1, d_model=24, num_heads=4))
    (pretrained_dir / "model_config.json").write_text(json.dumps(model.config.to_dict(), ensure_ascii=False), encoding="utf-8")
    torch.save({"schema_version": "unfolded-eq-v1", "state_dict": model.state_dict()}, pretrained_dir / "model_best.pt")

    policy_path = tmp_path / "empty_policy.pt"
    torch.save({"schema_version": "continuous-modulation-policy-v1", "state_dict": {}}, policy_path)

    result = subprocess.run(
        [
            sys.executable,
            "compare.py",
            "--config",
            "configs/continual_ppo.json",
            "--methods",
            "RL-Modulated Neural Block Equalizer",
            "--pretrained",
            str(pretrained_dir / "model_best.pt"),
            "--policy",
            str(policy_path),
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
            "multi_block",
            "--output-dir",
            str(tmp_path / "compare_empty_policy"),
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "state_dict" in result.stderr


def test_compare_rejects_empty_policy_checkpoint_with_equals_form(tmp_path):
    import json
    import subprocess
    import sys
    from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer

    pretrained_dir = tmp_path / "pretrained"
    pretrained_dir.mkdir()
    model = UnfoldedEqualizer(UnfoldedConfig(frame_len=512, max_delay=40, iterations=1, d_model=24, num_heads=4))
    (pretrained_dir / "model_config.json").write_text(json.dumps(model.config.to_dict(), ensure_ascii=False), encoding="utf-8")
    torch.save({"schema_version": "unfolded-eq-v1", "state_dict": model.state_dict()}, pretrained_dir / "model_best.pt")

    policy_path = tmp_path / "empty_policy_equals.pt"
    torch.save({"schema_version": "continuous-modulation-policy-v1", "state_dict": {}}, policy_path)

    result = subprocess.run(
        [
            sys.executable,
            "compare.py",
            "--config",
            "configs/continual_ppo.json",
            "--methods",
            "RL-Modulated Neural Block Equalizer",
            "--pretrained",
            str(pretrained_dir / "model_best.pt"),
            f"--policy={policy_path}",
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
            "multi_block",
            "--output-dir",
            str(tmp_path / "compare_empty_policy_equals"),
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "state_dict" in result.stderr


def test_compare_rl_modulated_is_reproducible_for_same_seed(tmp_path):
    import json
    import subprocess
    import sys
    from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer

    pretrained_dir = tmp_path / "pretrained"
    pretrained_dir.mkdir()
    model = UnfoldedEqualizer(UnfoldedConfig(frame_len=512, max_delay=40, iterations=1, d_model=24, num_heads=4))
    (pretrained_dir / "model_config.json").write_text(json.dumps(model.config.to_dict(), ensure_ascii=False), encoding="utf-8")
    torch.save({"schema_version": "unfolded-eq-v1", "state_dict": model.state_dict()}, pretrained_dir / "model_best.pt")

    def run_once(name: str) -> dict:
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
                "1",
                "--pilot-total",
                "64",
                "--pilot-layout",
                "multi_block",
                "--output-dir",
                str(tmp_path / name),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        return json.loads((tmp_path / name / "frame_metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])

    first = run_once("compare_repro_a")
    second = run_once("compare_repro_b")

    assert first["ber_data"] == second["ber_data"]
    assert first["reward"] == second["reward"]
    assert first["modulation_delta_norm"] == second["modulation_delta_norm"]
