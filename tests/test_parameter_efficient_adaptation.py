# -*- coding: utf-8 -*-
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.adaptation_controller import AdaptationController, AdaptationStrategy, make_strategy_table
from agent.adaptation_controller import OBS_DIM
from agent.neural_equalizer import AdapterEqualizer, EqualizerConfig
from env.comm_env import CommunicationEnv, EnvConfig
from env.frame_structure import FrameConfig
from env.frame_structure import FrameGenerator
from online_train import run_online_adaptation
from online_train import run_strategy_diagnostics
from online_train import run_pilot_overhead_study
from online_train import evaluate_mmse
from online_train import evaluate_traditional_baselines
from online_train import generate_fixed_eval_seeds
from online_train import build_oracle_action_dataset
from online_train import summarize_imitation_dataset
from online_train import mismatch_scenarios
from online_train import _build_pseudo_labels
from online_train import _build_policy_observation
from online_train import _load_pretrained_if_available
from online_train import infer_equalizer_kwargs_from_checkpoint
from online_train import policy_observation_dim
from online_train import _capture_model_state, _restore_model_state
from compare import run_snr_comparison
from compare import run_channel_baseline_comparison
from compare import run_mismatch_scenario_comparison
from compare import run_low_pilot_specialized_study
from compare import audit_rls_fairness
from pretrain import run_offline_pretraining
from pretrain import _sample_channel_config
from baseline.traditional_equalizers import make_traditional_equalizers


def _make_env(seed=7, snr_db=12.0):
    env = CommunicationEnv(EnvConfig(
        frame=FrameConfig(),
        channel=dict(num_taps=8, delay_spread=5, snr_db=snr_db, seed=seed),
        window_K=10,
        seed=seed,
    ))
    env.reset()
    return env


def test_adapter_equalizer_only_exposes_small_trainable_subset():
    model = AdapterEqualizer(EqualizerConfig(state_dim=45, d_model=16, n_heads=4, n_layers=1, adapter_rank=4))

    model.enable_parameter_efficient_tuning(train_output=True)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = model.trainable_parameter_count()
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]

    assert trainable_params < total_params * 0.25
    assert trainable_params == sum(p.numel() for p in model.trainable_parameters())
    assert any("adapter" in name for name in trainable_names)
    assert any("output_head" in name for name in trainable_names)

    states = torch.randn(2, 32, 45)
    logits, probs = model(states)
    assert logits.shape == (2, 32)
    assert probs.shape == (2, 32)


def test_frame_generator_uses_deterministic_training_and_pilot_bits():
    cfg = FrameConfig()
    gen = FrameGenerator(cfg)
    bits_a = gen.generate(np.random.default_rng(1))
    bits_b = gen.generate(np.random.default_rng(2))
    known_mask = gen.known_mask()
    data_mask = ~known_mask

    assert torch.equal(bits_a[known_mask], bits_b[known_mask])
    assert not torch.equal(bits_a[data_mask], bits_b[data_mask])


def test_environment_state_exposes_known_symbol_without_data_leakage():
    env = _make_env(seed=5)
    states = env.get_all_states()
    bits = env.get_true_bits()

    train_pos = 0
    pilot_pos = env.frame_cfg.pilot_positions[0]
    data_pos = env.frame_cfg.train_len + env.frame_cfg.pilot_len

    assert torch.allclose(states[train_pos, -3:], torch.tensor([1.0 - 2.0 * bits[train_pos].item(), 1.0, 0.0]))
    assert torch.allclose(states[pilot_pos, -3:], torch.tensor([1.0 - 2.0 * bits[pilot_pos].item(), 1.0, 0.0]))
    assert torch.allclose(states[data_pos, -3:], torch.tensor([0.0, 0.0, 1.0]))


def test_raw_state_builder_preserves_pure_neural_contract():
    env = _make_env(seed=6)
    states = env.get_all_states()
    data_pos = env.frame_cfg.train_len + env.frame_cfg.pilot_len

    assert states.shape == (env.frame_cfg.frame_len, 45)
    assert torch.allclose(states[data_pos, -3:], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.isfinite(states).all()


def test_environment_state_dim_tracks_window_k():
    env = CommunicationEnv(EnvConfig(
        frame=FrameConfig(),
        channel=dict(num_taps=16, delay_spread=10, snr_db=10.0, seed=77),
        window_K=16,
        seed=77,
    ))
    env.reset()
    states = env.get_all_states()
    assert env.config.state_dim == 69
    assert states.shape == (env.frame_cfg.frame_len, 69)


def test_environment_can_append_mmse_assisted_features():
    env = CommunicationEnv(EnvConfig(
        frame=FrameConfig(),
        channel=dict(num_taps=8, delay_spread=7, snr_db=10.0, seed=78),
        window_K=10,
        use_mmse_features=True,
        seed=78,
    ))
    env.reset()
    states = env.get_all_states()

    assert env.config.state_dim == 48
    assert states.shape == (env.frame_cfg.frame_len, 48)
    assert torch.isfinite(states[:, -6:-3]).all()
    assert torch.allclose(states[0, -3:], torch.tensor([1.0 - 2.0 * env.get_true_bits()[0].item(), 1.0, 0.0]))


def test_environment_can_apply_cfo_and_nonlinear_impairments():
    base = CommunicationEnv(EnvConfig(
        frame=FrameConfig(),
        channel=dict(num_taps=8, delay_spread=7, snr_db=30.0, seed=80),
        window_K=10,
        seed=80,
    ))
    impaired = CommunicationEnv(EnvConfig(
        frame=FrameConfig(),
        channel=dict(num_taps=8, delay_spread=7, snr_db=30.0, seed=80),
        window_K=10,
        seed=80,
        impairments=dict(cfo_norm=0.07, nonlinear_alpha=0.35),
    ))
    base.reset()
    impaired.reset()

    assert impaired.config.impairments["cfo_norm"] == 0.07
    assert torch.isfinite(impaired.get_rx_symbols()).all()
    assert not torch.allclose(base.get_rx_symbols(), impaired.get_rx_symbols())


def test_traditional_baselines_report_multiple_algorithms():
    env = _make_env(seed=79, snr_db=12.0)
    equalizers = make_traditional_equalizers(num_taps=8)
    metrics = evaluate_traditional_baselines(env, equalizers=equalizers)

    for name in ["matched_filter", "zero_forcing", "lms", "rls", "mmse"]:
        assert name in metrics
        assert 0.0 <= metrics[name]["BER_data"] <= 1.0
        assert 0.0 <= metrics[name]["BER_pilot"] <= 1.0


def test_channel_baseline_comparison_uses_all_standard_channels(tmp_path):
    results = run_channel_baseline_comparison(
        profiles=[None, "rician", "epa"],
        num_frames=1,
        seed=88,
        snr=10.0,
        output_dir=tmp_path,
    )

    assert set(results["channels"]) == {"rayleigh", "rician", "epa"}
    assert "mmse" in results["baselines"]
    assert "zero_forcing" in results["baselines"]
    assert os.path.exists(results["artifact"])


def test_fixed_eval_seeds_are_stable_and_reusable():
    first = generate_fixed_eval_seeds(base_seed=90, num_frames=4, profiles=[None, "rician"])
    second = generate_fixed_eval_seeds(base_seed=90, num_frames=4, profiles=[None, "rician"])

    assert first == second
    assert first["rayleigh"] == [90, 91, 92, 93]
    assert first["rician"] == [10090, 10091, 10092, 10093]


def test_oracle_action_dataset_can_feed_imitation_policy():
    dataset = build_oracle_action_dataset(
        num_frames=1,
        seed=91,
        snr=10.0,
        d_model=16,
        n_layers=1,
        adapter_rank=4,
        device="cpu",
        save_plots=False,
    )

    assert dataset["observations"].shape[0] == 1
    assert dataset["observations"].shape[1] == OBS_DIM
    assert dataset["labels"].shape == (1,)
    assert 0 <= int(dataset["labels"][0]) < len(make_strategy_table())


def test_imitation_dataset_summary_reports_accuracy_and_regret():
    dataset = build_oracle_action_dataset(
        num_frames=1,
        seed=93,
        snr=10.0,
        d_model=16,
        n_layers=1,
        adapter_rank=4,
        device="cpu",
        save_plots=False,
    )
    summary = summarize_imitation_dataset(dataset, predicted_actions=dataset["labels"])

    assert summary["num_samples"] == 1
    assert summary["action_accuracy"] == 1.0
    assert summary["oracle_regret"] == 0.0


def test_mismatch_scenarios_include_model_mismatch_cases():
    scenarios = mismatch_scenarios(snr=10.0)

    assert {"cfo", "doppler", "nonlinear", "low_pilot"}.issubset(set(scenarios))
    assert scenarios["cfo"]["impairments"]["cfo_norm"] != 0.0
    assert scenarios["doppler"]["channel"]["time_varying"] is True
    assert scenarios["nonlinear"]["impairments"]["nonlinear_alpha"] > 0.0
    assert scenarios["low_pilot"]["frame"].train_len < FrameConfig().train_len


def test_equalizer_can_apply_mmse_residual_correction():
    env = CommunicationEnv(EnvConfig(
        frame=FrameConfig(),
        channel=dict(num_taps=8, delay_spread=7, snr_db=10.0, seed=92),
        window_K=10,
        use_mmse_features=True,
        seed=92,
    ))
    env.reset()
    cfg = EqualizerConfig(
        state_dim=env.config.state_dim,
        d_model=16,
        n_heads=4,
        n_layers=1,
        dim_feedforward=32,
        adapter_rank=4,
        use_mmse_residual=True,
        mmse_feature_dim=3,
    )
    model = AdapterEqualizer(cfg)
    states = env.get_all_states().unsqueeze(0)

    logits, probs = model(states)

    assert hasattr(model, "residual_head")
    assert model.last_residual_correction is not None
    assert model.last_residual_correction.shape == logits.shape
    assert probs.shape == logits.shape


def test_equalizer_can_train_residual_head_without_output_head():
    cfg = EqualizerConfig(
        state_dim=48,
        d_model=16,
        n_heads=4,
        n_layers=1,
        dim_feedforward=32,
        adapter_rank=4,
        use_mmse_residual=True,
        mmse_feature_dim=3,
    )
    model = AdapterEqualizer(cfg)

    model.set_trainable_targets(train_adapter=False, train_output=False, train_sync=False, train_residual=True)
    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]

    assert trainable_names
    assert all(name.startswith("residual_head") for name in trainable_names)


def test_action_table_contains_residual_and_sync_focused_actions():
    strategies = make_strategy_table()
    names = {item.name for item in strategies}

    assert "residual-light" in names
    assert "sync-only" in names


def test_main_training_path_does_not_import_traditional_equalizer_state():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checked = [
        os.path.join(root, "pretrain.py"),
        os.path.join(root, "agent", "adaptation_controller.py"),
    ]
    for path in checked:
        text = open(path, "r", encoding="utf-8").read()
        assert "env.equalized_state" not in text
        assert "MMSEEqualizer" not in text


def test_offline_sampler_uses_variable_rayleigh_taps():
    rng = np.random.default_rng(42)
    samples = [_sample_channel_config(rng, seed=i, snr_min=10.0, snr_max=10.0) for i in range(200)]
    rayleigh_samples = [s for s in samples if s.get("type", "rayleigh") == "rayleigh" and "profile" not in s]
    tap_counts = {s.get("num_taps") for s in rayleigh_samples}

    assert len(rayleigh_samples) > 20
    assert len(tap_counts) >= 4
    assert not all(s.get("num_taps") == 16 and s.get("delay_spread") == 10 for s in rayleigh_samples)
    assert all(s.get("delay_spread") == s.get("num_taps") - 1 for s in rayleigh_samples)


def test_offline_sampler_includes_rician_channel():
    rng = np.random.default_rng(43)
    samples = [_sample_channel_config(rng, seed=i, snr_min=10.0, snr_max=10.0) for i in range(200)]
    rician_samples = [s for s in samples if s.get("type") == "rician"]

    assert rician_samples
    assert all("k_factor_db" in s for s in rician_samples)
    assert len({s.get("num_taps") for s in rician_samples}) >= 2
    assert all(s.get("delay_spread") == s.get("num_taps") - 1 for s in rician_samples)


def test_offline_sampler_respects_observation_window():
    rng = np.random.default_rng(44)
    samples = [_sample_channel_config(rng, seed=i, snr_min=10.0, snr_max=10.0, max_num_taps=11) for i in range(200)]
    neural_channels = [s for s in samples if "num_taps" in s]

    assert neural_channels
    assert all(s["num_taps"] <= 11 for s in neural_channels)


def test_rician_channel_outputs_valid_frame():
    env = CommunicationEnv(EnvConfig(
        frame=FrameConfig(),
        channel=dict(
            type="rician",
            num_taps=10,
            delay_spread=7,
            k_factor_db=6.0,
            snr_db=10.0,
            seed=123,
        ),
        window_K=10,
        seed=123,
    ))
    env.reset()

    rx = env.get_rx_symbols()
    states = env.get_all_states()
    env.set_snr(12.0)

    assert rx.shape == (env.frame_cfg.frame_len, 2)
    assert states.shape == (env.frame_cfg.frame_len, 45)
    assert torch.isfinite(rx).all()
    assert torch.isfinite(states).all()
    assert env.channel.snr_db == 12.0


def test_variable_rayleigh_channel_preserves_frame_length():
    env = CommunicationEnv(EnvConfig(
        frame=FrameConfig(),
        channel=dict(
            type="rayleigh",
            num_taps=6,
            delay_spread=20,
            snr_db=10.0,
            seed=124,
        ),
        window_K=10,
        seed=124,
    ))
    env.reset()

    assert env.get_rx_symbols().shape == (env.frame_cfg.frame_len, 2)


def test_equalizer_can_enable_neural_channel_encoder():
    cfg = EqualizerConfig(
        state_dim=45,
        d_model=16,
        n_heads=4,
        n_layers=1,
        dim_feedforward=32,
        adapter_rank=4,
        use_channel_encoder=True,
        channel_dim=8,
    )
    model = AdapterEqualizer(cfg)
    states = torch.randn(2, 32, 45)
    states[:, :16, -2] = 1.0
    states[:, :16, -3] = torch.sign(torch.randn(2, 16))

    logits, probs = model(states)

    assert hasattr(model, "channel_encoder")
    assert hasattr(model, "channel_film")
    assert logits.shape == (2, 32)
    assert probs.shape == (2, 32)


def test_equalizer_can_enable_sync_phase_delay_head():
    cfg = EqualizerConfig(
        state_dim=45,
        d_model=16,
        n_heads=4,
        n_layers=1,
        dim_feedforward=32,
        adapter_rank=4,
        use_sync_head=True,
        sync_dim=8,
        sync_delay_bins=5,
    )
    model = AdapterEqualizer(cfg)
    states = torch.randn(2, 32, 45)
    states[:, :16, -2] = 1.0
    states[:, :16, -3] = torch.sign(torch.randn(2, 16))

    logits, probs = model(states)

    assert hasattr(model, "sync_head")
    assert hasattr(model, "sync_film")
    assert logits.shape == (2, 32)
    assert probs.shape == (2, 32)


def test_equalizer_can_apply_cfo_latent_correction():
    cfg = EqualizerConfig(
        state_dim=45,
        d_model=16,
        n_heads=4,
        n_layers=1,
        dim_feedforward=32,
        adapter_rank=4,
        use_cfo_head=True,
    )
    model = AdapterEqualizer(cfg)
    states = torch.randn(2, 32, 45)
    states[:, :16, -2] = 1.0
    states[:, :16, -3] = torch.sign(torch.randn(2, 16))

    model.set_cfo_correction_strength(1.0)
    logits, probs = model(states)

    assert model.last_cfo_hat is not None
    assert model.last_cfo_hat.shape == (2,)
    assert logits.shape == (2, 32)
    assert probs.shape == (2, 32)


def test_adaptation_controller_reports_required_online_metrics():
    torch.manual_seed(1)
    np.random.seed(1)
    env = _make_env()
    model = AdapterEqualizer(EqualizerConfig(state_dim=45, d_model=16, n_heads=4, n_layers=1, adapter_rank=4))
    model.enable_parameter_efficient_tuning(train_output=True)
    controller = AdaptationController(model, frame_config=env.frame_cfg, device=torch.device("cpu"))

    result = controller.adapt_frame(
        env,
        AdaptationStrategy(name="adapter-fast", lr=1e-3, steps=1, train_adapter=True, train_output=True),
    )

    assert 0.0 <= result.ber_data <= 1.0
    assert 0.0 <= result.ber_pilot <= 1.0
    assert math.isfinite(result.pilot_loss)
    assert math.isfinite(result.reward)
    assert result.adapt_params == model.trainable_parameter_count()
    assert result.adapt_steps == 1
    assert result.latency_ms >= 0.0
    assert result.parameter_delta_norm >= 0.0


def test_adaptation_observation_uses_enhanced_pilot_only_features():
    env = _make_env()
    model = AdapterEqualizer(EqualizerConfig(state_dim=45, d_model=16, n_heads=4, n_layers=1, adapter_rank=4))
    controller = AdaptationController(model, frame_config=env.frame_cfg, device=torch.device("cpu"))

    obs = controller.build_observation(env, history={"last_latency_ms": 3.0})

    assert obs.shape == (OBS_DIM,)
    assert torch.isfinite(obs).all()


def test_pseudo_label_strategy_updates_with_pseudo_mask():
    env = _make_env()
    model = AdapterEqualizer(EqualizerConfig(state_dim=45, d_model=16, n_heads=4, n_layers=1, adapter_rank=4))
    controller = AdaptationController(model, frame_config=env.frame_cfg, device=torch.device("cpu"))
    pseudo_strategy = [item for item in make_strategy_table() if item.pseudo_label_gate != "off"][0]
    bits = env.get_true_bits()
    pseudo_mask = torch.zeros(env.frame_cfg.frame_len, dtype=torch.bool)
    pseudo_mask[env.frame_cfg.train_len + env.frame_cfg.pilot_len:env.frame_cfg.train_len + env.frame_cfg.pilot_len + 8] = True

    result = controller.adapt_frame(env, pseudo_strategy, pseudo_bits=bits, pseudo_mask=pseudo_mask)

    assert result.adapt_steps == pseudo_strategy.steps
    assert math.isfinite(result.reward)


def test_pseudo_gate_thresholds_change_selected_pseudo_labels():
    env = _make_env(seed=123)
    model = AdapterEqualizer(EqualizerConfig(state_dim=45, d_model=16, n_heads=4, n_layers=1, adapter_rank=4))
    device = torch.device("cpu")

    strict_bits, strict_mask, strict_stats = _build_pseudo_labels(
        model,
        env,
        device,
        neural_threshold=0.95,
        mmse_threshold=0.95,
        max_ratio=0.05,
        return_stats=True,
    )
    wide_bits, wide_mask, wide_stats = _build_pseudo_labels(
        model,
        env,
        device,
        neural_threshold=0.50,
        mmse_threshold=0.50,
        max_ratio=0.50,
        return_stats=True,
    )

    assert strict_bits.shape == wide_bits.shape
    assert strict_mask.sum().item() <= wide_mask.sum().item()
    assert strict_stats["pseudo_ratio"] <= wide_stats["pseudo_ratio"]
    assert 0.0 <= wide_stats["neural_mmse_disagreement"] <= 1.0


def test_action_probing_policy_observation_appends_candidate_summaries():
    env = _make_env(seed=124)
    model = AdapterEqualizer(EqualizerConfig(state_dim=45, d_model=16, n_heads=4, n_layers=1, adapter_rank=4))
    device = torch.device("cpu")
    theta_pre = _capture_model_state(model)
    strategies = make_strategy_table()
    controller = AdaptationController(model, env.frame_cfg, device)

    obs, probe_records = _build_policy_observation(
        controller,
        env,
        history={},
        model=model,
        theta_pre=theta_pre,
        strategies=strategies,
        use_sync_head=False,
        use_action_probing=True,
    )

    assert obs.shape == (policy_observation_dim(True, strategies),)
    assert len(probe_records) == len(strategies)
    assert "probe_pilot_loss" in probe_records[0]
    assert torch.isfinite(obs).all()


def test_probe_rule_selector_is_reported_by_strategy_diagnostics(tmp_path):
    results = run_strategy_diagnostics(
        num_frames=1,
        seed=126,
        snr=10.0,
        d_model=16,
        n_layers=1,
        adapter_rank=4,
        device="cpu",
        output_dir=tmp_path,
        save_plots=False,
        use_action_probing=True,
    )

    assert "probe_rule_selector" in results["methods"]
    assert results["methods"]["probe_rule_selector"]["probe_action_count"] == len(make_strategy_table())


def test_probe_aware_imitation_dataset_uses_expanded_observation():
    dataset = build_oracle_action_dataset(
        num_frames=1,
        seed=127,
        snr=10.0,
        d_model=16,
        n_layers=1,
        adapter_rank=4,
        device="cpu",
        save_plots=False,
        use_action_probing=True,
    )

    assert dataset["observations"].shape[1] == policy_observation_dim(True, make_strategy_table())


def test_disagreement_pseudo_gate_can_select_neural_confident_disagreement():
    env = _make_env(seed=125)
    device = torch.device("cpu")

    class AlwaysOneModel:
        def eval(self):
            return self

        def __call__(self, states):
            logits = torch.full(states.shape[:2], 8.0, dtype=torch.float32, device=states.device)
            return logits, torch.sigmoid(logits)

    _, pseudo_mask, stats = _build_pseudo_labels(
        AlwaysOneModel(),
        env,
        device,
        neural_threshold=0.90,
        mmse_threshold=1.00,
        max_ratio=0.25,
        gate="disagree-neural",
        return_stats=True,
    )

    assert pseudo_mask.sum().item() > 0
    assert stats["pseudo_disagreement_count"] == stats["pseudo_count"]
    assert stats["pseudo_ratio"] <= 0.25


def test_mismatched_checkpoint_returns_diagnostic_message(tmp_path):
    small = AdapterEqualizer(EqualizerConfig(state_dim=45, d_model=16, n_heads=4, n_layers=1, adapter_rank=4))
    large = AdapterEqualizer(EqualizerConfig(state_dim=45, d_model=32, n_heads=4, n_layers=1, adapter_rank=4))
    ckpt_path = tmp_path / "bad_model.pt"
    torch.save(large.state_dict(), ckpt_path)

    message = _load_pretrained_if_available(small, str(ckpt_path), torch.device("cpu"))

    assert "结构不匹配" in message


def test_checkpoint_shape_can_infer_matching_equalizer_kwargs(tmp_path):
    model = AdapterEqualizer(EqualizerConfig(
        state_dim=48,
        d_model=16,
        n_heads=4,
        n_layers=1,
        adapter_rank=4,
        use_channel_encoder=True,
        channel_dim=32,
        use_sync_head=True,
        sync_dim=16,
        sync_delay_bins=5,
        use_mmse_residual=True,
        mmse_feature_dim=3,
    ))
    ckpt_path = tmp_path / "model_best.pt"
    torch.save(model.state_dict(), ckpt_path)

    inferred = infer_equalizer_kwargs_from_checkpoint(str(ckpt_path))

    assert inferred["d_model"] == 16
    assert inferred["n_layers"] == 1
    assert inferred["adapter_rank"] == 4
    assert inferred["window_K"] == 10
    assert inferred["use_mmse_features"] is True
    assert inferred["use_channel_encoder"] is True
    assert inferred["channel_dim"] == 32
    assert inferred["use_sync_head"] is True


def test_online_adaptation_returns_peft_and_mmse_comparison_metrics(tmp_path):
    results = run_online_adaptation(
        num_frames=2,
        seed=11,
        snr=10.0,
        d_model=16,
        n_layers=1,
        adapter_rank=4,
        device="cpu",
        output_dir=tmp_path,
        save_plots=False,
        profile=None,
    )

    assert "peft" in results
    assert "mmse" in results
    for group in ("peft", "mmse"):
        assert 0.0 <= results[group]["BER_data"] <= 1.0
        assert 0.0 <= results[group]["BER_pilot"] <= 1.0
    assert results["peft"]["adapt_params"] > 0
    assert results["peft"]["adapt_steps"] >= 0
    assert results["peft"]["latency_ms"] >= 0.0
    assert "generalization" in results


def test_online_action_table_contains_cfo_control_actions():
    strategies = make_strategy_table()
    cfo_actions = [item for item in strategies if item.cfo_correction_strength > 0.0]

    assert cfo_actions
    assert any("cfo" in item.name for item in cfo_actions)


def test_strategy_diagnostics_returns_fixed_ppo_oracle_and_correlation(tmp_path):
    results = run_strategy_diagnostics(
        num_frames=1,
        seed=101,
        snr=10.0,
        d_model=16,
        n_layers=1,
        adapter_rank=4,
        device="cpu",
        output_dir=tmp_path,
        save_plots=False,
    )

    assert "mmse" in results
    assert "skip" in results["methods"]
    assert "both-fast" in results["methods"]
    assert "ppo_policy" in results["methods"]
    assert "pilot_oracle_action" in results["methods"]
    assert "data_oracle_action" in results["methods"]
    assert "delta_pilot_loss_vs_delta_ber_data" in results["correlations"]
    assert "pseudo_ratio" in results["methods"]["pseudo-both-fast"]
    assert results["methods"]["skip"]["pseudo_count"] == 0.0
    assert os.path.exists(results["artifacts"]["metrics"])


def test_pilot_overhead_study_exports_metrics(tmp_path):
    results = run_pilot_overhead_study(
        ratios=[0.5],
        num_frames=1,
        seed=102,
        snr=10.0,
        d_model=16,
        n_layers=1,
        adapter_rank=4,
        device="cpu",
        output_dir=tmp_path,
    )

    assert len(results["records"]) == 1
    assert results["records"][0]["known_ratio"] == 0.5
    assert os.path.exists(results["artifacts"]["metrics"])


def test_offline_pretraining_can_mix_low_pilot_overhead(tmp_path):
    result = run_offline_pretraining(
        num_steps=1,
        batch_size=1,
        seed=103,
        d_model=16,
        n_layers=1,
        adapter_rank=4,
        window_K=10,
        train_known_ratios=[0.25],
        save_dir=tmp_path,
        device="cpu",
        save_plots=False,
    )

    assert result["train_known_ratios"] == [0.25]
    assert os.path.exists(result["best_checkpoint"])


def test_offline_pretraining_supports_selective_distillation(tmp_path):
    result = run_offline_pretraining(
        num_steps=1,
        batch_size=1,
        seed=105,
        d_model=16,
        n_layers=1,
        adapter_rank=4,
        window_K=10,
        use_mmse_features=True,
        selective_distill_weight=0.1,
        save_dir=tmp_path,
        device="cpu",
        save_plots=False,
    )

    assert result["selective_distill_weight"] == 0.1
    assert os.path.exists(result["best_checkpoint"])


def test_mismatch_scenario_comparison_exports_metrics(tmp_path):
    results = run_mismatch_scenario_comparison(
        scenario_names=["cfo", "nonlinear"],
        num_frames=1,
        seed=104,
        snr=10.0,
        d_model=16,
        n_layers=1,
        adapter_rank=4,
        output_dir=tmp_path,
        device="cpu",
    )

    assert set(results["scenarios"]) == {"cfo", "nonlinear"}
    assert "peft" in results["metrics"]["cfo"]
    assert "mmse" in results["metrics"]["cfo"]
    assert os.path.exists(results["artifacts"]["metrics"])


def test_low_pilot_specialized_study_exports_cross_overhead_table(tmp_path):
    results = run_low_pilot_specialized_study(
        train_ratios=[0.25],
        eval_ratios=[0.25, 0.125],
        train_steps=1,
        eval_frames=1,
        seed=106,
        d_model=16,
        n_layers=1,
        adapter_rank=4,
        output_dir=tmp_path,
        device="cpu",
    )

    assert "0.25" in results["matrix"]
    assert "0.125" in results["matrix"]["0.25"]
    assert os.path.exists(results["artifacts"]["metrics"])


def test_rls_fairness_audit_is_insensitive_to_pilot_and_data_labels(tmp_path):
    result = audit_rls_fairness(
        num_frames=2,
        seed=107,
        snr=10.0,
        output_dir=tmp_path,
    )

    assert result["uses_training_only"] is True
    assert result["max_changed_output_delta"] < 1e-6
    assert "mean_BER_data" in result


def test_online_adaptation_can_restore_theta_pre_before_each_frame():
    model = AdapterEqualizer(EqualizerConfig(state_dim=45, d_model=16, n_heads=4, n_layers=1, adapter_rank=4))
    theta_pre = _capture_model_state(model)

    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.1)

    assert any(not torch.allclose(param, theta_pre[name]) for name, param in model.state_dict().items())

    _restore_model_state(model, theta_pre)

    assert all(torch.allclose(param, theta_pre[name]) for name, param in model.state_dict().items())


def test_mmse_bit_decision_matches_bpsk_mapping_at_high_snr():
    env = CommunicationEnv(EnvConfig(
        frame=FrameConfig(),
        channel=dict(num_taps=1, delay_spread=0, snr_db=30.0, seed=31),
        window_K=10,
        seed=31,
    ))
    env.reset()

    metrics = evaluate_mmse(env)

    assert metrics["BER_data"] < 0.2
    assert metrics["BER_pilot"] < 0.2


def test_snr_comparison_exports_visual_artifact(tmp_path):
    results = run_snr_comparison(
        snr_values=[8.0],
        num_frames=1,
        seed=21,
        d_model=16,
        n_layers=1,
        adapter_rank=4,
        output_dir=tmp_path,
        device="cpu",
    )

    assert results["snr_values"] == [8.0]
    assert len(results["peft_BER_data"]) == 1
    assert len(results["mmse_BER_data"]) == 1
    assert os.path.exists(results["artifact"])


def test_offline_pretraining_writes_adapter_equalizer_checkpoint(tmp_path):
    train_result = run_offline_pretraining(
        num_steps=1,
        batch_size=1,
        seed=33,
        d_model=16,
        n_layers=1,
        adapter_rank=4,
        save_dir=tmp_path,
        device="cpu",
        save_plots=False,
    )

    ckpt_path = tmp_path / "model_best.pt"
    assert ckpt_path.exists()
    assert train_result["best_checkpoint"] == str(ckpt_path)
    assert "best_data_ber" in train_result
    assert train_result["best_ber"] == train_result["best_data_ber"]

    model = AdapterEqualizer(EqualizerConfig(state_dim=45, d_model=16, n_heads=4, n_layers=1, dim_feedforward=32, adapter_rank=4))
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)

    online_result = run_online_adaptation(
        num_frames=1,
        seed=34,
        snr=10.0,
        d_model=16,
        n_layers=1,
        adapter_rank=4,
        pretrained=str(ckpt_path),
        output_dir=tmp_path / "online",
        save_plots=False,
        device="cpu",
    )
    assert 0.0 <= online_result["peft"]["BER_data"] <= 1.0
