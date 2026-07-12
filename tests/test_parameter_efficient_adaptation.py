# -*- coding: utf-8 -*-
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.adaptation_controller import AdaptationController, AdaptationStrategy
from agent.neural_equalizer import AdapterEqualizer, EqualizerConfig
from env.comm_env import CommunicationEnv, EnvConfig
from env.frame_structure import FrameConfig
from env.frame_structure import FrameGenerator
from online_train import run_online_adaptation
from online_train import evaluate_mmse
from online_train import _capture_model_state, _restore_model_state
from compare import run_snr_comparison
from pretrain import run_offline_pretraining
from pretrain import _sample_channel_config


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
