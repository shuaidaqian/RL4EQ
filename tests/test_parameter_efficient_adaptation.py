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
from online_train import run_online_adaptation
from online_train import evaluate_mmse
from compare import run_snr_comparison
from pretrain import run_offline_pretraining


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
