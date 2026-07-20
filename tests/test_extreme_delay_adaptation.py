# -*- coding: utf-8 -*-
"""极端长时延信道与在线参数高效适配的契约测试。"""

import json
import inspect
from pathlib import Path

import numpy as np
import torch

from env.extreme_delay_channel import ExtremeDelayChannel, ExtremeDelayChannelConfig
from env.comm_env import CommunicationEnv, EnvConfig
from env.frame_structure import (
    REGION_ADAPT_PILOT,
    REGION_DATA,
    REGION_REWARD_PILOT,
    FrameConfig,
    FrameGenerator,
)
from agent.neural_equalizer import ExtremeDelayEqualizer, EqualizerConfig
from agent.adaptation_controller import OBS_DIM, AdaptationController
from agent.adaptation_policy import ACTION_TABLE, PPOPolicy
from baseline.mmse_equalizer import LMMSEFIREqualizer
from baseline.traditional_equalizers import DFERLSEqualizer
from pretrain import load_pretrained_equalizer, run_offline_pretraining
from online_train import REQUIRED_METRICS, _configure_plot_font as configure_online_font
from online_train import run_online_training
from compare import _summarize_records, run_comparison


def test_frame_masks_cover_frame_without_overlap():
    cfg = FrameConfig()

    assert cfg.frame_len == 512
    assert cfg.adapt_pilot_len == 48
    assert cfg.reward_pilot_len == 16
    assert cfg.data_len == 448
    assert int(cfg.adapt_pilot_mask.sum()) == 48
    assert int(cfg.reward_pilot_mask.sum()) == 16
    assert int(cfg.data_mask.sum()) == 448
    assert torch.equal(
        cfg.adapt_pilot_mask | cfg.reward_pilot_mask | cfg.data_mask,
        torch.ones(cfg.frame_len, dtype=torch.bool),
    )
    assert not torch.any(cfg.adapt_pilot_mask & cfg.reward_pilot_mask)
    assert not torch.any(cfg.adapt_pilot_mask & cfg.data_mask)
    assert not torch.any(cfg.reward_pilot_mask & cfg.data_mask)
    assert torch.all(cfg.region_ids[cfg.adapt_pilot_mask] == REGION_ADAPT_PILOT)
    assert torch.all(cfg.region_ids[cfg.reward_pilot_mask] == REGION_REWARD_PILOT)
    assert torch.all(cfg.region_ids[cfg.data_mask] == REGION_DATA)


def test_pilot_sequence_changes_between_frames_but_is_reproducible():
    generator = FrameGenerator(FrameConfig(), seed=20260720)

    frame_a = generator.generate(frame_index=3)
    frame_b = generator.generate(frame_index=4)
    frame_a_repeat = generator.generate(frame_index=3)

    pilot_mask = generator.config.pilot_mask
    assert torch.equal(frame_a.bits, frame_a_repeat.bits)
    assert not torch.equal(frame_a.bits[pilot_mask], frame_b.bits[pilot_mask])
    assert torch.all(frame_a.adapt_pilot_symbols[~generator.config.adapt_pilot_mask] == 0)


def test_extreme_channel_samples_sparse_delays_with_required_endpoints():
    cfg = ExtremeDelayChannelConfig(
        max_delay_symbols=40,
        min_paths=3,
        max_paths=7,
        snr_db=100.0,
        seed=17,
    )
    channel = ExtremeDelayChannel(cfg)

    channel.reset_episode()

    assert 3 <= len(channel.delays) <= 7
    assert channel.delays[0] == 0
    assert channel.delays[-1] == 40
    assert len(set(channel.delays)) == len(channel.delays)
    assert torch.isclose(channel.taps.abs().square().sum(), torch.tensor(1.0), atol=1e-5)


def test_extreme_channel_seed_is_reproducible():
    cfg = ExtremeDelayChannelConfig(max_delay_symbols=30, seed=99, snr_db=20.0)
    first = ExtremeDelayChannel(cfg)
    second = ExtremeDelayChannel(cfg)

    first.reset_episode()
    second.reset_episode()

    assert first.delays == second.delays
    assert torch.allclose(first.taps, second.taps)


def test_extreme_channel_preserves_cross_frame_isi_memory():
    cfg = ExtremeDelayChannelConfig(
        max_delay_symbols=2,
        min_paths=2,
        max_paths=2,
        snr_db=200.0,
        gauss_markov_rho=1.0,
        seed=5,
    )
    channel = ExtremeDelayChannel(cfg)
    channel.reset_episode()
    channel.set_impulse_response(delays=[0, 2], taps=torch.tensor([0.0 + 0.0j, 1.0 + 0.0j]))

    first = torch.zeros(4, 2)
    first[-1, 0] = 1.0
    second = torch.zeros(4, 2)
    channel.transmit(first, add_noise=False)
    received = channel.transmit(second, add_noise=False)

    assert torch.allclose(received[1], torch.tensor([1.0, 0.0]))


def test_environment_generates_online_frame_with_channel_metadata():
    env = CommunicationEnv(
        EnvConfig(
            frame=FrameConfig(frame_len=96, adapt_pilot_len=18, reward_pilot_len=6),
            channel=ExtremeDelayChannelConfig(max_delay_symbols=20, seed=31),
            seed=31,
        )
    )

    env.reset_episode()
    received = env.next_frame()

    assert received.rx_symbols.shape == (96, 2)
    assert received.frame.bits.shape == (96,)
    assert received.max_delay_symbols == 20
    assert received.snr_db == env.channel.snr_db
    assert torch.isfinite(received.rx_symbols).all()


def _small_equalizer() -> ExtremeDelayEqualizer:
    return ExtremeDelayEqualizer(
        EqualizerConfig(
            d_model=32,
            n_heads=4,
            n_layers=2,
            dim_feedforward=64,
            adapter_rank=4,
            max_len=128,
            dilations=(1, 2, 4),
        )
    )


def test_equalizer_outputs_frame_logits_and_probabilities():
    model = _small_equalizer()
    cfg = FrameConfig(frame_len=96, adapt_pilot_len=18, reward_pilot_len=6)
    frame = FrameGenerator(cfg, seed=7).generate(0)
    rx = torch.randn(2, cfg.frame_len, 2)

    logits, probabilities = model(
        rx,
        frame.region_ids.unsqueeze(0).expand(2, -1),
        frame.adapt_pilot_symbols.unsqueeze(0).expand(2, -1),
        frame.adapt_pilot_mask.unsqueeze(0).expand(2, -1),
    )

    assert logits.shape == (2, cfg.frame_len)
    assert probabilities.shape == logits.shape
    assert torch.all((0.0 <= probabilities) & (probabilities <= 1.0))


def test_symbols_outside_adapt_mask_cannot_change_equalizer_output():
    torch.manual_seed(8)
    model = _small_equalizer().eval()
    cfg = FrameConfig(frame_len=96, adapt_pilot_len=18, reward_pilot_len=6)
    frame = FrameGenerator(cfg, seed=8).generate(0)
    rx = torch.randn(1, cfg.frame_len, 2)
    original = frame.adapt_pilot_symbols.unsqueeze(0)
    modified = original.clone()
    modified[:, ~frame.adapt_pilot_mask] = torch.randn_like(
        modified[:, ~frame.adapt_pilot_mask]
    )

    logits_a, _ = model(
        rx,
        frame.region_ids.unsqueeze(0),
        original,
        frame.adapt_pilot_mask.unsqueeze(0),
    )
    logits_b, _ = model(
        rx,
        frame.region_ids.unsqueeze(0),
        modified,
        frame.adapt_pilot_mask.unsqueeze(0),
    )

    assert torch.allclose(logits_a, logits_b)


def test_online_tuning_only_exposes_adapters_and_output_head():
    model = _small_equalizer()

    model.set_trainable_targets(train_adapters=True, train_output=True)
    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]

    assert trainable_names
    assert all("adapter" in name or name.startswith("output_head") for name in trainable_names)
    assert model.trainable_parameter_count() < sum(param.numel() for param in model.parameters()) * 0.25
    assert model.receptive_field >= 15


def test_action_table_matches_research_design():
    assert [action.name for action in ACTION_TABLE] == [
        "skip",
        "head-light",
        "adapters-light",
        "adapters+head-light",
        "adapters+head-fast",
        "adapters+head-deep",
        "rollback-last-good",
    ]


def test_policy_outputs_valid_action_and_value():
    policy = PPOPolicy(observation_dim=OBS_DIM, action_count=len(ACTION_TABLE), hidden_dim=32)
    observation = torch.zeros(OBS_DIM)

    action, log_prob, value = policy.sample_action(observation, deterministic=True)

    assert 0 <= action < len(ACTION_TABLE)
    assert log_prob.ndim == 0
    assert value.ndim == 0


def _small_received_frame(seed: int = 41):
    env = CommunicationEnv(
        EnvConfig(
            frame=FrameConfig(frame_len=96, adapt_pilot_len=18, reward_pilot_len=6),
            channel=ExtremeDelayChannelConfig(max_delay_symbols=12, snr_db=15.0, seed=seed),
            seed=seed,
        )
    )
    env.reset_episode()
    return env.next_frame()


def test_controller_observation_is_finite_and_does_not_use_data_labels():
    torch.manual_seed(42)
    model = _small_equalizer()
    controller = AdaptationController(model, device="cpu")
    received = _small_received_frame()

    observation_a = controller.build_observation(received)
    received.frame.bits[received.frame.data_mask] = 1.0 - received.frame.bits[received.frame.data_mask]
    observation_b = controller.build_observation(received)

    assert observation_a.shape == (OBS_DIM,)
    assert torch.isfinite(observation_a).all()
    assert torch.allclose(observation_a, observation_b)


def test_observation_path_does_not_read_reward_pilot_metrics():
    source = inspect.getsource(AdaptationController.build_observation)

    assert "_adapt_metrics" in source
    assert "_pilot_metrics" not in source
    assert "reward" not in source.lower().replace("last_reward", "")


def test_skip_action_uses_reward_pilot_without_parameter_update():
    torch.manual_seed(43)
    model = _small_equalizer()
    controller = AdaptationController(model, device="cpu")
    received = _small_received_frame(seed=43)

    result = controller.adapt_frame(received, ACTION_TABLE[0])

    assert result.action_name == "skip"
    assert result.adapt_steps == 0
    assert result.adapt_params == 0
    assert result.parameter_delta_norm == 0.0
    assert abs(result.reward - (result.reward_pilot_loss_before - result.reward_pilot_loss_after)) < 1e-7


def test_controller_can_restore_episode_anchor():
    model = _small_equalizer()
    controller = AdaptationController(model, device="cpu")
    controller.start_episode()
    original = model.capture_peft_state()
    with torch.no_grad():
        model.output_head.bias.add_(3.0)

    controller.end_episode()

    restored = model.capture_peft_state()
    assert all(torch.allclose(original[name], restored[name]) for name in original)


def test_traditional_baselines_only_require_received_signal_and_adapt_pilot():
    received = _small_received_frame(seed=44)
    frame = received.frame
    lmmse = LMMSEFIREqualizer(regularization=1e-2)
    rls = DFERLSEqualizer(forgetting_factor=0.995)

    lmmse_soft = lmmse.equalize(
        received.rx_symbols,
        frame.adapt_pilot_symbols,
        frame.adapt_pilot_mask,
        max_delay_symbols=received.max_delay_symbols,
        snr_db=received.snr_db,
    )
    rls_soft = rls.equalize(
        received.rx_symbols,
        frame.adapt_pilot_symbols,
        frame.adapt_pilot_mask,
        max_delay_symbols=received.max_delay_symbols,
        snr_db=received.snr_db,
    )

    assert lmmse_soft.shape == frame.bits.shape
    assert rls_soft.shape == frame.bits.shape
    assert torch.isfinite(lmmse_soft).all()
    assert torch.isfinite(rls_soft).all()


def _tiny_model_config() -> EqualizerConfig:
    return EqualizerConfig(
        d_model=16,
        n_heads=4,
        n_layers=1,
        dim_feedforward=32,
        adapter_rank=2,
        max_len=64,
        dilations=(1, 2),
        region_dim=4,
    )


def test_offline_pretraining_writes_strictly_loadable_checkpoint(tmp_path):
    result = run_offline_pretraining(
        stage_a_steps=1,
        stage_b_steps=1,
        batch_size=1,
        seed=51,
        save_dir=tmp_path,
        device="cpu",
        frame_len=64,
        pilot_lengths=(16,),
        delay_min=4,
        delay_max=6,
        snr_min=5.0,
        snr_max=10.0,
        validation_frames=1,
        model_config=_tiny_model_config(),
        save_plots=False,
    )

    best_path = Path(result["model_best"])
    assert best_path.exists()
    assert (tmp_path / "model_final.pt").exists()
    assert (tmp_path / "model_config.json").exists()
    assert (tmp_path / "pretrain_metrics.json").exists()
    metrics = json.loads((tmp_path / "pretrain_metrics.json").read_text(encoding="utf-8"))
    assert [item["stage"] for item in metrics["validation_history"]] == ["stage_a", "stage_b"]
    assert metrics["best_validation_BER_data"] == min(
        item["BER_data"] for item in metrics["validation_history"]
    )
    loaded, payload = load_pretrained_equalizer(best_path, device="cpu")
    assert isinstance(loaded, ExtremeDelayEqualizer)
    assert payload["model"]["d_model"] == 16


def test_online_training_reports_required_metrics(tmp_path):
    pretrained_dir = tmp_path / "pretrained"
    run_offline_pretraining(
        stage_a_steps=1,
        stage_b_steps=1,
        batch_size=1,
        seed=52,
        save_dir=pretrained_dir,
        device="cpu",
        frame_len=64,
        pilot_lengths=(16,),
        delay_min=4,
        delay_max=6,
        snr_min=8.0,
        snr_max=10.0,
        validation_frames=1,
        model_config=_tiny_model_config(),
        save_plots=False,
    )

    result = run_online_training(
        pretrained=pretrained_dir / "model_best.pt",
        output_dir=tmp_path / "online",
        num_episodes=1,
        frames_per_episode=1,
        seed=53,
        device="cpu",
        save_plots=False,
    )

    assert REQUIRED_METRICS.issubset(result["summary"])
    assert Path(result["records_path"]).exists()
    assert Path(result["policy_path"]).exists()


def test_comparison_exports_all_planned_methods(tmp_path):
    pretrained_dir = tmp_path / "pretrained_compare"
    run_offline_pretraining(
        stage_a_steps=1,
        stage_b_steps=1,
        batch_size=1,
        seed=54,
        save_dir=pretrained_dir,
        device="cpu",
        frame_len=64,
        pilot_lengths=(16,),
        delay_min=4,
        delay_max=6,
        snr_min=10.0,
        snr_max=10.0,
        validation_frames=1,
        model_config=_tiny_model_config(),
        save_plots=False,
    )

    result = run_comparison(
        pretrained=pretrained_dir / "model_best.pt",
        output_dir=tmp_path / "compare",
        delays=(4,),
        snrs=(10.0,),
        num_seeds=1,
        num_frames=1,
        device="cpu",
        save_plots=False,
    )

    assert set(result["methods"]) == {
        "lmmse_fir",
        "dfe_rls",
        "pretrained",
        "fixed_peft",
        "pilot_rule",
        "ppo_peft",
        "data_oracle",
    }
    assert Path(result["records_path"]).exists()
    assert Path(result["summary_path"]).exists()


def test_comparison_confidence_interval_uses_seed_means_not_frames():
    records = [
        {"method": "pretrained", "seed": 0, "BER_data": 0.1},
        {"method": "pretrained", "seed": 0, "BER_data": 0.5},
    ]

    summary = _summarize_records(records, methods=("pretrained",))

    assert summary["pretrained"]["seed_means"] == [0.3]
    assert summary["pretrained"]["std"] == 0.0
    assert summary["pretrained"]["ci95"] == 0.0


def test_repository_contains_only_new_research_entrypoints_and_standards():
    root = Path(__file__).resolve().parents[1]
    required = [
        "AGENTS.md",
        "README.md",
        "开发框架.md",
        "RL信道均衡研究分析.md",
        "configs/extreme_delay.json",
        "configs/eval_seeds.json",
        "pretrain.py",
        "online_train.py",
        "compare.py",
    ]
    obsolete = [
        "CLAUDE.md",
        "agent/actor_critic.py",
        "agent/ppo.py",
        "env/channel_models.py",
        "env/ldpc_coding.py",
        "logs/mmse_comparison.png",
        "logs/pftnet_snr.png",
        "logs/training_curve.png",
    ]

    assert all((root / relative).exists() for relative in required)
    assert all(not (root / relative).exists() for relative in obsolete)
    framework = (root / "开发框架.md").read_text(encoding="utf-8")
    analysis = (root / "RL信道均衡研究分析.md").read_text(encoding="utf-8")
    assert "Adapt Pilot" in framework and "Reward Pilot" in framework
    assert "EME 启发" in analysis and "BER_data" in analysis


def test_online_plotting_configures_chinese_font():
    import matplotlib.pyplot as plt

    configure_online_font()

    assert plt.rcParams["axes.unicode_minus"] is False
    assert plt.rcParams["font.sans-serif"][0] in {
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Arial Unicode MS",
    }
