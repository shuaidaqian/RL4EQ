import json
import inspect
import subprocess
import sys

import torch

from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from baseline.block_equalizers import DetectionResult
from env.frame_structure import FrameConfig, FrameGenerator
from training.curriculum import build_curriculum, load_config
from training.curriculum import CurriculumTrainer, supervised_equalization_loss
import training.meta_training as meta_training
from training.meta_training import (
    FixedGate,
    MetaTrainer,
    build_meta_episode,
    evaluate_best_fixed_level_b,
    evaluate_reward_data_alignment,
    first_order_inner_update,
)


def _tiny_identity_frame(frame_index: int = 0):
    frame = FrameGenerator(
        FrameConfig(frame_len=96, total_pilot=64, layout="multi_block", max_delay=4),
        seed=123,
    ).generate(frame_index)
    cir = torch.zeros(5, dtype=torch.complex64)
    cir[0] = 1.0 + 0.0j
    tail = torch.zeros(4, dtype=torch.complex64)
    return frame.with_channel_output(frame.tx_symbols.clone(), tail, cir)


def _tiny_model() -> UnfoldedEqualizer:
    return UnfoldedEqualizer(
        UnfoldedConfig(
            frame_len=96,
            max_delay=4,
            iterations=2,
            d_model=24,
            num_heads=4,
            adapter_rank=4,
            lora_rank=4,
        )
    )


def test_curriculum_is_always_pilot_conditioned_and_level_ordered():
    schedule = build_curriculum(load_config("configs/continual_ppo.json"))
    assert [phase.name for phase in schedule] == [
        "cir_level_a",
        "perfect_cir_level_a",
        "estimated_cir_level_a",
        "estimated_cir_level_b",
    ]
    assert all(phase.total_pilot in {64, 96, 128, 160} for phase in schedule)
    assert all(phase.layout in {"prefix", "two_block", "multi_block"} for phase in schedule)
    assert all(phase.uses_pilot_condition for phase in schedule)


def test_level_b_curriculum_phase_uses_configured_pilot_total():
    config = load_config("configs/continual_ppo.json")
    config["pilot_total"] = 128
    schedule = build_curriculum(config)
    level_b = [phase for phase in schedule if phase.name == "estimated_cir_level_b"][0]

    assert level_b.total_pilot == 128


def test_level_b_curriculum_phase_cycles_main_delay_snr_grid():
    config = load_config("configs/continual_ppo.json")
    config["main_delays"] = [20, 30, 40]
    config["main_snrs"] = [0, 5, 10, 15]
    schedule = build_curriculum(config)
    level_b = [phase for phase in schedule if phase.name == "estimated_cir_level_b"][0]

    sampled = {level_b.sample_delay_snr(index) for index in range(12)}

    assert sampled == {
        (20, 0.0),
        (20, 5.0),
        (20, 10.0),
        (20, 15.0),
        (30, 0.0),
        (30, 5.0),
        (30, 10.0),
        (30, 15.0),
        (40, 0.0),
        (40, 5.0),
        (40, 10.0),
        (40, 15.0),
    }


def test_curriculum_validate_level_b_runs_real_twelve_config_gate():
    config = load_config("configs/continual_ppo.json")
    config["validation_frames_per_config"] = 1
    trainer = CurriculumTrainer(config, device=torch.device("cpu"))

    validation = trainer._validate_level_b()

    assert validation["selection_metric"] == "mean_level_b_ber_data"
    assert validation["gate"] == "perfect_cir_bpsk_refine"
    assert validation["gate_threshold"] == 0.01
    assert len(validation["per_config"]) == 12
    assert all(row["frames"] == 1 for row in validation["per_config"])
    assert not all(row["ber_data"] == 1.0 for row in validation["per_config"])
    assert isinstance(validation["gate_pass"], bool)


def test_curriculum_step_loss_uses_real_channel_frames():
    source = inspect.getsource(CurriculumTrainer._step_loss)

    assert "del phase" not in source
    assert "CommunicationEnvironment" in source
    assert "frame.rx_symbols" in source
    assert "frame.bits" in source


def test_curriculum_step_loss_samples_episode_frame_offsets():
    source = inspect.getsource(CurriculumTrainer._step_loss)

    assert "curriculum_max_frame_offset" in source
    assert "env.next_frame()" in source
    assert "frame_offset" in source


def test_curriculum_reports_offline_nn_validation_metrics():
    config = load_config("configs/continual_ppo.json")
    config["validation_frames_per_config"] = 1
    config["model_validation_frames_per_config"] = 1
    trainer = CurriculumTrainer(config, device=torch.device("cpu"))

    metrics = trainer.train(stage="cir_level_a", steps=1, batch_size=1, accumulation_steps=1, use_amp=False)

    assert "offline_nn_validation" in metrics
    validation = metrics["offline_nn_validation"]
    assert validation["metric"] == "offline_nn_ber_data"
    assert validation["rows"]
    assert all("ber_data" in row for row in validation["rows"])
    assert all(0.0 <= row["ber_data"] <= 1.0 for row in validation["rows"])


def test_level_b_curriculum_step_loss_supports_mixed_delay_batch():
    config = load_config("configs/continual_ppo.json")
    config["main_delays"] = [20, 30]
    config["main_snrs"] = [10]
    config["model"] = {
        "frame_len": 512,
        "max_delay": 40,
        "iterations": 1,
        "d_model": 24,
        "num_heads": 4,
        "adapter_rank": 4,
        "lora_rank": 4,
    }
    trainer = CurriculumTrainer(config, device=torch.device("cpu"))
    phase = [phase for phase in build_curriculum(config) if phase.name == "estimated_cir_level_b"][0]

    loss = trainer._step_loss(phase, batch_size=2)

    assert torch.isfinite(loss)


def test_curriculum_cir_condition_augmentation_cycles_configured_sources():
    config = load_config("configs/continual_ppo.json")
    config["cir_condition_augmentation"] = {
        "enabled": True,
        "modes": ["true", "acquisition", "noisy", "dd_like"],
        "noise_std": 0.05,
        "dd_flip_probability": 0.02,
        "dd_blend_alpha": 0.2,
    }
    trainer = CurriculumTrainer(config, device=torch.device("cpu"))
    phase = [phase for phase in build_curriculum(config) if phase.name == "estimated_cir_level_b"][0]
    frame = _tiny_identity_frame(frame_index=1)
    acquisition = torch.zeros(5, dtype=torch.complex64)
    acquisition[0] = 1.0 + 0.0j

    sampled = [
        trainer._select_training_cir(
            phase=phase,
            frame=frame,
            acquisition_cir=acquisition,
            max_delay=4,
            snr_db=10.0,
            sample_index=index,
        )
        for index in range(4)
    ]

    assert [item.source for item in sampled] == ["true", "acquisition", "noisy", "dd_like"]
    assert all(item.cir.shape == acquisition.shape for item in sampled)
    assert all(torch.isfinite(item.cir.real).all() and torch.isfinite(item.cir.imag).all() for item in sampled)
    assert all(torch.isclose(torch.sum(torch.abs(item.cir) ** 2), torch.tensor(1.0), atol=1e-4) for item in sampled)


def test_curriculum_train_reports_condition_cir_source_counts():
    config = load_config("configs/continual_ppo.json")
    config["model"] = {
        "frame_len": 512,
        "max_delay": 40,
        "iterations": 1,
        "d_model": 24,
        "num_heads": 4,
        "adapter_rank": 4,
        "lora_rank": 4,
    }
    config["validation_frames_per_config"] = 1
    config["model_validation_frames_per_config"] = 1
    config["cir_condition_augmentation"] = {
        "enabled": True,
        "modes": ["true", "acquisition", "noisy", "dd_like"],
        "noise_std": 0.05,
        "dd_flip_probability": 0.02,
        "dd_blend_alpha": 0.2,
    }
    trainer = CurriculumTrainer(config, device=torch.device("cpu"))

    metrics = trainer.train(stage="estimated_cir_level_b", steps=1, batch_size=4, accumulation_steps=1, use_amp=False)

    assert metrics["condition_cir_sources"] == {
        "true": 1,
        "acquisition": 1,
        "noisy": 1,
        "dd_like": 1,
    }


def test_build_meta_episode_keeps_support_query_masks_and_hides_data_bits():
    frame = _tiny_identity_frame()

    episode = build_meta_episode(frame)

    assert torch.equal(episode.support_mask.cpu(), frame.adapt_mask)
    assert torch.equal(episode.query_mask.cpu(), frame.reward_mask | frame.data_mask)
    assert torch.equal(episode.receiver_view.adapt_mask.cpu(), frame.adapt_mask)
    assert not hasattr(episode.receiver_view, "data_bits")
    assert not hasattr(episode.receiver_view, "bits")
    assert not hasattr(episode, "data_bits")


def test_first_order_inner_update_uses_only_support_and_detaches_fast_weights():
    frame = _tiny_identity_frame()
    flipped_hidden = frame.with_replaced_hidden_labels(~frame.bits, ~frame.bits)
    model = _tiny_model()

    first = first_order_inner_update(model, build_meta_episode(frame), groups={"head"}, steps=2, lr=1e-2)
    second = first_order_inner_update(model, build_meta_episode(flipped_hidden), groups={"head"}, steps=2, lr=1e-2)

    assert first.fast_weights
    assert set(first.fast_weights) == set(second.fast_weights)
    for name, value in first.fast_weights.items():
        assert value.grad_fn is None
        assert not value.requires_grad
        assert torch.allclose(value, second.fast_weights[name], atol=1e-6)
    assert set(first.snapshot.tensors) == set(first.fast_weights)


def test_fixed_gate_enumerates_legal_grid_and_saves_best(tmp_path):
    gate = FixedGate(
        save_dir=tmp_path,
        group_grid=[{"head"}, {"adapter_lora"}],
        steps_grid=[1, 2],
        iterations_grid=[2, 4],
        lr_grid=[1e-3, 3e-4],
    )

    candidates = gate.enumerate_candidates()
    best = gate.select_best(lambda candidate: 0.5 + len(candidate.groups) * 0.01 + candidate.steps * 0.001)

    assert len(candidates) == 16
    assert all(candidate.groups <= FixedGate.LEGAL_GROUPS for candidate in candidates)
    assert all(candidate.steps in {1, 2, 4} for candidate in candidates)
    assert all(candidate.iterations in {2, 4, 6, 8} for candidate in candidates)
    best_path = tmp_path / "fixed_gate" / "best_fixed.json"
    assert best_path.exists()
    payload = json.loads(best_path.read_text(encoding="utf-8"))
    assert payload["best"]["score"] == best.score
    assert payload["gate_checked"] == len(candidates)


def test_best_fixed_gate_uses_real_reward_selection_and_data_gate(tmp_path):
    config = load_config("configs/continual_ppo.json")
    result = evaluate_best_fixed_level_b(
        config,
        output_dir=tmp_path,
        frames_per_config=1,
        seeds=[0],
        cg_grid=[32],
        refine_grid=[1, 2],
    )

    assert result["gate"] == "best_fixed_acquisition_cir"
    assert result["gate_threshold"] == 0.1
    assert isinstance(result["gate_pass"], bool)
    assert len(result["per_config"]) == 12
    assert any(row["ber_data"] >= 0.1 for row in result["per_config"])
    assert result["selected"]["selection_metric"] == "mean_reward_pilot_ber"
    assert (tmp_path / "fixed_gate" / "best_fixed.json").exists()


def test_best_fixed_selection_does_not_use_data_ber_as_tiebreaker(monkeypatch, tmp_path):
    metric_state = {"last_cg": None, "metric_index": 0}

    def fake_detect(rx, cir, soft_tail, noise_variance, cg_iterations, refine_iterations):
        del cir, noise_variance, refine_iterations
        metric_state["last_cg"] = int(cg_iterations)
        metric_state["metric_index"] = 0
        logits = torch.zeros(rx.shape, dtype=torch.float32)
        return DetectionResult(
            logits=logits,
            probabilities=torch.full_like(logits, 0.5),
            soft_tail=soft_tail.detach().clone(),
            iterations=int(cg_iterations),
        )

    def fake_bit_error_rate(logits, bits):
        del logits, bits
        metric_state["metric_index"] += 1
        if metric_state["metric_index"] == 1:
            return 0.05
        return 0.90 if metric_state["last_cg"] == 8 else 0.00

    monkeypatch.setattr(meta_training, "perfect_csi_bpsk_refine_detect", fake_detect)
    monkeypatch.setattr(meta_training, "bit_error_rate", fake_bit_error_rate)

    result = evaluate_best_fixed_level_b(
        {"main_delays": [20], "main_snrs": [10], "pilot_total": 128, "pilot_layout": "multi_block"},
        output_dir=tmp_path,
        frames_per_config=1,
        seeds=[0],
        cg_grid=[8, 16],
        refine_grid=[0],
    )

    assert result["selected"]["cg_iterations"] == 8
    assert result["selected"]["mean_ber_data"] == 0.90


def test_reward_data_alignment_uses_real_grouped_pairs(tmp_path):
    config = load_config("configs/continual_ppo.json")
    result = evaluate_reward_data_alignment(
        config,
        output_dir=tmp_path,
        frames_per_config=2,
        seeds=[0, 1],
        pilot_total=128,
        pilot_layout="prefix",
        action_grid=[(16, 0), (16, 1), (32, 1), (32, 2)],
    )

    assert result["gate"] == "reward_data_spearman"
    assert result["gate_threshold"] == 0.6
    assert result["pairing"] == "grouped_by_config_and_action"
    assert result["num_pairs"] == 48
    assert -1.0 <= result["spearman"] <= 1.0
    assert isinstance(result["gate_pass"], bool)
    assert (tmp_path / "reward_alignment" / "alignment.json").exists()


def test_meta_trainer_smoke_reports_gate_and_writes_checkpoint(tmp_path):
    config = load_config("configs/continual_ppo.json")
    config["model"] = _tiny_model().config.to_dict()
    trainer = MetaTrainer(config, device=torch.device("cpu"), save_dir=tmp_path)

    metrics = trainer.train(steps=1, batch_size=1, smoke=True)
    trainer.save(tmp_path, metrics)

    assert "gate_pass" in metrics
    assert "gate_checked" in metrics
    assert metrics["gate_checked"] >= 1
    assert (tmp_path / "model_best.pt").exists()
    assert (tmp_path / "model_final.pt").exists()
    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "model_config.json").exists()
    assert (tmp_path / "pretrain_metrics.json").exists()


def test_pretrain_smoke_writes_strict_loadable_checkpoint(tmp_path):
    save_dir = tmp_path / "pretrained"
    subprocess.run(
        [
            sys.executable,
            "pretrain.py",
            "--config",
            "configs/continual_ppo.json",
            "--stage",
            "all",
            "--steps",
            "2",
            "--batch-size",
            "1",
            "--accumulation-steps",
            "1",
            "--save-dir",
            str(save_dir),
        ],
        check=True,
    )
    payload = torch.load(save_dir / "model_best.pt", map_location="cpu", weights_only=False)
    config = UnfoldedConfig.from_dict(json.loads((save_dir / "model_config.json").read_text(encoding="utf-8")))
    model = UnfoldedEqualizer(config)
    model.load_state_dict(payload["state_dict"], strict=True)
    assert (save_dir / "pretrain_metrics.json").exists()
    assert payload["schema_version"] == "unfolded-eq-v1"


def test_pretrain_meta_smoke_falls_back_when_resume_checkpoint_missing(tmp_path):
    save_dir = tmp_path / "meta_pretrained"
    missing_resume = tmp_path / "missing" / "last.pt"

    subprocess.run(
        [
            sys.executable,
            "pretrain.py",
            "--config",
            "configs/continual_ppo.json",
            "--stage",
            "meta",
            "--steps",
            "2",
            "--batch-size",
            "1",
            "--resume",
            str(missing_resume),
            "--save-dir",
            str(save_dir),
        ],
        check=True,
    )

    metrics = json.loads((save_dir / "pretrain_metrics.json").read_text(encoding="utf-8"))
    payload = torch.load(save_dir / "model_best.pt", map_location="cpu", weights_only=False)
    config = UnfoldedConfig.from_dict(json.loads((save_dir / "model_config.json").read_text(encoding="utf-8")))
    model = UnfoldedEqualizer(config)
    model.load_state_dict(payload["state_dict"], strict=True)
    assert metrics["resume_loaded"] is False
    assert "gate_pass" in metrics
    assert "gate_checked" in metrics
    assert (save_dir / "model_final.pt").exists()
    assert (save_dir / "last.pt").exists()


def test_curriculum_trainer_can_resume_strict_checkpoint(tmp_path):
    config = load_config("configs/continual_ppo.json")
    config["validation_frames_per_config"] = 1
    config["model_validation_frames_per_config"] = 1
    trainer = CurriculumTrainer(config, device=torch.device("cpu"))
    metrics = trainer.train(stage="cir_level_a", steps=1, batch_size=1, accumulation_steps=1, use_amp=False)
    trainer.save(tmp_path / "source", metrics)

    resumed = CurriculumTrainer(config, device=torch.device("cpu"))
    before = {name: value.detach().clone() for name, value in trainer.model.state_dict().items()}

    assert resumed.load_resume(tmp_path / "source" / "last.pt") is True
    after = resumed.model.state_dict()
    assert all(torch.equal(before[name], after[name]) for name in before)
    assert resumed.load_resume(tmp_path / "missing.pt") is False


def test_curriculum_save_keeps_distinct_best_and_final_checkpoints(tmp_path):
    """训练后期退化时，model_best.pt 不能被最后一步权重覆盖。"""

    config = load_config("configs/continual_ppo.json")
    config["model"] = _tiny_model().config.to_dict()
    trainer = CurriculumTrainer(config, device=torch.device("cpu"))

    best_state = {name: torch.zeros_like(value) for name, value in trainer.model.state_dict().items()}
    trainer.best_model_state_dict = {name: value.clone() for name, value in best_state.items()}
    trainer.best_model_metric = 0.01
    trainer.best_model_validation = {"mean_ber_data": 0.01}
    with torch.no_grad():
        for parameter in trainer.model.parameters():
            parameter.add_(1.0)

    trainer.save(tmp_path, {"offline_nn_validation": {"mean_ber_data": 0.2}})

    best_payload = torch.load(tmp_path / "model_best.pt", map_location="cpu", weights_only=False)
    final_payload = torch.load(tmp_path / "model_final.pt", map_location="cpu", weights_only=False)
    for name, value in best_state.items():
        assert torch.equal(best_payload["state_dict"][name], value)
        assert not torch.equal(final_payload["state_dict"][name], value)


def test_curriculum_offline_validation_can_match_online_sequence_state():
    """Offline NN validation 需要能按真实在线序列维护 soft tail 与 DD-CIR。"""

    config = load_config("configs/continual_ppo.json")
    config["model"] = _tiny_model().config.to_dict()
    config["model_validation_sequence_state"] = True
    config["model_validation_seed_base"] = 70000
    config["model_validation_frames_per_config"] = 2
    config["model_validation_seeds"] = [0]
    config["model_validation_configs"] = [
        {"level": "B", "delay": 4, "snr_db": 10.0, "pilot_total": 64, "pilot_layout": "prefix", "cir": "decision_directed", "cir_alpha": 0.4}
    ]
    trainer = CurriculumTrainer(config, device=torch.device("cpu"))

    result = trainer._validate_offline_nn()

    row = result["rows"][0]
    assert row["sequence_state"] is True
    assert row["cir"] == "decision_directed"
    assert row["cir_update_uses_data_labels"] is False
    assert row["tail_mode"] == "soft"
    assert row["frames"] == 2
    assert row["seed_base"] == 70000


def test_supervised_equalization_loss_can_add_data_margin_penalty():
    """可选 margin loss 只强化 Data 区域的低置信判决边界。"""

    logits = torch.tensor([[0.1, 4.0, -4.0, -0.1]])
    bits = torch.tensor([[True, True, False, False]])
    adapt_mask = torch.tensor([[False, True, False, False]])
    reward_mask = torch.tensor([[False, False, True, False]])
    data_mask = torch.tensor([[True, False, False, True]])

    base = supervised_equalization_loss(
        logits,
        bits,
        adapt_mask,
        reward_mask,
        data_mask,
        data_margin_loss_weight=0.0,
        data_margin=1.0,
    )
    with_margin = supervised_equalization_loss(
        logits,
        bits,
        adapt_mask,
        reward_mask,
        data_mask,
        data_margin_loss_weight=0.5,
        data_margin=1.0,
    )

    assert with_margin > base
