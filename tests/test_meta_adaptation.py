import json
import subprocess
import sys

import torch

from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from env.frame_structure import FrameConfig, FrameGenerator
from training.curriculum import build_curriculum, load_config
from training.curriculum import CurriculumTrainer
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


def test_curriculum_validate_level_b_runs_real_nine_config_gate():
    config = load_config("configs/continual_ppo.json")
    config["validation_frames_per_config"] = 1
    trainer = CurriculumTrainer(config, device=torch.device("cpu"))

    validation = trainer._validate_level_b()

    assert validation["selection_metric"] == "mean_level_b_ber_data"
    assert validation["gate"] == "perfect_cir_bpsk_refine"
    assert validation["gate_threshold"] == 0.01
    assert len(validation["per_config"]) == 9
    assert all(row["frames"] == 1 for row in validation["per_config"])
    assert not all(row["ber_data"] == 1.0 for row in validation["per_config"])
    assert isinstance(validation["gate_pass"], bool)


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
    assert result["gate_pass"] is True
    assert len(result["per_config"]) == 9
    assert all(row["ber_data"] < 0.1 for row in result["per_config"])
    assert result["selected"]["selection_metric"] == "mean_reward_pilot_ber"
    assert (tmp_path / "fixed_gate" / "best_fixed.json").exists()


def test_reward_data_alignment_uses_real_grouped_pairs(tmp_path):
    config = load_config("configs/continual_ppo.json")
    result = evaluate_reward_data_alignment(
        config,
        output_dir=tmp_path,
        frames_per_config=2,
        seeds=[0, 1],
        pilot_total=128,
        pilot_layout="two_block",
        action_grid=[(16, 0), (16, 1), (32, 1), (32, 2)],
    )

    assert result["gate"] == "reward_data_spearman"
    assert result["gate_threshold"] == 0.6
    assert result["pairing"] == "grouped_by_config_and_action"
    assert result["num_pairs"] == 36
    assert result["spearman"] >= 0.6
    assert result["gate_pass"] is True
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
