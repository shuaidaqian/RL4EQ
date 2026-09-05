import json
import subprocess
import sys

import torch

from agent.cir_estimator import condition_from_cir
from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from env.frame_structure import FrameConfig, FrameGenerator
from training.online_meta_adaptation import (
    OnlineMetaTrainer,
    PilotMetaStepResult,
    build_meta_necessity_report,
    pilot_meta_train_step,
    run_meta_sequence_step,
)


def _frame(frame_index: int = 0):
    generated = FrameGenerator(
        FrameConfig(frame_len=96, total_pilot=64, layout="prefix", max_delay=4),
        seed=23,
    ).generate(frame_index)
    cir = torch.zeros(5, dtype=torch.complex64)
    cir[0] = 1.0 + 0.0j
    return generated.with_channel_output(generated.tx_symbols.clone(), torch.zeros(4, dtype=torch.complex64), cir)


def _model() -> UnfoldedEqualizer:
    return UnfoldedEqualizer(
        UnfoldedConfig(
            frame_len=96,
            max_delay=4,
            iterations=1,
            d_model=24,
            num_heads=4,
            adapter_rank=4,
            lora_rank=4,
            pilot_conditioned=True,
        )
    )


def _condition() -> object:
    cir = torch.zeros(1, 5, dtype=torch.complex64)
    cir[:, 0] = 1.0 + 0.0j
    return condition_from_cir(cir, snr_db=10.0)


def test_meta_step_uses_reward_pilot_as_post_adaptation_outer_target():
    torch.manual_seed(4)
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    result = pilot_meta_train_step(
        model,
        _frame(),
        _condition(),
        torch.zeros(1, 4, dtype=torch.complex64),
        optimizer,
        groups={"head"},
        inner_steps=1,
        inner_learning_rate=1e-2,
    )

    assert isinstance(result, PilotMetaStepResult)
    assert result.reward_pilot_count == 16
    assert torch.isfinite(torch.tensor(result.pre_adapt_reward_loss))
    assert torch.isfinite(torch.tensor(result.post_adapt_reward_loss))
    assert torch.isfinite(torch.tensor(result.meta_loss))
    assert result.outer_target == "reward_pilot"


def test_meta_outer_update_trains_base_model_while_inner_update_stays_peft_only():
    torch.manual_seed(8)
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    before = model.feature_proj.weight.detach().clone()

    pilot_meta_train_step(
        model,
        _frame(),
        _condition(),
        torch.zeros(1, 4, dtype=torch.complex64),
        optimizer,
        groups={"head"},
        inner_steps=1,
        inner_learning_rate=1e-2,
    )

    assert not torch.equal(before, model.feature_proj.weight.detach())
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_meta_step_is_invariant_to_hidden_data_labels():
    torch.manual_seed(7)
    original = _frame()
    hidden_changed = original.with_replaced_hidden_labels(original.bits, ~original.bits)
    first = _model()
    second = _model()
    second.load_state_dict(first.state_dict())
    first_optimizer = torch.optim.SGD(first.parameters(), lr=1e-3)
    second_optimizer = torch.optim.SGD(second.parameters(), lr=1e-3)

    result_a = pilot_meta_train_step(
        first, original, _condition(), torch.zeros(1, 4, dtype=torch.complex64), first_optimizer,
        groups={"head"}, inner_steps=1, inner_learning_rate=1e-2,
    )
    result_b = pilot_meta_train_step(
        second, hidden_changed, _condition(), torch.zeros(1, 4, dtype=torch.complex64), second_optimizer,
        groups={"head"}, inner_steps=1, inner_learning_rate=1e-2,
    )

    assert result_a.data_labels_used_online is False
    assert result_b.data_labels_used_online is False
    assert result_a.pre_adapt_reward_loss == result_b.pre_adapt_reward_loss
    for (name_a, value_a), (name_b, value_b) in zip(first.named_parameters(), second.named_parameters()):
        assert name_a == name_b
        if getattr(value_a, "_peft_group", None) == "head":
            assert torch.equal(value_a, value_b)


def test_rejected_meta_update_keeps_latest_soft_tail():
    first = _frame(0)
    second = _frame(1)
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    initial_tail = torch.zeros(4, dtype=torch.complex64)
    latest_tail = torch.full((4,), 0.75 + 0.0j, dtype=torch.complex64)

    result, tail = run_meta_sequence_step(
        model,
        first,
        _condition(),
        initial_tail,
        optimizer,
        groups={"head"},
        inner_steps=1,
        inner_learning_rate=1e-2,
        reward_guard=lambda before, after: False,
    )
    assert result.accepted is False

    _, next_tail = run_meta_sequence_step(
        model,
        second,
        _condition(),
        latest_tail,
        optimizer,
        groups={"head"},
        inner_steps=1,
        inner_learning_rate=1e-2,
        reward_guard=lambda before, after: False,
    )
    assert torch.equal(tail, initial_tail)
    assert torch.equal(next_tail, latest_tail)


def test_online_meta_trainer_uses_acquisition_condition_and_writes_metrics(tmp_path):
    config = {
        "main_delays": [4],
        "main_snrs": [10],
        "pilot_total": 64,
        "pilot_layout": "prefix",
        "impairment_profile": "clean",
        "model": _model().config.to_dict(),
        "online_meta_training": True,
        "meta_sequence_frames": 2,
        "meta_inner_steps": 1,
        "meta_inner_learning_rate": 1e-3,
        "meta_peft_groups": ["head"],
    }
    trainer = OnlineMetaTrainer(config, torch.device("cpu"), tmp_path)

    metrics = trainer.train(steps=1, batch_size=1, smoke=True)

    assert metrics["stage"] == "online_meta"
    assert metrics["condition_source"] == "acquisition_pilot"
    assert metrics["sequence_frames"] == 2
    assert len(metrics["history"]) == 2
    assert all(item["data_labels_used_online"] is False for item in metrics["history"])


def test_pretrain_cli_supports_online_meta_stage(tmp_path):
    config_path = tmp_path / "online_meta_config.json"
    config_path.write_text(
        json.dumps(
            {
                "main_delays": [4],
                "main_snrs": [10],
                "pilot_total": 64,
                "pilot_layout": "prefix",
                "impairment_profile": "clean",
                "model": _model().config.to_dict(),
                "online_meta_training": True,
                "meta_sequence_frames": 1,
                "meta_inner_steps": 1,
                "meta_inner_learning_rate": 1e-3,
                "meta_peft_groups": ["head"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    save_dir = tmp_path / "checkpoint"

    subprocess.run(
        [
            sys.executable,
            "pretrain.py",
            "--config",
            str(config_path),
            "--stage",
            "online_meta",
            "--steps",
            "1",
            "--save-dir",
            str(save_dir),
        ],
        check=True,
    )

    metrics = json.loads((save_dir / "pretrain_metrics.json").read_text(encoding="utf-8"))
    assert metrics["stage"] == "online_meta"
    assert (save_dir / "model_best.pt").exists()


def test_meta_necessity_report_has_paired_frame_bins():
    """元学习只有在后期和 heldout edge 的配对 Data BER 都改善时才推荐。"""

    rows = []
    for frame in range(1, 21):
        for method, ber in (
            ("Frozen Offline NN", 0.20 - 0.001 * frame),
            ("Pilot-SGD", 0.19 - 0.002 * frame),
            ("Meta-Pilot", 0.18 - 0.003 * frame),
        ):
            rows.append(
                {
                    "method": method,
                    "delay": 116,
                    "snr_db": 10.0,
                    "pilot_total": 128,
                    "pilot_layout": "prefix",
                    "seed": 0,
                    "frame": frame,
                    "state_split": "heldout_edge" if frame > 10 else "offline_train",
                    "ber_data": ber,
                    "data_labels_used_online": False,
                }
            )

    report = build_meta_necessity_report(rows)

    assert set(report["methods"]) == {"Frozen Offline NN", "Pilot-SGD", "Meta-Pilot"}
    assert set(report["bins"]) == {"early", "middle", "late"}
    assert report["bins"]["late"]["Meta-Pilot"]["count"] == 7
    assert report["heldout_edge"]["Meta-Pilot"]["count"] == 10
    assert report["paired"]["Meta-Pilot_vs_Pilot-SGD"]["late"]["mean_ber_delta"] < 0.0
    assert report["recommended"] is True
    assert report["data_labels_used_online"] is False
