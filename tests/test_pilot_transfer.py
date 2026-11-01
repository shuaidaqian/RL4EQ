# -*- coding: utf-8 -*-
"""验证真实在线更新一致性、隐藏信息隔离和离线外层梯度。"""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from agent.cir_estimator import condition_from_cir
from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from env.frame_structure import FrameConfig, FrameGenerator
from training.pilot_transfer import (
    PilotTransferTrainer, TransferContext, context_logits, differentiable_adapt, next_pilot_tail,
)
from training.windowed_discrete_ppo import _apply_adapt_peft_update


def _fixture():
    torch.manual_seed(91)
    model = UnfoldedEqualizer(UnfoldedConfig(
        frame_len=128, max_delay=4, iterations=1, d_model=16, num_heads=4,
        pilot_conditioned=True, enable_phase_correction_branch=True,
        phase_correction_initial_scale=1.0,
    )).eval()
    model.attach_online_phase_trend_adapter()
    frame = FrameGenerator(FrameConfig(frame_len=128, total_pilot=64, max_delay=4), seed=7).generate(0)
    positions = torch.arange(128)
    frame = replace(frame, rx_symbols=frame.tx_symbols * torch.exp(1j * (0.3 + 2 * torch.pi * 0.001 * positions)))
    cir = torch.tensor([1, 0, 0, 0, 0], dtype=torch.complex64)
    phase = torch.tensor([0.2, 0.0008, 0, 0] * 4)
    condition = condition_from_cir(cir, 10, phase_features=phase)
    return model, TransferContext(frame, condition, torch.zeros(4, dtype=torch.complex64))


@pytest.mark.parametrize("steps,radius", [(1, 0.05), (5, 0.05), (5, 0.001)])
def test_differentiable_adam_matches_actual_ppo_update(steps, radius):
    """高阶训练模拟的候选必须与真实窗口 PPO 更新数值一致。"""
    model, context = _fixture()
    # 重新构造以保留 Parameter 上的 PEFT 分组属性；deepcopy 不保留该属性。
    actual, _ = _fixture()
    fast, _ = differentiable_adapt(
        model, [context], learning_rate=0.0025, steps=steps,
        max_delta_norm=radius, create_graph=False,
    )
    _apply_adapt_peft_update(
        actual, context.frame, context.condition, context.tail,
        groups={"phase_trend"}, lr=0.0025, steps=steps,
        objective="pilot_signal_reconstruction", max_delta_norm=radius,
    )
    for name, value in actual.named_parameters():
        torch.testing.assert_close(fast[name], value, atol=1e-7, rtol=2e-5)


def test_inner_update_ignores_reward_data_labels_and_waveforms():
    model, context = _fixture()
    frame = context.frame
    hidden = ~frame.adapt_mask
    bits, tx, rx = frame.bits.clone(), frame.tx_symbols.clone(), frame.rx_symbols.clone()
    bits[hidden] = ~bits[hidden]
    tx[hidden] = -tx[hidden]
    rx[hidden] = 1000 + 1000j
    poisoned = replace(context, frame=replace(frame, bits=bits, tx_symbols=tx, rx_symbols=rx))
    results = [differentiable_adapt(
        model, [item], learning_rate=0.0025, steps=5, max_delta_norm=0.05,
    )[0] for item in (context, poisoned)]
    for name in results[0]:
        torch.testing.assert_close(results[0][name], results[1][name], rtol=0, atol=0)


@pytest.mark.parametrize("group", ["head", "adapter_lora_head", "phase"])
def test_neural_bce_inner_matches_formal_ppo_and_hides_unknown_information(group):
    """神经参数 BCE 内循环须与已有 PPO 动作一致，也不能通过注意力读取隐藏波形。"""
    model, context = _fixture()
    actual, _ = _fixture()
    fast, _ = differentiable_adapt(
        model, [context], learning_rate=0.0005, steps=1, max_delta_norm=0.05,
        groups={group}, objective="bce",
    )
    _apply_adapt_peft_update(actual, context.frame, context.condition, context.tail,
                            groups={group}, lr=0.0005, steps=1, objective="bce", max_delta_norm=0.05)
    for name, parameter in actual.named_parameters():
        torch.testing.assert_close(fast[name], parameter, atol=1e-7, rtol=2e-5)
    frame = context.frame
    hidden = ~frame.adapt_mask
    tx, rx, bits = frame.tx_symbols.clone(), frame.rx_symbols.clone(), frame.bits.clone()
    tx[hidden], rx[hidden], bits[hidden] = -tx[hidden], 1000 + 1000j, ~bits[hidden]
    poisoned = replace(context, frame=replace(frame, tx_symbols=tx, rx_symbols=rx, bits=bits))
    changed, _ = differentiable_adapt(model, [poisoned], learning_rate=0.0005,
                                     steps=1, max_delta_norm=0.05, groups={group}, objective="bce")
    for name in fast:
        torch.testing.assert_close(fast[name], changed[name], atol=0, rtol=0)


def test_head_post_update_outer_gradient_matches_finite_difference():
    model, context = _fixture()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith("online_phase_trend_adapter."))

    def loss():
        parameters, _ = differentiable_adapt(model, [context], learning_rate=0.0005,
                                             steps=1, max_delta_norm=0.05, create_graph=True,
                                             groups={"head"}, objective="bce")
        logits = context_logits(model, context, parameters)
        mask = context.frame.data_mask
        return F.binary_cross_entropy_with_logits(logits[mask], context.frame.bits[mask].float())

    gradient, = torch.autograd.grad(loss(), model.head.bias)
    initial = model.head.bias.detach().clone()
    values = []
    for sign in (-1, 1):
        with torch.no_grad():
            model.head.bias.copy_(initial + sign * 0.002)
        values.append(float(loss().detach()))
    finite_difference = (values[1] - values[0]) / 0.004
    assert float(gradient) == pytest.approx(finite_difference, abs=8e-5, rel=0.03)


def test_adaptation_has_no_inplace_parameter_changes():
    model, context = _fixture()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    fast, _ = differentiable_adapt(model, [context], learning_rate=0.0025, steps=2, max_delta_norm=0.05)
    assert not torch.equal(fast["online_phase_trend_adapter.raw_phase"], before["online_phase_trend_adapter.raw_phase"])
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_prefix_pilot_does_not_update_unobserved_tail_positions():
    """前缀没有覆盖帧尾时，严格保留历史尾状态，不能外推未知尾部。"""
    model, context = _fixture()
    context = replace(context, tail=torch.tensor([0.2, -0.3, 0.5, -0.8], dtype=torch.complex64))
    torch.testing.assert_close(next_pilot_tail(model, context, alpha=0.5), context.tail, rtol=0, atol=0)


def test_boundary_free_reconstruction_does_not_depend_on_unknown_tail():
    """完整已知前缀保留作卷积输入；仅损失评分窗跳过信道记忆长度。"""
    model, context = _fixture()
    cir = torch.tensor([0.8, 0.0, 0.0, 0.0, 0.6], dtype=torch.complex64)
    condition = replace(context.condition, complex_cir=cir.reshape(1, -1))
    context = replace(context, condition=condition)
    poisoned = replace(context, tail=torch.full_like(context.tail, 1000 + 1000j))
    updates = [differentiable_adapt(
        model, [item], learning_rate=0.0025, steps=5, max_delta_norm=0.05,
        score_start=model.config.max_delay,
    ) for item in (context, poisoned)]
    assert updates[0][1]["adapt_losses"] == updates[1][1]["adapt_losses"]
    for name in updates[0][0]:
        torch.testing.assert_close(updates[0][0][name], updates[1][0][name], rtol=0, atol=0)


def test_post_update_data_gradient_reaches_base_through_update_operator():
    """对基础相位尺度的外层导数与完整更新后的有限差分相符。"""
    model, context = _fixture()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name == "phase_correction_scale")

    def loss_at_current_scale():
        fast, _ = differentiable_adapt(
            model, [context], learning_rate=0.0025, steps=2,
            max_delta_norm=0.05, create_graph=True,
        )
        logits = context_logits(model, context, fast)
        mask = context.frame.data_mask
        return F.binary_cross_entropy_with_logits(logits[mask], context.frame.bits[mask].float())

    loss = loss_at_current_scale()
    derivative, = torch.autograd.grad(loss, model.phase_correction_scale)
    initial = float(model.phase_correction_scale.detach())
    samples = []
    epsilon = 0.002
    for offset in (-epsilon, epsilon):
        with torch.no_grad():
            model.phase_correction_scale.fill_(initial + offset)
        samples.append(float(loss_at_current_scale().detach()))
    finite_difference = (samples[1] - samples[0]) / (2 * epsilon)
    assert torch.isfinite(derivative)
    assert abs(float(derivative)) > 1e-6
    assert float(derivative) == pytest.approx(finite_difference, abs=8e-5, rel=0.03)


def test_saved_base_checkpoint_strict_loads_without_temporary_adapter(tmp_path):
    model, _ = _fixture()
    config = {"model": model.config.to_dict(), "pilot_layout": "prefix", "main_level": "B",
              "offline_state_split": "offline_train", "pilot_transfer": {}}
    trainer = PilotTransferTrainer(config, "cpu")
    trainer.save(tmp_path, {"objective": "post_adapt"})
    payload = torch.load(tmp_path / "model_best.pt", weights_only=False)
    standalone = UnfoldedEqualizer(model.config)
    standalone.load_state_dict(payload["state_dict"], strict=True)
    assert not any(name.startswith("online_phase_trend_adapter.") for name in payload["state_dict"])
    assert payload["selected_by"] == "fixed_training_budget"


def test_training_rejects_heldout_split():
    model, _ = _fixture()
    config = {"model": model.config.to_dict(), "pilot_layout": "prefix", "main_level": "B",
              "offline_state_split": "heldout_edge", "pilot_transfer": {}}
    with pytest.raises(ValueError, match="offline_train"):
        PilotTransferTrainer(config, "cpu")


def test_four_arm_summary_uses_seed_means_and_reports_interaction():
    from evaluation.pilot_transfer import summarize_transfer

    rows = []
    for branch, gains in (("baseline", [0.01, -0.01]), ("post_adapt", [0.03, 0.01])):
        for seed, gain in enumerate(gains):
            for frame_index in range(3):
                rows.append({"branch": branch, "snr_db": 5, "seed": seed, "frame": frame_index,
                             "ber_frozen": 0.1, "ber_online": 0.1 - gain,
                             "ber_before_update": 0.1, "ber_candidate": 0.1 - gain, "accepted": True})
    result, = summarize_transfer(rows)
    assert result["post_adapt"]["gain"]["seeds"] == 2
    assert result["post_adapt"]["gain"]["mean"] == pytest.approx(0.02)
    assert result["adaptation_gain_interaction"]["mean"] == pytest.approx(0.02)
    assert result["baseline"]["harmful_acceptances"] == 3
    assert result["post_vs_baseline_online_gain"]["mean"] == pytest.approx(0.02)


def test_four_arm_evaluation_pairs_inputs_and_keeps_reward_decisions_label_free(tmp_path, monkeypatch):
    """完整评估先结束 Pilot 决策；改变 Data 标签只应改变事后统计。"""
    from types import SimpleNamespace
    import evaluation.pilot_transfer as evaluation

    model, context = _fixture()
    config = {"model": model.config.to_dict(), "pilot_layout": "prefix", "main_level": "B",
              "offline_state_split": "offline_train", "pilot_transfer": {
                  "inner_learning_rate": 0.0025, "inner_steps": 2, "max_delta_norm": 0.05,
              }}
    trainer = PilotTransferTrainer(config, "cpu")
    trainer.model.load_state_dict(model.state_dict())
    checkpoint_dir = tmp_path / "checkpoint"
    trainer.save(checkpoint_dir, {"history": []})
    checkpoint = checkpoint_dir / "model_best.pt"
    frame = context.frame

    def episode(*args, frames, **kwargs):
        return (SimpleNamespace(state_metadata=lambda: {}), context.condition.complex_cir.reshape(-1),
                0, context.tail, [frame] * frames)

    monkeypatch.setattr(evaluation, "make_episode", episode)
    monkeypatch.setattr(evaluation, "make_context", lambda current, *args: replace(context, frame=current))
    decisions = []
    for poisoned in (False, True):
        if poisoned:
            bits = frame.bits.clone()
            bits[frame.data_mask] = ~bits[frame.data_mask]
            frame = replace(frame, bits=bits)
        directory = tmp_path / str(poisoned)
        summary = evaluation.run_transfer_comparison(
            config, baseline_checkpoint=checkpoint, post_checkpoint=checkpoint, seeds=[41001],
            snrs=[5], frames=2, device="cpu", output_dir=directory,
        )
        import json
        rows = [json.loads(line) for line in (directory / "frame_metrics.jsonl").read_text().splitlines()]
        decisions.append([(row["accepted"], row["reward_loss_improvement"], row["delta_norm"]) for row in rows])
        assert summary["per_snr"][0]["post_vs_baseline_online_gain"]["mean"] == 0
        assert len(rows) == 4
    assert decisions[0] == decisions[1]
