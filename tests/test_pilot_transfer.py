# -*- coding: utf-8 -*-
"""验证真实在线更新一致性、隐藏信息隔离和离线外层梯度。"""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

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
@pytest.mark.parametrize("receiver_input", ["adapt_only", "full_rx"])
def test_neural_bce_inner_matches_formal_ppo_and_hides_unknown_information(group, receiver_input):
    """两种输入协议的候选须一致，未知发送符号和标签始终不参与内循环。"""
    model, context = _fixture()
    actual, _ = _fixture()
    fast, _ = differentiable_adapt(
        model, [context], learning_rate=0.0005, steps=1, max_delta_norm=0.05,
        groups={group}, objective="bce", receiver_input=receiver_input,
    )
    _apply_adapt_peft_update(actual, context.frame, context.condition, context.tail,
                            groups={group}, lr=0.0005, steps=1, objective="bce", max_delta_norm=0.05,
                            receiver_input=receiver_input)
    for name, parameter in actual.named_parameters():
        torch.testing.assert_close(fast[name], parameter, atol=1e-7, rtol=2e-5)
    frame = context.frame
    hidden = ~frame.adapt_mask
    tx, rx, bits = frame.tx_symbols.clone(), frame.rx_symbols.clone(), frame.bits.clone()
    tx[hidden], bits[hidden] = -tx[hidden], ~bits[hidden]
    if receiver_input == "adapt_only":
        rx[hidden] = 1000 + 1000j
    poisoned = replace(context, frame=replace(frame, tx_symbols=tx, rx_symbols=rx, bits=bits))
    changed, _ = differentiable_adapt(model, [poisoned], learning_rate=0.0005,
                                     steps=1, max_delta_norm=0.05, groups={group}, objective="bce",
                                     receiver_input=receiver_input)
    for name in fast:
        torch.testing.assert_close(fast[name], changed[name], atol=0, rtol=0)


@pytest.mark.parametrize("group", ["head", "phase"])
@pytest.mark.parametrize("receiver_input", ["adapt_only", "full_rx"])
def test_neural_post_update_outer_gradient_matches_finite_difference(group, receiver_input):
    model, context = _fixture()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith("online_phase_trend_adapter."))

    def loss():
        # 与真实外层训练保持一致；CPU flash attention 也不支持这里的二阶导数。
        with sdpa_kernel(SDPBackend.MATH):
            parameters, _ = differentiable_adapt(model, [context], learning_rate=0.0005,
                                                 steps=1, max_delta_norm=0.05, create_graph=True,
                                                 groups={group}, objective="bce", receiver_input=receiver_input)
            logits = context_logits(model, context, parameters)
            mask = context.frame.data_mask
            return F.binary_cross_entropy_with_logits(logits[mask], context.frame.bits[mask].float())

    probe = model.head.bias if group == "head" else model.phase_correction_scale
    gradient, = torch.autograd.grad(loss(), probe)
    initial = probe.detach().clone()
    values = []
    for sign in (-1, 1):
        with torch.no_grad():
            probe.copy_(initial + sign * 0.002)
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


@pytest.mark.parametrize("identity_update", [False, True])
def test_offline_transfer_gradient_diagnostic_preserves_weights_and_identity_limit(monkeypatch, identity_update):
    """真实高阶梯度须有限；恒等内循环应严格退化为普通 Data 重加权。"""
    from scripts import diagnose_transfer_gradient as diagnostic

    model, context = _fixture()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith("online_phase_trend_adapter."))
    settings = {"inner_groups": ["phase"], "inner_objective": "bce",
                "inner_learning_rate": 0.0005, "inner_steps": 1, "max_delta_norm": 0.05}
    before = {name: value.clone() for name, value in model.state_dict().items()}
    if identity_update:
        monkeypatch.setattr(diagnostic, "differentiable_adapt",
                            lambda model, *args, **kwargs: (dict(model.named_parameters()), {"delta_norm": 0}))
    result = diagnostic.compare_gradients(model, context, settings)
    for label in ("all_base", "selected_group"):
        assert all(torch.isfinite(torch.tensor(value)) for value in result[label].values())
        if identity_update:
            assert result[label]["meta_residual_norm"] == 0
            assert result[label]["meta_residual_relative_to_reweighted"] == 0
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_transfer_gradient_diagnostic_rejects_heldout_before_loading_model(tmp_path):
    from scripts.diagnose_transfer_gradient import run_diagnostic

    with pytest.raises(ValueError, match="offline_train"):
        run_diagnostic({"offline_state_split": "heldout_edge"}, "missing.pt", [1], 1,
                       "cpu", tmp_path / "result.json")


@pytest.mark.parametrize("poison", ["labels", "waveform"])
def test_receiver_visibility_distinguishes_received_waveform_from_unknown_labels(poison):
    """整帧 Pilot 梯度可依赖接收波形，但不能依赖 Data 标签或未知发送符号。"""
    from scripts import diagnose_receiver_visibility as diagnostic

    model, context = _fixture()
    named = list(model.peft.named_group_parameters({"head", "phase"}))
    for _, parameter in named:
        parameter.requires_grad_(True)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    original = diagnostic.pilot_gradients(model, context, named)
    frame = context.frame
    bits, tx, rx = frame.bits.clone(), frame.tx_symbols.clone(), frame.rx_symbols.clone()
    if poison == "labels":
        bits[frame.data_mask] = ~bits[frame.data_mask]
        tx[frame.data_mask] = 300 + 400j
    else:
        rx[frame.data_mask] = 1000 + 1000j
    changed = diagnostic.pilot_gradients(model, replace(context, frame=replace(
        frame, bits=bits, tx_symbols=tx, rx_symbols=rx)), named)
    for view in ("adapt_only", "reward_pilot_rx"):
        assert original[view][0] == changed[view][0]
        for name, _ in named:
            torch.testing.assert_close(original[view][1][name], changed[view][1][name], atol=0, rtol=0)
    for view in ("adapt_full_rx", "reward_full_rx"):
        if poison == "labels":
            assert original[view][0] == changed[view][0]
            for name, _ in named:
                torch.testing.assert_close(original[view][1][name], changed[view][1][name], atol=0, rtol=0)
        else:
            assert any(not torch.equal(original[view][1][name], changed[view][1][name]) for name, _ in named)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_receiver_visibility_summary_is_finite_and_rejects_heldout(tmp_path):
    from scripts import diagnose_receiver_visibility as diagnostic

    model, context = _fixture()
    for _, parameter in model.peft.named_group_parameters({"head", "phase"}):
        parameter.requires_grad_(True)
    result = diagnostic.compare_visibility(model, context)
    assert set(result["groups"]) == {"head", "phase"}
    for values in result["groups"].values():
        assert all(value is None or torch.isfinite(torch.tensor(value)) for value in values.values())
    with pytest.raises(ValueError, match="offline_train"):
        diagnostic.run_diagnostic({"offline_state_split": "heldout_edge"}, "missing.pt", [1], 1,
                                  "cpu", tmp_path / "result.json")


def test_receiver_visibility_does_not_assign_direction_to_zero_gradient():
    from scripts.diagnose_receiver_visibility import _resolved_cosine

    assert _resolved_cosine(torch.zeros(3), torch.ones(3)) is None
    assert _resolved_cosine(torch.full((3,), 1e-12), torch.ones(3)) is None
    assert _resolved_cosine(torch.ones(3), torch.ones(3)) == pytest.approx(1)


def test_receiver_input_candidates_and_guards_ignore_data_labels_and_preserve_weights():
    """两种输入的候选及验收均不能用未知标签，候选不原地写回基础模型。"""
    from scripts.diagnose_receiver_input_updates import pilot_candidates_and_decisions

    model, context = _fixture()
    settings = {"inner_learning_rate": 0.0005, "inner_steps": 1, "max_delta_norm": 0.05}
    before = {name: value.clone() for name, value in model.state_dict().items()}
    frame = context.frame
    bits, tx = frame.bits.clone(), frame.tx_symbols.clone()
    bits[frame.data_mask] = ~bits[frame.data_mask]
    tx[frame.data_mask] = 300 + 400j
    poisoned = replace(context, frame=replace(frame, bits=bits, tx_symbols=tx))
    original = pilot_candidates_and_decisions(model, context, settings, ("head", "phase"))
    changed = pilot_candidates_and_decisions(model, poisoned, settings, ("head", "phase"))
    assert len(original[0]) == 8
    assert original[0] == changed[0]
    for key, fast in original[1].items():
        for name, value in fast.items():
            torch.testing.assert_close(value, changed[1][key][name], rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_full_receiver_input_reconstruction_is_rejected_without_mutation():
    from training.pilot_transfer import _AdaptObjective

    model, context = _fixture()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="只支持 Adapt BCE"):
        _AdaptObjective(model, receiver_input="full_rx")
    with pytest.raises(ValueError, match="只支持 Adapt BCE"):
        differentiable_adapt(model, [context], learning_rate=0.0005, steps=1,
                             max_delta_norm=0.05, receiver_input="full_rx")
    with pytest.raises(ValueError, match="只支持 Adapt BCE"):
        _apply_adapt_peft_update(model, context.frame, context.condition, context.tail,
                                groups={"phase_trend"}, lr=0.0005, steps=1,
                                objective="pilot_signal_reconstruction", max_delta_norm=0.05,
                                receiver_input="full_rx")
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_receiver_input_prototype_end_to_end_pairs_protocols(tmp_path, monkeypatch):
    from scripts import diagnose_receiver_input_updates as diagnostic

    model, context = _fixture()
    settings = {"inner_learning_rate": 0.0005, "inner_steps": 1, "max_delta_norm": 0.05,
                "inner_objective": "bce"}
    config = {"offline_state_split": "offline_train", "main_level": "B", "pilot_layout": "prefix",
              "main_snrs": [0, 5, 10, 15], "pilot_transfer": settings}
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"fixture")
    monkeypatch.setattr(diagnostic, "_load_branch", lambda *args: (model, {}))
    monkeypatch.setattr(diagnostic, "make_episode", lambda *args, frames, **kwargs: (
        None, context.condition.complex_cir.reshape(-1), 0, context.tail, [context.frame] * frames))
    monkeypatch.setattr(diagnostic, "make_context", lambda frame, *args: replace(context, frame=frame))
    result = diagnostic.run_diagnostic(config, checkpoint, [42000, 42001], 2, "cpu", tmp_path / "result.json")
    assert len(result["rows"]) == 128
    assert len(result["per_snr"]) == 32
    assert result["diagnostic_only"] and not result["main_aggregation_allowed"]
    assert not result["base_weights_modified"]
    for seed in result["seeds"]:
        for snr in config["main_snrs"]:
            for index in range(2):
                rows = [row for row in result["rows"] if (row["seed"], row["snr_db"], row["frame"]) == (seed, snr, index)]
                assert len(rows) == 8
                assert len({(row["input_sha256"], row["ber_frozen"], row["bce_frozen"]) for row in rows}) == 1
                for row in rows:
                    assert not row["data_labels_used_for_control"]
                    assert row["data_waveform_used_for_candidate"] == (row["receiver_input"] == "full_rx")
                    assert row["data_waveform_used_for_guard"] == (row["reward_input"] == "full_rx")
    for item in result["per_snr"]:
        assert item["committed_gain"]["seeds"] == 2
        if item["receiver_input"] == "adapt_only" and item["reward_input"] == "pilot_only":
            assert item["gain_vs_strict_reference"]["mean"] == 0
