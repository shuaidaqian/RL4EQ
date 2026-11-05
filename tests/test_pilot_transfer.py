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
    PilotTransferTrainer, TransferContext, context_logits, differentiable_adapt, next_pilot_tail, task_context,
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


@pytest.mark.parametrize("full_rx,prefix,persistent", [(False, False, False), (True, False, False), (True, True, False), (True, True, True)])
@pytest.mark.parametrize("burn_in", [0, 7])
@pytest.mark.parametrize("state_split", ["heldout_edge", "offline_train"])
def test_four_arm_evaluation_pairs_inputs_and_keeps_reward_decisions_label_free(tmp_path, monkeypatch, full_rx, prefix, persistent, burn_in, state_split):
    """完整评估先结束 Pilot 决策；改变 Data 标签只应改变事后统计。"""
    from types import SimpleNamespace
    import evaluation.pilot_transfer as evaluation

    model, context = _fixture()
    config = {"model": model.config.to_dict(), "pilot_layout": "prefix", "main_level": "B",
              "offline_state_split": "offline_train", "pilot_transfer": {
                  "inner_learning_rate": 0.0025, "inner_steps": 2, "max_delta_norm": 0.05,
                  "evaluation_history_frames": burn_in,
              }}
    if full_rx:
        config["pilot_transfer"].update(inner_receiver_input="full_rx", reward_receiver_input="full_rx",
                                       inner_objective="bce", inner_groups=["phase"])
    if prefix:
        config["pilot_transfer"].update(inner_task="prefix_boundary", reward_task="prefix_boundary")
    trainer = PilotTransferTrainer(config, "cpu")
    trainer.model.load_state_dict(model.state_dict())
    checkpoint_dir = tmp_path / "checkpoint"
    trainer.save(checkpoint_dir, {"history": []})
    checkpoint = checkpoint_dir / "model_best.pt"
    frame = context.frame

    def episode(*args, frames, **kwargs):
        assert frames == burn_in + 2
        return (SimpleNamespace(state_metadata=lambda: {}), context.condition.complex_cir.reshape(-1),
                0, context.tail, [replace(frame, frame_index=index) for index in range(frames)])

    monkeypatch.setattr(evaluation, "make_episode", episode)
    monkeypatch.setattr(evaluation, "make_context", lambda current, *args: replace(context, frame=current))
    decisions = []
    for poisoned in (False, True):
        if poisoned:
            bits = frame.bits.clone()
            tx = frame.tx_symbols.clone()
            bits[frame.data_mask] = ~bits[frame.data_mask]
            tx[frame.data_mask] = 1000 + 1000j
            frame = replace(frame, bits=bits, tx_symbols=tx)
        directory = tmp_path / str(poisoned)
        summary = evaluation.run_transfer_comparison(
            config, baseline_checkpoint=checkpoint, post_checkpoint=checkpoint, seeds=[41001],
            snrs=[5], frames=2, device="cpu", output_dir=directory,
            persistent=persistent, state_split=state_split,
        )
        import json
        rows = [json.loads(line) for line in (directory / "frame_metrics.jsonl").read_text().splitlines()]
        decisions.append([(row["accepted"], row["reward_loss_improvement"], row["delta_norm"]) for row in rows])
        assert summary["per_snr"][0]["post_vs_baseline_online_gain"]["mean"] == 0
        assert len(rows) == 4
        assert summary["manifest"]["evaluation_history_frames"] == burn_in
        assert summary["manifest"]["state_split"] == state_split
        assert all(row["channel_frame_index"] == burn_in + row["frame"] for row in rows)
        assert all(row["data_waveform_used_online"] == (full_rx and not prefix) for row in rows)
        assert all(not row["data_labels_used_online"] for row in rows)
        assert all(row["episode_delta_norm_online"] <= 0.050001 for row in rows)
        assert all(row["episode_delta_norm_before"] == 0 for row in rows if row["frame"] == 0)
        if persistent:
            for first, second in zip(rows[::2], rows[1::2]):
                assert first["episode_delta_norm_online"] == second["episode_delta_norm_before"]
    assert decisions[0] == decisions[1]


@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("objective", ["baseline", "post_adapt"])
def test_trainer_routes_full_receiver_input_and_records_actual_protocol(monkeypatch, objective, prefix):
    import training.pilot_transfer as transfer

    model, context = _fixture()
    settings = {"inner_learning_rate": 0.0005, "inner_steps": 1, "max_delta_norm": 0.05,
                "inner_objective": "bce", "inner_groups": ["phase"],
                "inner_receiver_input": "full_rx", "reward_receiver_input": "full_rx"}
    config = {"model": model.config.to_dict(), "pilot_layout": "prefix", "main_level": "B",
              "offline_state_split": "offline_train", "pilot_transfer": settings}
    if prefix:
        settings.update(inner_task="prefix_boundary", reward_task="prefix_boundary")
    trainer = PilotTransferTrainer(config, "cpu", objective=objective)
    trainer.model.load_state_dict(model.state_dict())
    monkeypatch.setattr(trainer, "_sample", lambda: (context, {}, 10, 44000))
    calls = []
    actual = transfer.differentiable_adapt

    def spy(*args, **kwargs):
        calls.append((kwargs["receiver_input"], kwargs["create_graph"]))
        expected, start = task_context(context, "prefix_boundary" if prefix else "full_frame", role="adapt")
        assert args[1][0].frame.rx_symbols.numel() == expected.frame.rx_symbols.numel()
        assert kwargs["score_start"] == start
        return actual(*args, **kwargs)

    monkeypatch.setattr(transfer, "differentiable_adapt", spy)
    result = trainer.train(1)
    assert calls == [("full_rx", objective == "post_adapt")]
    assert result["inner_uses_data_waveform"] == (not prefix)
    assert not result["inner_uses_data_labels"]
    assert result["inner_receiver_input"] == result["reward_receiver_input"] == "full_rx"
    assert len(result["history"]) == 1
    assert all(torch.isfinite(value).all() for value in trainer.model.state_dict().values())


@pytest.mark.parametrize("settings", [
    {"inner_receiver_input": "unknown"}, {"reward_receiver_input": "unknown"},
    {"inner_receiver_input": "full_rx", "inner_objective": "pilot_signal_reconstruction"},
])
def test_receiver_input_configuration_rejects_unsupported_combinations(settings):
    from training.pilot_transfer import receiver_inputs

    with pytest.raises(ValueError):
        receiver_inputs(settings)


@pytest.mark.parametrize("settings", [
    {"inner_task": "unknown"}, {"reward_task": "unknown"},
    {"inner_task": "prefix_boundary"}, {"reward_task": "prefix_boundary"},
])
def test_receiver_task_rejects_invalid_or_masked_short_window(settings):
    from training.pilot_transfer import receiver_tasks

    with pytest.raises(ValueError):
        receiver_tasks(settings)


def test_four_arm_rejects_training_task_mismatch(tmp_path, monkeypatch):
    import evaluation.pilot_transfer as evaluation

    model, _ = _fixture()
    settings = {"inner_receiver_input": "full_rx", "reward_receiver_input": "full_rx", "inner_objective": "bce"}
    config = {"pilot_transfer": {**settings, "inner_task": "prefix_boundary", "reward_task": "prefix_boundary"}}
    monkeypatch.setattr(evaluation, "_load_branch", lambda *args: (
        model, {"metrics": {"stage": "pilot_transfer", "settings": settings}}))
    with pytest.raises(ValueError, match="接收任务不匹配"):
        evaluation.run_transfer_comparison(config, baseline_checkpoint=tmp_path / "old.pt",
            post_checkpoint=tmp_path / "old.pt", seeds=[82001], snrs=[10], frames=1,
            device="cpu", output_dir=tmp_path / "result")


def test_four_arm_evaluation_rejects_training_receiver_protocol_mismatch(tmp_path, monkeypatch):
    import evaluation.pilot_transfer as evaluation

    model, _ = _fixture()
    config = {"pilot_transfer": {"inner_receiver_input": "full_rx", "reward_receiver_input": "full_rx",
                                 "inner_objective": "bce"}}
    monkeypatch.setattr(evaluation, "_load_branch", lambda *args: (
        model, {"metrics": {"stage": "pilot_transfer", "settings": {"inner_objective": "bce"}}}))
    with pytest.raises(ValueError, match="接收输入协议不匹配"):
        evaluation.run_transfer_comparison(config, baseline_checkpoint=tmp_path / "old.pt",
            post_checkpoint=tmp_path / "old.pt", seeds=[60001], snrs=[10], frames=1,
            device="cpu", output_dir=tmp_path / "result")


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


def test_error_geometry_decomposes_branches_without_hidden_information():
    from scripts.diagnose_error_geometry import receiver_outputs, region_masks, score_outputs

    model, context = _fixture()
    model.config = replace(model.config, physics_warm_start_iterations=2, neural_residual_scale=0.1)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    heads = []
    hook = model.head.register_forward_hook(lambda module, args, output: heads.append(output.detach()))
    logits, physics = receiver_outputs(model, context)
    hook.remove()
    torch.testing.assert_close(logits, physics + 0.1 * heads[-1].squeeze(), atol=1e-6, rtol=1e-6)
    frame = context.frame
    bits, tx = frame.bits.clone(), frame.tx_symbols.clone()
    bits[frame.data_mask], tx[frame.data_mask] = ~bits[frame.data_mask], 1000 + 1000j
    poisoned = replace(context, frame=replace(frame, bits=bits, tx_symbols=tx,
                        true_cir=torch.full((5,), 1000 + 1000j), tail_symbols=torch.full((4,), 1000 + 1000j)))
    changed = receiver_outputs(model, poisoned)
    for first, second in zip((logits, physics), changed):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    masks = region_masks(frame, 4)
    assert torch.equal(sum(mask.int() for key, mask in masks.items() if key.startswith("data_")), frame.data_mask.int())
    assert torch.equal(sum(mask.int() for mask in masks.values()), torch.ones_like(frame.bits, dtype=torch.int))
    scored = score_outputs(frame, logits, physics, 4)
    assert sum(r["bits"] for r in scored) == frame.bits.numel()
    for row in scored:
        assert row["physics_errors"] - row["neural_fixed_errors"] + row["neural_introduced_errors"] == row["errors"]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def _symbol_fixture():
    base, context = _fixture()
    base.config = replace(base.config, physics_warm_start_iterations=2, neural_residual_scale=0.1)
    model = UnfoldedEqualizer(replace(base.config, enable_symbol_residual_branch=True)).eval()
    model.attach_online_phase_trend_adapter()
    state = model.state_dict()
    state.update(base.state_dict())
    model.load_state_dict(state, strict=True)
    return base, model, context


def test_symbol_residual_initialization_preserves_output_and_is_bounded():
    from agent.unfolded_equalizer import SymbolResidualDenoiser

    base, model, context = _symbol_fixture()
    torch.testing.assert_close(context_logits(base, context), context_logits(model, context), atol=0, rtol=0)
    selected = model.peft.named_group_parameters({"symbol_residual_head"})
    assert sum(value.numel() for _, value in selected) == 17
    denoiser = SymbolResidualDenoiser()
    with torch.no_grad():
        denoiser.head.bias.fill_(100)
    correction = denoiser(torch.randn(2, 128, 7))
    assert correction.shape == (2, 128)
    assert bool((correction.abs() <= 0.5).all())


def test_symbol_head_matches_actual_adamw_and_ignores_unknown_information():
    _, model, context = _symbol_fixture()
    _, actual, _ = _symbol_fixture()
    actual.load_state_dict(model.state_dict())
    fast, _ = differentiable_adapt(model, [context], learning_rate=0.0005, steps=1, max_delta_norm=0.05,
                                   groups={"symbol_residual_head"}, objective="bce", receiver_input="full_rx")
    _apply_adapt_peft_update(actual, context.frame, context.condition, context.tail, groups={"symbol_residual_head"},
                            lr=0.0005, steps=1, objective="bce", max_delta_norm=0.05, receiver_input="full_rx")
    for name, value in actual.named_parameters():
        torch.testing.assert_close(fast[name], value, atol=1e-7, rtol=2e-5)
    frame = context.frame
    tx, bits = frame.tx_symbols.clone(), frame.bits.clone()
    tx[~frame.adapt_mask], bits[~frame.adapt_mask] = 1000 + 1000j, ~bits[~frame.adapt_mask]
    poisoned = replace(context, frame=replace(frame, tx_symbols=tx, bits=bits))
    altered, _ = differentiable_adapt(model, [poisoned], learning_rate=0.0005, steps=1, max_delta_norm=0.05,
                                      groups={"symbol_residual_head"}, objective="bce", receiver_input="full_rx")
    for name in fast:
        torch.testing.assert_close(fast[name], altered[name], atol=0, rtol=0)


def test_symbol_transfer_migration_and_outer_freeze(tmp_path, monkeypatch):
    base, model, context = _symbol_fixture()
    checkpoint = tmp_path / "base.pt"
    torch.save({"state_dict": {name: value for name, value in base.state_dict().items()
                               if not name.startswith("online_phase_trend_adapter.")},
                "model_config": base.config.to_dict()}, checkpoint)
    settings = {"inner_learning_rate": 0.0005, "inner_steps": 1, "max_delta_norm": 0.05,
                "inner_objective": "bce", "inner_groups": ["symbol_residual_head"],
                "inner_receiver_input": "full_rx", "reward_receiver_input": "full_rx",
                "outer_groups": ["symbol_residual"], "outer_learning_rate": 0.001}
    config = {"model": model.config.to_dict(), "pilot_layout": "prefix", "main_level": "B",
              "offline_state_split": "offline_train", "pilot_transfer": settings}
    trainer = PilotTransferTrainer(config, "cpu", objective="post_adapt")
    with pytest.raises(ValueError, match="参数结构不一致"):
        trainer.load_resume(checkpoint)
    trainer.settings["initialize_symbol_residual_from_base"] = True
    trainer.load_resume(checkpoint)
    # 同为推理模式时比较；可微分支会改变 attention 的后端选择与舍入路径。
    with torch.no_grad():
        torch.testing.assert_close(context_logits(base, context), context_logits(trainer.model, context), atol=0, rtol=0)
    before = {name: value.clone() for name, value in trainer.model.state_dict().items()}
    monkeypatch.setattr(trainer, "_sample", lambda: (context, {}, 10, 46000))
    trainer.train(1)
    for name, value in trainer.model.state_dict().items():
        assert bool(torch.isfinite(value).all())
        if not name.startswith("symbol_residual."):
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    assert any(not torch.equal(value, before[name]) for name, value in trainer.model.state_dict().items()
               if name.startswith("symbol_residual.head."))
    trainer.save(tmp_path / "saved", {})
    payload = torch.load(tmp_path / "saved" / "model_best.pt", weights_only=False)
    standalone = UnfoldedEqualizer(UnfoldedConfig.from_dict(payload["model_config"]))
    standalone.load_state_dict(payload["state_dict"], strict=True)


@pytest.mark.parametrize("task", ["full_frame", "prefix_boundary"])
def test_symbol_head_outer_gradient_matches_nonzero_finite_difference(task):
    _, model, context = _symbol_fixture()
    frame = context.frame
    offset = 1.2 * torch.linspace(-1, 1, frame.rx_symbols.numel())
    context = replace(context, frame=replace(frame, rx_symbols=frame.rx_symbols + offset))
    for _, parameter in model.peft.named_group_parameters({"symbol_residual"}):
        parameter.requires_grad_(True)

    def loss():
        with sdpa_kernel(SDPBackend.MATH):
            inner, start = task_context(context, task, role="adapt")
            fast, _ = differentiable_adapt(model, [inner], learning_rate=0.0005, steps=1,
                max_delta_norm=0.05, create_graph=True, groups={"symbol_residual_head"},
                objective="bce", receiver_input="full_rx", score_start=start)
            logits = context_logits(model, context, fast)
            return F.binary_cross_entropy_with_logits(logits[frame.data_mask], frame.bits[frame.data_mask].float())

    probe = model.symbol_residual.head.bias
    gradient, = torch.autograd.grad(loss(), probe)
    initial = probe.detach().clone()
    values = []
    for sign in (-1, 1):
        with torch.no_grad():
            probe.copy_(initial + sign * 0.001)
        values.append(float(loss().detach()))
    finite_difference = (values[1] - values[0]) / 0.002
    assert abs(float(gradient)) > 1e-4
    assert float(gradient) == pytest.approx(finite_difference, abs=2e-3, rel=0.03)


def test_prefix_task_uses_true_short_operator_and_hides_unknown_labels():
    from scripts.diagnose_prefix_task import candidates_and_guards, prefix_context
    from env.linear_operator import LinearChannelOperator

    _, model, context = _symbol_fixture()
    # 非零已训练 head 保证短窗路径不是零模块的平凡特例。
    with torch.no_grad():
        model.symbol_residual.head.weight.fill_(0.02)
    settings = {"inner_learning_rate": 0.0005, "inner_steps": 1, "max_delta_norm": 0.05}
    before = {name: value.clone() for name, value in model.state_dict().items()}
    original = candidates_and_guards(model, context, settings)
    frame = context.frame
    bits, tx = frame.bits.clone(), frame.tx_symbols.clone()
    bits[frame.data_mask], tx[frame.data_mask] = ~bits[frame.data_mask], 1000 + 1000j
    poisoned = replace(context, frame=replace(frame, bits=bits, tx_symbols=tx))
    altered = candidates_and_guards(model, poisoned, settings)
    assert len(original[0]) == 4 and original[0] == altered[0]
    for task, parameters in original[1].items():
        for name, value in parameters.items():
            torch.testing.assert_close(value, altered[1][task][name], rtol=0, atol=0)
    end = int(torch.nonzero(frame.adapt_mask).max()) + 1
    short = prefix_context(context, end)
    assert short.frame.rx_symbols.numel() == end
    assert not bool(short.frame.data_mask.any())
    assert context_logits(model, short).numel() == end
    cir = torch.tensor([0.6, 0, 0, 0, 0.8], dtype=torch.complex64)
    vector = torch.randn(end, dtype=torch.complex64)
    op = LinearChannelOperator(end, 4)
    actual = op.adjoint(op.forward(vector, cir, torch.zeros(4, dtype=torch.complex64)), cir)
    full = LinearChannelOperator(128, 4)
    padded = F.pad(vector, (0, 128 - end))
    wrong = full.adjoint(full.forward(padded, cir, torch.zeros(4, dtype=torch.complex64)), cir)[:end]
    assert not torch.allclose(actual, wrong)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("window", [1, 4])
def test_prefix_alignment_is_read_only_and_data_labels_do_not_change_candidate(window):
    from scripts.diagnose_prefix_alignment import gradient_alignment

    _, model, context = _symbol_fixture()
    with torch.no_grad():
        model.symbol_residual.head.weight.fill_(0.02)
    settings = {"inner_learning_rate": 0.005, "inner_steps": 1, "max_delta_norm": 0.05,
                "inner_groups": ["symbol_residual_head"], "inner_objective": "bce",
                "inner_receiver_input": "full_rx", "reward_receiver_input": "full_rx",
                "inner_task": "prefix_boundary", "reward_task": "prefix_boundary"}
    before = {n: v.clone() for n, v in model.state_dict().items()}
    flags = {n: p.requires_grad for n, p in model.named_parameters()}
    original = gradient_alignment(model, context, settings, history=[context] * window)
    bits = context.frame.bits.clone()
    bits[context.frame.data_mask] = ~bits[context.frame.data_mask]
    poisoned = replace(context, frame=replace(context.frame, bits=bits))
    altered = gradient_alignment(model, poisoned, settings, history=[poisoned] * window)
    assert original["adapt_losses"] == altered["adapt_losses"]
    assert original["delta_norm"] == altered["delta_norm"]
    assert original["adapt"] == altered["adapt"]
    assert original["reward"] == altered["reward"]
    assert original["data"] != altered["data"]
    assert original["parameter_count"] == 17
    assert original["history_frames"] == window
    for role in ("adapt", "reward", "data", "data_tail", "data_interior"):
        assert all(torch.isfinite(torch.tensor(v)) for v in original[role].values())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert {n: p.requires_grad for n, p in model.named_parameters()} == flags


def test_pilot_state_equations_ignore_hidden_waveforms_labels_and_unknown_tail():
    from scripts.diagnose_pilot_state_transfer import pilot_equations, posterior_candidate

    _, model, context = _symbol_fixture()
    frame = context.frame
    hidden = ~frame.adapt_mask
    bits, tx, rx = frame.bits.clone(), frame.tx_symbols.clone(), frame.rx_symbols.clone()
    bits[hidden], tx[hidden], rx[hidden] = ~bits[hidden], 1000 + 1000j, -1000 + 1000j
    poisoned = replace(context, tail=torch.full_like(context.tail, 1000j),
        frame=replace(frame, bits=bits, tx_symbols=tx, rx_symbols=rx))
    first, second = pilot_equations(model, context), pilot_equations(model, poisoned)
    for actual, expected in zip(first, second):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert first[0].shape == (int(frame.adapt_mask.sum()) - context.tail.numel(), context.tail.numel() + 1)
    prior = context.condition.complex_cir.reshape(-1)
    design = 10 * torch.eye(prior.numel(), dtype=torch.complex64)
    exact, _ = posterior_candidate([(design, design @ prior)], context, acquisition_samples=124, rho_frame=.999)
    torch.testing.assert_close(exact, prior, rtol=0, atol=0)
    target = prior * 1.1
    estimate, _ = posterior_candidate([(design, design @ target)], context, acquisition_samples=124, rho_frame=.999)
    assert float((estimate - target).norm()) < float((prior - target).norm())


def test_pilot_state_candidate_guard_hides_data_labels_and_preserves_model():
    from scripts.diagnose_pilot_state_transfer import pilot_equations, state_candidates_and_guards

    _, model, context = _symbol_fixture()
    config = {"frame_len": 128, "max_delay": 4, "rho_frame": .999}
    before = {n: v.clone() for n, v in model.state_dict().items()}
    equations = [pilot_equations(model, context)] * 8
    original = state_candidates_and_guards(model, context, equations, [1, 8], config)
    frame = context.frame
    bits, tx = frame.bits.clone(), frame.tx_symbols.clone()
    bits[frame.data_mask], tx[frame.data_mask] = ~bits[frame.data_mask], 1000 + 1000j
    poisoned = replace(context, frame=replace(frame, bits=bits, tx_symbols=tx))
    altered = state_candidates_and_guards(model, poisoned, equations, [1, 8], config)
    assert original[2] == altered[2]
    torch.testing.assert_close(original[0], altered[0], rtol=0, atol=0)
    for window in [1, 8]:
        torch.testing.assert_close(original[1][window], altered[1][window], rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def _channel_state_fixture():
    _, base, context = _symbol_fixture()
    model = UnfoldedEqualizer(replace(base.config, enable_channel_state_branch=True)).eval()
    model.attach_online_phase_trend_adapter()
    state = model.state_dict()
    state.update(base.state_dict())
    model.load_state_dict(state, strict=True)
    return base, model, context


def _channel_state_settings():
    return {"inner_learning_rate": .005, "inner_steps": 1, "max_delta_norm": .05,
        "inner_objective": "channel_state_signal", "reward_objective": "channel_state_signal",
        "inner_groups": ["channel_state_head"], "outer_groups": ["channel_state"],
        "inner_receiver_input": "full_rx", "reward_receiver_input": "full_rx", "outer_learning_rate": .001}


def test_neural_channel_state_zero_initialization_counts_and_direction_bound():
    base, model, context = _channel_state_fixture()
    with torch.no_grad():
        torch.testing.assert_close(context_logits(base, context), context_logits(model, context), rtol=0, atol=0)
    assert sum(p.numel() for _, p in model.peft.named_group_parameters({"channel_state"})) == 449
    assert sum(p.numel() for _, p in model.peft.named_group_parameters({"channel_state_head"})) == 17
    with torch.no_grad():
        model.channel_state.head.bias.fill_(100)
        frame = context.frame
        corrected = model._apply_phase_correction(torch.stack((frame.rx_symbols.real, frame.rx_symbols.imag), dim=-1).unsqueeze(0), context.condition)
        updated = model._channel_state_condition(corrected, context.condition,
            frame.tx_symbols.unsqueeze(0), frame.adapt_mask.unsqueeze(0), frame.frame_index)
        assert float((updated.complex_cir - context.condition.complex_cir).norm()) <= 1.000001


@pytest.mark.parametrize("objective", ["channel_state_signal", "channel_state_map"])
def test_neural_channel_state_adamw_matches_and_hides_reward_data_and_tail(objective):
    from training.pilot_transfer import ChannelStateSignalObjective

    _, model, context = _channel_state_fixture()
    _, actual, _ = _channel_state_fixture()
    settings = _channel_state_settings()
    for net in (model, actual):
        net.set_trainable_groups({"channel_state_head"})
    fast, info = differentiable_adapt(model, [context], learning_rate=.005, steps=1, max_delta_norm=.05,
        groups=settings["inner_groups"], objective=objective, receiver_input="full_rx")
    selected = [p for _, p in actual.peft.named_group_parameters({"channel_state_head"})]
    optimizer = torch.optim.AdamW(selected, lr=.005)
    loss = ChannelStateSignalObjective(actual, include_prior=objective == "channel_state_map")(context)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(selected, 1.)
    optimizer.step()
    assert info["delta_norm"] > 0
    for name, parameter in actual.named_parameters():
        torch.testing.assert_close(fast[name], parameter, atol=1e-7, rtol=2e-5)
    frame = context.frame
    bits, tx, rx = frame.bits.clone(), frame.tx_symbols.clone(), frame.rx_symbols.clone()
    hidden = ~frame.adapt_mask
    bits[hidden], tx[hidden], rx[hidden] = ~bits[hidden], 1000j, -1000j
    poisoned = replace(context, tail=torch.full_like(context.tail, 1000j),
        frame=replace(frame, bits=bits, tx_symbols=tx, rx_symbols=rx))
    altered, _ = differentiable_adapt(model, [poisoned], learning_rate=.005, steps=1, max_delta_norm=.05,
        groups=settings["inner_groups"], objective=objective, receiver_input="full_rx")
    for name in fast:
        torch.testing.assert_close(fast[name], altered[name], atol=0, rtol=0)


def test_neural_channel_state_reward_signal_loss_hides_data_information():
    from training.pilot_transfer import channel_state_reward_loss

    _, model, context = _channel_state_fixture()
    with torch.no_grad():
        model.channel_state.head.bias.fill_(.1)
    frame = context.frame
    bits, tx, rx = frame.bits.clone(), frame.tx_symbols.clone(), frame.rx_symbols.clone()
    bits[frame.data_mask], tx[frame.data_mask], rx[frame.data_mask] = ~bits[frame.data_mask], 1000j, -1000j
    poisoned = replace(context, frame=replace(frame, bits=bits, tx_symbols=tx, rx_symbols=rx))
    torch.testing.assert_close(channel_state_reward_loss(model, context), channel_state_reward_loss(model, poisoned), rtol=0, atol=0)


@pytest.mark.parametrize("objective", ["channel_state_signal", "channel_state_map"])
@pytest.mark.parametrize("future_target", [False, True])
def test_neural_channel_state_post_update_outer_gradient_matches_finite_difference(objective, future_target):
    _, model, context = _channel_state_fixture()
    frame = context.frame
    context = replace(context, frame=replace(frame,
        rx_symbols=frame.rx_symbols + 1.2 * torch.linspace(-1, 1, frame.rx_symbols.numel())))
    target = (replace(context, frame=replace(context.frame, frame_index=8,
              rx_symbols=context.frame.rx_symbols * (.9 + .1j))) if future_target else context)
    model.set_trainable_groups({"channel_state"})

    def loss():
        with sdpa_kernel(SDPBackend.MATH):
            fast, _ = differentiable_adapt(model, [context], learning_rate=.005, steps=1,
                max_delta_norm=.05, create_graph=True, groups={"channel_state_head"},
                objective=objective, receiver_input="full_rx")
            logits = context_logits(model, target, fast)
            return F.binary_cross_entropy_with_logits(logits[frame.data_mask], frame.bits[frame.data_mask].float())

    probe = model.channel_state.head.bias
    gradient, = torch.autograd.grad(loss(), probe)
    initial = probe.detach().clone()
    values = []
    for sign in (-1, 1):
        with torch.no_grad():
            probe.copy_(initial + sign * .001)
        values.append(float(loss().detach()))
    assert abs(float(gradient)) > 1e-5
    assert float(gradient) == pytest.approx((values[1] - values[0]) / .002, abs=2e-3, rel=.05)


def test_neural_channel_state_migration_training_and_strict_checkpoint(tmp_path, monkeypatch):
    base, model, context = _channel_state_fixture()
    checkpoint = tmp_path / "base.pt"
    torch.save({"state_dict": {n: p for n, p in base.state_dict().items()
        if not n.startswith("online_phase_trend_adapter.")}, "model_config": base.config.to_dict()}, checkpoint)
    settings = _channel_state_settings()
    config = {"model": model.config.to_dict(), "pilot_layout": "prefix", "main_level": "B",
              "offline_state_split": "offline_train", "pilot_transfer": settings}
    trainer = PilotTransferTrainer(config, "cpu", objective="post_adapt")
    with pytest.raises(ValueError, match="参数结构不一致"):
        trainer.load_resume(checkpoint)
    trainer.settings["initialize_channel_state_from_base"] = True
    trainer.load_resume(checkpoint)
    with torch.no_grad():
        torch.testing.assert_close(context_logits(base, context), context_logits(trainer.model, context), rtol=0, atol=0)
    before = {n: p.clone() for n, p in trainer.model.state_dict().items()}
    monkeypatch.setattr(trainer, "_sample", lambda: (context, {}, 10, 76000))
    metrics = trainer.train(1)
    assert metrics["outer_trainable_parameter_count"] == 449
    assert not metrics["inner_uses_data_waveform"]
    for name, parameter in trainer.model.state_dict().items():
        assert torch.isfinite(parameter).all()
        if not name.startswith("channel_state."):
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    assert any(not torch.equal(p, before[n]) for n, p in trainer.model.state_dict().items() if n.startswith("channel_state.head."))
    trainer.save(tmp_path / "saved", metrics)
    payload = torch.load(tmp_path / "saved/model_best.pt", weights_only=False)
    standalone = UnfoldedEqualizer(UnfoldedConfig.from_dict(payload["model_config"]))
    standalone.load_state_dict(payload["state_dict"], strict=True)


@pytest.mark.parametrize("reward_input", ["full_rx", "pilot_only"])
def test_neural_channel_state_four_arm_guard_is_label_free(tmp_path, monkeypatch, reward_input):
    import json
    from types import SimpleNamespace
    import evaluation.pilot_transfer as evaluation

    _, model, context = _channel_state_fixture()
    settings = _channel_state_settings()
    settings["reward_receiver_input"] = reward_input
    config = {"model": model.config.to_dict(), "pilot_layout": "prefix", "main_level": "B",
              "offline_state_split": "offline_train", "pilot_transfer": settings}
    trainer = PilotTransferTrainer(config, "cpu")
    trainer.model.load_state_dict(model.state_dict())
    trainer.save(tmp_path / "checkpoint", {"stage": "pilot_transfer", "settings": settings, "history": []})
    checkpoint = tmp_path / "checkpoint/model_best.pt"
    frame = context.frame

    def episode(*args, frames, **kwargs):
        return SimpleNamespace(state_metadata=lambda: {}), context.condition.complex_cir.reshape(-1), 0, context.tail, [frame] * frames

    monkeypatch.setattr(evaluation, "make_episode", episode)
    monkeypatch.setattr(evaluation, "make_context", lambda current, *args: replace(context, frame=current))
    decisions = []
    for poison in [False, True]:
        if poison:
            bits, tx = frame.bits.clone(), frame.tx_symbols.clone()
            bits[frame.data_mask], tx[frame.data_mask] = ~bits[frame.data_mask], 1000j
            frame = replace(frame, bits=bits, tx_symbols=tx)
        directory = tmp_path / str(poison)
        evaluation.run_transfer_comparison(config, baseline_checkpoint=checkpoint, post_checkpoint=checkpoint,
            seeds=[86001], snrs=[5], frames=2, device="cpu", output_dir=directory, persistent=True)
        rows = [json.loads(line) for line in (directory / "frame_metrics.jsonl").read_text().splitlines()]
        assert all(not r["data_labels_used_online"] and not r["data_waveform_used_for_update"] for r in rows)
        assert all(r["data_waveform_used_for_guard"] == (reward_input == "full_rx") for r in rows)
        assert all(r["data_waveform_used_online"] == (reward_input == "full_rx") for r in rows)
        assert all(r["reward_objective"] == "channel_state_signal" for r in rows)
        decisions.append([(r["accepted"], r["reward_loss_before"], r["reward_loss_after"],
                           r["reward_ber_before"], r["reward_ber_after"], r["delta_norm"],
                           r["episode_delta_norm_online"]) for r in rows])
    assert decisions[0] == decisions[1]


def test_channel_state_capacity_audit_hides_data_labels_and_preserves_weights():
    from scripts.diagnose_channel_state_capacity import candidates_and_guards

    _, model, context = _channel_state_fixture()
    with torch.no_grad():
        model.channel_state.head.bias.fill_(.1)
    before = {n: v.clone() for n, v in model.state_dict().items()}
    original = candidates_and_guards(model, context, ("channel_state_signal", "channel_state_map"))
    frame = context.frame
    bits, tx = frame.bits.clone(), frame.tx_symbols.clone()
    bits[frame.data_mask], tx[frame.data_mask] = ~bits[frame.data_mask], 1000j
    altered = candidates_and_guards(model, replace(context, frame=replace(frame, bits=bits, tx_symbols=tx)), ("channel_state_signal", "channel_state_map"))
    assert original[2] == altered[2]
    assert len(original[2]) == 6
    for name in original[1]:
        torch.testing.assert_close(original[1][name], altered[1][name], rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_channel_state_map_preserves_posterior_stationary_point():
    from training.pilot_transfer import ChannelStateSignalObjective

    _, model, context = _channel_state_fixture()
    model.set_trainable_groups({"channel_state_head"})
    with torch.no_grad():
        model.channel_state.head.bias.copy_(torch.atanh(torch.tensor([.5])))
    parameters = [p for _, p in model.peft.named_group_parameters({"channel_state_head"})]
    map_loss = ChannelStateSignalObjective(model, include_prior=True)(context)
    likelihood = ChannelStateSignalObjective(model)(context)
    map_gradient = torch.cat([g.flatten() for g in torch.autograd.grad(map_loss, parameters)])
    signal_gradient = torch.cat([g.flatten() for g in torch.autograd.grad(likelihood, parameters)])
    assert float(map_gradient.norm()) < 1e-5
    assert float(signal_gradient.norm()) > 1e-4
    with pytest.raises(ValueError, match="先验惩罚"):
        ChannelStateSignalObjective(model, role="reward", include_prior=True)


def test_retention_pilot_rules_ignore_data_labels_and_preserve_state():
    """保留或撤销只由 Pilot 决定；候选生成和事后比较均不改写离线权重。"""
    from scripts.diagnose_channel_state_retention import pilot_comparison

    _, model, context = _channel_state_fixture()
    with torch.no_grad():
        model.channel_state.head.bias.fill_(.1)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    fast, update = differentiable_adapt(model, [context], learning_rate=.02, steps=3,
        max_delta_norm=.2, groups={"channel_state_head"}, objective="channel_state_signal", receiver_input="full_rx")
    fast = {name: value.detach() for name, value in fast.items()}
    original = pilot_comparison(model, context, fast, update["delta_norm"])
    frame = context.frame
    bits, tx = frame.bits.clone(), frame.tx_symbols.clone()
    bits[frame.data_mask], tx[frame.data_mask] = ~bits[frame.data_mask], 1000j
    changed = pilot_comparison(model, replace(context, frame=replace(frame, bits=bits, tx_symbols=tx)), fast, update["delta_norm"])
    assert original[1] == changed[1]
    assert not (original[1]["revalidate_keep"] and original[1]["rollback"])
    for name in original[0]:
        torch.testing.assert_close(original[0][name], changed[0][name], atol=0, rtol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_retention_rejects_heldout_and_output_reuse_before_loading(tmp_path):
    from scripts.diagnose_channel_state_retention import run

    config = {"offline_state_split": "heldout_edge", "main_level": "B", "pilot_layout": "prefix"}
    with pytest.raises(ValueError, match="训练分布"):
        run(config, "missing.pt", [79000], "cpu", tmp_path / "new.json")
    config["offline_state_split"] = "offline_train"
    output = tmp_path / "old.json"
    output.write_text("保留原始结果", encoding="utf-8")
    with pytest.raises(ValueError, match="新的输出文件"):
        run(config, "missing.pt", [79000], "cpu", output)
    assert output.read_text(encoding="utf-8") == "保留原始结果"


@pytest.mark.parametrize("offsets", [[], [1], [0, -1], [0, 1, 1], [0, 2, 1], [0, 1.5], [0, True]])
def test_future_outer_targets_reject_invalid_offsets(offsets):
    model, _ = _fixture()
    config = {"model": model.config.to_dict(), "pilot_layout": "prefix", "main_level": "B",
              "offline_state_split": "offline_train", "pilot_transfer": {"post_target_frame_offsets": offsets}}
    with pytest.raises(ValueError, match="外层帧偏移"):
        PilotTransferTrainer(config, "cpu")


@pytest.mark.parametrize("objective", ["baseline", "post_adapt"])
def test_future_outer_targets_share_exposure_and_keep_inner_causal(monkeypatch, objective):
    import training.pilot_transfer as transfer
    from types import SimpleNamespace

    _, model, context = _channel_state_fixture()
    settings = {**_channel_state_settings(), "post_target_frame_offsets": [0, 8], "history_frame_choices": [7]}
    config = {"model": model.config.to_dict(), "pilot_layout": "prefix", "main_level": "B",
              "offline_state_split": "offline_train", "main_snrs": [10], "pilot_transfer": settings}
    trainer = PilotTransferTrainer(config, "cpu", objective=objective)
    trainer.model.load_state_dict(model.state_dict())

    def episode(*args, frames, **kwargs):
        assert frames == 16
        return SimpleNamespace(state_metadata=lambda: {}), None, None, context.tail, [
            replace(context.frame, frame_index=index) for index in range(frames)]

    monkeypatch.setattr(transfer, "make_episode", episode)
    monkeypatch.setattr(transfer, "make_context", lambda frame, *args: replace(context, frame=frame))
    sample = trainer._sample()[0]
    assert sample.frame.frame_index == 7
    assert [c.frame.frame_index for c in sample.offline_post_contexts] == [15]
    hidden = replace(sample.offline_post_contexts[0], frame=replace(sample.offline_post_contexts[0].frame,
                     bits=~sample.frame.bits, tx_symbols=-sample.frame.tx_symbols,
                     rx_symbols=torch.full_like(sample.frame.rx_symbols, 1000j)))
    options = dict(learning_rate=.005, steps=1, max_delta_norm=.05,
                   groups={"channel_state_head"}, objective="channel_state_signal", receiver_input="full_rx")
    fast, info = differentiable_adapt(model, [sample], **options)
    changed, changed_info = differentiable_adapt(model, [replace(sample, offline_post_contexts=(hidden,))], **options)
    assert info == changed_info
    for name in fast:
        torch.testing.assert_close(fast[name], changed[name], rtol=0, atol=0)
    visited = []
    original_logits = transfer.context_logits

    def observe(model, current, *args, **kwargs):
        visited.append(current.frame.frame_index)
        return original_logits(model, current, *args, **kwargs)

    monkeypatch.setattr(trainer, "_sample", lambda: (sample, {}, 10, 80000))
    monkeypatch.setattr(transfer, "context_logits", observe)
    metrics = trainer.train(1)
    assert visited == [7, 15, 7, 15]
    assert metrics["post_target_frame_offsets"] == [0, 8]
    assert metrics["history"][0]["post_target_frame_indices"] == [7, 15]


def test_capacity_age_summary_does_not_mix_ages_or_accept_duplicate_seeds():
    from scripts.diagnose_channel_state_capacity import summarize_by_age

    rows = []
    for seed in [1, 2, 3]:
        for age, gain in [(0, .01), (4095, -.02)]:
            rows.append({"snr_db": 10, "frame": age, "action": "medium:channel_state_signal",
                "seed": seed, "accepted": True, "ber_frozen": .1, "ber_candidate": .1 - gain,
                "ber_committed": .1 - gain, "bce_frozen": .2, "bce_candidate": .2 - gain, "cir_step_norm": .02})
    summary = summarize_by_age(rows, [1, 2, 3])
    assert [r["frame"] for r in summary] == [0, 4095]
    assert [r["ber_committed_gain"]["mean"] for r in summary] == pytest.approx([.01, -.02])
    assert all(r["ber_committed_gain"]["seeds"] == 3 for r in summary)
    with pytest.raises(ValueError, match="恰好一个"):
        summarize_by_age(rows + rows[:1], [1, 2, 3])


def test_channel_state_windows_hide_all_data_labels_and_preserve_weights():
    from scripts.diagnose_channel_state_windows import window_candidates

    _, model, context = _channel_state_fixture()
    with torch.no_grad():
        model.channel_state.head.bias.fill_(.1)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    contexts = [replace(context, frame=replace(context.frame, frame_index=index)) for index in [7, 8]]
    original = window_candidates(model, contexts, windows=(1, 2))
    changed_contexts = []
    for current in contexts:
        frame = current.frame
        bits, tx = frame.bits.clone(), frame.tx_symbols.clone()
        bits[frame.data_mask], tx[frame.data_mask] = ~bits[frame.data_mask], 1000j
        changed_contexts.append(replace(current, frame=replace(frame, bits=bits, tx_symbols=tx)))
    changed = window_candidates(model, changed_contexts, windows=(1, 2))
    assert original[2] == changed[2]
    assert len(original[2]) == 4
    torch.testing.assert_close(original[0], changed[0], atol=0, rtol=0)
    for window in [1, 2]:
        torch.testing.assert_close(original[1][window], changed[1][window], atol=0, rtol=0)
    # Reward 标签可改变验收评分，但绝不能改变 Adapt 产生的候选。
    reward_poison = []
    for current in contexts:
        frame = current.frame
        bits, tx = frame.bits.clone(), frame.tx_symbols.clone()
        bits[frame.reward_mask], tx[frame.reward_mask] = ~bits[frame.reward_mask], -tx[frame.reward_mask]
        reward_poison.append(replace(current, frame=replace(frame, bits=bits, tx_symbols=tx)))
    altered = window_candidates(model, reward_poison, windows=(1, 2))
    for window in [1, 2]:
        torch.testing.assert_close(original[1][window], altered[1][window], atol=0, rtol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


def test_channel_state_window_one_matches_original_medium_candidate_and_guard():
    from scripts.diagnose_channel_state_windows import window_candidates
    from scripts.diagnose_channel_state_capacity import candidates_and_guards, ACTIONS

    _, model, context = _channel_state_fixture()
    original = candidates_and_guards(model, context, actions=(ACTIONS[1],))
    current = window_candidates(model, [context], windows=(1,))
    assert original[2][0]["accepted"] == current[2][0]["accepted"]
    for field in ["reward_loss_before", "reward_loss_after", "reward_ber_before", "reward_ber_after", "delta_norm"]:
        assert original[2][0][field] == current[2][0][field]
    torch.testing.assert_close(original[1]["medium:channel_state_signal"], current[1][1], atol=0, rtol=0)
    with pytest.raises(ValueError, match="历史必须足够"):
        window_candidates(model, [context], windows=(1, 8))
    with pytest.raises(ValueError, match="连续已接收帧"):
        window_candidates(model, [context, replace(context, frame=replace(context.frame, frame_index=2))], windows=(1, 2))


def test_long_age_offline_cache_pairs_branches_and_rejects_wrong_protocol(tmp_path, monkeypatch):
    import training.pilot_transfer as transfer
    from types import SimpleNamespace

    _, model, context = _channel_state_fixture()
    settings = {**_channel_state_settings(), "post_target_frame_offsets": [0, 8],
                "history_frame_choices": [7], "offline_sample_cache_dir": str(tmp_path / "samples")}
    config = {"model": model.config.to_dict(), "pilot_layout": "prefix", "main_level": "B",
              "offline_state_split": "offline_train", "main_snrs": [10], "pilot_transfer": settings}
    generated = []

    def episode(*args, frames, **kwargs):
        generated.append(frames)
        return SimpleNamespace(state_metadata=lambda: {"paired": True}), context.condition.complex_cir.flatten(), 0., context.tail, [
            replace(context.frame, frame_index=index) for index in range(frames)]

    monkeypatch.setattr(transfer, "make_episode", episode)
    monkeypatch.setattr(transfer, "make_context", lambda frame, *args: replace(context, frame=frame))
    trainers = [PilotTransferTrainer(config, "cpu", objective=objective) for objective in ["baseline", "post_adapt"]]
    samples = [trainer._sample() for trainer in trainers]
    assert generated == [16]
    assert trainers[0].sample_cache_misses == trainers[1].sample_cache_hits == 1
    assert trainers[0].last_sample_cache_key == trainers[1].last_sample_cache_key
    assert samples[0][1:] == samples[1][1:]
    for left, right in zip((samples[0][0],) + samples[0][0].offline_post_contexts,
                           (samples[1][0],) + samples[1][0].offline_post_contexts):
        assert left.frame.frame_index == right.frame.frame_index
        for name in ["rx_symbols", "bits", "tx_symbols", "adapt_mask", "reward_mask", "data_mask"]:
            torch.testing.assert_close(getattr(left.frame, name), getattr(right.frame, name), rtol=0, atol=0)
    cache = next((tmp_path / "samples").glob("*.pt"))
    payload = torch.load(cache, weights_only=False)
    payload["protocol"]["seed"] += 1
    torch.save(payload, cache)
    trainers[1].sample_index = 0
    with pytest.raises(ValueError, match="缓存协议"):
        trainers[1]._sample()


def test_channel_state_geometry_jacobian_matches_finite_difference():
    from scripts.diagnose_channel_state_geometry import state_jacobian, real_vector

    _, model, context = _channel_state_fixture()
    with torch.no_grad():
        model.channel_state.head.bias.fill_(.1)
    _, jacobian, theta, state = state_jacobian(model, context)
    direction = torch.linspace(-1, 1, theta.numel())
    direction = direction / direction.norm()
    finite = real_vector(state(theta + .001 * direction) - state(theta - .001 * direction)) / .002
    assert jacobian.shape == (10, 17)
    assert float(finite.norm()) > 1e-5
    torch.testing.assert_close(jacobian @ direction, finite, atol=3e-5, rtol=.01)


def test_channel_state_geometry_data_derivatives_do_not_generate_candidates_or_mutate_weights():
    from scripts.diagnose_channel_state_geometry import analyze_geometry

    _, model, context = _channel_state_fixture()
    with torch.no_grad():
        model.channel_state.head.bias.fill_(.1)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    original = analyze_geometry(model, context)
    frame = context.frame
    bits, tx = frame.bits.clone(), frame.tx_symbols.clone()
    bits[frame.data_mask], tx[frame.data_mask] = ~bits[frame.data_mask], 1000j
    changed = analyze_geometry(model, replace(context, frame=replace(frame, bits=bits, tx_symbols=tx)))
    for field in ["singular_values", "rank_relative_1e5", "first_singular_energy_fraction",
                  "adapt_gradient_norm", "actual_cir_step_norm", "actual_head_delta_norm"]:
        assert original[field] == changed[field]
    assert not original["data_gradient_used_for_action"]
    assert 0 <= original["offline_data_reachable_energy_fraction"] <= 1 + 1e-8
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    assert all(value.grad is None for value in model.parameters())


def _channel_memory_fixture():
    _, source, context = _channel_state_fixture()
    with torch.no_grad():
        source.channel_state.head.bias.fill_(.2)
    model = UnfoldedEqualizer(replace(source.config, enable_channel_state_memory=True)).eval()
    model.attach_online_phase_trend_adapter()
    state = model.state_dict()
    state.update(source.state_dict())
    model.load_state_dict(state, strict=True)
    return source, model, context


def test_channel_memory_zero_head_preserves_frozen_after_scale_changes_and_has_full_cir_rank():
    from scripts.diagnose_channel_state_capacity import _StateOutput
    from scripts.diagnose_channel_state_geometry import real_vector
    from torch.func import functional_call

    source, model, context = _channel_memory_fixture()
    assert sum(p.numel() for _, p in model.peft.named_group_parameters({"channel_state_memory"})) == 10
    assert sum(p.numel() for _, p in model.peft.named_group_parameters({"channel_state_memory_scale"})) == 34
    with torch.no_grad():
        model.channel_state.memory_scale.weight.fill_(3)
        model.channel_state.memory_scale.bias.fill_(-2)
        torch.testing.assert_close(context_logits(source, context), context_logits(model, context), atol=0, rtol=0)
    wrapper = _StateOutput(model)
    name, parameter = model.peft.named_group_parameters({"channel_state_memory"})[0]
    theta = parameter.detach().flatten()
    jac = torch.autograd.functional.jacobian(lambda value: real_vector(functional_call(wrapper,
        {"model." + name: value.view_as(parameter)}, (context,))), theta, vectorize=True)
    assert jac.shape == (10, 10) and int(torch.linalg.matrix_rank(jac)) == 10
    assert bool(torch.isfinite(jac).all())


@pytest.mark.parametrize("objective", ["channel_state_signal", "channel_state_map"])
def test_channel_memory_updates_match_adamw_ignore_hidden_labels_and_keep_base(objective):
    from training.pilot_transfer import ChannelStateSignalObjective

    _, model, context = _channel_memory_fixture()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    options = dict(learning_rate=.02, steps=3, max_delta_norm=.8,
                   groups={"channel_state_memory"}, objective=objective, receiver_input="full_rx")
    fast, info = differentiable_adapt(model, [context], **options)
    frame = context.frame
    bits, tx, rx = frame.bits.clone(), frame.tx_symbols.clone(), frame.rx_symbols.clone()
    hidden = ~frame.adapt_mask
    bits[hidden], tx[hidden], rx[hidden] = ~bits[hidden], 1000j, -1000j
    altered, changed = differentiable_adapt(model, [replace(context, frame=replace(frame, bits=bits, tx_symbols=tx, rx_symbols=rx))], **options)
    assert info == changed
    for name in fast:
        torch.testing.assert_close(fast[name], altered[name], atol=0, rtol=0)
    _, actual, _ = _channel_memory_fixture()
    actual.load_state_dict(model.state_dict())
    actual.set_trainable_groups({"channel_state_memory"})
    name, parameter = actual.peft.named_group_parameters({"channel_state_memory"})[0]
    start = parameter.detach().clone()
    optimizer = torch.optim.AdamW([parameter], lr=.02)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        ChannelStateSignalObjective(actual, include_prior=objective == "channel_state_map")(context).backward()
        optimizer.step()
    with torch.no_grad():
        delta = parameter - start
        parameter.copy_(start + delta * (.8 / delta.norm().clamp_min(1e-12)).clamp(max=1.))
    torch.testing.assert_close(fast[name], parameter, atol=2e-7, rtol=3e-5)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


def test_channel_memory_explicit_migration_and_offline_scale_training_keep_frozen(tmp_path, monkeypatch):
    source, model, context = _channel_memory_fixture()
    checkpoint = tmp_path / "source.pt"
    torch.save({"model_config": source.config.to_dict(), "state_dict": {name: value for name, value in source.state_dict().items()
        if not name.startswith("online_phase_trend_adapter.")}}, checkpoint)
    settings = {**_channel_state_settings(), "inner_groups": ["channel_state_memory"],
                "outer_groups": ["channel_state_memory_scale"], "inner_learning_rate": .02,
                "inner_steps": 3, "max_delta_norm": .8}
    config = {"model": model.config.to_dict(), "pilot_layout": "prefix", "main_level": "B",
              "offline_state_split": "offline_train", "pilot_transfer": settings}
    trainer = PilotTransferTrainer(config, "cpu", objective="post_adapt")
    with pytest.raises(ValueError, match="参数结构不一致"):
        trainer.load_resume(checkpoint)
    trainer.settings["initialize_channel_state_memory_from_base"] = True
    trainer.load_resume(checkpoint)
    before = {name: value.clone() for name, value in trainer.model.state_dict().items()}
    monkeypatch.setattr(trainer, "_sample", lambda: (context, {}, 10, 98000))
    metrics = trainer.train(1)
    assert metrics["outer_trainable_parameter_count"] == 34
    assert metrics["initialized_modules"] == ["channel_state_memory"]
    with torch.no_grad():
        torch.testing.assert_close(context_logits(source, context), context_logits(trainer.model, context), atol=0, rtol=0)
    assert any(not torch.equal(value, before[name]) for name, value in trainer.model.state_dict().items()
               if name.startswith("channel_state.memory_scale."))
    for name, value in trainer.model.state_dict().items():
        if not name.startswith("channel_state.memory_scale."):
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    trainer.save(tmp_path / "saved", metrics)
    payload = torch.load(tmp_path / "saved/model_best.pt", weights_only=False)
    standalone = UnfoldedEqualizer(UnfoldedConfig.from_dict(payload["model_config"]))
    standalone.load_state_dict(payload["state_dict"], strict=True)


def test_channel_memory_capacity_guards_ignore_data_labels_and_never_commit_candidates():
    from scripts.diagnose_channel_state_capacity import candidates_and_guards
    from scripts.diagnose_channel_memory_capacity import ACTIONS

    _, model, context = _channel_memory_fixture()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    options = dict(objectives=("channel_state_signal", "channel_state_map"), actions=ACTIONS,
                   groups=("channel_state_memory",))
    original = candidates_and_guards(model, context, **options)
    frame = context.frame
    bits, tx = frame.bits.clone(), frame.tx_symbols.clone()
    bits[frame.data_mask], tx[frame.data_mask] = ~bits[frame.data_mask], 1000j
    changed = candidates_and_guards(model, replace(context, frame=replace(frame, bits=bits, tx_symbols=tx)), **options)
    assert original[2] == changed[2]
    assert len(original[2]) == 4
    for name in original[1]:
        torch.testing.assert_close(original[1][name], changed[1][name], atol=0, rtol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


def test_channel_memory_offline_gradient_diagnostic_cannot_mutate_online_state():
    from scripts.diagnose_channel_memory_capacity import offline_scale_gradients

    _, model, context = _channel_memory_fixture()
    model.set_trainable_groups({"channel_state_memory_scale"})
    context = replace(context, frame=replace(context.frame,
        rx_symbols=context.frame.rx_symbols + 1.2 * torch.linspace(-1, 1, context.frame.rx_symbols.numel())))
    future = replace(context, frame=replace(context.frame, frame_index=8))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    rows = offline_scale_gradients(model, [context, future])
    assert len(rows) == 2
    assert all(row["base_gradient_norm"] == 0 and row["post_gradient_norm"] > 1e-7 for row in rows)
    assert all(not row["data_gradients_used_for_control"] and len(row["post_gradient"]) == 34 for row in rows)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_channel_memory_history_uses_last_frame_and_ignores_both_frames_hidden_labels():
    from scripts.diagnose_channel_state_windows import window_candidates

    _, model, context = _channel_memory_fixture()
    future = replace(context, frame=replace(context.frame, frame_index=8,
                     rx_symbols=context.frame.rx_symbols * (.9 + .1j)))
    contexts = [context, future]
    options = dict(windows=(1, 2), groups=("channel_state_memory",), objective="channel_state_map",
                   max_delta_norm=.8, require_consecutive=False)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    original = window_candidates(model, contexts, **options)
    altered = []
    for current in contexts:
        frame = current.frame
        bits, tx = frame.bits.clone(), frame.tx_symbols.clone()
        bits[frame.data_mask], tx[frame.data_mask] = ~bits[frame.data_mask], 1000j
        altered.append(replace(current, frame=replace(frame, bits=bits, tx_symbols=tx)))
    changed = window_candidates(model, altered, **options)
    assert original[2] == changed[2] and len(original[2]) == 4
    with torch.no_grad():
        torch.testing.assert_close(original[0], context_logits(model, future), atol=0, rtol=0)
    for window in [1, 2]:
        torch.testing.assert_close(original[1][window], changed[1][window], atol=0, rtol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    with pytest.raises(ValueError, match="严格递增"):
        window_candidates(model, contexts[::-1], **options)


def test_level_b_pilot_history_fills_algebraic_rank_without_using_data():
    from scripts.diagnose_channel_memory_history import pilot_spectrum

    generator = FrameGenerator(FrameConfig(frame_len=1024, total_pilot=256, reward_pilot_total=32,
                                           max_delay=116, layout="prefix"), seed=7)
    condition = condition_from_cir(torch.zeros(117, dtype=torch.complex64), 10)
    contexts = [TransferContext(generator.generate(index), condition, torch.zeros(116, dtype=torch.complex64))
                for index in [0, 8]]
    single, pooled = pilot_spectrum(contexts[:1]), pilot_spectrum(contexts)
    assert single["rows"] == single["complex_rank"] == 108 and single["complex_nullity"] == 9
    assert pooled["rows"] == 216 and pooled["complex_rank"] == 117 and pooled["complex_nullity"] == 0
    changed = []
    for context in contexts:
        frame = context.frame
        bits, tx, rx = frame.bits.clone(), frame.tx_symbols.clone(), frame.rx_symbols.clone()
        bits[frame.data_mask], tx[frame.data_mask], rx[:] = ~bits[frame.data_mask], 1000j, -1000j
        changed.append(replace(context, frame=replace(frame, bits=bits, tx_symbols=tx, rx_symbols=rx)))
    assert pilot_spectrum(changed) == pooled


def test_formal_frozen_and_ppo_paths_preserve_physical_age_instead_of_local_counter():
    import compare
    from types import SimpleNamespace
    from env.comm_env import ReceiverState
    from agent.discrete_safe_policy import DiscreteSafePolicy, safe_modulation_actions
    from training.windowed_discrete_ppo import WindowedDiscreteOnlineState, WindowRewardAccumulator, run_windowed_discrete_frame

    _, model, context = _channel_memory_fixture()
    frame = replace(context.frame, frame_index=1023)
    actions = safe_modulation_actions(len(model.blocks), action_names=["identity"])
    cir = context.condition.complex_cir.flatten()
    ages = []
    hook = model.channel_state.register_forward_pre_hook(lambda module, args: ages.append(args[-1]))
    try:
        frozen = compare.NeuralMethodState(cir=cir, receiver_state=ReceiverState(context.tail.clone()),
            model=model, modulation=actions[0].modulation, pretrained_loaded=True, condition_source="pilot_cir_phase")
        compare._run_new_or_single_method("Frozen Offline NN", frame, 10., frozen, 4, 1, 8)
        assert ages == [1023, 1023]
        ages.clear()
        policy = DiscreteSafePolicy(observation_dim=4, action_count=1, hidden_size=8)
        state = WindowedDiscreteOnlineState(cir=cir, receiver_state=ReceiverState(context.tail.clone()),
            model=model, policy=policy, optimizer=torch.optim.AdamW(policy.parameters()), encoder=SimpleNamespace(),
            actions=actions, hidden=policy.initial_hidden(), window_size=4, previous_window_reward=0.,
            last_action_delta_norm=0., rollout=[], current_action=actions[0], frames_in_window=1,
            current_accumulator=WindowRewardAccumulator(0.))
        row = run_windowed_discrete_frame(state, frame, context.condition, 10., 1, 8)
        assert ages and set(ages) == {1023}
        assert row["channel_frame_index"] == 1023 and row["frame"] == 1
    finally:
        hook.remove()


@pytest.mark.parametrize("objective", ["channel_state_signal", "channel_state_map"])
@pytest.mark.parametrize("future_target", [False, True])
def test_channel_memory_offline_scale_gradient_matches_finite_difference(objective, future_target):
    _, model, context = _channel_memory_fixture()
    # 避免无噪声 toy 已接近零 BCE，使导数核验退化为数值零。
    context = replace(context, frame=replace(context.frame,
        rx_symbols=context.frame.rx_symbols + 1.2 * torch.linspace(-1, 1, context.frame.rx_symbols.numel())))
    model.set_trainable_groups({"channel_state_memory_scale"})
    target = (replace(context, frame=replace(context.frame, frame_index=8,
              rx_symbols=context.frame.rx_symbols * (.9 + .1j))) if future_target else context)

    def loss():
        with sdpa_kernel(SDPBackend.MATH):
            fast, _ = differentiable_adapt(model, [context], learning_rate=.02, steps=3,
                max_delta_norm=.8, create_graph=True, groups={"channel_state_memory"},
                objective=objective, receiver_input="full_rx")
            logits = context_logits(model, target, fast)
            mask = target.frame.data_mask
            return F.binary_cross_entropy_with_logits(logits[mask], target.frame.bits[mask].float())

    probe = model.channel_state.memory_scale.bias
    gradient, = torch.autograd.grad(loss(), probe)
    start = probe.detach().clone()
    values = []
    for sign in [-1, 1]:
        with torch.no_grad():
            probe.copy_(start)
            probe[0].add_(sign * .002)
        values.append(float(loss().detach()))
    assert abs(float(gradient[0])) > 1e-7
    assert float(gradient[0]) == pytest.approx((values[1] - values[0]) / .004, abs=2e-4, rel=.05)
