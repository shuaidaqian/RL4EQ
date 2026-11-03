# -*- coding: utf-8 -*-
"""跨帧联合观测、连续相位和在线 Data 标签隔离的机制验证。"""

from dataclasses import replace

import pytest
import torch

from agent.cir_estimator import condition_from_cir
from agent.discrete_safe_policy import DiscreteSafePolicy, safe_modulation_actions
from agent.rl_modulator import ModulationObservationEncoder
from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from env.comm_env import CommEnvConfig, CommunicationEnvironment, ReceiverState
from env.linear_operator import CrossFrameLinearOperator
from training.online_adaptation import PilotDrivenOnlineAdapter, _frame_lookahead_model_kwargs
from training.windowed_discrete_ppo import WindowedDiscreteOnlineState, run_windowed_discrete_frame


def _model():
    torch.manual_seed(198)
    return UnfoldedEqualizer(UnfoldedConfig(
        frame_len=128, max_delay=8, iterations=1, d_model=8, num_heads=2,
        physics_warm_start_iterations=12, physics_warm_start_scale=1.,
        pilot_conditioned=True, enable_phase_correction_branch=True,
        phase_correction_initial_scale=1.,
    )).eval()


def _episode():
    env = CommunicationEnvironment(CommEnvConfig(
        frame_len=128, max_delay=8, total_pilot=64, layout="prefix", seed=7813,
    ))
    start = env.reset_episode()
    return env, start


def _poison_data(frame):
    # 保留已接收波形和合法 Pilot，只毒化仿真器隐藏的 Data 真值。
    bits, symbols = frame.bits.clone(), frame.tx_symbols.clone()
    bits[frame.data_mask] = ~bits[frame.data_mask]
    symbols[frame.data_mask] = complex(float("nan"), float("nan"))
    return replace(frame, bits=bits, tx_symbols=symbols)


def test_joint_observation_recovers_unobservable_pure_delay_tail():
    model = _model()
    cir = torch.zeros(1, 9, dtype=torch.complex64)
    cir[:, 8] = 1.
    condition = replace(condition_from_cir(cir, 60.), noise_variance=torch.tensor([1e-6]))
    tx = torch.where(torch.arange(128) % 2 == 0, 1., -1.).to(torch.complex64)[None]
    future = torch.ones(1, 8, dtype=torch.complex64)
    tail = torch.zeros(1, 8, dtype=torch.complex64)
    joint = CrossFrameLinearOperator(128, 8, 8).forward(tx, future, cir, tail)
    plain = model._physics_warm_start_logits(joint[:, :128], condition, tail)
    extended = model._physics_warm_start_logits(joint[:, :128], condition, tail, joint[:, 128:], future)
    assert torch.equal(plain[:, -8:], torch.zeros(1, 8))
    assert torch.equal(extended > 0, tx.real > 0)
    assert bool((extended[:, -8:].abs() > 10).all())


def test_phase_correction_uses_one_continuous_coordinate_for_suffix(monkeypatch):
    model = _model()
    model.attach_online_phase_trend_adapter()
    with torch.no_grad():
        model.online_phase_trend_adapter.raw_cfo.fill_(.3)
    cir = torch.zeros(1, 9, dtype=torch.complex64)
    cir[:, 0] = 1.
    condition = condition_from_cir(cir, 10.)
    rx = torch.randn(1, 136, dtype=torch.complex64)
    iq = torch.stack((rx.real, rx.imag), dim=-1)
    expected = model.online_phase_trend_adapter(model._apply_phase_correction(iq, condition))
    captured = {}

    def capture(current, condition, tail, **kwargs):
        captured["rx"] = torch.cat((current, kwargs["lookahead_rx_symbols"]), dim=1)
        return current.real * 0

    monkeypatch.setattr(model, "_physics_warm_start_logits", capture)
    model(iq[:, :128], condition, torch.zeros(1, 128, dtype=torch.long),
          torch.zeros(1, 8, dtype=torch.complex64),
          adapt_symbols=torch.ones(1, 128, dtype=torch.complex64),
          adapt_mask=torch.ones(1, 128, dtype=torch.bool),
          lookahead_rx_symbols=rx[:, 128:],
          lookahead_adapt_symbols=torch.ones(1, 8, dtype=torch.complex64))
    torch.testing.assert_close(captured["rx"], torch.complex(expected[..., 0], expected[..., 1]))


def test_future_hidden_data_does_not_change_received_prefix(monkeypatch):
    clean, _ = _episode()
    poisoned, _ = _episode()
    transmit = poisoned._transmit_next_frame

    def receive_then_poison():
        frame = transmit()
        return _poison_data(frame) if frame.frame_index == 1 else frame

    monkeypatch.setattr(poisoned, "_transmit_next_frame", receive_then_poison)
    first, second = clean.next_frame(8), poisoned.next_frame(8)
    assert torch.equal(first.lookahead_rx_symbols, second.lookahead_rx_symbols)
    assert torch.equal(first.lookahead_adapt_symbols, second.lookahead_adapt_symbols)
    for name in ("lookahead_bits", "lookahead_tx_symbols", "lookahead_data_mask"):
        assert not hasattr(second.receiver_view(), name)


def test_cross_frame_action_gradient_reward_and_state_ignore_current_data():
    env, start = _episode()
    frame = env.next_frame(8)
    condition = condition_from_cir(frame.true_cir.unsqueeze(0), 10.)

    def run(item):
        model = _model()
        actions = safe_modulation_actions(len(model.blocks), action_names=["identity", "peft_head_light"])
        encoder = ModulationObservationEncoder()
        policy = DiscreteSafePolicy(len(encoder.FIELDS), len(actions), hidden_size=8)
        with torch.no_grad():
            policy.logits.weight.zero_()
            policy.logits.bias.copy_(torch.tensor([-100., 100.]))
        state = WindowedDiscreteOnlineState(
            cir=frame.true_cir.clone(), receiver_state=ReceiverState(start.initial_soft_tail.clone()),
            model=model, policy=policy, optimizer=torch.optim.AdamW(policy.parameters(), lr=1e-4),
            encoder=encoder, actions=actions, hidden=policy.initial_hidden(), window_size=1,
            previous_window_reward=0., last_action_delta_norm=0., rollout=[],
            reward_receiver_input="full_rx", adapt_receiver_input="full_rx",
        )
        gradients = []
        model.set_trainable_groups({"head"})
        handle = model.head.weight.register_hook(lambda grad: gradients.append(grad.detach().clone()))
        row = run_windowed_discrete_frame(state, item, condition, 10., 1, 1)
        handle.remove()
        return state, row, gradients

    left, row_a, gradients_a = run(frame)
    right, row_b, gradients_b = run(_poison_data(frame))
    assert row_a["policy_action_name"] == row_b["policy_action_name"] == "peft_head_light"
    assert gradients_a and len(gradients_a) == len(gradients_b)
    for a, b in zip(gradients_a, gradients_b):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    evaluation_labels = {"ber_data", "pre_update_ber_data", "data_bce_before_eval_only",
                         "data_bce_after_eval_only", "data_bce_improvement_eval_only"}
    for key in row_a.keys() - evaluation_labels:
        assert row_a[key] == row_b[key], key
    assert row_a["ber_data"] + row_b["ber_data"] == pytest.approx(1.)
    torch.testing.assert_close(left.receiver_state.soft_tail, right.receiver_state.soft_tail, atol=0, rtol=0)
    torch.testing.assert_close(left.current_observation, right.current_observation, atol=0, rtol=0)
    for key, value in left.model.state_dict().items():
        torch.testing.assert_close(value, right.model.state_dict()[key], atol=0, rtol=0)
    for key, value in left.policy.state_dict().items():
        torch.testing.assert_close(value, right.policy.state_dict()[key], atol=0, rtol=0)


def test_pilot_adapter_forwards_suffix_before_during_and_after_update(monkeypatch):
    env, start = _episode()
    frame = env.next_frame(8)
    model = _model()
    original = model.forward
    calls = []

    def capture(*args, **kwargs):
        calls.append(kwargs.get("lookahead_rx_symbols"))
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward", capture)
    adapter = PilotDrivenOnlineAdapter(model, groups={"head"}, learning_rate=.001, steps=1)
    result = adapter.adapt(frame, condition_from_cir(frame.true_cir.unsqueeze(0), 10.),
                           start.initial_soft_tail, lookahead_kwargs=_frame_lookahead_model_kwargs(frame))
    assert result.adapt_pilot_count == int(frame.adapt_mask.sum())
    assert len(calls) == 3
    assert all(torch.equal(value, frame.lookahead_rx_symbols.unsqueeze(0)) for value in calls)


def test_offline_inner_loop_matches_online_cross_frame_update_and_ignores_data():
    from training.pilot_transfer import TransferContext, differentiable_adapt, prefix_context
    from training.windowed_discrete_ppo import _apply_adapt_peft_update

    env, start = _episode()
    frame = env.next_frame(8)
    context = TransferContext(frame, condition_from_cir(frame.true_cir.unsqueeze(0), 10.), start.initial_soft_tail)
    expected, _ = differentiable_adapt(_model(), [context], learning_rate=.001, steps=1,
        max_delta_norm=.05, groups={"head"}, objective="bce", receiver_input="full_rx", create_graph=False)
    poisoned, _ = differentiable_adapt(_model(), [replace(context, frame=_poison_data(frame))],
        learning_rate=.001, steps=1, max_delta_norm=.05, groups={"head"}, objective="bce",
        receiver_input="full_rx", create_graph=False)
    actual = _model()
    _apply_adapt_peft_update(actual, frame, context.condition, context.tail, groups={"head"},
        lr=.001, steps=1, max_delta_norm=.05, objective="bce", receiver_input="full_rx",
        lookahead_kwargs=_frame_lookahead_model_kwargs(frame))
    for name, value in actual.named_parameters():
        torch.testing.assert_close(value, expected[name], atol=1e-7, rtol=2e-5)
        torch.testing.assert_close(expected[name], poisoned[name], atol=0, rtol=0)
    # 真正缩短到 Pilot 的诊断窗与下一帧后缀不再相邻，必须清除后缀。
    short = prefix_context(context, 48)
    assert short.frame.lookahead_rx_symbols is None
    assert short.frame.lookahead_adapt_symbols is None
    assert short.frame.lookahead_frame_index is None
