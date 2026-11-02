# -*- coding: utf-8 -*-
"""复记忆正式入口的接收信息、同步与配置契约。"""

import json
from dataclasses import replace

import pytest
import torch

from agent.discrete_safe_policy import DiscreteSafePolicy, safe_modulation_actions
from agent.rl_modulator import ModulationObservationEncoder
from env.comm_env import ReceiverState
from tests.test_pilot_transfer import _channel_memory_fixture
from training import windowed_discrete_ppo as runner


def _state(model, context):
    actions = safe_modulation_actions(len(model.blocks), action_names=["identity", "peft_memory_map_small"],
                                      include_channel_state_memory=True)
    encoder = ModulationObservationEncoder(include_pilot_state=False)
    policy = DiscreteSafePolicy(len(encoder.FIELDS), len(actions), hidden_size=8)
    with torch.no_grad():
        policy.logits.weight.zero_()
        policy.logits.bias.copy_(torch.tensor([-100., 100.]))
    return runner.WindowedDiscreteOnlineState(
        cir=context.condition.complex_cir.flatten().clone(), receiver_state=ReceiverState(context.tail.clone()),
        model=model, policy=policy, optimizer=torch.optim.AdamW(policy.parameters(), lr=.001),
        encoder=encoder, actions=actions, hidden=policy.initial_hidden(), window_size=2,
        previous_window_reward=0., last_action_delta_norm=0., rollout=[], reward_receiver_input="full_rx")


def test_full_rx_memory_control_ignores_data_labels_across_ppo_windows(monkeypatch):
    _, model, context = _channel_memory_fixture()
    clean = _state(model, context)
    _, other_model, _ = _channel_memory_fixture()
    other_model.load_state_dict(model.state_dict())
    poisoned = _state(other_model, context)
    poisoned.policy.load_state_dict(clean.policy.state_dict())
    frame = context.frame
    bits, tx = frame.bits.clone(), frame.tx_symbols.clone()
    bits[frame.data_mask], tx[frame.data_mask] = ~bits[frame.data_mask], 1000j
    changed = replace(frame, bits=bits, tx_symbols=tx)
    seen = []
    original = runner.refine_logits_with_tail

    def capture(model, rx, *args, **kwargs):
        seen.append(rx.detach().clone())
        return original(model, rx, *args, **kwargs)

    monkeypatch.setattr(runner, "refine_logits_with_tail", capture)
    for index in range(1, 5):
        a = runner.run_windowed_discrete_frame(clean, replace(frame, frame_index=1022 + index),
                                               context.condition, 10., index, 2)
        b = runner.run_windowed_discrete_frame(poisoned, replace(changed, frame_index=1022 + index),
                                               context.condition, 10., index, 2)
        for key in ("reward", "policy_loss", "policy_action_probabilities", "peft_frame_accepted",
                    "window_reward_rollback", "trajectory_peft_delta_norm_after_reward"):
            assert a[key] == b[key]
        assert a["peft_delta_norm"] > 0
        assert a["data_waveform_used_online"] is True
        assert a["data_labels_used_online"] is False
        assert a["trajectory_peft_delta_norm_after_reward"] <= .800001
    # 第一次为 Adapt observation，随后两次 Reward 前向必须收到完整波形。
    assert not seen[0][0, ~frame.adapt_mask].any()
    full_rx = torch.stack((frame.rx_symbols.real, frame.rx_symbols.imag), -1).unsqueeze(0)
    for rx in seen[1:3]:
        torch.testing.assert_close(rx, full_rx, atol=0, rtol=0)
    for left, right in ((clean.model, poisoned.model), (clean.policy, poisoned.policy)):
        for key, value in left.state_dict().items():
            torch.testing.assert_close(value, right.state_dict()[key], atol=0, rtol=0)


def test_online_runner_retains_acquisition_cfo_and_honors_config(tmp_path, monkeypatch):
    _, model, _ = _channel_memory_fixture()
    config = {"model": model.config.to_dict(), "rl_action_names": ["identity"],
              "online_ppo_learning_rate": .001, "online_ppo_pilot_state_embedding": False,
              "rl_reward_receiver_input": "full_rx", "impairment_profile": "cfo_phase_tiny"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    observed = []
    original = runner.estimate_phase_residual_vector

    def acquisition(frame, delay, **kwargs):
        cir = torch.zeros(delay + 1, dtype=torch.complex64)
        cir[0] = 1
        return cir, .0007

    def phase(*args, **kwargs):
        observed.append(kwargs["cfo_hint"])
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "estimate_acquisition_cir_with_cfo", acquisition)
    monkeypatch.setattr(runner, "estimate_phase_residual_vector", phase)
    result = runner.run_windowed_discrete_online(path, 2, 1, tmp_path / "out", delays=[4], snrs=[10],
        pilot_total=64, pilot_layout="prefix", window_size=2, update_interval=2, device="cpu")
    assert observed == [.0007, .0007]
    assert all(row["policy_learning_rate"] == .001 and not row["pilot_state_embedding_used"]
               and row["acquisition_cfo"] == .0007 and row["reward_receiver_input"] == "full_rx"
               for row in result["rows"])


def test_memory_contract_checks_actual_layout_and_receiver_protocol():
    _, model, _ = _channel_memory_fixture()
    actions = safe_modulation_actions(len(model.blocks), include_channel_state_memory=True)
    with pytest.raises(ValueError, match="prefix"):
        runner.validate_discrete_receiver_contract(model, actions, "two_block", "full_rx")
    with pytest.raises(ValueError, match="Reward"):
        runner.validate_discrete_receiver_contract(model, actions, "prefix", "unknown")
