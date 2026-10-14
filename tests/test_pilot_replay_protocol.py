# -*- coding: utf-8 -*-
"""防止诊断信道、相位条件和重构目标偏离实际接收流程。"""

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from agent.cir_estimator import condition_from_cir
from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from compare import _neural_phase_features, _reward_signal_reconstruction_loss
from env.frame_structure import FrameConfig, FrameGenerator
from evaluation.pilot_replay_protocol import PilotReplayEpisode, build_pilot_replay_episode
from evaluation.research_diagnostics import _pilot_signal_reconstruction_loss_for_diagnostic
from scripts.replay_cumulative_state_mapping import _adapt_mapping, _reward_gate
from training.online_adaptation import _pilot_signal_reconstruction_loss


def _phase_fixture(known_phase=0.3):
    model = UnfoldedEqualizer(UnfoldedConfig(
        frame_len=128, max_delay=4, iterations=1, d_model=16, num_heads=4,
        enable_phase_correction_branch=True, phase_correction_initial_scale=1.0,
    ))
    model.attach_online_phase_trend_adapter()
    frame = FrameGenerator(FrameConfig(frame_len=128, total_pilot=64, max_delay=4), seed=7).generate(0)
    position = torch.arange(128)
    frame = replace(frame, rx_symbols=frame.tx_symbols * torch.exp(1j * (0.3 + 2 * torch.pi * 0.001 * position)))
    cir = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.complex64)
    features = torch.tensor([known_phase, 0.001, 0.0, 0.0] * 4)
    condition = condition_from_cir(cir, 10, phase_features=features)
    return model, frame, condition, torch.zeros(4, dtype=torch.complex64)


def test_reconstruction_does_not_correct_known_phase_twice():
    """已有补偿正确时，Adapter 应收到零残差，而不是再拟合原始相位。"""
    model, frame, condition, tail = _phase_fixture()
    view = frame.receiver_view()
    loss = _pilot_signal_reconstruction_loss(
        model, condition, view.adapt_symbols, view.rx_symbols, view.adapt_mask, tail,
    )
    diagnostic = _pilot_signal_reconstruction_loss_for_diagnostic(model, frame, condition, tail)
    reward = _reward_signal_reconstruction_loss(model, frame, condition, tail)
    assert float(loss.detach()) < 1e-12
    assert float(diagnostic.detach()) == pytest.approx(float(loss.detach()))
    assert float(reward) < 1e-12
    loss.backward()
    assert abs(float(model.online_phase_trend_adapter.raw_phase.grad)) < 1e-6


@pytest.mark.parametrize("objective", ["signal_reconstruction", "bce", "joint"])
def test_cumulative_mapping_learns_residual_and_ignores_hidden_labels(objective):
    """扰动所有 Reward/Data 标签不应改变累计更新；主干和直接相位参数必须冻结。"""
    model, frame, condition, tail = _phase_fixture(known_phase=0.2)
    hidden = ~frame.adapt_mask
    changed_bits, changed_tx = frame.bits.clone(), frame.tx_symbols.clone()
    changed_bits[hidden] = ~changed_bits[hidden]
    changed_tx[hidden] = -changed_tx[hidden]
    poisoned = replace(frame, bits=changed_bits, tx_symbols=changed_tx)
    second = copy.deepcopy(model)
    original = copy.deepcopy(model.state_dict())
    embedding = torch.tensor([0.4, 0.2, 0.1, 0.5, 0.8, 0.3])
    results = []
    for current, current_frame in ((model, frame), (second, poisoned)):
        results.append(_adapt_mapping(current, [
            {"frame": current_frame, "condition": condition, "tail": tail, "embedding": embedding},
        ], lr=0.005, steps=3, max_delta_norm=0.05, proximal_weight=0.01, objective=objective))
    if objective == "signal_reconstruction":
        assert results[0]["adapt_reconstruction_after"] < results[0]["adapt_reconstruction_before"]
    elif objective == "bce":
        assert results[0]["adapt_bce_after"] < results[0]["adapt_bce_before"]
    else:
        before = (results[0]["adapt_reconstruction_before"] / results[0]["joint_reconstruction_scale"]
                  + results[0]["adapt_bce_before"] / results[0]["joint_bce_scale"])
        after = (results[0]["adapt_reconstruction_after"] / results[0]["joint_reconstruction_scale"]
                 + results[0]["adapt_bce_after"] / results[0]["joint_bce_scale"])
        assert after < before
    assert results[0]["state_mapping_delta_norm"] > 0
    assert results[0]["direct_phase_cfo_delta_norm"] == 0
    assert results[0]["updated_parameter_count"] == 14
    assert results[0]["state_mapping_delta_norm"] <= 0.050001
    for name, parameter in model.state_dict().items():
        assert torch.equal(parameter, second.state_dict()[name])
        if not name.startswith("online_phase_trend_adapter.state_to_phase_cfo."):
            assert torch.equal(parameter, original[name])


def test_direct_phase_cfo_update_changes_only_adapter_parameters():
    model, frame, condition, tail = _phase_fixture(known_phase=0.2)
    adapter = model.online_phase_trend_adapter
    original_mapping = copy.deepcopy(adapter.state_to_phase_cfo.state_dict())
    result = _adapt_mapping(model, [{
        "frame": frame, "condition": condition, "tail": tail,
        "embedding": torch.tensor([0.4, 0.2, 0.1, 0.5, 0.8, 0.3]),
    }], lr=0.005, steps=3, max_delta_norm=0.05, proximal_weight=0.01,
        objective="signal_reconstruction", update_scope="direct_phase_cfo")
    assert result["updated_parameter_scope"] == "direct_phase_cfo"
    assert result["updated_parameter_count"] == 2
    assert result["direct_phase_cfo_delta_norm"] > 0
    assert result["state_mapping_delta_norm"] == 0
    assert all(torch.equal(value, adapter.state_to_phase_cfo.state_dict()[key])
               for key, value in original_mapping.items())


def test_channel_residual_update_changes_only_cir_adapter_parameters():
    model, frame, condition, tail = _phase_fixture(known_phase=0.2)
    model.attach_online_channel_residual_adapter()
    original_phase = copy.deepcopy(model.online_phase_trend_adapter.state_dict())
    result = _adapt_mapping(model, [{
        "frame": frame, "condition": condition, "tail": tail,
        "embedding": torch.tensor([0.4, 0.2, 0.1, 0.5, 0.8, 0.3]),
    }], lr=0.005, steps=3, max_delta_norm=0.05, proximal_weight=0.01,
        objective="signal_reconstruction", update_scope="channel_residual")
    assert result["updated_parameter_scope"] == "channel_residual"
    assert result["updated_parameter_count"] == 10
    assert result["channel_residual_delta_norm"] > 0
    assert result["max_parameter_delta_norm"] == pytest.approx(0.05)
    assert result["state_mapping_delta_norm"] == 0
    assert result["direct_phase_cfo_delta_norm"] == 0
    assert all(torch.equal(value, model.online_phase_trend_adapter.state_dict()[key])
               for key, value in original_phase.items())


@pytest.mark.parametrize("source", ["acquisition", "pilot_phase", "pilot_cir_phase"])
def test_replay_condition_matches_compare_and_ignores_hidden_labels(source):
    _, frame, _, tail = _phase_fixture()
    cir = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.complex64)
    initial_features = torch.arange(16, dtype=torch.float32)
    episode = PilotReplayEpisode(SimpleNamespace(config=SimpleNamespace(snr_db=10)),
                                 None, cir, 0.001, initial_features, source, False)
    expected = _neural_phase_features(
        condition_source=source, receiver_view=frame.receiver_view(), cir=cir,
        soft_tail=tail, acquisition_cfo=0.001, acquisition_phase_features=initial_features,
    )
    condition = episode.condition(frame, tail)
    assert torch.equal(condition.latent_residual[0, :16], expected)
    poisoned = replace(frame, bits=~frame.bits, tx_symbols=torch.where(
        frame.adapt_mask, frame.tx_symbols, -frame.tx_symbols,
    ))
    assert torch.equal(condition.latent_residual, episode.condition(poisoned, tail).latent_residual)
    if source == "acquisition":
        assert torch.equal(condition.latent_residual[0, :16], initial_features)


def test_replay_uses_frozen_eme_config_and_explicit_condition_override(monkeypatch):
    """完整物理配置必须传入 replay，不能静默退回 legacy 信道。"""
    import evaluation.pilot_replay_protocol as protocol
    experiment = json.loads(Path("configs/online_phase_signal_state_embedding_formal_moderate.json").read_text(encoding="utf-8"))
    model_config = UnfoldedConfig.from_dict(experiment["model"])
    monkeypatch.setattr(protocol, "estimate_acquisition_cir_with_cfo", lambda *args, **kwargs: (
        torch.ones(117, dtype=torch.complex64), 0.001,
    ))
    monkeypatch.setattr(protocol, "estimate_phase_residual_vector", lambda *args, **kwargs: torch.zeros(16))
    episode = build_pilot_replay_episode(
        experiment, model_config, snr=10, seed=1, delay=116, pilot_total=256,
        reward_pilot_total=32, gap_seconds=0, device="cpu", condition_source="pilot_cir_phase",
    )
    assert episode.env.config.profile_name == "eme_long_memory_v2"
    assert episode.env.config.frame_len == 1024
    assert episode.env.config.max_delay == 116
    assert episode.env.config.layout == "prefix"
    assert episode.env.config.total_pilot == 256
    assert episode.env.config.reward_pilot_total == 32
    assert episode.env.config.acquisition_to_data_gap_seconds == 0
    assert episode.condition_source == "pilot_cir_phase"
    assert episode.history_cleared_after_gap is False
    assert experiment["online_condition_source"] == "acquisition"


def test_replay_can_apply_physical_eme_acquisition_gap_without_changing_profile(monkeypatch):
    import evaluation.pilot_replay_protocol as protocol
    experiment = json.loads(Path("configs/online_phase_signal_state_embedding_formal_moderate.json").read_text(encoding="utf-8"))
    model_config = UnfoldedConfig.from_dict(experiment["model"])
    monkeypatch.setattr(protocol, "estimate_acquisition_cir_with_cfo", lambda *args, **kwargs: (
        torch.ones(117, dtype=torch.complex64), 0.001,
    ))
    monkeypatch.setattr(protocol, "estimate_phase_residual_vector", lambda *args, **kwargs: torch.zeros(16))
    episode = build_pilot_replay_episode(
        experiment, model_config, snr=10, seed=2, delay=116, pilot_total=256,
        reward_pilot_total=32, gap_seconds=600, device="cpu", condition_source="acquisition",
    )
    assert episode.env.config.profile_name == "eme_long_memory_v2"
    assert episode.env.config.acquisition_to_data_gap_seconds == 600
    assert episode.condition_source == "acquisition"
    assert episode.history_cleared_after_gap is True


def test_reward_gate_rejects_conflicting_or_invalid_history():
    args = dict(min_improvement=0.0005, cross_frame_tolerance=1e-5,
                snr=10, freeze_below_snr=5)
    assert _reward_gate([0.001, 0.002], **args)
    assert not _reward_gate([0.01, -0.001], **args)
    assert not _reward_gate([float("nan")], **args)
    assert not _reward_gate([], **args)
    assert not _reward_gate([0.001], **{**args, "snr": 0})
    assert _reward_gate([0.002, -0.0005], **args, aggregation="aggregate")
    assert not _reward_gate([0.0002, -0.0001], **args, aggregation="aggregate")
    assert _reward_gate([0.0001, 0.0], **{**args, "min_improvement": 0.0}, aggregation="aggregate")
    from scripts.replay_cumulative_state_mapping import _reward_error_gate
    assert _reward_error_gate([1, -1], "aggregate")
    assert not _reward_error_gate([1, -1], "per_frame")


def test_joint_relative_reward_uses_only_relative_pilot_improvements(monkeypatch):
    import scripts.replay_cumulative_state_mapping as replay

    def loss(model, item, score):
        values = {
            ("reference", "signal_reconstruction"): 8.0,
            ("candidate", "signal_reconstruction"): 6.0,
            ("reference", "bce"): 0.2,
            ("candidate", "bce"): 0.1,
        }
        return values[(model, score)]

    monkeypatch.setattr(replay, "_reward_loss", loss)
    gain = replay._reward_gain("reference", "candidate", {}, "joint_relative")
    assert gain == pytest.approx(0.5 * (0.25 + 0.5))
    assert replay._reward_gain("reference", "candidate", {}, "bce_relative") == pytest.approx(0.5)


@pytest.mark.parametrize("reward_score", ["bce", "bce_relative", "signal_reconstruction", "joint_relative"])
def test_future_reward_validation_precedes_disjoint_data_evaluation(monkeypatch, reward_score):
    """未来验收帧不参与梯度，也不纳入 Data 指标；验收决定先于 Data 标签读取。"""
    import scripts.replay_cumulative_state_mapping as replay

    model, frame, condition, tail = _phase_fixture()
    frames = iter([replace(frame, frame_index=index) for index in range(8)])
    episode = SimpleNamespace(
        env=SimpleNamespace(next_frame=lambda: next(frames)),
        start=SimpleNamespace(initial_soft_tail=tail), cir=condition.complex_cir,
        condition=lambda *args: condition, metadata=lambda: {},
    )
    monkeypatch.setattr(replay, "build_pilot_replay_episode", lambda *args, **kwargs: episode)
    monkeypatch.setattr(replay, "_build_equalizer", lambda *args: model)
    monkeypatch.setattr(replay, "pilot_state_embedding", lambda *args: torch.zeros(6))
    monkeypatch.setattr(replay, "_logits", lambda current, item: torch.zeros(128))
    events = []

    def adapt(current, contexts, **kwargs):
        indices = [item["frame"].frame_index for item in contexts]
        events.append(("adapt", indices))
        return {}

    def gate(gains, **kwargs):
        events.append(("gate", len(gains)))
        return False

    def metrics(before, after, current_frame):
        events.append(("data", current_frame.frame_index))
        return {}

    monkeypatch.setattr(replay, "_adapt_mapping", adapt)
    monkeypatch.setattr(replay, "_reward_gate", gate)
    monkeypatch.setattr(replay, "_data_metrics", metrics)
    rows = replay._run_trial(
        experiment={}, model_config=model.config, checkpoint=Path("unused.pt"), device="cpu",
        snr=10, seed=0, frames_count=8, train_counts=[1, 3], delay=4,
        pilot_total=64, reward_pilot_total=16, gap_seconds=0, lr=0.005, steps=1,
        validation_frames=2,
        reward_score=reward_score,
    )
    assert [(kind, value) for kind, value in events if kind == "adapt"] == [
        ("adapt", [0]), ("adapt", [0, 1, 2]),
    ]
    assert all(value == 2 for kind, value in events if kind == "gate")
    assert all(value >= 5 for kind, value in events if kind == "data")
    assert events[0][0] == "adapt" and events[1][0] == "gate" and events[2][0] == "data"
    assert all(row["reward_validation_frame_indices"] == [3, 4] for row in rows)
    assert all(row["reward_gate_mode"] == "future_reward_only" for row in rows)
    assert all(set(row["train_frame_indices"]).isdisjoint(row["reward_validation_frame_indices"])
               and row["test_frame_index"] >= 5 for row in rows)


def test_signal_reward_uses_remaining_error_in_noise_units():
    from scripts.replay_cumulative_state_mapping import _reward_loss

    model, frame, condition, tail = _phase_fixture(known_phase=0.2)
    item = {"frame": frame, "condition": condition, "tail": tail, "embedding": torch.zeros(6)}
    expected = 2 * (1 - torch.cos(torch.tensor(0.1))) / condition.noise_variance.mean()
    assert _reward_loss(model, item, "signal_reconstruction") == pytest.approx(float(expected), abs=1e-6)


@pytest.mark.parametrize("aggregation", ["per_frame", "aggregate"])
def test_hold_monitor_aggregation_controls_single_frame_rollback(monkeypatch, aggregation):
    """逐帧门控会回滚单帧退化；聚合门控只按窗口均值判断。"""
    import scripts.replay_cumulative_state_mapping as replay

    model, frame, condition, tail = _phase_fixture()
    frames = iter([replace(frame, frame_index=index) for index in range(8)])
    episode = SimpleNamespace(
        env=SimpleNamespace(next_frame=lambda: next(frames)),
        start=SimpleNamespace(initial_soft_tail=tail), cir=condition.complex_cir,
        condition=lambda *args: condition, metadata=lambda: {},
    )
    monkeypatch.setattr(replay, "build_pilot_replay_episode", lambda *args, **kwargs: episode)
    monkeypatch.setattr(replay, "_build_equalizer", lambda *args: model)
    monkeypatch.setattr(replay, "pilot_state_embedding", lambda *args: torch.zeros(6))
    monkeypatch.setattr(replay, "_logits", lambda current, item: torch.ones(128) * (
        2.0 if getattr(current, "candidate_ready", False) else 1.0
    ))

    def adapt(current, contexts, **kwargs):
        current.candidate_ready = True
        return {}

    def reward(current, item, score):
        if not getattr(current, "candidate_ready", False):
            return 1.0
        # 验收帧均有益，首个测试帧恶化，之后连续恢复。
        return 1.01 if item["frame"].frame_index == 5 else 0.98

    monkeypatch.setattr(replay, "_adapt_mapping", adapt)
    monkeypatch.setattr(replay, "_reward_loss", reward)
    rows = replay._run_trial(
        experiment={}, model_config=model.config, checkpoint=Path("unused.pt"), device="cpu",
        snr=10, seed=0, frames_count=8, train_counts=[3], delay=4,
        pilot_total=64, reward_pilot_total=16, gap_seconds=0, lr=0.005, steps=1,
        validation_frames=2, reward_score="joint_relative", monitor_reward_window=2,
        reward_aggregation=aggregation,
    )
    assert [row["test_frame_index"] for row in rows] == [5, 6, 7]
    assert all(row["initial_reward_gate_accepted"] for row in rows)
    if aggregation == "per_frame":
        assert [row["monitor_reward_rollback"] for row in rows] == [True, False, False]
        assert all(not row["reward_gate_accepted"] for row in rows)
        assert all(row["data_bce_improvement"] == 0.0 for row in rows)
    else:
        assert [row["monitor_reward_rollback"] for row in rows] == [False, False, False]
        assert all(row["reward_gate_accepted"] for row in rows)
        assert all(row["data_bce_improvement"] != 0.0 for row in rows)
    assert all(row["candidate_data_logits_abs_mean_delta"] > 0 for row in rows)
    assert [row["monitor_reward_frame_indices"] for row in rows] == [[4, 5], [5, 6], [6, 7]]
