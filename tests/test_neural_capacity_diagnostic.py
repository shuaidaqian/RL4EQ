# -*- coding: utf-8 -*-
"""现有神经分支诊断应保留基础模型，并明确记录实际候选和评分范围。"""

from dataclasses import replace

import torch

from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from env.frame_structure import FrameConfig, FrameGenerator
from scripts import diagnose_head_transfer as diagnostic


def test_existing_phase_diagnostic_does_not_change_model_or_register_action(monkeypatch, tmp_path):
    torch.manual_seed(752)
    model = UnfoldedEqualizer(UnfoldedConfig(
        frame_len=128, max_delay=4, iterations=1, d_model=16, num_heads=4,
        enable_phase_correction_branch=True, phase_correction_initial_scale=1,
        pilot_conditioned=True,
    )).eval()
    model.attach_online_phase_trend_adapter()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    frame = FrameGenerator(FrameConfig(frame_len=128, total_pilot=64, max_delay=4), seed=12).generate(0)
    positions = torch.arange(128)
    frame = replace(frame, rx_symbols=frame.tx_symbols * torch.exp(1j * (0.2 + 0.002 * positions)))
    cir = torch.tensor([1, 0, 0, 0, 0], dtype=torch.complex64)
    tail = torch.zeros(4, dtype=torch.complex64)
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"diagnostic-source")
    monkeypatch.setattr(diagnostic, "_load_branch", lambda *args: (model, {}))
    monkeypatch.setattr(diagnostic, "make_episode", lambda *args, **kwargs: (None, cir, 0, tail, [frame]))
    config = {"main_snrs": [10], "pilot_transfer": {
        "inner_groups": ["head"], "inner_objective": "bce", "inner_learning_rate": 0.0005,
        "inner_steps": 1, "max_delta_norm": 0.05,
    }}
    before = {name: value.clone() for name, value in model.state_dict().items()}
    result = diagnostic.run_diagnostic(config, checkpoint, [9], 1, "cpu", tmp_path / "output.json",
                                       groups=["phase"], score_start=4)
    assert result["groups"] == ["phase"]
    assert result["score_start"] == 4
    assert result["candidate_only_no_rl_action_registered"] is True
    assert result["data_labels_used_for_online_decisions"] is False
    assert len(result["rows"]) == 1
    assert len(result["rows"][0]["input_sha256"]) == 64
    assert result["per_snr"][0]["committed_gain"]["ci95"] is None
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


def test_phase_trend_group_is_supported_by_update_direction_diagnostic(monkeypatch, tmp_path):
    torch.manual_seed(753)
    model = UnfoldedEqualizer(UnfoldedConfig(
        frame_len=128, max_delay=4, iterations=1, d_model=16, num_heads=4,
        enable_phase_correction_branch=True, phase_correction_initial_scale=1,
        pilot_conditioned=True,
    )).eval()
    model.attach_online_phase_trend_adapter()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    frame = FrameGenerator(FrameConfig(frame_len=128, total_pilot=64, max_delay=4), seed=13).generate(0)
    cir = torch.tensor([1, 0, 0, 0, 0], dtype=torch.complex64)
    tail = torch.zeros(4, dtype=torch.complex64)
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"diagnostic-source")
    monkeypatch.setattr(diagnostic, "_load_branch", lambda *args: (model, {}))
    monkeypatch.setattr(diagnostic, "make_episode", lambda *args, **kwargs: (None, cir, 0, tail, [frame]))
    config = {"main_snrs": [10], "pilot_transfer": {
        "inner_groups": ["phase_trend"], "inner_objective": "bce", "inner_learning_rate": 0.0005,
        "inner_steps": 1, "max_delta_norm": 0.05,
    }}
    result = diagnostic.run_diagnostic(config, checkpoint, [10], 1, "cpu", tmp_path / "phase-trend.json",
                                       groups=["phase_trend"])
    assert result["groups"] == ["phase_trend"]
    assert result["selected_parameter_count"] == 16
    assert result["data_labels_used_for_online_decisions"] is False
