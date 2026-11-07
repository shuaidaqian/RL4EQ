# -*- coding: utf-8 -*-
"""公平传统基线比较：同 episode/波形/Pilot/acquisition，传统方法不读 Data 标签。"""

import json
import math
from dataclasses import replace
from pathlib import Path

import torch

from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from baseline.traditional_equalizers import run_traditional_equalizer
from scripts.compare_traditional_baselines import (
    TRADITIONAL_COMPARISON_METHODS,
    run_comparison,
)
from training.pilot_transfer import make_episode

CONFIG = json.loads(
    Path("configs/eme_long_memory_v2_online_dd_tail_v1.json").read_text(encoding="utf-8")
)


def _model():
    torch.manual_seed(7)
    return UnfoldedEqualizer(UnfoldedConfig(
        frame_len=1024,
        max_delay=116,
        iterations=1,
        d_model=8,
        num_heads=2,
        pilot_conditioned=True,
        enable_phase_correction_branch=True,
        phase_correction_initial_scale=1.0,
    )).eval()


def test_comparison_uses_same_episode_and_no_data_label_baselines():
    """所有方法共享同一帧波形；传统方法不读 Reward/Data 标签。"""

    model = _model()
    methods = ("frozen", "bayes", "CFO+DD-Phase LMMSE-FIR", "CFO+DD-Phase DFE-RLS", "SC-FDE-MMSE")
    rows = run_comparison(CONFIG, model, seed=70001, snr=5, frames=1, methods=methods, device="cpu")

    assert len(rows) == len(methods)
    assert {row["seed"] for row in rows} == {70001}
    assert {row["frame_index"] for row in rows} == {0}
    assert len({row["rx_sha256"] for row in rows}) == 1
    for row in rows:
        assert math.isfinite(row["ber_data"])
        if row["method"] in TRADITIONAL_COMPARISON_METHODS:
            assert row["uses_neural_network"] is False
            assert row["uses_reward_or_data_labels"] is False
        else:
            assert row["uses_neural_network"] is True
            assert row["uses_reward_or_data_labels"] is False


def test_traditional_decisions_ignore_data_bits():
    """翻转 Data 比特不得改变传统均衡器的判决。"""

    model = _model()
    _, cir, _, tail, frames = make_episode(
        CONFIG, model, seed=70002, snr=5, frames=1,
        state_split="offline_train", device=torch.device("cpu"),
    )
    frame = frames[0]
    first = run_traditional_equalizer(
        "CFO+DD-Phase LMMSE-FIR", frame.receiver_view(), cir, tail, 5,
    )
    poisoned_bits = frame.bits.clone()
    poisoned_bits[frame.data_mask] = ~poisoned_bits[frame.data_mask]
    second = run_traditional_equalizer(
        "CFO+DD-Phase LMMSE-FIR",
        replace(frame, bits=poisoned_bits).receiver_view(),
        cir, tail, 5,
    )
    torch.testing.assert_close(first.logits, second.logits, rtol=0, atol=0)
