# -*- coding: utf-8 -*-
"""Bayes CIR 逐帧成本 profile：组件耗时有限，tail 复用与独立前向一致。"""

import json
import math
from pathlib import Path

import torch

from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from scripts.profile_bayes_cost import run_profile

CONFIG = json.loads(
    Path("configs/eme_long_memory_v2_online_dd_tail_v1.json").read_text(encoding="utf-8")
)


def _model():
    torch.manual_seed(13)
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


def test_profile_components_are_finite_and_tail_reuse_matches():
    model = _model()
    rows = run_profile(CONFIG, model, seed=160001, snr=0, frames=1, device="cpu")

    assert len(rows) == 1
    row = rows[0]
    for key in (
        "acquisition_ms", "forward_ms", "pilot_ms", "data_ms",
        "tail_forward_ms", "tail_reuse_ms", "total_forward_ms", "total_reuse_ms",
    ):
        assert math.isfinite(row[key])
        assert row[key] >= 0.0
    assert row["tail_match"] is True
    assert row["total_reuse_ms"] <= row["total_forward_ms"]
