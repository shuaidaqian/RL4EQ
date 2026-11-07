# -*- coding: utf-8 -*-
"""warm-start/AMP 扫描：组件耗时有限，tail 复用与 dd_soft 一致。"""

import json
import math
from pathlib import Path

import torch

from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from scripts.profile_bayes_cost_sweep import run_sweep

CONFIG = json.loads(
    Path("configs/eme_long_memory_v2_online_dd_tail_v1.json").read_text(encoding="utf-8")
)


def _model():
    torch.manual_seed(17)
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


def test_sweep_warm_start_and_amp_are_finite():
    model = _model()
    rows = run_sweep(
        CONFIG, model, seeds=[160201], snr=0, frames=1,
        variants=((8, False), (4, True)), device="cpu",
    )
    assert len(rows) == 2
    for row in rows:
        for key in ("ber_data", "nmse", "forward_ms", "total_ms"):
            assert math.isfinite(row[key])
            assert row[key] >= 0.0
        assert row["tail_match"] is True
