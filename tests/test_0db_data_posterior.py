# -*- coding: utf-8 -*-
"""0 dB Data 后验变体：同 episode、同波形，oracle 之外不读 Data 标签。"""

import json
import math
from pathlib import Path

import pytest
import torch

from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from scripts.diagnose_0db_data_posterior import run_variant

CONFIG = json.loads(
    Path("configs/eme_long_memory_v2_online_dd_tail_v1.json").read_text(encoding="utf-8")
)


def _model():
    torch.manual_seed(11)
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


def test_variants_share_episode_and_label_boundaries():
    """所有变体共享同一帧波形；只有 oracle 变体使用 Data 标签。"""

    model = _model()
    variants = ("soft", "em2", "block_phase", "oracle")
    rows = run_variant(CONFIG, model, seed=140001, snr=0, frames=1, variants=variants, device="cpu")

    assert len(rows) == len(variants)
    assert {row["frame_index"] for row in rows} == {0}
    assert len({row["rx_sha256"] for row in rows}) == 1
    for row in rows:
        assert math.isfinite(row["ber_data"])
        assert math.isfinite(row["nmse"])
        assert row["uses_reward_or_data_labels"] is (row["variant"] == "oracle")


def test_causal_timing_does_not_redecode_current_frame():
    """因果时序：首帧判决必须早于本帧后验更新，故与 Frozen 完全一致。"""

    model = _model()
    rows = run_variant(
        CONFIG, model, seed=140001, snr=0, frames=1,
        variants=("soft", "block_phase"), device="cpu", timing="causal",
    )
    assert rows and all(row["timing"] == "causal" for row in rows)
    torch.testing.assert_close(
        torch.tensor([row["gain"] for row in rows]), torch.zeros(len(rows)), rtol=0, atol=0,
    )

    rows_same = run_variant(
        CONFIG, model, seed=140001, snr=0, frames=1,
        variants=("soft", "block_phase"), device="cpu", timing="same_frame",
    )
    assert all(row["timing"] == "same_frame" for row in rows_same)


def test_unknown_timing_is_rejected():
    with pytest.raises(ValueError, match="时序"):
        run_variant(None, None, seed=1, snr=0, frames=1, timing="transductive")
