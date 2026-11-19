# -*- coding: utf-8 -*-
"""0 dB Data 后验变体：同 episode、同波形，oracle 之外不读 Data 标签。"""

import json
import math
from pathlib import Path

import pytest
import torch

from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
import scripts.diagnose_0db_data_posterior as diagnose
from scripts.diagnose_0db_data_posterior import run_variant
from training.bayesian_cir import acquisition_prior
from training.pilot_transfer import make_context, make_episode

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


def test_em_same_frame_scores_with_final_cir(monkeypatch):
    """same_frame 下 EM 的判决必须来自最终 CIR 的重解码，而不是倒数第二次更新。"""

    model = _model()
    device = torch.device("cpu")
    _, _, cfo, tail, frames, acquisition = make_episode(
        CONFIG, model, seed=140001, snr=0, frames=1,
        state_split="offline_train", device=device, return_acquisition=True,
    )
    frame = frames[0]
    max_delay = int(CONFIG["max_delay"])
    noise_variance = 1.0
    prior_mean, prior_covariance = acquisition_prior(
        acquisition, cfo, max_delay, noise_variance, ridge=1e-2,
    )
    context = make_context(frame, prior_mean, cfo, tail, 0)

    recorded = []
    real_make_context = diagnose.make_context

    def spy(frame_, cir, cfo_, tail_, snr_):
        recorded.append(cir.detach().clone())
        return real_make_context(frame_, cir, cfo_, tail_, snr_)

    monkeypatch.setattr(diagnose, "make_context", spy)
    final_cir, _, _, _ = diagnose._em_posterior(
        model, frame, context, tail, prior_mean, prior_covariance, cfo, 0,
        max_delay, noise_variance, 0.0, cfo, 2, False, redecode_final=True,
    )
    assert len(recorded) == 2
    torch.testing.assert_close(recorded[-1], final_cir, rtol=0, atol=0)


def test_unknown_timing_is_rejected():
    with pytest.raises(ValueError, match="时序"):
        run_variant(None, None, seed=1, snr=0, frames=1, timing="transductive")
