# -*- coding: utf-8 -*-

import torch

from env.frame_structure import FrameConfig, FrameGenerator
from scripts.replay_pilot_subwindow_split import split_pilot_frame


def test_pilot_subwindow_split_keeps_prefix_and_suffix_disjoint():
    frame = FrameGenerator(
        FrameConfig(frame_len=128, total_pilot=64, layout="prefix", max_delay=4),
        seed=7,
    ).generate(0)

    train_frame, holdout_mask, train_count, holdout_count = split_pilot_frame(frame, 0.5)

    assert train_count == 24
    assert holdout_count == 24
    assert int(train_frame.adapt_mask.sum()) == train_count
    assert int(holdout_mask.sum()) == holdout_count
    assert not bool(torch.any(train_frame.adapt_mask & holdout_mask))
    assert torch.equal(
        torch.nonzero(train_frame.adapt_mask).reshape(-1),
        torch.arange(train_count),
    )
    assert torch.equal(
        torch.nonzero(holdout_mask).reshape(-1),
        torch.arange(train_count, train_count + holdout_count),
    )
    assert train_frame.reward_mask.sum() == 0
    assert train_frame.data_mask.sum() == frame.data_mask.sum()


def test_pilot_subwindow_split_rejects_invalid_ratio():
    frame = FrameGenerator(
        FrameConfig(frame_len=128, total_pilot=64, layout="prefix", max_delay=4),
        seed=7,
    ).generate(0)

    for ratio in (0.0, 1.0, -0.1, 1.1):
        try:
            split_pilot_frame(frame, ratio)
        except ValueError:
            pass
        else:
            raise AssertionError("非法 holdout_ratio 未被拒绝。")
