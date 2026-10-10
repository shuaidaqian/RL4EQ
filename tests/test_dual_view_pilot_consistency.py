# -*- coding: utf-8 -*-

import torch

from env.frame_structure import Frame
from scripts.replay_dual_view_pilot_consistency import _split_adapt_frame


def _tiny_frame() -> Frame:
    length = 16
    adapt_mask = torch.zeros(length, dtype=torch.bool)
    adapt_mask[:8] = True
    reward_mask = torch.zeros(length, dtype=torch.bool)
    reward_mask[8:10] = True
    data_mask = ~(adapt_mask | reward_mask)
    bits = torch.arange(length, dtype=torch.float32) % 2
    tx_symbols = torch.where(bits > 0.5, 1.0, -1.0).to(torch.complex64)
    return Frame(
        frame_index=0,
        bits=bits,
        tx_symbols=tx_symbols,
        rx_symbols=tx_symbols.clone(),
        adapt_mask=adapt_mask,
        reward_mask=reward_mask,
        data_mask=data_mask,
        model_region_ids=torch.zeros(length, dtype=torch.long),
        adapt_block_lengths=[8],
    )


def test_dual_view_split_keeps_pilot_only_and_disjoint_masks():
    first, second = _split_adapt_frame(_tiny_frame())

    assert int(first.adapt_mask.sum()) == 4
    assert int(second.adapt_mask.sum()) == 4
    assert not bool(torch.any(first.adapt_mask & second.adapt_mask))
    assert torch.equal(first.reward_mask, second.reward_mask)
    assert torch.equal(first.data_mask, second.data_mask)

