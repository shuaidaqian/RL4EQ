# -*- coding: utf-8 -*-
"""多块 Pilot 帧结构与接收端可见视图。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


REGION_ADAPT = 1
REGION_UNKNOWN = 0


@dataclass(frozen=True)
class FrameConfig:
    frame_len: int = 512
    total_pilot: int = 128
    layout: str = "multi_block"
    max_delay: int = 40

    def __post_init__(self) -> None:
        if self.total_pilot not in {64, 96, 128, 160}:
            raise ValueError("total_pilot 必须是 64/96/128/160。")
        if self.layout not in {"prefix", "two_block", "multi_block"}:
            raise ValueError("layout 必须是 prefix/two_block/multi_block。")
        if self.frame_len <= self.total_pilot:
            raise ValueError("frame_len 必须大于 total_pilot。")
        if self.total_pilot % 4 != 0:
            raise ValueError("total_pilot 必须能按 3:1 切分。")


@dataclass(frozen=True)
class ReceiverFrameView:
    rx_symbols: torch.Tensor
    adapt_symbols: torch.Tensor
    adapt_mask: torch.Tensor
    model_region_ids: torch.Tensor


@dataclass(frozen=True)
class Frame:
    frame_index: int
    bits: torch.Tensor
    tx_symbols: torch.Tensor
    rx_symbols: torch.Tensor
    adapt_mask: torch.Tensor
    reward_mask: torch.Tensor
    data_mask: torch.Tensor
    model_region_ids: torch.Tensor
    adapt_block_lengths: Sequence[int]
    tail_symbols: torch.Tensor | None = None
    true_cir: torch.Tensor | None = None

    @property
    def unknown_region_mask(self) -> torch.Tensor:
        return self.reward_mask | self.data_mask

    def receiver_view(self) -> ReceiverFrameView:
        adapt_symbols = torch.zeros_like(self.tx_symbols)
        adapt_symbols[self.adapt_mask] = self.tx_symbols[self.adapt_mask]
        return ReceiverFrameView(
            rx_symbols=self.rx_symbols.clone(),
            adapt_symbols=adapt_symbols,
            adapt_mask=self.adapt_mask.clone(),
            model_region_ids=self.model_region_ids.clone(),
        )

    def offline_view(self) -> "Frame":
        return self

    def with_channel_output(
        self,
        rx_symbols: torch.Tensor,
        tail_symbols: torch.Tensor,
        true_cir: torch.Tensor,
    ) -> "Frame":
        return Frame(
            frame_index=self.frame_index,
            bits=self.bits,
            tx_symbols=self.tx_symbols,
            rx_symbols=rx_symbols,
            adapt_mask=self.adapt_mask,
            reward_mask=self.reward_mask,
            data_mask=self.data_mask,
            model_region_ids=self.model_region_ids,
            adapt_block_lengths=self.adapt_block_lengths,
            tail_symbols=tail_symbols,
            true_cir=true_cir,
        )

    def with_replaced_hidden_labels(self, reward_bits: torch.Tensor, data_bits: torch.Tensor) -> "Frame":
        bits = self.bits.clone()
        bits[self.reward_mask] = reward_bits[self.reward_mask].to(bits.dtype)
        bits[self.data_mask] = data_bits[self.data_mask].to(bits.dtype)
        tx_symbols = _bpsk(bits)
        return Frame(
            frame_index=self.frame_index,
            bits=bits,
            tx_symbols=tx_symbols,
            rx_symbols=self.rx_symbols,
            adapt_mask=self.adapt_mask,
            reward_mask=self.reward_mask,
            data_mask=self.data_mask,
            model_region_ids=self.model_region_ids,
            adapt_block_lengths=self.adapt_block_lengths,
            tail_symbols=self.tail_symbols,
            true_cir=self.true_cir,
        )


class FrameGenerator:
    """按 frame index 可复现生成整帧 BPSK 和 Pilot mask。"""

    def __init__(self, config: FrameConfig, seed: int = 0):
        self.config = config
        self.seed = seed

    def generate(self, frame_index: int) -> Frame:
        generator = torch.Generator(device="cpu").manual_seed(self.seed + frame_index * 104_729)
        bits = torch.randint(0, 2, (self.config.frame_len,), generator=generator, dtype=torch.int64).bool()
        adapt_mask, reward_mask, adapt_lengths = self._build_masks()
        data_mask = ~(adapt_mask | reward_mask)
        region_ids = torch.full((self.config.frame_len,), REGION_UNKNOWN, dtype=torch.long)
        region_ids[adapt_mask] = REGION_ADAPT
        tx_symbols = _bpsk(bits)
        return Frame(
            frame_index=frame_index,
            bits=bits,
            tx_symbols=tx_symbols,
            rx_symbols=torch.zeros_like(tx_symbols),
            adapt_mask=adapt_mask,
            reward_mask=reward_mask,
            data_mask=data_mask,
            model_region_ids=region_ids,
            adapt_block_lengths=tuple(adapt_lengths),
        )

    def _build_masks(self) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        adapt_total = 3 * self.config.total_pilot // 4
        reward_total = self.config.total_pilot // 4
        adapt_mask = torch.zeros(self.config.frame_len, dtype=torch.bool)
        reward_mask = torch.zeros(self.config.frame_len, dtype=torch.bool)
        adapt_blocks, reward_blocks = self._layout_blocks(adapt_total, reward_total)
        for start, length in adapt_blocks:
            adapt_mask[start : start + length] = True
        for start, length in reward_blocks:
            reward_mask[start : start + length] = True
        return adapt_mask, reward_mask, [length for _, length in adapt_blocks]

    def _layout_blocks(self, adapt_total: int, reward_total: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        if self.config.layout == "prefix":
            return [(0, adapt_total)], [(adapt_total, reward_total)]
        if self.config.layout == "two_block":
            first_adapt = 48
            second_adapt = adapt_total - first_adapt
            midpoint = self.config.frame_len // 2
            return [(0, first_adapt), (midpoint, second_adapt)], [
                (first_adapt, reward_total // 2),
                (midpoint + second_adapt, reward_total - reward_total // 2),
            ]
        first_adapt = 48
        remaining = adapt_total - first_adapt
        second_adapt = remaining // 2
        third_adapt = remaining - second_adapt
        reward_first = reward_total // 3
        reward_second = reward_total // 3
        reward_third = reward_total - reward_first - reward_second
        one_third = self.config.frame_len // 3
        two_third = 2 * self.config.frame_len // 3
        return [(0, first_adapt), (one_third, second_adapt), (two_third, third_adapt)], [
            (first_adapt, reward_first),
            (one_third + second_adapt, reward_second),
            (two_third + third_adapt, reward_third),
        ]


def _bpsk(bits: torch.Tensor) -> torch.Tensor:
    values = bits.to(torch.float32) * 2.0 - 1.0
    return torch.complex(values, torch.zeros_like(values))
