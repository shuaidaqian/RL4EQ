# -*- coding: utf-8 -*-
"""Pilot 留出式单载波 BPSK 帧结构。"""

from dataclasses import dataclass, field

import numpy as np
import torch


REGION_ADAPT_PILOT = 0
REGION_REWARD_PILOT = 1
REGION_DATA = 2


@dataclass
class FrameConfig:
    """定义 ``Adapt Pilot | Reward Pilot | Data`` 帧。"""

    frame_len: int = 512
    adapt_pilot_len: int = 48
    reward_pilot_len: int = 16
    modulation: str = "bpsk"
    data_len: int = field(init=False)
    adapt_pilot_mask: torch.Tensor = field(init=False, repr=False)
    reward_pilot_mask: torch.Tensor = field(init=False, repr=False)
    data_mask: torch.Tensor = field(init=False, repr=False)
    pilot_mask: torch.Tensor = field(init=False, repr=False)
    region_ids: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.modulation.lower() != "bpsk":
            raise ValueError("当前研究主线只支持 BPSK。")
        if self.frame_len <= 0 or self.adapt_pilot_len <= 0 or self.reward_pilot_len <= 0:
            raise ValueError("帧长与两类 Pilot 长度必须为正数。")
        self.data_len = self.frame_len - self.adapt_pilot_len - self.reward_pilot_len
        if self.data_len <= 0:
            raise ValueError("Pilot 总长度必须小于帧长。")

        adapt_end = self.adapt_pilot_len
        reward_end = adapt_end + self.reward_pilot_len
        self.adapt_pilot_mask = torch.zeros(self.frame_len, dtype=torch.bool)
        self.reward_pilot_mask = torch.zeros(self.frame_len, dtype=torch.bool)
        self.data_mask = torch.zeros(self.frame_len, dtype=torch.bool)
        self.adapt_pilot_mask[:adapt_end] = True
        self.reward_pilot_mask[adapt_end:reward_end] = True
        self.data_mask[reward_end:] = True
        self.pilot_mask = self.adapt_pilot_mask | self.reward_pilot_mask
        self.region_ids = torch.full((self.frame_len,), REGION_DATA, dtype=torch.long)
        self.region_ids[self.adapt_pilot_mask] = REGION_ADAPT_PILOT
        self.region_ids[self.reward_pilot_mask] = REGION_REWARD_PILOT

    @classmethod
    def from_total_pilot(cls, total_pilot_len: int, frame_len: int = 512) -> "FrameConfig":
        if total_pilot_len % 4 != 0:
            raise ValueError("Pilot 总长度必须能按 3:1 切分。")
        return cls(
            frame_len=frame_len,
            adapt_pilot_len=total_pilot_len * 3 // 4,
            reward_pilot_len=total_pilot_len // 4,
        )

    def summary(self) -> str:
        return (
            f"帧长={self.frame_len}, Adapt Pilot={self.adapt_pilot_len}, "
            f"Reward Pilot={self.reward_pilot_len}, Data={self.data_len}"
        )


@dataclass
class FrameData:
    """单帧发送端数据及接收端已知信息。"""

    bits: torch.Tensor
    tx_symbols: torch.Tensor
    region_ids: torch.Tensor
    adapt_pilot_symbols: torch.Tensor
    adapt_pilot_mask: torch.Tensor
    reward_pilot_mask: torch.Tensor
    data_mask: torch.Tensor


class FrameGenerator:
    """生成随帧变化、可由相同种子复现的 PN Pilot。"""

    def __init__(self, config: FrameConfig, seed: int = 20260720):
        self.config = config
        self.seed = int(seed)

    def generate(self, frame_index: int = 0) -> FrameData:
        rng = np.random.default_rng(self.seed + int(frame_index) * 104729)
        bits = torch.from_numpy(
            rng.integers(0, 2, size=self.config.frame_len).astype(np.float32)
        )
        tx_symbols = self.modulate(bits)
        adapt_symbols = torch.zeros(self.config.frame_len, dtype=torch.float32)
        adapt_symbols[self.config.adapt_pilot_mask] = (
            1.0 - 2.0 * bits[self.config.adapt_pilot_mask]
        )
        return FrameData(
            bits=bits,
            tx_symbols=tx_symbols,
            region_ids=self.config.region_ids.clone(),
            adapt_pilot_symbols=adapt_symbols,
            adapt_pilot_mask=self.config.adapt_pilot_mask.clone(),
            reward_pilot_mask=self.config.reward_pilot_mask.clone(),
            data_mask=self.config.data_mask.clone(),
        )

    @staticmethod
    def modulate(bits: torch.Tensor) -> torch.Tensor:
        symbols = 1.0 - 2.0 * bits.float()
        return torch.stack([symbols, torch.zeros_like(symbols)], dim=-1)
