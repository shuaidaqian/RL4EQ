# -*- coding: utf-8 -*-
"""
帧结构模块 — 定义 RL4EQ 系统的帧格式和比特序列生成

参考 P-FTNet 论文的参数配置，帧结构为 (总长 512):
  | 训练序列(128) | 导频块1(64) | 数据块1(128) | 导频块2(64) | 数据块2(128) |

已知位 = 训练(128) + 导频(2*64) = 256 位 (50%)
数据位 = 2*128 = 256 位 (50%)
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np
import torch


@dataclass
class FrameConfig:
    """单帧结构配置类。
    
    自动计算数据区长度和导频位置，确保帧结构合法。
    """
    frame_len: int = 512                 # 帧总长度 (P-FTNet 标准帧长)
    train_len: int = 128                 # 帧首训练序列长度
    pilot_len: int = 64                  # 每个导频块的长度
    num_pilots: int = 2                  # 导频块数量
    modulation: str = "bpsk"             # BPSK 调制

    data_len: int = 0
    pilot_positions: List[int] = field(default_factory=list)

    def __post_init__(self):
        remaining = self.frame_len - self.train_len
        self.data_len = remaining - self.num_pilots * self.pilot_len
        assert self.data_len >= 0
        assert self.data_len % self.num_pilots == 0
        self._assign_pilot_positions()

    def _assign_pilot_positions(self):
        data_per_block = self.data_len // self.num_pilots
        self.pilot_positions = []
        pos = self.train_len
        for _ in range(self.num_pilots):
            self.pilot_positions.append(pos)
            pos += self.pilot_len + data_per_block

    def is_training(self, pos: int) -> bool:
        return 0 <= pos < self.train_len

    def is_pilot(self, pos: int) -> bool:
        return any(start <= pos < start + self.pilot_len for start in self.pilot_positions)

    def is_data(self, pos: int) -> bool:
        return not (self.is_training(pos) or self.is_pilot(pos))

    def bit_type(self, pos: int) -> str:
        if self.is_training(pos):
            return "train"
        elif self.is_pilot(pos):
            return "pilot"
        else:
            return "data"

    def summary(self) -> str:
        return (f"帧: L={self.frame_len}, 训练={self.train_len}, "
                f"导频({self.num_pilots}x{self.pilot_len})={self.num_pilots*self.pilot_len}, "
                f"数据={self.data_len}")


class FrameGenerator:
    """帧生成器 — 产生比特序列和 BPSK 调制符号。"""

    def __init__(self, config: FrameConfig):
        self.config = config
        self._known_pattern = self._build_known_pattern()

    def generate(self, rng=None) -> torch.Tensor:
        if rng is None:
            rng = np.random.default_rng()
        bits = rng.integers(0, 2, size=self.config.frame_len).astype(np.float32)
        bits = torch.from_numpy(bits)
        mask = self.known_mask()
        bits[mask] = self._known_pattern[mask]
        return bits

    def modulate(self, bits: torch.Tensor) -> torch.Tensor:
        symbols = 1.0 - 2.0 * bits
        return torch.stack([symbols, torch.zeros_like(symbols)], dim=-1)

    def known_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.config.frame_len, dtype=torch.bool)
        mask[:self.config.train_len] = True
        for start in self.config.pilot_positions:
            mask[start:start + self.config.pilot_len] = True
        return mask

    def known_bits(self, bits: torch.Tensor) -> torch.Tensor:
        known = bits.clone()
        known[~self.known_mask()] = 0.0
        return known

    def _build_known_pattern(self) -> torch.Tensor:
        """生成接收端已知的确定性训练/导频 PN 序列。"""
        rng = np.random.default_rng(20260709)
        pn = rng.integers(0, 2, size=self.config.frame_len)
        return torch.from_numpy(pn.astype(np.float32))


def frame_config_for_known_ratio(ratio: float, frame_len: int = 512, num_pilots: int = 2) -> FrameConfig:
    """按已知位比例构造低 pilot overhead 帧结构。"""
    known_total = max(4, int(round(frame_len * float(ratio))))
    pilot_len = max(2, known_total // (num_pilots * 2))
    train_len = max(4, known_total - num_pilots * pilot_len)
    return FrameConfig(frame_len=frame_len, train_len=train_len, pilot_len=pilot_len, num_pilots=num_pilots)


def test_frame():
    cfg = FrameConfig(frame_len=512, train_len=128, pilot_len=64, num_pilots=2)
    gen = FrameGenerator(cfg)
    bits = gen.generate()
    print(cfg.summary())
    print(f"导频位置: {cfg.pilot_positions}")
    known_mask = gen.known_mask()
    print(f"已知位: {known_mask.sum().item()} ({known_mask.sum().item()/512*100:.0f}%)")
    sym = gen.modulate(bits)
    print(f"调制后形状: {list(sym.shape)}")
    print("帧结构自测通过。")


if __name__ == "__main__":
    test_frame()
