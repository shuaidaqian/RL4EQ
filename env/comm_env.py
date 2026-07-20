# -*- coding: utf-8 -*-
"""极端长时延均衡实验的帧级通信环境。"""

from dataclasses import dataclass, field

import numpy as np
import torch

from env.extreme_delay_channel import ExtremeDelayChannel, ExtremeDelayChannelConfig
from env.frame_structure import FrameConfig, FrameData, FrameGenerator


@dataclass
class EnvConfig:
    frame: FrameConfig = field(default_factory=FrameConfig)
    channel: ExtremeDelayChannelConfig = field(default_factory=ExtremeDelayChannelConfig)
    seed: int = 42
    random_warmup: bool = True


@dataclass
class ReceivedFrame:
    """仿真接收帧；真实比特只供离线训练和评估使用。"""

    frame: FrameData
    rx_symbols: torch.Tensor
    snr_db: float
    max_delay_symbols: int
    path_count: int


class CommunicationEnv:
    """以 episode 为单位维持信道路径与跨帧 ISI。"""

    def __init__(self, config: EnvConfig):
        self.config = config
        self.frame_generator = FrameGenerator(config.frame, seed=config.seed)
        self.channel = ExtremeDelayChannel(config.channel)
        self._rng = np.random.default_rng(config.seed + 1)
        self._frame_index = 0

    def reset_episode(self) -> None:
        self._frame_index = 0
        warmup = None
        if self.config.random_warmup:
            bits = torch.from_numpy(
                self._rng.integers(
                    0,
                    2,
                    size=self.config.channel.max_delay_symbols,
                ).astype(np.float32)
            )
            warmup = FrameGenerator.modulate(bits)
        self.channel.reset_episode(warmup_symbols=warmup)

    def next_frame(self) -> ReceivedFrame:
        if not self.channel.delays:
            self.reset_episode()
        frame = self.frame_generator.generate(self._frame_index)
        rx_symbols = self.channel.transmit(frame.tx_symbols)
        self._frame_index += 1
        return ReceivedFrame(
            frame=frame,
            rx_symbols=rx_symbols,
            snr_db=float(self.channel.snr_db),
            max_delay_symbols=int(self.config.channel.max_delay_symbols),
            path_count=len(self.channel.delays),
        )
