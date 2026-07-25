# -*- coding: utf-8 -*-
"""整帧缓冲通信 episode 环境。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from env.extreme_delay_channel import ExtremeDelayChannel, ExtremeDelayChannelConfig
from env.frame_structure import Frame, FrameConfig, FrameGenerator


@dataclass(frozen=True)
class CommEnvConfig:
    level: str = "B"
    max_delay: int = 40
    snr_db: float = 10.0
    rho: float = 0.99
    total_pilot: int = 128
    layout: str = "multi_block"
    seed: int = 42


@dataclass(frozen=True)
class EpisodeStart:
    warmup_symbols: torch.Tensor
    acquisition: Frame
    initial_soft_tail: torch.Tensor


@dataclass
class ReceiverState:
    soft_tail: torch.Tensor

    def clone(self) -> "ReceiverState":
        return ReceiverState(self.soft_tail.clone())

    def update_tail(self, soft_tail: torch.Tensor) -> None:
        self.soft_tail = soft_tail.clone()


class CommunicationEnvironment:
    """提供 acquisition frame 和普通帧的仿真环境。"""

    def __init__(self, config: CommEnvConfig):
        self.config = config
        self.channel = ExtremeDelayChannel(
            ExtremeDelayChannelConfig(
                level=config.level,
                max_delay=config.max_delay,
                snr_db=config.snr_db,
                rho=config.rho,
                seed=config.seed,
            )
        )
        self.frame_generator = FrameGenerator(
            FrameConfig(total_pilot=config.total_pilot, layout=config.layout, max_delay=config.max_delay),
            seed=config.seed + 1_000,
        )
        self._frame_index = 0
        self._last_tail = torch.zeros(config.max_delay, dtype=torch.complex64)

    def reset_episode(self) -> EpisodeStart:
        warmup = self._known_symbols(self.config.max_delay, self.config.seed + 2_000)
        self.channel.reset_episode(warmup)
        acquisition_bits = self._known_symbols(512, self.config.seed + 3_000)
        acquisition = Frame(
            frame_index=-1,
            bits=(acquisition_bits.real > 0),
            tx_symbols=acquisition_bits,
            rx_symbols=acquisition_bits.clone(),
            adapt_mask=torch.ones(512, dtype=torch.bool),
            reward_mask=torch.zeros(512, dtype=torch.bool),
            data_mask=torch.zeros(512, dtype=torch.bool),
            model_region_ids=torch.ones(512, dtype=torch.long),
            adapt_block_lengths=(512,),
        )
        rx = self.channel.transmit(acquisition.tx_symbols, add_noise=True)
        acquisition = acquisition.with_channel_output(rx, warmup[-self.config.max_delay :], self.channel.last_cir_used())
        self._last_tail = acquisition.tx_symbols[-self.config.max_delay :].clone()
        self._frame_index = 0
        return EpisodeStart(
            warmup_symbols=warmup,
            acquisition=acquisition,
            initial_soft_tail=self._last_tail.clone(),
        )

    def next_frame(self) -> Frame:
        frame = self.frame_generator.generate(self._frame_index)
        tail = self._last_tail.clone()
        rx = self.channel.transmit(frame.tx_symbols, add_noise=True)
        received = frame.with_channel_output(rx, tail, self.channel.last_cir_used())
        self._last_tail = frame.tx_symbols[-self.config.max_delay :].clone()
        self._frame_index += 1
        return received

    @staticmethod
    def _known_symbols(length: int, seed: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        bits = torch.randint(0, 2, (length,), generator=generator, dtype=torch.int64).bool()
        values = bits.to(torch.float32) * 2.0 - 1.0
        return torch.complex(values, torch.zeros_like(values))


def environment_schema() -> str:
    """返回新路线环境契约版本。"""

    return "continual-ppo-env-v1"
