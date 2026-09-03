# -*- coding: utf-8 -*-
"""整帧缓冲通信 episode 环境。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from env.eme_reference import EME_FULL_RADAR_DEPTH_SECONDS
from env.extreme_delay_channel import ExtremeDelayChannel, ExtremeDelayChannelConfig
from env.frame_structure import Frame, FrameConfig, FrameGenerator


@dataclass(frozen=True)
class CommEnvConfig:
    level: str = "B"
    max_delay: int = 40
    snr_db: float = 10.0
    rho: float = 0.99
    total_pilot: int = 128
    layout: str = "prefix"
    seed: int = 42
    impairment_profile: str = "clean"
    cfo_cycles_per_symbol: float = 0.0
    phase_noise_std: float = 0.0
    profile_name: str = "legacy_sparse_v1"
    sample_rate_hz: float | None = None
    symbol_rate_hz: float | None = None
    frame_len: int = 512
    max_delay_seconds: float = EME_FULL_RADAR_DEPTH_SECONDS
    coherence_time_seconds: float | None = None
    strong_path_count: tuple[int, int] | None = None
    diffuse_energy_ratio: tuple[float, float] | None = None
    include_anomalous_scatterer: bool = False
    acquisition_to_data_gap_seconds: float = 0.0
    state_split: str | None = None
    state_ranges: Mapping[str, Any] | None = None
    reward_pilot_total: int | None = None

    def __post_init__(self) -> None:
        if self.profile_name in {"eme_measurement_v1", "eme_long_memory_v2"} and self.layout != "prefix":
            raise ValueError(f"{self.profile_name} 仅允许 prefix Pilot layout。")


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
                profile_name=config.profile_name,
                level=config.level,
                max_delay=config.max_delay,
                snr_db=config.snr_db,
                rho=config.rho,
                seed=config.seed,
                impairment_profile=config.impairment_profile,
                cfo_cycles_per_symbol=config.cfo_cycles_per_symbol,
                phase_noise_std=config.phase_noise_std,
                sample_rate_hz=config.sample_rate_hz,
                symbol_rate_hz=config.symbol_rate_hz,
                frame_len=config.frame_len,
                max_delay_seconds=config.max_delay_seconds,
                coherence_time_seconds=config.coherence_time_seconds,
                strong_path_count=config.strong_path_count,
                diffuse_energy_ratio=config.diffuse_energy_ratio,
                include_anomalous_scatterer=config.include_anomalous_scatterer,
                acquisition_to_data_gap_seconds=config.acquisition_to_data_gap_seconds,
                state_split=config.state_split,
                state_ranges=config.state_ranges,
            )
        )
        self.frame_generator = FrameGenerator(
            FrameConfig(
                frame_len=config.frame_len,
                total_pilot=config.total_pilot,
                layout=config.layout,
                max_delay=self.channel.max_delay,
                reward_pilot_total=config.reward_pilot_total,
            ),
            seed=config.seed + 1_000,
        )
        self._frame_index = 0
        self._last_tail = torch.zeros(self.channel.max_delay, dtype=torch.complex64)

    def reset_episode(self) -> EpisodeStart:
        tail_length = self.channel.max_delay
        warmup = self._known_symbols(tail_length, self.config.seed + 2_000)
        self.channel.reset_episode(warmup)
        acquisition_bits = self._known_symbols(self.config.frame_len, self.config.seed + 3_000)
        acquisition = Frame(
            frame_index=-1,
            bits=(acquisition_bits.real > 0),
            tx_symbols=acquisition_bits,
            rx_symbols=acquisition_bits.clone(),
            adapt_mask=torch.ones(self.config.frame_len, dtype=torch.bool),
            reward_mask=torch.zeros(self.config.frame_len, dtype=torch.bool),
            data_mask=torch.zeros(self.config.frame_len, dtype=torch.bool),
            model_region_ids=torch.ones(self.config.frame_len, dtype=torch.long),
            adapt_block_lengths=(self.config.frame_len,),
        )
        rx = self.channel.transmit(acquisition.tx_symbols, add_noise=True)
        acquisition = acquisition.with_channel_output(
            rx, warmup[-tail_length:], self.channel.last_cir_used()
        )
        self.channel.advance_after_acquisition()
        self._last_tail = torch.cat((warmup, acquisition.tx_symbols))[-tail_length:].clone()
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
        self._last_tail = torch.cat((tail, frame.tx_symbols))[-self.channel.max_delay :].clone()
        self._frame_index += 1
        return received

    def state_metadata(self) -> dict[str, Any]:
        """返回当前 episode 的信道状态元数据，仅用于实验记录。"""

        return self.channel.state_metadata()

    @staticmethod
    def _known_symbols(length: int, seed: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        bits = torch.randint(0, 2, (length,), generator=generator, dtype=torch.int64).bool()
        values = bits.to(torch.float32) * 2.0 - 1.0
        return torch.complex(values, torch.zeros_like(values))


def environment_schema() -> str:
    """返回新路线环境契约版本。"""

    return "continual-ppo-env-v1"
