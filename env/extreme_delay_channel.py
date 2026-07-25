# -*- coding: utf-8 -*-
"""分层极端长时延稀疏信道与跨帧 ISI。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from env.channel_profiles import ChannelLevel, ChannelProfileConfig, sample_profile


@dataclass(frozen=True)
class ExtremeDelayChannelConfig:
    level: str = "B"
    max_delay: int = 40
    snr_db: float = 10.0
    rho: float = 0.99
    seed: int = 42

    def __post_init__(self) -> None:
        ChannelLevel(self.level)
        if self.max_delay < 1:
            raise ValueError("max_delay 必须为正。")
        if not 0.0 <= self.rho <= 1.0:
            raise ValueError("rho 必须位于 [0, 1]。")


class ExtremeDelayChannel:
    """具有固定 support、缓慢 tap 漂移和跨帧 ISI 的复基带信道。"""

    def __init__(self, config: ExtremeDelayChannelConfig):
        self.config = config
        self.noise_variance = float(10.0 ** (-config.snr_db / 10.0))
        self._rng = np.random.default_rng(config.seed)
        self._torch_rng = torch.Generator(device="cpu").manual_seed(config.seed + 10_000)
        self._delays: tuple[int, ...] = ()
        self._base_cir = torch.zeros(config.max_delay + 1, dtype=torch.complex64)
        self._current_cir = torch.zeros(config.max_delay + 1, dtype=torch.complex64)
        self._history = torch.zeros(config.max_delay, dtype=torch.complex64)

    @property
    def delays(self) -> tuple[int, ...]:
        return self._delays

    @property
    def tap_power(self) -> torch.Tensor:
        return torch.abs(self._current_cir[self._current_cir != 0]) ** 2

    def reset_episode(self, known_warmup: torch.Tensor) -> None:
        """采样 episode 基准 tap 并设置已知发送历史。"""

        warmup = known_warmup.detach().to(dtype=torch.complex64, device="cpu")
        if warmup.numel() < self.config.max_delay:
            raise ValueError("known_warmup 长度必须不小于 max_delay。")
        profile = sample_profile(
            ChannelProfileConfig(
                level=ChannelLevel(self.config.level),
                max_delay=self.config.max_delay,
                seed=self.config.seed,
            )
        )
        self._delays = tuple(profile.delays)
        self._base_cir = torch.zeros(self.config.max_delay + 1, dtype=torch.complex64)
        self._base_cir[list(self._delays)] = torch.as_tensor(profile.taps, dtype=torch.complex64)
        self._current_cir = self._normalize_cir(self._base_cir.clone())
        self._history = warmup[-self.config.max_delay :].clone()
        self._rng = np.random.default_rng(self.config.seed)
        self._torch_rng = torch.Generator(device="cpu").manual_seed(self.config.seed + 10_000)

    def transmit(self, symbols: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        """施加线性卷积、跨帧历史、固定 Es/N0 噪声，并在帧末更新 tap。"""

        if self._history.numel() != self.config.max_delay:
            raise RuntimeError("请先调用 reset_episode。")
        tx = symbols.detach().to(dtype=torch.complex64, device="cpu")
        rx = self._convolve_with_history(tx, self._current_cir)
        if add_noise:
            rx = rx + self._sample_noise(tx.numel())
        combined = torch.cat([self._history, tx])
        self._history = combined[-self.config.max_delay :].clone()
        self._evolve_taps()
        return rx

    def true_cir(self) -> torch.Tensor:
        """仅向离线监督与 Perfect-CSI 诊断暴露真实 CIR。"""

        return self._current_cir.clone()

    def _convolve_with_history(self, symbols: torch.Tensor, cir: torch.Tensor) -> torch.Tensor:
        padded = torch.cat([self._history, symbols])
        frame_len = symbols.numel()
        origin = self.config.max_delay
        rx = torch.zeros(frame_len, dtype=torch.complex64)
        indices = torch.arange(frame_len)
        for delay in self._delays:
            rx += cir[delay] * padded[origin + indices - delay]
        return rx

    def _sample_noise(self, frame_len: int) -> torch.Tensor:
        std = float(np.sqrt(self.noise_variance / 2.0))
        real = torch.randn(frame_len, generator=self._torch_rng, dtype=torch.float32) * std
        imag = torch.randn(frame_len, generator=self._torch_rng, dtype=torch.float32) * std
        return torch.complex(real, imag)

    def _evolve_taps(self) -> None:
        rho = self.config.rho
        if rho >= 1.0:
            return
        support = list(self._delays)
        current = self._current_cir[support]
        base = self._base_cir[support]
        innovation_scale = np.sqrt(max(0.0, 1.0 - rho**2))
        base_power = torch.abs(base).clamp_min(1e-6)
        real = torch.randn(len(support), generator=self._torch_rng) / np.sqrt(2.0)
        imag = torch.randn(len(support), generator=self._torch_rng) / np.sqrt(2.0)
        innovation = torch.complex(real, imag).to(torch.complex64) * base_power
        updated = rho * current + (1.0 - rho) * base + innovation_scale * innovation
        next_cir = torch.zeros_like(self._current_cir)
        next_cir[support] = updated
        self._current_cir = self._normalize_cir(next_cir)

    @staticmethod
    def _normalize_cir(cir: torch.Tensor) -> torch.Tensor:
        power = torch.sum(torch.abs(cir) ** 2).clamp_min(1e-12)
        return cir / torch.sqrt(power)
