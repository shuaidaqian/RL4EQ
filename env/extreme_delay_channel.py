# -*- coding: utf-8 -*-
"""分层极端长时延稀疏信道与跨帧 ISI。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch

from env.channel_profiles import ChannelLevel, ChannelProfileConfig, sample_profile
from env.eme_channel_profiles import EMEChannelProfileConfig, sample_eme_profile
from env.eme_reference import EME_FULL_RADAR_DEPTH_SECONDS
from env.impairments import (
    PhaseImpairmentSettings,
    apply_phase_impairment,
    settings_from_profile,
    validate_cfo_cycles_per_symbol,
    validate_impairment_profile,
    validate_phase_noise_std,
)


@dataclass(frozen=True)
class ExtremeDelayChannelConfig:
    level: str = "B"
    max_delay: int = 40
    snr_db: float = 10.0
    rho: float = 0.99
    seed: int = 42
    impairment_profile: str = "clean"
    cfo_cycles_per_symbol: float = 0.0
    phase_noise_std: float = 0.0
    profile_name: str = "legacy_sparse_v1"
    sample_rate_hz: float | None = None
    symbol_rate_hz: float | None = None
    frame_len: int | None = None
    max_delay_seconds: float = EME_FULL_RADAR_DEPTH_SECONDS
    coherence_time_seconds: float | None = None
    strong_path_count: tuple[int, int] | None = None
    diffuse_energy_ratio: tuple[float, float] | None = None
    include_anomalous_scatterer: bool = False
    _eme_profile_config: EMEChannelProfileConfig | None = field(
        init=False, repr=False, compare=False, default=None
    )

    def __post_init__(self) -> None:
        if isinstance(self.snr_db, (bool, np.bool_)):
            raise ValueError("snr_db 必须为有限非布尔数。")
        try:
            snr_db = float(self.snr_db)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("snr_db 必须为有限非布尔数。") from exc
        if not math.isfinite(snr_db):
            raise ValueError("snr_db 必须为有限非布尔数。")
        cfo_cycles_per_symbol = validate_cfo_cycles_per_symbol(
            self.cfo_cycles_per_symbol
        )
        phase_noise_std = validate_phase_noise_std(self.phase_noise_std)
        validate_impairment_profile(self.impairment_profile)
        object.__setattr__(self, "snr_db", snr_db)
        object.__setattr__(
            self, "cfo_cycles_per_symbol", cfo_cycles_per_symbol
        )
        object.__setattr__(self, "phase_noise_std", phase_noise_std)
        ChannelLevel(self.level)
        if self.profile_name not in {"legacy_sparse_v1", "eme_measurement_v1"}:
            raise ValueError("profile_name 必须是 legacy_sparse_v1 或 eme_measurement_v1。")
        if self.profile_name == "legacy_sparse_v1":
            if self.max_delay < 1:
                raise ValueError("max_delay 必须为正。")
            if not 0.0 <= self.rho <= 1.0:
                raise ValueError("rho 必须位于 [0, 1]。")
            return

        required_fields = (
            "sample_rate_hz",
            "symbol_rate_hz",
            "frame_len",
            "coherence_time_seconds",
            "strong_path_count",
            "diffuse_energy_ratio",
        )
        for field_name in required_fields:
            if getattr(self, field_name) is None:
                raise ValueError(f"eme_measurement_v1 要求配置 {field_name}。")

        coherence = self.coherence_time_seconds
        if isinstance(coherence, (bool, np.bool_)):
            raise ValueError("coherence_time_seconds 必须为正且有限。")
        try:
            coherence = float(coherence)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("coherence_time_seconds 必须为正且有限。") from exc
        if not math.isfinite(coherence) or coherence <= 0.0:
            raise ValueError("coherence_time_seconds 必须为正且有限。")

        eme_config = EMEChannelProfileConfig(
            level=self.level,
            sample_rate_hz=self.sample_rate_hz,
            symbol_rate_hz=self.symbol_rate_hz,
            frame_len=self.frame_len,
            max_delay_seconds=self.max_delay_seconds,
            strong_path_count=self.strong_path_count,
            diffuse_energy_ratio=self.diffuse_energy_ratio,
            include_anomalous_scatterer=self.include_anomalous_scatterer,
            seed=self.seed,
        )
        object.__setattr__(self, "sample_rate_hz", eme_config.sample_rate_hz)
        object.__setattr__(self, "symbol_rate_hz", eme_config.symbol_rate_hz)
        object.__setattr__(self, "frame_len", eme_config.frame_len)
        object.__setattr__(self, "max_delay_seconds", eme_config.max_delay_seconds)
        object.__setattr__(self, "coherence_time_seconds", coherence)
        object.__setattr__(self, "strong_path_count", eme_config.strong_path_count)
        object.__setattr__(self, "diffuse_energy_ratio", eme_config.diffuse_energy_ratio)
        object.__setattr__(self, "max_delay", eme_config.max_delay_samples)
        object.__setattr__(self, "_eme_profile_config", eme_config)

    @property
    def samples_per_symbol(self) -> float | None:
        if self._eme_profile_config is None:
            return None
        return self._eme_profile_config.samples_per_symbol

    @property
    def frame_duration_seconds(self) -> float | None:
        if self._eme_profile_config is None:
            return None
        return self._eme_profile_config.frame_len / self._eme_profile_config.sample_rate_hz

    @property
    def rho_frame(self) -> float:
        if self._eme_profile_config is None:
            return self.rho
        return math.exp(-self.frame_duration_seconds / self.coherence_time_seconds)


class ExtremeDelayChannel:
    """具有固定 support、缓慢 tap 漂移和跨帧 ISI 的复基带信道。"""

    def __init__(self, config: ExtremeDelayChannelConfig):
        self.config = config
        self.noise_variance = float(10.0 ** (-config.snr_db / 10.0))
        self._rng = np.random.default_rng(config.seed)
        self._tap_rng = torch.Generator(device="cpu").manual_seed(config.seed + 10_000)
        self._noise_rng = torch.Generator(device="cpu").manual_seed(config.seed + 20_000)
        self._phase_rng = torch.Generator(device="cpu").manual_seed(config.seed + 30_000)
        self._delays: tuple[int, ...] = ()
        self._base_cir = torch.zeros(self.max_delay + 1, dtype=torch.complex64)
        self._current_cir = torch.zeros(self.max_delay + 1, dtype=torch.complex64)
        self._last_cir_used = torch.zeros(self.max_delay + 1, dtype=torch.complex64)
        self._history = torch.zeros(self.max_delay, dtype=torch.complex64)
        self._diffuse_mask = torch.zeros(self.max_delay + 1, dtype=torch.bool)
        self._profile_metadata: Mapping[str, Any] | None = None
        self._global_symbol_index = 0
        self._phase_state = 0.0
        self._impairment = PhaseImpairmentSettings()

    @property
    def delays(self) -> tuple[int, ...]:
        return self._delays

    @property
    def max_delay(self) -> int:
        return self.config.max_delay

    @property
    def tap_power(self) -> torch.Tensor:
        return torch.abs(self._current_cir[self._current_cir != 0]) ** 2

    def reset_episode(self, known_warmup: torch.Tensor) -> None:
        """采样 episode 基准 tap 并设置已知发送历史。"""

        warmup = known_warmup.detach().to(dtype=torch.complex64, device="cpu")
        if warmup.numel() < self.max_delay:
            raise ValueError("known_warmup 长度必须不小于 max_delay。")
        if self.config.profile_name == "eme_measurement_v1":
            profile = sample_eme_profile(self.config._eme_profile_config)
            self._delays = tuple(int(delay) for delay in profile.strong_delays)
            self._base_cir = torch.as_tensor(
                np.array(profile.cir, copy=True), dtype=torch.complex64
            )
            self._diffuse_mask = torch.as_tensor(
                np.array(profile.diffuse_mask, copy=True), dtype=torch.bool
            )
            self._profile_metadata = profile.metadata
        else:
            profile = sample_profile(
                ChannelProfileConfig(
                    level=ChannelLevel(self.config.level),
                    max_delay=self.max_delay,
                    seed=self.config.seed,
                )
            )
            self._delays = tuple(profile.delays)
            self._base_cir = torch.zeros(self.max_delay + 1, dtype=torch.complex64)
            self._base_cir[list(self._delays)] = torch.as_tensor(profile.taps, dtype=torch.complex64)
            self._diffuse_mask = torch.zeros(self.max_delay + 1, dtype=torch.bool)
            self._profile_metadata = None
        self._current_cir = self._normalize_cir(self._base_cir.clone())
        self._last_cir_used = torch.zeros_like(self._current_cir)
        self._history = warmup[-self.max_delay :].clone()
        self._rng = np.random.default_rng(self.config.seed)
        self._tap_rng = torch.Generator(device="cpu").manual_seed(self.config.seed + 10_000)
        self._noise_rng = torch.Generator(device="cpu").manual_seed(self.config.seed + 20_000)
        self._phase_rng = torch.Generator(device="cpu").manual_seed(self.config.seed + 30_000)
        self._global_symbol_index = 0
        self._phase_state = 0.0
        self._impairment = settings_from_profile(
            self.config.impairment_profile,
            self._rng,
            cfo_cycles_per_symbol=self.config.cfo_cycles_per_symbol,
            phase_noise_std=self.config.phase_noise_std,
        )

    def transmit(self, symbols: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        """施加线性卷积、跨帧历史、固定 Es/N0 噪声，并在帧末更新 tap。"""

        if self._history.numel() != self.max_delay:
            raise RuntimeError("请先调用 reset_episode。")
        tx = symbols.detach().to(dtype=torch.complex64, device="cpu")
        rx = self._convolve_with_history(tx, self._current_cir)
        self._last_cir_used = self._current_cir.clone()
        rx, self._phase_state = apply_phase_impairment(
            rx,
            global_start=self._global_symbol_index,
            settings=self._impairment,
            phase_state=self._phase_state,
            phase_rng=self._phase_rng,
        )
        if add_noise:
            rx = rx + self._sample_noise(tx.numel())
        combined = torch.cat([self._history, tx])
        self._history = combined[-self.max_delay :].clone()
        self._global_symbol_index += int(tx.numel())
        self._evolve_taps()
        return rx

    def true_cir(self) -> torch.Tensor:
        """仅向离线监督与 Perfect-CSI 诊断暴露真实 CIR。"""

        return self._current_cir.clone()

    def last_cir_used(self) -> torch.Tensor:
        """返回最近一次 transmit 实际用于生成接收帧的 CIR。"""

        return self._last_cir_used.clone()

    def _convolve_with_history(self, symbols: torch.Tensor, cir: torch.Tensor) -> torch.Tensor:
        padded = torch.cat([self._history, symbols])
        frame_len = symbols.numel()
        origin = self.max_delay
        rx = torch.zeros(frame_len, dtype=torch.complex64)
        indices = torch.arange(frame_len)
        nonzero_delays = torch.nonzero(cir != 0, as_tuple=False).flatten().tolist()
        for delay in nonzero_delays:
            rx += cir[delay] * padded[origin + indices - delay]
        return rx

    def _sample_noise(self, frame_len: int) -> torch.Tensor:
        std = float(np.sqrt(self.noise_variance / 2.0))
        real = torch.randn(frame_len, generator=self._noise_rng, dtype=torch.float32) * std
        imag = torch.randn(frame_len, generator=self._noise_rng, dtype=torch.float32) * std
        return torch.complex(real, imag)

    def _evolve_taps(self) -> None:
        if self.config.profile_name == "eme_measurement_v1":
            self._evolve_eme_taps()
            return

        rho = self.config.rho
        if rho >= 1.0:
            return
        support = list(self._delays)
        current = self._current_cir[support]
        base = self._base_cir[support]
        innovation_scale = np.sqrt(max(0.0, 1.0 - rho**2))
        base_power = torch.abs(base).clamp_min(1e-6)
        real = torch.randn(len(support), generator=self._tap_rng) / np.sqrt(2.0)
        imag = torch.randn(len(support), generator=self._tap_rng) / np.sqrt(2.0)
        innovation = torch.complex(real, imag).to(torch.complex64) * base_power
        updated = rho * current + (1.0 - rho) * base + innovation_scale * innovation
        next_cir = torch.zeros_like(self._current_cir)
        next_cir[support] = updated
        self._current_cir = self._normalize_cir(next_cir)

    def _evolve_eme_taps(self) -> None:
        """按物理复增益递推；瞬时总功率允许波动，长期期望由基准 tap 功率约束。"""

        rho = self.config.rho_frame
        if rho >= 1.0:
            return
        support = torch.nonzero(self._base_cir != 0, as_tuple=False).flatten()
        current = self._current_cir[support]
        base_amplitude = torch.abs(self._base_cir[support])
        real = torch.randn(support.numel(), generator=self._tap_rng) / np.sqrt(2.0)
        imag = torch.randn(support.numel(), generator=self._tap_rng) / np.sqrt(2.0)
        innovation = torch.complex(real, imag).to(torch.complex64) * base_amplitude
        updated = rho * current + math.sqrt(max(0.0, 1.0 - rho**2)) * innovation
        next_cir = torch.zeros_like(self._current_cir)
        next_cir[support] = updated
        self._current_cir = next_cir

    @staticmethod
    def _normalize_cir(cir: torch.Tensor) -> torch.Tensor:
        power = torch.sum(torch.abs(cir) ** 2).clamp_min(1e-12)
        return cir / torch.sqrt(power)
