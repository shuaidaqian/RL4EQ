# -*- coding: utf-8 -*-
"""Level A/B/C 可控稀疏长回波信道 profile。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np


class ChannelLevel(str, Enum):
    A = "A"
    B = "B"
    C = "C"


class ProfileSamplingError(RuntimeError):
    """受约束信道 profile 采样失败。"""


@dataclass(frozen=True)
class ChannelProfileConfig:
    level: ChannelLevel
    max_delay: int
    seed: int
    spectral_grid: int = 2048
    max_attempts: int = 10_000

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "level", ChannelLevel(self.level))
        except ValueError as exc:
            raise ValueError(f"未知信道层级：{self.level}") from exc
        if self.max_delay < 2:
            raise ValueError("max_delay 必须至少为 2，才能包含 0、内部径和最大时延径。")
        if self.spectral_grid < 16:
            raise ValueError("spectral_grid 过小，无法形成稳定频谱指标。")
        if self.max_attempts < 1:
            raise ValueError("max_attempts 必须为正数。")


@dataclass(frozen=True)
class ChannelProfile:
    level: ChannelLevel
    delays: Sequence[int]
    taps: np.ndarray
    strongest_gap_db: float
    max_delay_relative_db: float
    delayed_energy_ratio: float
    notch_depth_db: float
    condition_proxy: float


def classify_for_aggregation(level: ChannelLevel | str) -> str:
    """返回该层级在论文统计中的汇总类别。"""

    parsed = ChannelLevel(level)
    return "pressure" if parsed is ChannelLevel.C else "main"


def sample_profile(config: ChannelProfileConfig) -> ChannelProfile:
    """按层级约束采样一个稀疏复 CIR profile。"""

    rng = np.random.default_rng(config.seed)
    for _ in range(config.max_attempts):
        path_count = _sample_path_count(config.level, config.max_delay, rng)
        if path_count is None:
            continue
        delays = _sample_delays(config.max_delay, path_count, rng)
        powers = _sample_powers(config.level, delays, rng)
        if powers is None:
            continue
        phases = rng.uniform(-np.pi, np.pi, size=path_count)
        taps = np.sqrt(powers).astype(np.float64) * np.exp(1j * phases)
        taps = taps.astype(np.complex128)
        profile = _build_profile(config.level, delays, taps, config.spectral_grid)
        if _satisfies_level(profile):
            return profile
    raise ProfileSamplingError(
        f"无法在 {config.max_attempts} 次内采样满足约束的 profile："
        f"level={config.level.value}, max_delay={config.max_delay}, seed={config.seed}"
    )


def _sample_path_count(level: ChannelLevel, max_delay: int, rng: np.random.Generator) -> int | None:
    ranges = {
        ChannelLevel.A: (3, 5),
        ChannelLevel.B: (3, 7),
        ChannelLevel.C: (3, 10),
    }
    low, high = ranges[level]
    feasible_high = min(high, max_delay + 1)
    if feasible_high < low:
        return None
    return int(rng.integers(low, feasible_high + 1))


def _sample_delays(max_delay: int, path_count: int, rng: np.random.Generator) -> list[int]:
    interior_count = path_count - 2
    if interior_count <= 0:
        return [0, max_delay]
    interior = rng.choice(np.arange(1, max_delay), size=interior_count, replace=False)
    return [0, *sorted(int(x) for x in interior), max_delay]


def _sample_powers(level: ChannelLevel, delays: Sequence[int], rng: np.random.Generator) -> np.ndarray | None:
    path_count = len(delays)
    powers = np.zeros(path_count, dtype=np.float64)
    if level is ChannelLevel.A:
        delayed_ratio = rng.uniform(0.10, 0.35)
        main_power = 1.0 - delayed_ratio
        max_relative_db = rng.uniform(-20.0, -10.0)
        max_power = main_power * 10.0 ** (max_relative_db / 10.0)
        second_gap_db = rng.uniform(6.0, 15.0)
        second_power = main_power * 10.0 ** (-second_gap_db / 10.0)
        remaining = delayed_ratio - max_power - second_power
        if remaining < 0.0 or path_count < 4:
            return None
        powers[0] = main_power
        powers[-1] = max_power
        powers[1] = second_power
        if path_count > 3:
            weights = rng.dirichlet(np.ones(path_count - 3))
            powers[2:-1] = remaining * weights
        return _normalize_power(powers)

    if level is ChannelLevel.B:
        delayed_ratio = rng.uniform(0.35, 0.75)
        main_power = 1.0 - delayed_ratio
        max_relative_db = rng.uniform(-6.0, 0.0)
        max_power = main_power * 10.0 ** (max_relative_db / 10.0)
        remaining = delayed_ratio - max_power
        if remaining < 0.0:
            return None
        powers[0] = main_power
        powers[-1] = max_power
        if path_count > 2:
            weights = rng.dirichlet(np.ones(path_count - 2))
            powers[1:-1] = remaining * weights
        return _normalize_power(powers)

    delayed_ratio = rng.uniform(0.50, 0.90)
    powers[0] = 1.0 - delayed_ratio
    if path_count > 1:
        powers[1:] = delayed_ratio * rng.dirichlet(np.ones(path_count - 1))
    return _normalize_power(powers)


def _normalize_power(powers: np.ndarray) -> np.ndarray:
    total = float(np.sum(powers))
    if total <= 0.0:
        raise ValueError("功率总和必须为正。")
    return powers / total


def _build_profile(
    level: ChannelLevel,
    delays: Sequence[int],
    taps: np.ndarray,
    spectral_grid: int,
) -> ChannelProfile:
    powers = np.abs(taps) ** 2
    powers = powers / np.sum(powers)
    taps = taps / np.sqrt(np.sum(np.abs(taps) ** 2))
    sorted_powers = np.sort(powers)[::-1]
    strongest_gap_db = float(10.0 * np.log10((sorted_powers[0] + 1e-12) / (sorted_powers[1] + 1e-12)))
    max_delay_relative_db = float(10.0 * np.log10((powers[-1] + 1e-12) / (powers[0] + 1e-12)))
    delayed_energy_ratio = float(np.sum(powers[1:]))
    response = _frequency_response(delays, taps, spectral_grid)
    magnitude = np.abs(response)
    min_mag = float(np.min(magnitude))
    max_mag = float(np.max(magnitude))
    notch_depth_db = float(20.0 * np.log10((min_mag + 1e-12) / (max_mag + 1e-12)))
    condition_proxy = float(((max_mag + 1e-12) / (min_mag + 1e-12)) ** 2)
    return ChannelProfile(
        level=level,
        delays=tuple(int(x) for x in delays),
        taps=taps,
        strongest_gap_db=strongest_gap_db,
        max_delay_relative_db=max_delay_relative_db,
        delayed_energy_ratio=delayed_energy_ratio,
        notch_depth_db=notch_depth_db,
        condition_proxy=condition_proxy,
    )


def _frequency_response(delays: Sequence[int], taps: np.ndarray, spectral_grid: int) -> np.ndarray:
    omega = 2.0 * np.pi * np.arange(spectral_grid, dtype=np.float64) / spectral_grid
    delay_array = np.asarray(delays, dtype=np.float64)
    exponent = np.exp(-1j * omega[:, None] * delay_array[None, :])
    return exponent @ taps


def _satisfies_level(profile: ChannelProfile) -> bool:
    path_count = len(profile.delays)
    if profile.delays[0] != 0 or profile.delays[-1] <= 0:
        return False
    if not np.isclose(np.sum(np.abs(profile.taps) ** 2), 1.0, atol=1e-6):
        return False
    if profile.level is ChannelLevel.A:
        return (
            3 <= path_count <= 5
            and 6.0 <= profile.strongest_gap_db <= 15.0
            and -20.0 <= profile.max_delay_relative_db <= -10.0
            and 0.10 <= profile.delayed_energy_ratio <= 0.35
        )
    if profile.level is ChannelLevel.B:
        return (
            3 <= path_count <= 7
            and 0.0 <= profile.strongest_gap_db <= 6.0
            and profile.max_delay_relative_db >= -10.0
            and 0.35 <= profile.delayed_energy_ratio <= 0.75
        )
    return 3 <= path_count <= 10 and 0.50 <= profile.delayed_energy_ratio <= 0.90
