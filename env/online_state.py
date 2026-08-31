# -*- coding: utf-8 -*-
"""EME 在线状态划分、合法性校验与确定性采样。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any, Mapping

import numpy as np

from env.eme_reference import EME_FULL_RADAR_DEPTH_SECONDS


_SUPPORTED_PROFILES = {"eme_measurement_v1", "eme_long_memory_v2"}


def _pair(value: Any, field_name: str, cast: type) -> tuple[Any, Any]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} 必须是长度为 2 的数组。")
    try:
        low, high = cast(value[0]), cast(value[1])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} 包含非法数值。") from exc
    return low, high


def _integer_pair(value: Any, field_name: str) -> tuple[int, int]:
    low, high = _pair(value, field_name, int)
    if low < 2 or low > high:
        raise ValueError(f"{field_name} 必须满足 2 <= low <= high。")
    return low, high


def _float_pair(value: Any, field_name: str) -> tuple[float, float]:
    low, high = _pair(value, field_name, float)
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f"{field_name} 必须为有限且升序的范围。")
    return low, high


def _bounded_pair(
    value: Any,
    field_name: str,
    lower: float,
    upper: float,
    base: tuple[float, float],
) -> tuple[float, float]:
    pair = _float_pair(value, field_name)
    if pair[0] < lower or pair[1] > upper:
        raise ValueError(f"{field_name} 必须位于 [{lower}, {upper}] 内。")
    if pair[0] < base[0] or pair[1] > base[1]:
        raise ValueError(f"{field_name} 不得超出冻结 profile 范围 {base}。")
    return pair


@dataclass(frozen=True)
class OnlineStateSplit:
    """一个共享物理 profile 内的当前隐状态采样范围。"""

    name: str
    profile_name: str
    max_delay_seconds: float
    strong_path_count: tuple[int, int]
    diffuse_energy_ratio: tuple[float, float]
    cfo_abs_range: tuple[float, float]
    phase_noise_std_range: tuple[float, float]
    pilot_layout: str = "prefix"

    def sample(self, seed: int) -> dict[str, float | int | str]:
        """从当前 split 确定性采样一个 episode 级状态实例。"""

        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("state split sample 的 seed 必须为非负整数。")
        rng = np.random.default_rng(int(seed))
        path_count = int(
            rng.integers(self.strong_path_count[0], self.strong_path_count[1] + 1)
        )
        diffuse = float(rng.uniform(*self.diffuse_energy_ratio))
        cfo_abs = float(rng.uniform(*self.cfo_abs_range))
        phase_noise_std = float(rng.uniform(*self.phase_noise_std_range))
        sign = -1.0 if float(rng.uniform()) < 0.5 else 1.0
        return {
            "state_split": self.name,
            "profile_name": self.profile_name,
            "max_delay_seconds": self.max_delay_seconds,
            "strong_path_count": path_count,
            "diffuse_energy_ratio": diffuse,
            "cfo_abs": cfo_abs,
            "cfo_cycles_per_symbol": sign * cfo_abs,
            "phase_noise_std": phase_noise_std,
            "pilot_layout": self.pilot_layout,
        }

    def to_metadata(self) -> dict[str, Any]:
        """导出不含随机实例的 split 元数据。"""

        return {
            "name": self.name,
            "profile_name": self.profile_name,
            "max_delay_seconds": self.max_delay_seconds,
            "strong_path_count": list(self.strong_path_count),
            "diffuse_energy_ratio": list(self.diffuse_energy_ratio),
            "cfo_abs_range": list(self.cfo_abs_range),
            "phase_noise_std_range": list(self.phase_noise_std_range),
            "pilot_layout": self.pilot_layout,
        }


def load_online_state_split(
    experiment: Mapping[str, Any], split_name: str
) -> OnlineStateSplit:
    """从实验配置加载一个受冻结 profile 约束的在线状态 split。"""

    if not isinstance(split_name, str) or not split_name:
        raise ValueError("split_name 必须为非空字符串。")
    profile_name = str(experiment.get("profile_name", experiment.get("channel_profile", "")))
    if profile_name not in _SUPPORTED_PROFILES:
        raise ValueError(f"在线状态 split 不支持 profile={profile_name}。")
    if str(experiment.get("channel_profile", profile_name)) != profile_name:
        raise ValueError("channel_profile 必须与 profile_name 一致。")
    max_delay_seconds = float(experiment.get("max_delay_seconds", -1.0))
    if max_delay_seconds != EME_FULL_RADAR_DEPTH_SECONDS:
        raise ValueError("在线状态 split 不得改变冻结的 max_delay_seconds=0.0116。")
    splits = experiment.get("online_state_splits")
    if not isinstance(splits, Mapping) or split_name not in splits:
        raise ValueError(f"配置中不存在 online_state_splits[{split_name!r}]。")
    payload = splits[split_name]
    if not isinstance(payload, Mapping):
        raise ValueError(f"online_state_splits[{split_name!r}] 必须为对象。")
    if "max_delay_seconds" in payload:
        payload_delay = float(payload["max_delay_seconds"])
        if payload_delay != EME_FULL_RADAR_DEPTH_SECONDS:
            raise ValueError("online 状态 split 不得覆盖冻结的 max_delay_seconds。")
    if "profile_name" in payload and str(payload["profile_name"]) != profile_name:
        raise ValueError("online 状态 split 不得改变 profile_name。")

    base_paths = _integer_pair(experiment.get("strong_path_count"), "strong_path_count")
    base_diffuse = _bounded_pair(
        experiment.get("diffuse_energy_ratio"),
        "diffuse_energy_ratio",
        0.0,
        1.0,
        (0.0, 1.0),
    )
    paths = _integer_pair(payload.get("strong_path_count"), "state strong_path_count")
    if paths[0] < base_paths[0] or paths[1] > base_paths[1]:
        raise ValueError(f"state strong_path_count 不得超出冻结 profile 范围 {base_paths}。")
    diffuse = _bounded_pair(
        payload.get("diffuse_energy_ratio"),
        "state diffuse_energy_ratio",
        0.0,
        1.0,
        base_diffuse,
    )
    cfo = _bounded_pair(
        payload.get("cfo_abs_range"),
        "cfo_abs_range",
        0.0,
        0.5,
        (0.0, 0.5),
    )
    phase = _bounded_pair(
        payload.get("phase_noise_std_range"),
        "phase_noise_std_range",
        0.0,
        float("inf"),
        (0.0, float("inf")),
    )
    layout = str(payload.get("pilot_layout", "prefix"))
    if layout != "prefix":
        raise ValueError("在线状态 split 不得改变 prefix Pilot layout。")
    return OnlineStateSplit(
        name=split_name,
        profile_name=profile_name,
        max_delay_seconds=max_delay_seconds,
        strong_path_count=paths,
        diffuse_energy_ratio=diffuse,
        cfo_abs_range=cfo,
        phase_noise_std_range=phase,
        pilot_layout=layout,
    )
