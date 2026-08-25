# -*- coding: utf-8 -*-
"""由文献包络约束的稀疏强径与弱弥散 EME 信道 profile。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from env.eme_reference import (
    EME_FULL_RADAR_DEPTH_SECONDS,
    load_evans_1965_envelope,
    physical_delay_samples,
)


_AGGREGATION_BY_LEVEL = {"A": "sanity", "B": "main", "C": "pressure"}
_ANOMALOUS_SCATTERER_DOI = "10.1029/JZ067i012p04881"


def _immutable_array(values: Any, dtype: np.dtype[Any]) -> np.ndarray:
    contiguous = np.ascontiguousarray(values, dtype=dtype)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def _finite_pair(
    value: tuple[float, float], field_name: str
) -> tuple[float, float]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{field_name} 必须是包含两个值的 tuple。")
    try:
        low, high = (float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} 必须包含有限数值。") from exc
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f"{field_name} 必须是有限且按升序排列的范围。")
    return low, high


def _freeze_metadata_value(value: Any, path: str = "metadata") -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} 的键必须为字符串，无法可靠冻结。")
            frozen[key] = _freeze_metadata_value(item, f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_metadata_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError(f"{path} 的 object ndarray 无法可靠冻结。")
        return _immutable_array(value, value.dtype)
    if isinstance(value, np.generic):
        return _freeze_metadata_value(value.item(), path)
    if value is None or isinstance(value, (str, bytes, bool, int, float, complex)):
        return value
    raise ValueError(f"{path} 包含无法可靠冻结的可变或未知类型。")


def _positive_integer(value: Any, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{field_name} 必须为非布尔正整数。")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{field_name} 必须为非布尔正整数。")
    return normalized


@dataclass(frozen=True)
class EMEChannelProfileConfig:
    """符号率等效 EME profile 的显式建模候选范围。"""

    level: str
    sample_rate_hz: float
    symbol_rate_hz: float
    frame_len: int
    strong_path_count: tuple[int, int]
    diffuse_energy_ratio: tuple[float, float]
    seed: int
    max_delay_seconds: float = EME_FULL_RADAR_DEPTH_SECONDS
    include_anomalous_scatterer: bool = False
    anomalous_power_gain: tuple[float, float] = (7.0, 8.0)

    def __post_init__(self) -> None:
        if not isinstance(self.level, str) or self.level not in _AGGREGATION_BY_LEVEL:
            raise ValueError("level 必须为 A、B 或 C。")

        if isinstance(self.sample_rate_hz, (bool, np.bool_)):
            raise ValueError("sample_rate_hz 不得为布尔值。")
        if isinstance(self.symbol_rate_hz, (bool, np.bool_)):
            raise ValueError("symbol_rate_hz 不得为布尔值。")
        try:
            sample_rate = float(self.sample_rate_hz)
            symbol_rate = float(self.symbol_rate_hz)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("采样率和符号率必须为正且有限。") from exc
        if (
            not math.isfinite(sample_rate)
            or not math.isfinite(symbol_rate)
            or sample_rate <= 0.0
            or symbol_rate <= 0.0
        ):
            raise ValueError("采样率和符号率必须为正且有限。")
        if sample_rate != symbol_rate:
            raise ValueError("当前符号率等效模型要求 sample_rate_hz 等于 symbol_rate_hz。")

        if (
            isinstance(self.frame_len, bool)
            or not isinstance(self.frame_len, Integral)
            or self.frame_len <= 0
        ):
            raise ValueError("frame_len 必须为正整数。")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or self.seed < 0:
            raise ValueError("seed 必须为非负整数。")
        if not isinstance(self.include_anomalous_scatterer, bool):
            raise ValueError("include_anomalous_scatterer 必须为布尔值。")

        if isinstance(self.max_delay_seconds, (bool, np.bool_)):
            raise ValueError("max_delay_seconds 必须严格等于 0.0116。")
        try:
            max_delay_seconds = float(self.max_delay_seconds)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("max_delay_seconds 必须严格等于 0.0116。") from exc
        if (
            not math.isfinite(max_delay_seconds)
            or max_delay_seconds != EME_FULL_RADAR_DEPTH_SECONDS
        ):
            raise ValueError("max_delay_seconds 必须严格等于 0.0116。")
        max_delay = physical_delay_samples(
            sample_rate, EME_FULL_RADAR_DEPTH_SECONDS
        )

        path_range = self.strong_path_count
        if not isinstance(path_range, tuple) or len(path_range) != 2:
            raise ValueError("strong_path_count 必须是包含两个整数的 tuple。")
        low_paths, high_paths = path_range
        if any(isinstance(item, bool) or not isinstance(item, Integral) for item in path_range):
            raise ValueError("strong_path_count 必须包含整数。")
        if low_paths < 2 or low_paths > high_paths:
            raise ValueError("strong_path_count 必须满足 2 <= low <= high。")

        diffuse_range = _finite_pair(self.diffuse_energy_ratio, "diffuse_energy_ratio")
        if diffuse_range[0] < 0.0 or diffuse_range[1] >= 1.0:
            raise ValueError("diffuse_energy_ratio 必须位于 [0, 1) 内。")

        available_strong_taps = max_delay + 1
        if diffuse_range[1] > 0.0:
            available_strong_taps -= 1
        if high_paths > available_strong_taps:
            raise ValueError(
                "strong_path_count 超过当前时延支撑上的可用 tap 数。"
            )

        anomalous_range = _finite_pair(
            self.anomalous_power_gain, "anomalous_power_gain"
        )
        if anomalous_range[0] < 7.0 or anomalous_range[1] > 8.0:
            raise ValueError("anomalous_power_gain 必须位于 [7, 8] 内。")

        object.__setattr__(self, "sample_rate_hz", sample_rate)
        object.__setattr__(self, "symbol_rate_hz", symbol_rate)
        object.__setattr__(self, "frame_len", int(self.frame_len))
        object.__setattr__(
            self, "max_delay_seconds", EME_FULL_RADAR_DEPTH_SECONDS
        )
        object.__setattr__(self, "strong_path_count", (int(low_paths), int(high_paths)))
        object.__setattr__(self, "diffuse_energy_ratio", diffuse_range)
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "anomalous_power_gain", anomalous_range)

    @property
    def samples_per_symbol(self) -> float:
        """当前符号率等效模型固定每符号一个采样点。"""

        return 1.0

    @property
    def max_delay_samples(self) -> int:
        """按物理时延上界向上取整得到离散最大时延。"""

        return physical_delay_samples(
            self.sample_rate_hz, EME_FULL_RADAR_DEPTH_SECONDS
        )


@dataclass(frozen=True)
class EMEChannelProfile:
    """冻结的复 CIR 及其结构化文献、聚合与采样语义。"""

    cir: np.ndarray
    strong_delays: np.ndarray
    diffuse_mask: np.ndarray
    diffuse_energy_ratio: float
    effective_taps_90: int
    max_delay_samples: int
    aggregation: str
    anomalous_delay: int | None
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        max_delay_samples = _positive_integer(
            self.max_delay_samples, "max_delay_samples"
        )
        effective_taps_90 = _positive_integer(
            self.effective_taps_90, "effective_taps_90"
        )
        if self.anomalous_delay is None:
            anomalous_delay = None
        else:
            anomalous_delay = _positive_integer(
                self.anomalous_delay, "anomalous_delay"
            )
            if anomalous_delay > max_delay_samples:
                raise ValueError("anomalous_delay 不得超过 max_delay_samples。")

        try:
            raw_strong_delays = np.asarray(self.strong_delays, dtype=object)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("strong_delays 必须是一维非布尔整数序列。") from exc
        if raw_strong_delays.ndim != 1:
            raise ValueError("strong_delays 必须是一维非布尔整数序列。")
        if any(
            isinstance(item, (bool, np.bool_)) or not isinstance(item, Integral)
            for item in raw_strong_delays
        ):
            raise ValueError("strong_delays 的每个元素必须为非布尔整数。")
        normalized_delays = [int(item) for item in raw_strong_delays]
        if any(delay < 0 or delay > max_delay_samples for delay in normalized_delays):
            raise ValueError("strong_delays 必须位于完整时延支撑内。")
        strong_delays = np.asarray(normalized_delays, dtype=np.int64)

        try:
            cir = np.asarray(self.cir, dtype=np.complex128)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("CIR 必须是一维有限复数数组。") from exc
        if cir.ndim != 1:
            raise ValueError("CIR 必须是一维有限复数数组。")
        if cir.size != max_delay_samples + 1:
            raise ValueError("CIR 必须覆盖 max_delay_samples 定义的完整时延支撑。")
        if not np.all(np.isfinite(cir)):
            raise ValueError("CIR 的所有复数元素必须有限。")
        powers = np.abs(cir) ** 2
        total_energy = float(np.sum(powers))
        if not math.isfinite(total_energy) or total_energy <= 0.0:
            raise ValueError("CIR 总能量必须为有限非零值。")
        if not math.isclose(total_energy, 1.0, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("CIR 总能量必须约等于 1。")

        try:
            raw_diffuse_mask = np.asarray(self.diffuse_mask, dtype=object)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("diffuse_mask 必须是一维布尔数组。") from exc
        if raw_diffuse_mask.ndim != 1 or raw_diffuse_mask.size != cir.size:
            raise ValueError("diffuse_mask 必须是一维且与 CIR 等长。")
        if any(
            not isinstance(item, (bool, np.bool_)) for item in raw_diffuse_mask
        ):
            raise ValueError("diffuse_mask 的每个元素必须为布尔值。")
        diffuse_mask = np.asarray(
            [bool(item) for item in raw_diffuse_mask], dtype=np.bool_
        )

        if strong_delays.size < 2 or np.any(np.diff(strong_delays) <= 0):
            raise ValueError("强径时延必须至少含两条且严格递增。")
        if strong_delays[0] != 0:
            raise ValueError("强径时延必须从 0 开始且不得越过最大时延。")
        if np.any(diffuse_mask[strong_delays]):
            raise ValueError("弥散 mask 不得覆盖强径。")
        if np.any(powers[strong_delays] <= 0.0):
            raise ValueError("每条强径必须具有非零能量。")
        strong_mask = np.zeros(cir.size, dtype=np.bool_)
        strong_mask[strong_delays] = True
        if np.any(powers[~(strong_mask | diffuse_mask)] != 0.0):
            raise ValueError("强径与弥散支撑之外不得存在非零能量。")

        if isinstance(self.diffuse_energy_ratio, (bool, np.bool_)):
            raise ValueError("diffuse_energy_ratio 必须为 [0, 1) 内的有限数值。")
        try:
            diffuse_energy_ratio = float(self.diffuse_energy_ratio)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "diffuse_energy_ratio 必须为 [0, 1) 内的有限数值。"
            ) from exc
        if (
            not math.isfinite(diffuse_energy_ratio)
            or diffuse_energy_ratio < 0.0
            or diffuse_energy_ratio >= 1.0
        ):
            raise ValueError("diffuse_energy_ratio 必须为 [0, 1) 内的有限数值。")
        actual_diffuse_energy = float(np.sum(powers[diffuse_mask]))
        if not math.isclose(
            diffuse_energy_ratio,
            actual_diffuse_energy,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("diffuse_energy_ratio 必须等于 mask 内的实际能量。")

        if effective_taps_90 > cir.size:
            raise ValueError("effective_taps_90 必须位于 [1, CIR 长度] 内。")
        recomputed_effective_taps = _effective_taps_90(cir)
        if effective_taps_90 != recomputed_effective_taps:
            raise ValueError("effective_taps_90 必须等于从 CIR 重算的值。")

        if self.aggregation not in _AGGREGATION_BY_LEVEL.values():
            raise ValueError("aggregation 必须为 sanity、main 或 pressure。")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata 必须为映射。")
        metadata = _freeze_metadata_value(self.metadata)
        anomaly_metadata = metadata.get("anomalous_scatterer")
        if not isinstance(anomaly_metadata, Mapping):
            raise ValueError("anomalous_scatterer metadata 必须为映射。")
        metadata_enabled = anomaly_metadata.get("enabled")
        metadata_delay = anomaly_metadata.get("delay")
        if not isinstance(metadata_enabled, bool):
            raise ValueError("anomalous_scatterer metadata.enabled 必须为布尔值。")
        if metadata_delay is not None and (
            isinstance(metadata_delay, bool) or not isinstance(metadata_delay, Integral)
        ):
            raise ValueError("anomalous_scatterer metadata.delay 必须为整数或 None。")
        normalized_metadata_delay = (
            None if metadata_delay is None else int(metadata_delay)
        )
        if anomalous_delay is None:
            if metadata_enabled or normalized_metadata_delay is not None:
                raise ValueError("anomalous_delay 必须与异常 metadata 保持一致。")
        else:
            if anomalous_delay == 0 or anomalous_delay not in strong_delays:
                raise ValueError("anomalous_delay 必须是非零 strong delay。")
            if not metadata_enabled or normalized_metadata_delay != anomalous_delay:
                raise ValueError("anomalous_delay 必须与异常 metadata 保持一致。")

        object.__setattr__(self, "cir", _immutable_array(cir, np.dtype(np.complex128)))
        object.__setattr__(
            self,
            "strong_delays",
            _immutable_array(strong_delays, np.dtype(np.int64)),
        )
        object.__setattr__(
            self,
            "diffuse_mask",
            _immutable_array(diffuse_mask, np.dtype(np.bool_)),
        )
        object.__setattr__(self, "diffuse_energy_ratio", diffuse_energy_ratio)
        object.__setattr__(self, "effective_taps_90", effective_taps_90)
        object.__setattr__(self, "max_delay_samples", max_delay_samples)
        object.__setattr__(self, "anomalous_delay", anomalous_delay)
        object.__setattr__(self, "metadata", metadata)


def _envelope_linear_power(max_delay_samples: int, sample_rate_hz: float) -> tuple[np.ndarray, Any]:
    envelope = load_evans_1965_envelope()
    observed = envelope.observed_mask
    delay_anchors = np.concatenate(
        (envelope.delay_seconds[observed], envelope.delay_seconds[-1:])
    )
    observed_midpoint_db = (
        envelope.upper_power_db[observed] + envelope.lower_power_db[observed]
    ) / 2.0
    power_db_anchors = np.concatenate(
        (observed_midpoint_db, envelope.upper_power_db[-1:])
    )
    tap_delays_seconds = (
        np.arange(max_delay_samples + 1, dtype=np.float64) / sample_rate_hz
    )
    tap_power_db = np.interp(
        tap_delays_seconds,
        delay_anchors,
        power_db_anchors,
        left=power_db_anchors[0],
        right=power_db_anchors[-1],
    )
    return np.power(10.0, tap_power_db / 10.0), envelope


def _weighted_choice(
    rng: np.random.Generator,
    candidates: np.ndarray,
    count: int,
    weights: np.ndarray,
) -> np.ndarray:
    if count == 0:
        return np.empty(0, dtype=np.int64)
    probabilities = weights / np.sum(weights)
    return np.asarray(
        rng.choice(candidates, size=count, replace=False, p=probabilities),
        dtype=np.int64,
    )


def _sample_strong_delays(
    rng: np.random.Generator,
    max_delay_samples: int,
    path_count: int,
    envelope_power: np.ndarray,
) -> np.ndarray:
    long_start = math.ceil(0.75 * max_delay_samples)
    long_candidates = np.arange(long_start, max_delay_samples + 1, dtype=np.int64)
    long_delay = int(
        _weighted_choice(
            rng,
            long_candidates,
            1,
            envelope_power[long_candidates],
        )[0]
    )
    remaining = np.setdiff1d(
        np.arange(1, max_delay_samples + 1, dtype=np.int64),
        np.asarray([long_delay], dtype=np.int64),
        assume_unique=True,
    )
    others = _weighted_choice(
        rng,
        remaining,
        path_count - 2,
        envelope_power[remaining],
    )
    return np.asarray(sorted((0, long_delay, *others.tolist())), dtype=np.int64)


def _effective_taps_90(cir: np.ndarray) -> int:
    descending_power = np.sort(np.abs(cir) ** 2)[::-1]
    return int(np.searchsorted(np.cumsum(descending_power), 0.9, side="left") + 1)


def sample_eme_profile(config: EMEChannelProfileConfig) -> EMEChannelProfile:
    """按显式候选范围采样可完全复现的文献包络约束 EME profile。"""

    if not isinstance(config, EMEChannelProfileConfig):
        raise ValueError("config 必须是 EMEChannelProfileConfig。")

    rng = np.random.default_rng(config.seed)
    max_delay = config.max_delay_samples
    envelope_power, envelope = _envelope_linear_power(
        max_delay, config.sample_rate_hz
    )
    path_count = int(
        rng.integers(config.strong_path_count[0], config.strong_path_count[1] + 1)
    )
    strong_delays = _sample_strong_delays(
        rng, max_delay, path_count, envelope_power
    )

    diffuse_low, diffuse_high = config.diffuse_energy_ratio
    diffuse_ratio = (
        diffuse_low
        if diffuse_low == diffuse_high
        else float(rng.uniform(diffuse_low, diffuse_high))
    )

    strong_power = np.array(envelope_power[strong_delays], copy=True)
    anomalous_delay: int | None = None
    anomalous_gain: float | None = None
    if config.include_anomalous_scatterer:
        anomalous_index = int(rng.integers(1, strong_delays.size))
        anomalous_delay = int(strong_delays[anomalous_index])
        gain_low, gain_high = config.anomalous_power_gain
        anomalous_gain = (
            gain_low if gain_low == gain_high else float(rng.uniform(gain_low, gain_high))
        )
        strong_power[anomalous_index] *= anomalous_gain

    strong_power *= (1.0 - diffuse_ratio) / np.sum(strong_power)
    strong_phase = rng.uniform(-np.pi, np.pi, size=strong_delays.size)
    cir = np.zeros(max_delay + 1, dtype=np.complex128)
    cir[strong_delays] = np.sqrt(strong_power) * np.exp(1j * strong_phase)

    diffuse_mask = np.zeros(max_delay + 1, dtype=np.bool_)
    if diffuse_ratio > 0.0:
        diffuse_mask[:] = True
        diffuse_mask[strong_delays] = False
        diffuse_noise = (
            rng.standard_normal(np.count_nonzero(diffuse_mask))
            + 1j * rng.standard_normal(np.count_nonzero(diffuse_mask))
        ) / math.sqrt(2.0)
        diffuse_taps = diffuse_noise * np.sqrt(envelope_power[diffuse_mask])
        diffuse_taps *= math.sqrt(
            diffuse_ratio / float(np.sum(np.abs(diffuse_taps) ** 2))
        )
        cir[diffuse_mask] = diffuse_taps

    cir /= math.sqrt(float(np.sum(np.abs(cir) ** 2)))
    actual_diffuse_ratio = float(np.sum(np.abs(cir[diffuse_mask]) ** 2))
    aggregation = _AGGREGATION_BY_LEVEL[config.level]
    metadata = {
        "modeling_ranges": {
            "strong_path_count": config.strong_path_count,
            "diffuse_energy_ratio": config.diffuse_energy_ratio,
            "semantics": "configured_modeling_candidates",
            "direct_eme_measurement": False,
        },
        "envelope_evidence": {
            "source_doi": envelope.source_doi,
            "observed_power_rule": "midpoint_db",
            "power_domain": "linear_power_after_db_conversion",
            "delay_sampling_weight": "envelope_linear_power",
            "endpoint_point_kind": envelope.point_kind[-1],
            "endpoint_power_rule": "upper_hold",
            "lower_endpoint_semantics": "right_censored_limit",
            "lower_endpoint_used_as_exact_power": False,
        },
        "anomalous_scatterer": {
            "enabled": config.include_anomalous_scatterer,
            "delay": anomalous_delay,
            "power_gain": anomalous_gain,
            "source_doi": (
                _ANOMALOUS_SCATTERER_DOI
                if config.include_anomalous_scatterer
                else None
            ),
        },
        "include_in_main_average": aggregation == "main",
    }
    return EMEChannelProfile(
        cir=cir,
        strong_delays=strong_delays,
        diffuse_mask=diffuse_mask,
        diffuse_energy_ratio=actual_diffuse_ratio,
        effective_taps_90=_effective_taps_90(cir),
        max_delay_samples=max_delay,
        aggregation=aggregation,
        anomalous_delay=anomalous_delay,
        metadata=metadata,
    )
