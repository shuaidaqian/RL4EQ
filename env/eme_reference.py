# -*- coding: utf-8 -*-
"""Evans 1965 月球雷达回波参考数据与物理时延换算。"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EME_FULL_RADAR_DEPTH_SECONDS = 0.0116

_REFERENCE_SCHEMA_VERSION = "eme-reference-v1"
_SOURCE_DOI = "10.6028/jres.069d.195"
_SOURCE_FIGURE = "Evans 1965 Fig. 8"
_REFERENCE_DIR = Path(__file__).resolve().parents[1] / "data" / "eme"
_ENVELOPE_PATH = _REFERENCE_DIR / "evans_1965_fig8_envelope.csv"
_MANIFEST_PATH = _REFERENCE_DIR / "reference_manifest.json"
_REQUIRED_CSV_COLUMNS = {
    "delay_ms",
    "power_db_3p6cm",
    "power_db_68cm",
    "point_kind",
    "source_figure",
    "digitization_note",
}
_ALLOWED_POINT_KINDS = {"digitized", "support_extension"}


def _immutable_array(values: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    contiguous = np.ascontiguousarray(values, dtype=dtype)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype)


def _is_close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-15)


@dataclass(frozen=True)
class EMEUpperEndpointPolicy:
    """上包络在物理支撑端点的延伸策略。"""

    mode: str
    value_db: float
    observed: bool

    def __post_init__(self) -> None:
        if self.mode != "hold_last_observed":
            raise ValueError("manifest 上端点模式必须为 hold_last_observed。")
        if not math.isfinite(self.value_db):
            raise ValueError("manifest 上端点功率必须有限。")
        if self.observed is not False:
            raise ValueError("manifest 上端点必须标记为非观测值。")


@dataclass(frozen=True)
class EMELowerEndpointPolicy:
    """下包络在物理支撑端点的右删失策略。"""

    mode: str
    censoring_limit_db: float
    observed: bool

    def __post_init__(self) -> None:
        if self.mode != "right_censored":
            raise ValueError("manifest 下端点模式必须为 right_censored。")
        if not math.isfinite(self.censoring_limit_db):
            raise ValueError("manifest 下端点删失边界必须有限。")
        if self.observed is not False:
            raise ValueError("manifest 下端点必须标记为非观测值。")


@dataclass(frozen=True)
class EMEEndpointExtensionPolicy:
    """物理雷达深度端点的机器可审计策略。"""

    kind: str
    delay_seconds: float
    upper: EMEUpperEndpointPolicy
    lower: EMELowerEndpointPolicy

    def __post_init__(self) -> None:
        if self.kind != "support_extension":
            raise ValueError("manifest 端点 kind 必须为 support_extension。")
        if not math.isfinite(self.delay_seconds) or not _is_close(
            self.delay_seconds, EME_FULL_RADAR_DEPTH_SECONDS
        ):
            raise ValueError("manifest 端点时延必须为 0.0116 秒。")
        if not isinstance(self.upper, EMEUpperEndpointPolicy) or not isinstance(
            self.lower, EMELowerEndpointPolicy
        ):
            raise ValueError("manifest 端点上下界策略类型无效。")


@dataclass(frozen=True)
class EMEEchoEnvelope:
    """不可变的 EME 回波功率上下包络及其观测语义。"""

    delay_seconds: np.ndarray
    upper_power_db: np.ndarray
    lower_power_db: np.ndarray
    point_kind: tuple[str, ...]
    observed_mask: np.ndarray
    endpoint_extension_policy: EMEEndpointExtensionPolicy
    source_doi: str

    def __post_init__(self) -> None:
        try:
            delay = np.asarray(self.delay_seconds, dtype=np.float64)
            upper = np.asarray(self.upper_power_db, dtype=np.float64)
            lower = np.asarray(self.lower_power_db, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("时延和功率数组必须可转换为浮点数组。") from exc

        if delay.ndim != 1 or upper.ndim != 1 or lower.ndim != 1:
            raise ValueError("时延和功率数组必须是一维数组。")
        if delay.size == 0 or upper.size == 0 or lower.size == 0:
            raise ValueError("时延和功率数组不能为空。")
        if not (delay.size == upper.size == lower.size):
            raise ValueError("时延和功率数组必须等长。")
        if not (
            np.all(np.isfinite(delay))
            and np.all(np.isfinite(upper))
            and np.all(np.isfinite(lower))
        ):
            raise ValueError("时延和功率数组必须全部有限。")
        if np.any(np.diff(delay) <= 0.0):
            raise ValueError("参考数据时延必须严格递增。")
        if not _is_close(delay[0], 0.0):
            raise ValueError("参考数据首个时延必须为 0。")
        if not _is_close(delay[-1], EME_FULL_RADAR_DEPTH_SECONDS):
            raise ValueError("参考数据末端时延必须为 0.0116 秒。")
        if np.any(lower > upper):
            raise ValueError("下包络功率不得高于上包络功率。")

        try:
            point_kind = tuple(self.point_kind)
        except TypeError as exc:
            raise ValueError("point_kind 必须是可迭代序列。") from exc
        if len(point_kind) != delay.size:
            raise ValueError("point_kind 必须与功率数组等长。")
        if any(kind not in _ALLOWED_POINT_KINDS for kind in point_kind):
            raise ValueError("point_kind 只能为 digitized 或 support_extension。")
        extension_indices = [
            index for index, kind in enumerate(point_kind) if kind == "support_extension"
        ]
        if extension_indices != [delay.size - 1]:
            raise ValueError("support_extension 必须唯一且位于末行。")

        observed_mask = np.asarray(self.observed_mask)
        if observed_mask.ndim != 1 or observed_mask.size != delay.size:
            raise ValueError("observed_mask 必须是一维等长数组。")
        if observed_mask.dtype != np.dtype(np.bool_):
            raise ValueError("observed_mask 必须为布尔数组。")
        expected_mask = np.asarray(
            [kind == "digitized" for kind in point_kind], dtype=np.bool_
        )
        if not np.array_equal(observed_mask, expected_mask):
            raise ValueError("observed_mask 必须与 point_kind 的观测语义一致。")

        policy = self.endpoint_extension_policy
        if not isinstance(policy, EMEEndpointExtensionPolicy):
            raise ValueError("端点策略类型无效。")
        if policy.kind != point_kind[-1] or not _is_close(policy.delay_seconds, delay[-1]):
            raise ValueError("端点策略与 CSV 末行的类型或时延不一致。")
        if not _is_close(policy.upper.value_db, upper[-1]):
            raise ValueError("端点策略与 CSV 末行的上包络功率不一致。")
        if not _is_close(policy.lower.censoring_limit_db, lower[-1]):
            raise ValueError("端点策略与 CSV 末行的下包络删失边界不一致。")
        if self.source_doi != _SOURCE_DOI:
            raise ValueError("source_doi 与 Evans 1965 DOI 不一致。")

        object.__setattr__(self, "delay_seconds", _immutable_array(delay, np.dtype(np.float64)))
        object.__setattr__(self, "upper_power_db", _immutable_array(upper, np.dtype(np.float64)))
        object.__setattr__(self, "lower_power_db", _immutable_array(lower, np.dtype(np.float64)))
        object.__setattr__(self, "point_kind", point_kind)
        object.__setattr__(
            self,
            "observed_mask",
            _immutable_array(observed_mask, np.dtype(np.bool_)),
        )


def _load_csv_rows(envelope_path: Path) -> list[dict[str, str]]:
    with envelope_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("CSV 缺少表头。")
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError("CSV 表头包含重复列。")
        missing_columns = _REQUIRED_CSV_COLUMNS.difference(fieldnames)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"CSV 缺少必需列：{missing}")
        rows = list(reader)

    if not rows:
        raise ValueError("CSV 参考数据不能为空。")
    if any(row.get(column) is None for row in rows for column in _REQUIRED_CSV_COLUMNS):
        raise ValueError("CSV 行缺少必需字段值。")

    point_kinds = [row["point_kind"] for row in rows]
    if any(kind not in _ALLOWED_POINT_KINDS for kind in point_kinds):
        raise ValueError("CSV point_kind 只能为 digitized 或 support_extension。")
    extension_indices = [
        index for index, kind in enumerate(point_kinds) if kind == "support_extension"
    ]
    if extension_indices != [len(rows) - 1]:
        raise ValueError("CSV support_extension 必须唯一且位于末行。")
    return rows


def _mapping_field(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"manifest 缺少 {context}.{key}。")
    return mapping[key]


def _load_manifest_policy(manifest_path: Path) -> tuple[str, EMEEndpointExtensionPolicy]:
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest 顶层必须是对象。")

    expected_identity = {
        "schema_version": _REFERENCE_SCHEMA_VERSION,
        "doi": _SOURCE_DOI,
        "source_figure": _SOURCE_FIGURE,
    }
    for key, expected in expected_identity.items():
        if manifest.get(key) != expected:
            raise ValueError(f"manifest {key} 必须为 {expected}。")

    raw_policy = _mapping_field(manifest, "endpoint_extension_policy", "")
    if not isinstance(raw_policy, Mapping):
        raise ValueError("manifest endpoint_extension_policy 必须是对象。")
    raw_upper = _mapping_field(raw_policy, "upper", "endpoint_extension_policy")
    raw_lower = _mapping_field(raw_policy, "lower", "endpoint_extension_policy")
    if not isinstance(raw_upper, Mapping) or not isinstance(raw_lower, Mapping):
        raise ValueError("manifest 端点上下界策略必须是对象。")

    try:
        upper = EMEUpperEndpointPolicy(
            mode=_mapping_field(raw_upper, "mode", "endpoint_extension_policy.upper"),
            value_db=float(
                _mapping_field(raw_upper, "value_db", "endpoint_extension_policy.upper")
            ),
            observed=_mapping_field(
                raw_upper, "observed", "endpoint_extension_policy.upper"
            ),
        )
        lower = EMELowerEndpointPolicy(
            mode=_mapping_field(raw_lower, "mode", "endpoint_extension_policy.lower"),
            censoring_limit_db=float(
                _mapping_field(
                    raw_lower,
                    "censoring_limit_db",
                    "endpoint_extension_policy.lower",
                )
            ),
            observed=_mapping_field(
                raw_lower, "observed", "endpoint_extension_policy.lower"
            ),
        )
        policy = EMEEndpointExtensionPolicy(
            kind=_mapping_field(raw_policy, "kind", "endpoint_extension_policy"),
            delay_seconds=float(
                _mapping_field(raw_policy, "delay_seconds", "endpoint_extension_policy")
            ),
            upper=upper,
            lower=lower,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, ValueError) and "manifest" in str(exc):
            raise
        raise ValueError("manifest endpoint_extension_policy 数值或结构无效。") from exc
    return str(manifest["doi"]), policy


def _load_evans_1965_envelope_from_paths(
    envelope_path: Path, manifest_path: Path
) -> EMEEchoEnvelope:
    rows = _load_csv_rows(envelope_path)
    source_doi, policy = _load_manifest_policy(manifest_path)

    try:
        delays_ms = np.asarray([float(row["delay_ms"]) for row in rows], dtype=np.float64)
        curve_3p6cm = np.asarray(
            [float(row["power_db_3p6cm"]) for row in rows], dtype=np.float64
        )
        curve_68cm = np.asarray(
            [float(row["power_db_68cm"]) for row in rows], dtype=np.float64
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("CSV 时延或功率字段必须是有限数值。") from exc

    point_kind = tuple(row["point_kind"] for row in rows)
    return EMEEchoEnvelope(
        delay_seconds=delays_ms / 1_000.0,
        upper_power_db=np.maximum(curve_3p6cm, curve_68cm),
        lower_power_db=np.minimum(curve_3p6cm, curve_68cm),
        point_kind=point_kind,
        observed_mask=np.asarray(
            [kind == "digitized" for kind in point_kind], dtype=np.bool_
        ),
        endpoint_extension_policy=policy,
        source_doi=source_doi,
    )


def load_evans_1965_envelope() -> EMEEchoEnvelope:
    """从模块相对位置加载 Evans 1965 图 8 的稀疏包络锚点。"""

    return _load_evans_1965_envelope_from_paths(_ENVELOPE_PATH, _MANIFEST_PATH)


def physical_delay_samples(sample_rate_hz: float, max_delay_seconds: float) -> int:
    """将最大物理时延按向上取整换算为离散采样点数。"""

    try:
        sample_rate = float(sample_rate_hz)
        max_delay = float(max_delay_seconds)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("采样率和最大时延必须为正且有限。") from exc
    if (
        not math.isfinite(sample_rate)
        or not math.isfinite(max_delay)
        or sample_rate <= 0.0
        or max_delay <= 0.0
    ):
        raise ValueError("采样率和最大时延必须为正且有限。")

    product = sample_rate * max_delay
    if not math.isfinite(product) or product <= 0.0:
        raise ValueError("采样率与最大时延的乘积必须为正且有限。")
    return math.ceil(product)
