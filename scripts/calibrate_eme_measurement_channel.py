# -*- coding: utf-8 -*-
"""EME 测量约束信道的统计与传统均衡器校准入口。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, module=r"matplotlib(\..*)?"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline.traditional_equalizers import (
    TRADITIONAL_BASELINES,
    TraditionalPhaseState,
    estimate_acquisition_cir_with_cfo,
    run_traditional_equalizer,
)
from env.comm_env import CommEnvConfig, CommunicationEnvironment
from env.eme_reference import load_evans_1965_envelope, physical_delay_samples
from env.extreme_delay_channel import ExtremeDelayChannel, ExtremeDelayChannelConfig


SCHEMA_VERSION = "eme-measurement-channel-candidates-v1"
RANGE_SEMANTICS = "configured_modeling_candidates_not_direct_eme_measurements"
SELECTION_ORDER = [
    "physical_delay",
    "support_stability",
    "cross_frame_error",
    "frame_lag_correlation",
    "envelope_rmse",
    "traditional_ber",
]
DEFAULT_FIXED = {
    "profile_name": "eme_measurement_v1",
    "level": "B",
    "max_delay_seconds": 0.0116,
    "sample_rate_hz": 2000,
    "symbol_rate_hz": 2000,
    "frame_len": 512,
    "main_snrs": [0, 5, 10, 15],
    "pilot_layout": "prefix",
    "pilot_total": 128,
    "sanity_impairment_profile": "clean",
}
DEFAULT_TRADITIONAL_METHODS = [
    "CFO+DD-Phase LMMSE-FIR",
    "CFO+DD-Phase DFE-RLS",
]
TOP_LEVEL_FIELDS = {
    "schema_version",
    "traditional_only",
    "proposed_methods_included",
    "fixed",
    "traditional_methods",
    "selection_order",
    "candidates",
}
FIXED_FIELDS = set(DEFAULT_FIXED)
CANDIDATE_FIELDS = {
    "candidate_id",
    "strong_path_count",
    "diffuse_energy_ratio",
    "coherence_time_seconds",
    "include_anomalous_scatterer",
    "impairment_profile",
    "residual_cfo_limit",
    "aggregation_role",
    "semantics",
}


def load_candidate_config(path: str | Path) -> dict:
    """读取并严格校验显式候选配置。"""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取候选配置：{source}") from exc
    if not isinstance(payload, dict):
        raise ValueError("候选配置顶层必须是对象。")
    _require_exact_keys(payload, TOP_LEVEL_FIELDS, "候选配置顶层")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version 必须是 {SCHEMA_VERSION}。")
    if payload["traditional_only"] is not True:
        raise ValueError("traditional_only 必须为 true。")
    if payload["proposed_methods_included"] is not False:
        raise ValueError("proposed_methods_included 必须为 false。")
    if payload["selection_order"] != SELECTION_ORDER:
        raise ValueError("selection_order 与固定选择顺序不一致。")
    fixed = payload["fixed"]
    if not isinstance(fixed, dict):
        raise ValueError("fixed 必须是对象。")
    _require_exact_keys(fixed, FIXED_FIELDS, "fixed")
    if fixed != DEFAULT_FIXED:
        raise ValueError("fixed 必须与 EME Level B 主配置完全一致。")
    _validate_methods(payload["traditional_methods"])
    _validate_candidates(payload["candidates"], require_standard_count=True)
    return payload


def run_calibration(
    candidates: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    frames: int,
    output_dir: str | Path,
    fixed: Mapping[str, Any] | None = None,
    traditional_methods: Sequence[str] | None = None,
    snrs: Sequence[float] | None = None,
) -> dict:
    """运行真实信道统计诊断与信息受限的传统 baseline 校准。"""

    normalized_fixed = _validate_runtime_fixed(DEFAULT_FIXED if fixed is None else fixed)
    normalized_candidates = _validate_candidates(
        candidates, require_standard_count=False
    )
    normalized_seeds = _validate_seeds(seeds)
    normalized_frames = _positive_integer(frames, "frames")
    methods = _validate_methods(
        DEFAULT_TRADITIONAL_METHODS
        if traditional_methods is None
        else traditional_methods
    )
    normalized_snrs = _validate_snrs(
        normalized_fixed["main_snrs"] if snrs is None else snrs
    )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    statistics = []
    pdp_series = []
    for candidate in normalized_candidates:
        statistic, plot_data = _measure_channel_statistics(
            candidate,
            normalized_fixed,
            normalized_seeds,
            normalized_frames,
        )
        statistics.append(statistic)
        pdp_series.append(plot_data)

    traditional_ber = _measure_traditional_ber(
        normalized_candidates,
        normalized_fixed,
        normalized_seeds,
        normalized_frames,
        methods,
        normalized_snrs,
    )
    payload = {
        "schema_version": "eme-measurement-channel-calibration-v1",
        "traditional_only": True,
        "proposed_methods_included": False,
        "uses_neural_network": False,
        "uses_rl": False,
        "uses_true_channel_for_statistics": True,
        "data_labels_used_for_adaptation": False,
        "max_delay_seconds": float(normalized_fixed["max_delay_seconds"]),
        "selection_order": list(SELECTION_ORDER),
        "range_semantics": RANGE_SEMANTICS,
        "fixed": normalized_fixed,
        "seeds": normalized_seeds,
        "frames": normalized_frames,
        "snrs": normalized_snrs,
        "traditional_methods": methods,
        "statistics": statistics,
        "traditional_ber": traditional_ber,
    }
    _write_json(target / "summary.json", payload)
    _write_csv(target / "channel_statistics.csv", statistics)
    _write_csv(target / "traditional_ber.csv", traditional_ber)
    _write_pdp_plot(target / "pdp_comparison.png", pdp_series)
    return payload


def _measure_channel_statistics(
    candidate: Mapping[str, Any],
    fixed: Mapping[str, Any],
    seeds: Sequence[int],
    frames: int,
) -> tuple[dict, dict]:
    cir_power_samples: list[np.ndarray] = []
    path_counts: list[int] = []
    supports: list[dict] = []
    effective_taps: list[int] = []
    diffuse_ratios: list[float] = []
    lag_correlations: list[complex] = []
    cross_errors: list[float] = []
    support_changed_frames = 0
    max_delay = physical_delay_samples(
        float(fixed["sample_rate_hz"]), float(fixed["max_delay_seconds"])
    )

    for seed in seeds:
        config = _diagnostic_channel_config(candidate, fixed, seed)
        channel = ExtremeDelayChannel(config)
        channel.reset_episode(torch.zeros(max_delay, dtype=torch.complex64))
        initial_support = tuple(channel.delays)
        supports.append({"seed": int(seed), "delays": list(initial_support)})
        path_counts.append(len(initial_support))
        previous_cir: torch.Tensor | None = None
        for _ in range(frames):
            impulse = torch.zeros(int(fixed["frame_len"]), dtype=torch.complex64)
            impulse[0] = 1.0 + 0.0j
            received = channel.transmit(impulse, add_noise=False)
            cir = channel.last_cir_used()
            support_changed_frames += int(tuple(channel.delays) != initial_support)
            cross_errors.append(
                float(torch.max(torch.abs(received[: cir.numel()] - cir)).item())
            )
            if previous_cir is not None:
                denominator = torch.linalg.vector_norm(previous_cir) * torch.linalg.vector_norm(cir)
                correlation = torch.sum(torch.conj(previous_cir) * cir) / denominator.clamp_min(1e-12)
                lag_correlations.append(complex(correlation.item()))
            previous_cir = cir
            power = torch.abs(cir).square().cpu().numpy().astype(np.float64)
            cir_power_samples.append(power)
            effective_taps.append(_effective_taps_90(power))
            strong_mask = np.zeros(power.size, dtype=bool)
            strong_mask[list(initial_support)] = True
            diffuse_ratios.append(float(np.sum(power[~strong_mask])))

    empirical_power = np.mean(np.stack(cir_power_samples, axis=0), axis=0)
    empirical_pdp_db = _relative_power_db(empirical_power)
    delay_seconds = np.arange(max_delay + 1, dtype=np.float64) / float(
        fixed["sample_rate_hz"]
    )
    reference_pdp_db = _reference_midpoint_with_upper_hold(delay_seconds)
    envelope_rmse = float(
        np.sqrt(np.mean(np.square(empirical_pdp_db - reference_pdp_db)))
    )
    mean_lag = (
        sum(lag_correlations, 0.0j) / len(lag_correlations)
        if lag_correlations
        else 0.0j
    )
    statistic = {
        "candidate_id": str(candidate["candidate_id"]),
        "physical_delay_seconds": float(fixed["max_delay_seconds"]),
        "discrete_delay_samples": int(max_delay),
        "strong_path_count": float(np.mean(path_counts)),
        "strong_path_count_observed_range": [min(path_counts), max(path_counts)],
        "strong_path_support": supports,
        "support_changed_frames": int(support_changed_frames),
        "effective_taps_90": float(np.mean(effective_taps)),
        "diffuse_energy_ratio": float(np.mean(diffuse_ratios)),
        "rho_frame_target": float(config.rho_frame),
        "frame_lag_correlation": {
            "real": float(mean_lag.real),
            "imag": float(mean_lag.imag),
            "magnitude": float(abs(mean_lag)),
        },
        "envelope_rmse_db": envelope_rmse,
        "cross_frame_impulse_error": float(max(cross_errors, default=0.0)),
        "aggregation_role": str(candidate["aggregation_role"]),
        "uses_true_channel_for_statistics": True,
        "semantics": str(candidate["semantics"]),
    }
    plot_data = {
        "candidate_id": str(candidate["candidate_id"]),
        "delay_seconds": delay_seconds,
        "empirical_pdp_db": empirical_pdp_db,
        "reference_pdp_db": reference_pdp_db,
    }
    return statistic, plot_data


def _measure_traditional_ber(
    candidates: Sequence[Mapping[str, Any]],
    fixed: Mapping[str, Any],
    seeds: Sequence[int],
    frames: int,
    methods: Sequence[str],
    snrs: Sequence[float],
) -> list[dict]:
    totals: dict[tuple[str, str, float], dict[str, int]] = defaultdict(
        lambda: {"errors": 0, "bits": 0}
    )
    max_delay = physical_delay_samples(
        float(fixed["sample_rate_hz"]), float(fixed["max_delay_seconds"])
    )
    for candidate in candidates:
        cfo_limit = float(candidate["residual_cfo_limit"])
        for snr_db in snrs:
            for seed in seeds:
                environment = CommunicationEnvironment(
                    _environment_config(candidate, fixed, seed, snr_db)
                )
                start = environment.reset_episode()
                cir, _ = estimate_acquisition_cir_with_cfo(
                    start.acquisition,
                    max_delay,
                    cfo_limit=cfo_limit,
                )
                soft_tails = {
                    method: start.initial_soft_tail.clone() for method in methods
                }
                phase_states = {
                    method: TraditionalPhaseState() for method in methods
                }
                for _ in range(frames):
                    frame = environment.next_frame()
                    receiver_view = frame.receiver_view()
                    for method in methods:
                        result = run_traditional_equalizer(
                            method,
                            receiver_view,
                            cir,
                            soft_tails[method],
                            float(snr_db),
                            phase_state=phase_states[method],
                            residual_cfo_limit=cfo_limit,
                        )
                        soft_tails[method] = result.soft_tail.clone()
                        # Data 标签只在算法完成后用于离线 BER 计数。
                        data_logits = result.logits[frame.data_mask]
                        data_bits = frame.bits[frame.data_mask]
                        errors = int(
                            torch.sum((data_logits > 0) != data_bits).item()
                        )
                        key = (
                            str(candidate["candidate_id"]),
                            str(method),
                            float(snr_db),
                        )
                        totals[key]["errors"] += errors
                        totals[key]["bits"] += int(data_bits.numel())

    rows = []
    for (candidate_id, method, snr_db), counts in sorted(totals.items()):
        bits = int(counts["bits"])
        rows.append(
            {
                "candidate_id": candidate_id,
                "method": method,
                "snr_db": float(snr_db),
                "ber": float(counts["errors"] / bits) if bits else 0.0,
                "bit_errors": int(counts["errors"]),
                "bits": bits,
                "data_labels_used_for_adaptation": False,
                "uses_neural_network": False,
                "uses_rl": False,
            }
        )
    return rows


def _diagnostic_channel_config(
    candidate: Mapping[str, Any], fixed: Mapping[str, Any], seed: int
) -> ExtremeDelayChannelConfig:
    return ExtremeDelayChannelConfig(
        profile_name=str(fixed["profile_name"]),
        level=str(fixed["level"]),
        snr_db=100.0,
        seed=int(seed),
        impairment_profile=str(fixed["sanity_impairment_profile"]),
        sample_rate_hz=float(fixed["sample_rate_hz"]),
        symbol_rate_hz=float(fixed["symbol_rate_hz"]),
        frame_len=int(fixed["frame_len"]),
        max_delay_seconds=float(fixed["max_delay_seconds"]),
        coherence_time_seconds=float(candidate["coherence_time_seconds"]),
        strong_path_count=tuple(candidate["strong_path_count"]),
        diffuse_energy_ratio=tuple(candidate["diffuse_energy_ratio"]),
        include_anomalous_scatterer=bool(
            candidate["include_anomalous_scatterer"]
        ),
    )


def _environment_config(
    candidate: Mapping[str, Any],
    fixed: Mapping[str, Any],
    seed: int,
    snr_db: float,
) -> CommEnvConfig:
    return CommEnvConfig(
        profile_name=str(fixed["profile_name"]),
        level=str(fixed["level"]),
        snr_db=float(snr_db),
        seed=int(seed),
        impairment_profile=str(candidate["impairment_profile"]),
        sample_rate_hz=float(fixed["sample_rate_hz"]),
        symbol_rate_hz=float(fixed["symbol_rate_hz"]),
        frame_len=int(fixed["frame_len"]),
        max_delay_seconds=float(fixed["max_delay_seconds"]),
        coherence_time_seconds=float(candidate["coherence_time_seconds"]),
        strong_path_count=tuple(candidate["strong_path_count"]),
        diffuse_energy_ratio=tuple(candidate["diffuse_energy_ratio"]),
        include_anomalous_scatterer=bool(
            candidate["include_anomalous_scatterer"]
        ),
        total_pilot=int(fixed["pilot_total"]),
        layout=str(fixed["pilot_layout"]),
    )


def _reference_midpoint_with_upper_hold(delay_seconds: np.ndarray) -> np.ndarray:
    envelope = load_evans_1965_envelope()
    observed = envelope.observed_mask
    anchor_delays = np.concatenate(
        (envelope.delay_seconds[observed], envelope.delay_seconds[-1:])
    )
    observed_midpoints = (
        envelope.upper_power_db[observed] + envelope.lower_power_db[observed]
    ) / 2.0
    anchor_power_db = np.concatenate(
        (observed_midpoints, envelope.upper_power_db[-1:])
    )
    reference = np.interp(
        delay_seconds,
        anchor_delays,
        anchor_power_db,
        left=anchor_power_db[0],
        right=anchor_power_db[-1],
    )
    return reference - reference[0]


def _relative_power_db(power: np.ndarray) -> np.ndarray:
    normalized = power / max(float(power[0]), np.finfo(np.float64).tiny)
    return 10.0 * np.log10(np.maximum(normalized, np.finfo(np.float64).tiny))


def _effective_taps_90(power: np.ndarray) -> int:
    normalized = power / max(float(np.sum(power)), np.finfo(np.float64).tiny)
    cumulative = np.cumsum(np.sort(normalized)[::-1])
    return int(np.searchsorted(cumulative, 0.9, side="left") + 1)


def _write_pdp_plot(path: Path, series: Sequence[Mapping[str, Any]]) -> None:
    envelope = load_evans_1965_envelope()
    observed = envelope.observed_mask
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    observed_ms = envelope.delay_seconds[observed] * 1000.0
    axis.fill_between(
        observed_ms,
        envelope.lower_power_db[observed],
        envelope.upper_power_db[observed],
        color="0.85",
        label="Evans observed envelope",
    )
    for item in series:
        axis.plot(
            np.asarray(item["delay_seconds"]) * 1000.0,
            item["empirical_pdp_db"],
            marker="o",
            markersize=3,
            label=str(item["candidate_id"]),
        )
    axis.plot(
        [observed_ms[-1], envelope.delay_seconds[-1] * 1000.0],
        [envelope.upper_power_db[observed][-1], envelope.upper_power_db[-1]],
        linestyle="--",
        color="black",
        linewidth=1.2,
        label="Upper-hold support extension",
    )
    axis.set_xlabel("Delay from leading tap (ms)")
    axis.set_ylabel("Relative power (dB)")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def _validate_runtime_fixed(fixed: Mapping[str, Any]) -> dict:
    if not isinstance(fixed, Mapping):
        raise ValueError("fixed 必须是映射。")
    normalized = dict(fixed)
    _require_exact_keys(normalized, FIXED_FIELDS, "fixed")
    if normalized["profile_name"] != "eme_measurement_v1":
        raise ValueError("profile_name 必须是 eme_measurement_v1。")
    if normalized["level"] != "B":
        raise ValueError("level 必须是 B。")
    if float(normalized["max_delay_seconds"]) != 0.0116:
        raise ValueError("max_delay_seconds 必须是 0.0116。")
    if float(normalized["sample_rate_hz"]) != 2000.0:
        raise ValueError("sample_rate_hz 必须是 2000。")
    if float(normalized["symbol_rate_hz"]) != 2000.0:
        raise ValueError("symbol_rate_hz 必须是 2000。")
    if normalized["pilot_layout"] != "prefix":
        raise ValueError("pilot_layout 必须是 prefix。")
    if normalized["sanity_impairment_profile"] != "clean":
        raise ValueError("sanity_impairment_profile 必须是 clean。")
    frame_len = _positive_integer(normalized["frame_len"], "frame_len")
    pilot_total = _positive_integer(normalized["pilot_total"], "pilot_total")
    if frame_len <= pilot_total:
        raise ValueError("frame_len 必须大于 pilot_total。")
    normalized["frame_len"] = frame_len
    normalized["pilot_total"] = pilot_total
    normalized["main_snrs"] = _validate_snrs(normalized["main_snrs"])
    return normalized


def _validate_candidates(
    candidates: Sequence[Mapping[str, Any]], *, require_standard_count: bool
) -> list[dict]:
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise ValueError("candidates 必须是候选对象列表。")
    if require_standard_count and not 3 <= len(candidates) <= 4:
        raise ValueError("正式候选必须显式列出 3 至 4 个。")
    if not candidates:
        raise ValueError("candidates 不得为空。")
    normalized = []
    identifiers = set()
    max_paths = physical_delay_samples(2000.0, 0.0116) + 1
    for index, raw_candidate in enumerate(candidates):
        if not isinstance(raw_candidate, Mapping):
            raise ValueError(f"candidates[{index}] 必须是对象。")
        candidate = dict(raw_candidate)
        _require_exact_keys(candidate, CANDIDATE_FIELDS, f"candidates[{index}]")
        identifier = candidate["candidate_id"]
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("candidate_id 必须是非空字符串。")
        if identifier in identifiers:
            raise ValueError("candidate_id 必须唯一。")
        identifiers.add(identifier)
        strong_range = _integer_range(
            candidate["strong_path_count"], "strong_path_count"
        )
        if strong_range[0] < 2 or strong_range[1] > max_paths:
            raise ValueError("strong_path_count 超出物理支撑。")
        diffuse_range = _float_range(
            candidate["diffuse_energy_ratio"], "diffuse_energy_ratio"
        )
        if diffuse_range[0] < 0.0 or diffuse_range[1] >= 1.0:
            raise ValueError("diffuse_energy_ratio 必须位于 [0, 1) 内。")
        coherence = _finite_float(
            candidate["coherence_time_seconds"], "coherence_time_seconds"
        )
        if coherence <= 0.0:
            raise ValueError("coherence_time_seconds 必须为正。")
        if not isinstance(candidate["include_anomalous_scatterer"], bool):
            raise ValueError("include_anomalous_scatterer 必须为布尔值。")
        if candidate["impairment_profile"] not in {"clean", "cfo_phase_tiny"}:
            raise ValueError("候选 impairment_profile 仅允许 clean/cfo_phase_tiny。")
        cfo_limit = _finite_float(
            candidate["residual_cfo_limit"], "residual_cfo_limit"
        )
        if cfo_limit <= 0.0 or cfo_limit >= 0.5:
            raise ValueError("residual_cfo_limit 必须位于 (0, 0.5) 内。")
        if candidate["aggregation_role"] not in {"sanity", "main"}:
            raise ValueError("aggregation_role 仅允许 sanity/main。")
        if (
            candidate["impairment_profile"] == "clean"
            and candidate["aggregation_role"] != "sanity"
        ):
            raise ValueError("clean 候选不得进入 Level B 主平均。")
        if candidate["semantics"] != RANGE_SEMANTICS:
            raise ValueError("候选范围 semantics 不正确。")
        candidate["strong_path_count"] = list(strong_range)
        candidate["diffuse_energy_ratio"] = list(diffuse_range)
        candidate["coherence_time_seconds"] = coherence
        candidate["residual_cfo_limit"] = cfo_limit
        normalized.append(candidate)
    if require_standard_count:
        has_sanity = any(
            item["impairment_profile"] == "clean"
            and item["aggregation_role"] == "sanity"
            for item in normalized
        )
        has_main = any(
            item["impairment_profile"] == "cfo_phase_tiny"
            and item["aggregation_role"] == "main"
            for item in normalized
        )
        if not has_sanity or not has_main:
            raise ValueError("正式候选必须同时含 clean sanity 与 cfo_phase_tiny main。")
    return normalized


def _validate_methods(methods: Sequence[str]) -> list[str]:
    if isinstance(methods, (str, bytes)) or not isinstance(methods, Sequence):
        raise ValueError("traditional_methods 必须是方法名列表。")
    normalized = list(methods)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("traditional_methods 不得为空或重复。")
    unknown = [method for method in normalized if method not in TRADITIONAL_BASELINES]
    if unknown:
        raise ValueError(f"发现非传统 baseline：{unknown}")
    return normalized


def _validate_seeds(seeds: Sequence[int]) -> list[int]:
    if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence):
        raise ValueError("seeds 必须是整数列表。")
    normalized = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seeds 必须是整数列表。")
        normalized.append(int(seed))
    if not normalized:
        raise ValueError("seeds 不得为空。")
    return normalized


def _validate_snrs(snrs: Sequence[float]) -> list[float]:
    if isinstance(snrs, (str, bytes)) or not isinstance(snrs, Sequence):
        raise ValueError("snrs 必须是数值列表。")
    normalized = [_finite_float(value, "snr") for value in snrs]
    if not normalized:
        raise ValueError("snrs 不得为空。")
    return normalized


def _require_exact_keys(
    mapping: Mapping[str, Any], expected: set[str], field_name: str
) -> None:
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{field_name} 字段不匹配；缺失={missing}，额外={extra}。"
        )


def _integer_range(value: Any, field_name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} 必须是长度为 2 的整数范围。")
    if any(isinstance(item, bool) or not isinstance(item, (int, np.integer)) for item in value):
        raise ValueError(f"{field_name} 必须是长度为 2 的整数范围。")
    low, high = int(value[0]), int(value[1])
    if low > high:
        raise ValueError(f"{field_name} 必须满足 low <= high。")
    return low, high


def _float_range(value: Any, field_name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} 必须是长度为 2 的数值范围。")
    low = _finite_float(value[0], field_name)
    high = _finite_float(value[1], field_name)
    if low > high:
        raise ValueError(f"{field_name} 必须满足 low <= high。")
    return low, high


def _positive_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field_name} 必须是正整数。")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{field_name} 必须是正整数。")
    return normalized


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} 必须是有限数值。")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} 必须是有限数值。") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} 必须是有限数值。")
    return normalized


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校准 EME 测量约束信道候选。")
    parser.add_argument(
        "--config",
        default="configs/eme_measurement_channel_candidates.json",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--output-dir", default="logs/eme_measurement_channel")
    parser.add_argument("--snrs", nargs="+", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_candidate_config(args.config)
    payload = run_calibration(
        config["candidates"],
        seeds=args.seeds,
        frames=args.frames,
        output_dir=args.output_dir,
        fixed=config["fixed"],
        traditional_methods=config["traditional_methods"],
        snrs=args.snrs,
    )
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir)),
                "candidate_count": len(payload["statistics"]),
                "traditional_rows": len(payload["traditional_ber"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
