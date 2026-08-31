# -*- coding: utf-8 -*-
"""把实验 JSON 严格适配为实际通信环境配置。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from env.comm_env import CommEnvConfig
from env.eme_reference import physical_delay_samples
from env.online_state import load_online_state_split


_EME_PROFILE = "eme_measurement_v1"
_EME_LONG_MEMORY_PROFILE = "eme_long_memory_v2"
_EME_REQUIRED_FIELDS = (
    "profile_name",
    "sample_rate_hz",
    "symbol_rate_hz",
    "frame_len",
    "max_delay_seconds",
    "coherence_time_seconds",
    "strong_path_count",
    "diffuse_energy_ratio",
    "include_anomalous_scatterer",
)


def _required(experiment: Mapping[str, Any], field: str) -> Any:
    if field not in experiment:
        raise ValueError(f"EME 冻结配置缺少字段：{field}")
    return experiment[field]


def _pair(value: Any, field: str, cast: type) -> tuple[Any, Any]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field} 必须是长度为 2 的数组。")
    try:
        return cast(value[0]), cast(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 包含非法数值。") from exc


def _validate_eme_dimensions(experiment: Mapping[str, Any], physical_delay: int) -> None:
    configured_delay = int(_required(experiment, "max_delay"))
    if configured_delay != physical_delay:
        raise ValueError(
            f"max_delay 必须由物理时标得到 {physical_delay}，配置值为 {configured_delay}。"
        )
    main_delays = experiment.get("main_delays")
    if main_delays != [physical_delay]:
        raise ValueError(f"main_delays 必须严格等于 [{physical_delay}]。")
    model = experiment.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("EME 冻结配置缺少 model。")
    model_delay = int(model.get("max_delay", -1))
    if model_delay != physical_delay:
        raise ValueError(
            f"model.max_delay 配置值 {model_delay} 与物理时延 {physical_delay} 冲突。"
        )
    frame_len = int(_required(experiment, "frame_len"))
    model_frame_len = int(model.get("frame_len", -1))
    if model_frame_len != frame_len:
        raise ValueError(
            f"model.frame_len 配置值 {model_frame_len} 与信道 frame_len {frame_len} 冲突。"
        )


def _build_eme_config(
    experiment: Mapping[str, Any],
    *,
    level: str,
    snr_db: float,
    seed: int,
    max_delay: int | None,
    total_pilot: int | None,
    pilot_layout: str | None,
    impairment_profile: str | None,
    state_split: str | None,
) -> CommEnvConfig:
    for field in _EME_REQUIRED_FIELDS:
        _required(experiment, field)
    if experiment.get("eme_physical_fields_passthrough") != "implemented":
        raise ValueError("eme_physical_fields_passthrough 必须为 implemented，禁止回落 legacy 信道。")
    profile_name = str(experiment["profile_name"])
    if profile_name not in {_EME_PROFILE, _EME_LONG_MEMORY_PROFILE}:
        raise ValueError(f"profile_name 必须为 {_EME_PROFILE} 或 {_EME_LONG_MEMORY_PROFILE}。")
    if str(experiment.get("channel_profile", "")) != profile_name:
        raise ValueError("channel_profile 必须与 profile_name 一致。")

    frozen_level = str(experiment.get("level", experiment.get("main_level", "B")))
    if str(level) != frozen_level:
        raise ValueError(f"level 必须使用冻结主场景 {frozen_level}，收到 {level}。")
    allowed_snrs = [float(value) for value in _required(experiment, "main_snrs")]
    if float(snr_db) not in allowed_snrs:
        raise ValueError(f"snr_db={float(snr_db):g} 不在冻结主矩阵 {allowed_snrs} 中。")

    sample_rate_hz = float(experiment["sample_rate_hz"])
    symbol_rate_hz = float(experiment["symbol_rate_hz"])
    max_delay_seconds = float(experiment["max_delay_seconds"])
    physical_delay = physical_delay_samples(sample_rate_hz, max_delay_seconds)
    _validate_eme_dimensions(experiment, physical_delay)
    if max_delay is not None and int(max_delay) != physical_delay:
        raise ValueError(
            f"max_delay 冻结值为 {physical_delay}，CLI/调用值为 {int(max_delay)}，禁止覆盖。"
        )

    layout = str(pilot_layout if pilot_layout is not None else experiment.get("pilot_layout", "prefix"))
    if layout != "prefix":
            raise ValueError(f"{profile_name} 只允许 prefix Pilot layout。")
    selected_pilot = int(
        total_pilot if total_pilot is not None else experiment.get("pilot_total", 128)
    )
    allowed_pilots = [
        int(value)
        for value in experiment.get(
            "pilot_sweep_totals",
            [int(experiment.get("pilot_total", 128))],
        )
    ]
    if selected_pilot not in allowed_pilots:
        raise ValueError(
            f"total_pilot={selected_pilot} 不在冻结候选 {allowed_pilots} 中。"
        )
    frozen_impairment = str(_required(experiment, "impairment_profile"))
    selected_impairment = str(
        impairment_profile if impairment_profile is not None else frozen_impairment
    )
    if selected_impairment != frozen_impairment:
        raise ValueError(
            "impairment_profile 冻结值为 "
            f"{frozen_impairment}，CLI/调用值为 {selected_impairment}，禁止覆盖。"
        )

    frame_len = int(experiment["frame_len"])
    coherence_time_seconds = float(experiment["coherence_time_seconds"])
    acquisition_to_data_gap_seconds = float(
        experiment.get("acquisition_to_data_gap_seconds", 0.0)
    )
    expected_rho = math.exp(-(frame_len / sample_rate_hz) / coherence_time_seconds)
    configured_rho = float(_required(experiment, "rho_frame"))
    alias_rho = float(_required(experiment, "rho"))
    if not math.isclose(configured_rho, expected_rho, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"rho_frame={configured_rho} 与物理时标计算值 {expected_rho} 不一致。"
        )
    if not math.isclose(alias_rho, configured_rho, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("rho 必须与 rho_frame 完全一致。")

    selected_split = None
    state_ranges = None
    if state_split is not None:
        selected_split = load_online_state_split(experiment, state_split)
        strong_path_count = selected_split.strong_path_count
        diffuse_energy_ratio = selected_split.diffuse_energy_ratio
        state_ranges = {
            "cfo_abs_range": selected_split.cfo_abs_range,
            "phase_noise_std_range": selected_split.phase_noise_std_range,
        }
    else:
        strong_path_count = _pair(experiment["strong_path_count"], "strong_path_count", int)
        diffuse_energy_ratio = _pair(experiment["diffuse_energy_ratio"], "diffuse_energy_ratio", float)
    anomalous = experiment["include_anomalous_scatterer"]
    if not isinstance(anomalous, bool):
        raise ValueError("include_anomalous_scatterer 必须是布尔值。")

    return CommEnvConfig(
        level=frozen_level,
        max_delay=physical_delay,
        snr_db=float(snr_db),
        rho=configured_rho,
        total_pilot=selected_pilot,
        layout=layout,
        seed=int(seed),
        impairment_profile=selected_impairment,
        profile_name=profile_name,
        sample_rate_hz=sample_rate_hz,
        symbol_rate_hz=symbol_rate_hz,
        frame_len=frame_len,
        max_delay_seconds=max_delay_seconds,
        coherence_time_seconds=coherence_time_seconds,
        acquisition_to_data_gap_seconds=acquisition_to_data_gap_seconds,
        strong_path_count=strong_path_count,
        diffuse_energy_ratio=diffuse_energy_ratio,
        include_anomalous_scatterer=anomalous,
        state_split=None if selected_split is None else selected_split.name,
        state_ranges=state_ranges,
    )


def build_comm_env_config(
    experiment: Mapping[str, Any],
    *,
    level: str,
    snr_db: float,
    seed: int,
    max_delay: int | None = None,
    total_pilot: int | None = None,
    pilot_layout: str | None = None,
    impairment_profile: str | None = None,
    state_split: str | None = None,
) -> CommEnvConfig:
    """构造实际环境配置；EME 配置不允许静默回落 legacy。"""

    channel_profile = experiment.get("channel_profile")
    profile_name = experiment.get("profile_name")
    if channel_profile is not None and profile_name is not None and str(channel_profile) != str(profile_name):
        raise ValueError(
            f"channel_profile={channel_profile} 与 profile_name={profile_name} 不一致。"
        )
    profile = str(channel_profile if channel_profile is not None else profile_name or "legacy_sparse_v1")
    if profile in {_EME_PROFILE, _EME_LONG_MEMORY_PROFILE}:
        return _build_eme_config(
            experiment,
            level=level,
            snr_db=snr_db,
            seed=seed,
            max_delay=max_delay,
            total_pilot=total_pilot,
            pilot_layout=pilot_layout,
            impairment_profile=impairment_profile,
            state_split=state_split,
        )
    if profile != "legacy_sparse_v1":
        raise ValueError(f"信道 profile={profile} 不支持。")
    return CommEnvConfig(
        level=str(level),
        max_delay=int(max_delay if max_delay is not None else experiment.get("max_delay", 40)),
        snr_db=float(snr_db),
        rho=float(experiment.get("rho", 0.99)),
        total_pilot=int(total_pilot if total_pilot is not None else experiment.get("pilot_total", 128)),
        layout=str(pilot_layout if pilot_layout is not None else experiment.get("pilot_layout", "prefix")),
        seed=int(seed),
        impairment_profile=str(
            impairment_profile
            if impairment_profile is not None
            else experiment.get("impairment_profile", "clean")
        ),
        frame_len=int(experiment.get("frame_len", experiment.get("model", {}).get("frame_len", 512))),
        state_split=state_split,
    )


def effective_channel_metadata(config: CommEnvConfig) -> dict[str, Any]:
    """从已构造配置导出可审计的实际信道元数据。"""

    return {
        "profile_name": str(config.profile_name),
        "level": str(config.level),
        "max_delay": int(config.max_delay),
        "sample_rate_hz": None if config.sample_rate_hz is None else float(config.sample_rate_hz),
        "symbol_rate_hz": None if config.symbol_rate_hz is None else float(config.symbol_rate_hz),
        "frame_len": int(config.frame_len),
        "max_delay_seconds": float(config.max_delay_seconds),
        "coherence_time_seconds": (
            None if config.coherence_time_seconds is None else float(config.coherence_time_seconds)
        ),
        "acquisition_to_data_gap_seconds": float(config.acquisition_to_data_gap_seconds),
        "rho_frame": float(config.rho),
        "strong_path_count": (
            None if config.strong_path_count is None else list(config.strong_path_count)
        ),
        "diffuse_energy_ratio": (
            None if config.diffuse_energy_ratio is None else list(config.diffuse_energy_ratio)
        ),
        "include_anomalous_scatterer": bool(config.include_anomalous_scatterer),
        "impairment_profile": str(config.impairment_profile),
        "pilot_layout": str(config.layout),
        "pilot_total": int(config.total_pilot),
        "state_split": config.state_split,
    }


def validate_model_dimensions(
    model_config: Mapping[str, Any] | Any,
    env_config: CommEnvConfig,
) -> None:
    """校验实际加载的模型维度与已经构造的物理环境一致。"""

    if isinstance(model_config, Mapping):
        model_delay = int(model_config.get("max_delay", -1))
        model_frame_len = int(model_config.get("frame_len", -1))
    else:
        model_delay = int(getattr(model_config, "max_delay", -1))
        model_frame_len = int(getattr(model_config, "frame_len", -1))
    if model_delay != int(env_config.max_delay):
        raise ValueError(
            f"实际模型 max_delay={model_delay} 与信道 max_delay={int(env_config.max_delay)} 冲突。"
        )
    if model_frame_len != int(env_config.frame_len):
        raise ValueError(
            "实际模型 frame_len="
            f"{model_frame_len} 与信道 frame_len={int(env_config.frame_len)} 冲突。"
        )
    if env_config.profile_name in {_EME_PROFILE, _EME_LONG_MEMORY_PROFILE}:
        phase_segments = int(
            model_config.get("phase_correction_segments", 4)
            if isinstance(model_config, Mapping)
            else getattr(model_config, "phase_correction_segments", 4)
        )
        if phase_segments != 4:
            raise ValueError(
                "EME phase conditioner 必须使用 4 个 block，"
                f"以匹配 phase residual 特征生成器；收到 {phase_segments}。"
            )
