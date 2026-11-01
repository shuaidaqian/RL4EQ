# -*- coding: utf-8 -*-
"""把离线冻结的传统校准设置接入统一比较，不读取神经结果选参数。"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from baseline.linear_mmse import zero_padded_wiener_fde
from baseline.synchronization_compensation import apply_phase_correction
from baseline.traditional_equalizers import (
    TraditionalResult, _dfe_rls, _estimate_to_logits, _lmmse_fir, _tail_from_estimate,
    estimate_phase_residual_vector,
)
from env.experiment_config import build_comm_env_config


CALIBRATED_METHODS = {
    "LMMSE-FIR": "fir", "DFE-RLS": "dfe", "SC-FDE-MMSE": "padded_fde",
}


@dataclass(frozen=True)
class FrozenTraditionalCalibration:
    """保留原始校准来源，按实际环境而非神经结构校验适用范围。"""

    source_config: dict
    selection: dict
    sha256: str
    path: str
    schema_version: str = "traditional-calibration-protocol-v1"

    def validate_environment(self, environment):
        expected = build_comm_env_config(
            self.source_config, level="B", snr_db=environment.snr_db,
            seed=environment.seed, max_delay=int(self.source_config["max_delay"]),
            total_pilot=int(self.source_config["pilot_total"]),
            reward_pilot_total=int(self.source_config["reward_pilot_total"]),
            pilot_layout="prefix", state_split=environment.state_split,
        )
        if asdict(expected) != asdict(environment):
            raise ValueError("传统校准协议与实际信道/Pilot 预算不匹配，不得静默复用。")
        if environment.seed in self.selection["calibration_seeds"]:
            raise ValueError("统一比较的实际信道 seed 与传统校准 seed 重叠。")

    def settings_for(self, method, snr_db):
        if method not in CALIBRATED_METHODS:
            raise ValueError("该方法没有冻结的传统校准设置。")
        key = format(float(snr_db), "g")
        entries = self.selection["by_snr"].get(key, [])
        selected = [tuple(item) for item in entries if item[0] == CALIBRATED_METHODS[method]]
        if len(selected) != 1:
            raise ValueError("每个 SNR/传统方法必须恰有一项冻结设置。")
        return selected[0]


def load_frozen_traditional_calibration(path):
    source = Path(path)
    content = source.read_bytes()
    payload = json.loads(content)
    if payload.get("schema_version") != "traditional-calibration-protocol-v1":
        raise ValueError("不支持的传统校准协议版本。")
    if payload.get("method_mapping") != CALIBRATED_METHODS:
        raise ValueError("传统校准方法映射与已实现的冻结协议不一致。")
    config, selection = payload["source_config"], payload["selection"]
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
    if signature != selection.get("config_sha256"):
        raise ValueError("传统校准的来源配置 SHA256 不匹配。")
    if selection.get("schema_version") != "linear-baseline-frozen-selection-v1":
        raise ValueError("传统校准来源选择的版本不匹配。")
    calibration = FrozenTraditionalCalibration(config, selection, hashlib.sha256(content).hexdigest(), str(source.resolve()))
    for snr in config["main_snrs"]:
        for method in CALIBRATED_METHODS:
            kind, length, regularizer = calibration.settings_for(method, snr)
            if kind == "fir" and (length not in {41, 117, 233} or regularizer not in {1, 16, 224}):
                raise ValueError("FIR 设置超出原校准网格。")
            if kind == "dfe" and (length, regularizer) not in {(41, 5), (117, 116)}:
                raise ValueError("DFE 设置超出原校准网格。")
            if kind == "padded_fde" and (length, regularizer) != (0, 0):
                raise ValueError("零填充 FDE 不包含额外可调参数。")
    return calibration


def run_calibrated_traditional_equalizer(method, receiver_view, cir, soft_tail, snr_db,
                                         *, acquisition_cfo, calibration):
    """使用冻结设置、acquisition CFO 和当前 Adapt 相位，保留各自历史递推。"""
    if not math.isfinite(float(acquisition_cfo)):
        raise ValueError("传统同步需要有限的 acquisition CFO 估计。")
    kind, length, regularizer = calibration.settings_for(method, snr_db)
    phase = estimate_phase_residual_vector(receiver_view, cir, soft_tail, cfo_hint=acquisition_cfo)
    corrected = apply_phase_correction(receiver_view.rx_symbols, float(phase[0]), acquisition_cfo)
    variance = torch.tensor(10 ** (-float(snr_db) / 10), device=corrected.device, dtype=torch.float32)
    if kind == "fir":
        estimate, iterations = _lmmse_fir(corrected, receiver_view.adapt_mask,
                                         receiver_view.adapt_symbols, length, variance * regularizer)
    elif kind == "dfe":
        estimate, iterations = _dfe_rls(corrected, receiver_view.adapt_mask,
                                       receiver_view.adapt_symbols, length, feedback_len=regularizer)
    else:
        estimate = zero_padded_wiener_fde(corrected, cir, soft_tail, variance)
        iterations = corrected.numel()
    return TraditionalResult(
        method=method, logits=_estimate_to_logits(estimate, variance),
        soft_tail=_tail_from_estimate(estimate, soft_tail.numel()), iterations=iterations,
        extra={
            "traditional": True, "uses_neural_network": False, "uses_rl": False,
            "uses_reward_or_data_labels": False, "shared_placeholder_kernel": False,
            "algorithm": f"frozen_pilot_cfo_{kind}",
            "traditional_calibration_protocol": calibration.schema_version,
            "traditional_calibration_applied": True,
            "traditional_calibration_sha256": calibration.sha256,
            "phase_compensation": "acquisition_cfo_prior+adapt_phase_median",
            "estimated_cfo_cycles_per_symbol": float(acquisition_cfo),
            "estimated_phase0": float(phase[0]), "filter_length": length,
            "ridge_multiplier": regularizer if kind == "fir" else None,
            "feedback_length": regularizer if kind == "dfe" else None,
            "fde_boundary": "zero_padded_wiener_approximation" if kind == "padded_fde" else None,
        },
    )
