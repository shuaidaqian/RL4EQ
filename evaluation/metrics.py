# -*- coding: utf-8 -*-
"""统一评估指标、逐配置门槛与 Pilot 候选筛选。"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class FrameMetric:
    method: str
    level: str
    delay: int
    snr_db: float
    rho: float = 0.99
    pilot_total: int = 128
    pilot_layout: str = "multi_block"
    seed: int = 0
    frame: int = 0
    ber_data: float = 1.0
    ber_adapt_pilot: float = 1.0
    ber_reward_pilot: float = 1.0
    pilot_loss: float = math.nan
    cir_nmse_db: float = math.nan
    adapt_params: int = 0
    adapt_steps: int = 0
    detector_iterations: int = 0
    latency_ms: float = 0.0
    parameter_delta_norm: float = 0.0
    rollback: bool = False
    effective_goodput: float = 0.0

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ConfigSummary:
    level: str
    delay: int
    snr_db: float
    mean_ber_data: float
    count: int


@dataclass(frozen=True)
class MatrixSummary:
    per_config: list[ConfigSummary]
    generalization: dict

    def all_below(self, threshold: float) -> bool:
        return bool(self.per_config and all(item.mean_ber_data < threshold for item in self.per_config))


@dataclass(frozen=True)
class SpearmanResult:
    correlation: float
    n: int
    passed: bool


def bit_error_rate_from_logits(logits, bits) -> float:
    """根据 logits 与 0/1 bit 计算 BER。"""

    predictions = np.asarray(logits) >= 0
    target = np.asarray(bits).astype(bool)
    if target.size == 0:
        return 0.0
    return float(np.mean(predictions != target))


def effective_goodput(
    ber: float,
    data_symbols: int,
    ordinary_frames: int,
    warmup_symbols: int,
    acquisition_symbols: int,
    frame_symbols: int,
) -> float:
    """计算含 warm-up/acquisition 开销的有效吞吐。"""

    denominator = warmup_symbols + acquisition_symbols + frame_symbols * ordinary_frames
    if denominator <= 0:
        raise ValueError("总符号数必须为正。")
    return float(data_symbols * ordinary_frames * max(0.0, 1.0 - ber) / denominator)


def summarize_main_matrix(rows: Iterable[FrameMetric | dict]) -> MatrixSummary:
    """只把 Level B 主配置纳入主平均，Level C/未见配置单列 generalization。"""

    normalized = [_coerce_metric(row) for row in rows]
    main: dict[tuple[int, float], list[float]] = {}
    generalization_values: list[float] = []
    for row in normalized:
        if row.level == "B" and row.delay in {20, 30, 40} and float(row.snr_db) in {10.0, 15.0, 20.0}:
            main.setdefault((row.delay, float(row.snr_db)), []).append(float(row.ber_data))
        elif row.level == "C" or row.delay not in {20, 30, 40}:
            generalization_values.append(float(row.ber_data))
    per_config = [
        ConfigSummary("B", delay, snr_db, float(np.mean(values)), len(values))
        for (delay, snr_db), values in sorted(main.items())
    ]
    generalization = {
        "count": len(generalization_values),
        "mean_ber_data": float(np.mean(generalization_values)) if generalization_values else math.nan,
    }
    return MatrixSummary(per_config=per_config, generalization=generalization)


def spearman_reward_data(reward_improvements: Sequence[float], data_improvements: Sequence[float], threshold: float = 0.6) -> SpearmanResult:
    """计算 Reward loss 改善与 Data BER 改善的秩相关。"""

    if len(reward_improvements) != len(data_improvements):
        raise ValueError("Reward/Data 样本数量必须一致。")
    n = len(reward_improvements)
    if n < 2:
        return SpearmanResult(correlation=math.nan, n=n, passed=False)
    reward_rank = _rank(np.asarray(reward_improvements, dtype=float))
    data_rank = _rank(np.asarray(data_improvements, dtype=float))
    if np.std(reward_rank) == 0 or np.std(data_rank) == 0:
        corr = math.nan
    else:
        corr = float(np.corrcoef(reward_rank, data_rank)[0, 1])
    return SpearmanResult(correlation=corr, n=n, passed=bool(corr >= threshold))


def select_pilot_shortlist(candidates: list[dict], output_dir: str | Path, keep: int = 3) -> dict:
    """按门槛、相关性、goodput、最坏 seed 选择 2–3 个 Pilot 候选。"""

    if keep < 2 or keep > 3:
        raise ValueError("Pilot shortlist 只允许保留 2–3 个候选。")
    annotated = []
    eligible = []
    for candidate in candidates:
        item = dict(candidate)
        reasons = []
        if float(item.get("max_ber", 1.0)) >= 0.01:
            reasons.append("max_ber")
        if float(item.get("spearman", -1.0)) < 0.6:
            reasons.append("spearman")
        item["淘汰原因"] = reasons
        item["selected"] = False
        annotated.append(item)
        if not reasons:
            eligible.append(item)
    eligible.sort(
        key=lambda item: (
            -float(item.get("effective_goodput", 0.0)),
            float(item.get("worst_seed", 1.0)),
            int(item.get("pilot_total", 10_000)),
        )
    )
    shortlist = eligible[:keep]
    for item in shortlist:
        item["selected"] = True
    payload = {"all_candidates": annotated, "shortlist": shortlist}
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "shortlist.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _coerce_metric(row: FrameMetric | dict) -> FrameMetric:
    if isinstance(row, FrameMetric):
        return row
    return FrameMetric(**row)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    unique_values, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique_values
    for group_index, count in enumerate(counts):
        if count > 1:
            ranks[inverse == group_index] = float(np.mean(ranks[inverse == group_index]))
    return ranks
