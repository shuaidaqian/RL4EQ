# -*- coding: utf-8 -*-
"""传统 LMMSE-FIR 与 DFE 的标签隔离占位实现。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LegacyResult:
    method: str
    input_hash: str
    ber_data: float
    latency_ms: float


def legacy_lmmse_fir(observable_frame) -> LegacyResult:
    return LegacyResult("Legacy LMMSE-FIR", observable_frame.observable_hash, 0.08, 0.1)


def legacy_dfe(observable_frame) -> LegacyResult:
    return LegacyResult("Legacy DFE", observable_frame.observable_hash, 0.09, 0.1)
