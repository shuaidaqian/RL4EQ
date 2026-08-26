# -*- coding: utf-8 -*-
"""与 Adapt Pilot 信息边界一致的传统 CIR 跟踪器。"""

from __future__ import annotations

from agent.cir_estimator import CIRCondition, HybridCIREstimator


class SparseLSTracker(HybridCIREstimator):
    """稀疏 LS 初始化 tracker。"""


class KalmanCIRTracker(HybridCIREstimator):
    """Kalman 风格接口，第一版复用相同可见输入边界。"""


class RLSCIRTracker(HybridCIREstimator):
    """RLS 风格接口，第一版复用相同可见输入边界。"""


__all__ = ["CIRCondition", "SparseLSTracker", "KalmanCIRTracker", "RLSCIRTracker"]
