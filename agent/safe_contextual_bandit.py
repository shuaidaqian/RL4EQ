# -*- coding: utf-8 -*-
"""只调度在线 PEFT 更新的保守上下文 Bandit。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class SafeUpdateAction:
    """一个不直接生成网络参数的离散安全动作。"""

    name: str
    groups: frozenset[str]
    learning_rate_scale: float
    max_delta_scale: float
    hold_frames: int
    min_confidence: float


def _default_actions() -> tuple[SafeUpdateAction, ...]:
    return (
        SafeUpdateAction("skip", frozenset(), 0.0, 0.0, 1, 0.0),
        SafeUpdateAction("phase_weak", frozenset({"phase"}), 0.25, 0.35, 1, 0.05),
        SafeUpdateAction("head_weak", frozenset({"head"}), 0.25, 0.5, 2, 0.25),
        SafeUpdateAction("head_nominal", frozenset({"head"}), 1.0, 1.0, 2, 0.40),
        SafeUpdateAction("film_nominal", frozenset({"conditioner_film"}), 1.0, 0.75, 2, 0.40),
        SafeUpdateAction("joint_nominal", frozenset({"head", "conditioner_film", "phase"}), 0.75, 0.75, 4, 0.60),
    )


class SafeContextualBandit:
    """在小动作空间中进行保守的 Pilot reward 调度。"""

    DEFAULT_FEATURES = (
        "adapt_loss",
        "pilot_confidence",
        "residual_cfo",
        "phase_slope",
        "cir_drift",
        "snr_db",
        "reward_trend",
        "rollback_rate",
        "consecutive_rejections",
        "parameter_delta_norm",
    )

    def __init__(
        self,
        actions: tuple[SafeUpdateAction, ...] | None = None,
        feature_names: tuple[str, ...] = DEFAULT_FEATURES,
        exploration: float = 0.15,
        ridge: float = 1.0,
        seed: int = 0,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("Bandit seed 必须为非负整数。")
        if not feature_names:
            raise ValueError("Bandit feature_names 不能为空。")
        if exploration < 0.0 or not math.isfinite(float(exploration)):
            raise ValueError("Bandit exploration 必须为非负有限数。")
        if ridge <= 0.0 or not math.isfinite(float(ridge)):
            raise ValueError("Bandit ridge 必须为正有限数。")
        self.actions = tuple(actions or _default_actions())
        if not self.actions or len({action.name for action in self.actions}) != len(self.actions):
            raise ValueError("Bandit actions 必须非空且名称唯一。")
        self.feature_names = tuple(str(name) for name in feature_names)
        self.exploration = float(exploration)
        self.ridge = float(ridge)
        self._rng = np.random.default_rng(int(seed))
        dimension = len(self.feature_names) + 1
        self._matrices = {
            action.name: np.eye(dimension, dtype=np.float64) * self.ridge
            for action in self.actions
        }
        self._vectors = {
            action.name: np.zeros(dimension, dtype=np.float64)
            for action in self.actions
        }
        self._counts = {action.name: 0 for action in self.actions}
        self._reward_sums = {action.name: 0.0 for action in self.actions}
        self._accepted_counts = {action.name: 0 for action in self.actions}
        self.data_labels_used_online = False

    def action(self, name: str) -> SafeUpdateAction:
        """按名称返回动作。"""

        for action in self.actions:
            if action.name == name:
                return action
        raise ValueError(f"未知 Bandit 动作：{name}")

    def select(
        self,
        context: Mapping[str, float],
        allowed_names: set[str] | None = None,
    ) -> SafeUpdateAction:
        """根据 Pilot/历史统计选择一个经过安全盾牌筛选的动作。"""

        confidence = float(context.get("pilot_confidence", 0.0))
        if not math.isfinite(confidence):
            raise ValueError("pilot_confidence 必须为有限数。")
        confidence = max(0.0, min(1.0, confidence))
        candidates = [
            action
            for action in self.actions
            if action.min_confidence <= confidence
            and (allowed_names is None or action.name in allowed_names)
        ]
        if not candidates:
            candidates = [self.action("skip")]
        features = self._features(context)
        scores: list[float] = []
        for action in candidates:
            name = action.name
            count = self._counts[name]
            if count == 0:
                score = self._prior(action)
            else:
                matrix = self._matrices[name]
                vector = self._vectors[name]
                try:
                    inverse = np.linalg.solve(matrix, np.eye(matrix.shape[0]))
                except np.linalg.LinAlgError:
                    inverse = np.linalg.pinv(matrix)
                mean = float(features @ np.linalg.solve(matrix, vector))
                uncertainty = float(np.sqrt(max(0.0, features @ inverse @ features)))
                score = mean + self.exploration * uncertainty
            scores.append(score)
        best_index = int(np.argmax(np.asarray(scores, dtype=np.float64)))
        return candidates[best_index]

    def update(
        self,
        action_name: str,
        context: Mapping[str, float],
        reward: float,
        accepted: bool,
    ) -> None:
        """使用动作后的 Reward Pilot 反馈更新统计。"""

        self.action(action_name)
        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError("Bandit reward 必须为有限数。")
        features = self._features(context)
        self._matrices[action_name] += np.outer(features, features)
        self._vectors[action_name] += features * reward
        self._counts[action_name] += 1
        self._reward_sums[action_name] += reward
        self._accepted_counts[action_name] += int(bool(accepted))

    def statistics(self, action_name: str) -> dict[str, float]:
        """返回动作统计，用于实验审计。"""

        self.action(action_name)
        count = self._counts[action_name]
        return {
            "count": float(count),
            "reward_sum": float(self._reward_sums[action_name]),
            "accepted_count": float(self._accepted_counts[action_name]),
            "accepted_rate": float(self._accepted_counts[action_name] / max(1, count)),
        }

    def _features(self, context: Mapping[str, float]) -> np.ndarray:
        values = [1.0]
        for name in self.feature_names:
            value = float(context.get(name, 0.0))
            if not math.isfinite(value):
                raise ValueError(f"Bandit context[{name}] 必须为有限数。")
            values.append(self._normalize_feature(name, value))
        return np.asarray(values, dtype=np.float64)

    @staticmethod
    def _normalize_feature(name: str, value: float) -> float:
        """把不同物理量压到相近数值范围，避免线性模型被 SNR 主导。"""

        scales = {
            "adapt_loss": 1.0,
            "pilot_confidence": 1.0,
            "residual_cfo": 0.001,
            "phase_slope": 0.1,
            "cir_drift": 0.1,
            "snr_db": 20.0,
            "reward_trend": 1.0,
            "rollback_rate": 1.0,
            "consecutive_rejections": 4.0,
            "parameter_delta_norm": 1.0,
        }
        normalized = value / scales.get(name, 1.0)
        return float(max(-5.0, min(5.0, normalized)))

    @staticmethod
    def _prior(action: SafeUpdateAction) -> float:
        priors = {
            "skip": 0.0,
            "phase_weak": 0.02,
            "head_weak": 0.01,
            "head_nominal": 0.005,
            "film_nominal": 0.004,
            "joint_nominal": -0.01,
        }
        return priors.get(action.name, 0.0)
