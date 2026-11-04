# -*- coding: utf-8 -*-
"""仅从合法验证Pilot统计估计神经更新的离线Data风险。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


FEATURE_NAMES = (
    "验证BCE均值改善", "验证BCE改善中位数", "验证BCE改善下十分位",
    "验证符号改善比例", "验证硬BER改善", "更新前平均绝对logit",
    "更新后平均绝对logit", "符号翻转比例", "平均绝对logit变化",
    "参数候选增量范数", "Adapt梯度范数",
)


def validation_risk_features(before_logits, after_logits, observation,
                             coefficient_delta_norm, adapt_gradient_norm):
    """返回仅依赖更新前后合法验证Pilot和Adapt优化状态的特征。"""
    mask = observation.validation_mask
    labels = observation.validation_labels[mask].float()
    before = before_logits[mask].float()
    after = after_logits[mask].float()
    before_bce = F.binary_cross_entropy_with_logits(before, labels, reduction="none")
    after_bce = F.binary_cross_entropy_with_logits(after, labels, reduction="none")
    gains = before_bce - after_bce
    before_error = ((before >= 0) != labels.bool()).float()
    after_error = ((after >= 0) != labels.bool()).float()
    return torch.stack((
        gains.mean(), torch.quantile(gains, .5), torch.quantile(gains, .1),
        (gains > 0).float().mean(), (before_error - after_error).mean(),
        before.abs().mean(), after.abs().mean(), ((before >= 0) != (after >= 0)).float().mean(),
        (after - before).abs().mean(),
        torch.as_tensor(coefficient_delta_norm, device=before.device, dtype=before.dtype),
        torch.as_tensor(adapt_gradient_norm, device=before.device, dtype=before.dtype),
    ))


@dataclass(frozen=True)
class PilotRiskRegressor:
    """训练后固定的线性岭回归器；输出预测Data BCE改善，正数代表获益。"""

    mean: torch.Tensor
    scale: torch.Tensor
    weight: torch.Tensor
    bias: float
    ridge: float

    @classmethod
    def fit(cls, features, targets, ridge=0.1):
        features = torch.as_tensor(features, dtype=torch.float64, device="cpu")
        targets = torch.as_tensor(targets, dtype=torch.float64, device="cpu").flatten()
        if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
            raise ValueError("验证Pilot风险模型输入特征维数不符。")
        if features.shape[0] != targets.numel() or features.shape[0] < 2:
            raise ValueError("风险模型训练样本不足或标签数量不匹配。")
        if not bool(torch.isfinite(features).all() and torch.isfinite(targets).all()):
            raise FloatingPointError("风险模型训练数据包含非有限值。")
        if ridge <= 0:
            raise ValueError("岭回归正则必须为正数。")
        mean = features.mean(dim=0)
        scale = features.std(dim=0, unbiased=False).clamp_min(1e-8)
        normalized = (features - mean) / scale
        design = torch.cat((normalized, torch.ones((features.shape[0], 1), dtype=features.dtype)), dim=1)
        penalty = torch.eye(design.shape[1], dtype=design.dtype) * ridge
        penalty[-1, -1] = 0.
        solution = torch.linalg.solve(design.T @ design + penalty, design.T @ targets)
        return cls(mean.float(), scale.float(), solution[:-1].float(), float(solution[-1]), float(ridge))

    def predict(self, features):
        features = torch.as_tensor(features, dtype=self.mean.dtype, device=self.mean.device)
        return ((features - self.mean) / self.scale) @ self.weight + self.bias

    def to_dict(self):
        return dict(feature_names=list(FEATURE_NAMES), mean=self.mean.cpu().tolist(),
            scale=self.scale.cpu().tolist(), weight=self.weight.cpu().tolist(),
            bias=self.bias, ridge=self.ridge, target="Frozen与候选动作的Data BCE差；正值代表改善",
            decision_rule="仅当既有验证Pilot硬守门接受且预测Data BCE改善大于0时保留")

    @classmethod
    def from_dict(cls, value, device="cpu"):
        if value.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("风险模型特征定义与当前代码不一致。")
        return cls(torch.tensor(value["mean"], dtype=torch.float32, device=device),
            torch.tensor(value["scale"], dtype=torch.float32, device=device),
            torch.tensor(value["weight"], dtype=torch.float32, device=device),
            float(value["bias"]), float(value["ridge"]))

    def __call__(self, features):
        return self.predict(features)
