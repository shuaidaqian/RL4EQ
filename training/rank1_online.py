# -*- coding: utf-8 -*-
"""固定 rank-1 子空间上的 Adapt Pilot 阻尼 Gauss–Newton 候选。"""

from __future__ import annotations

from collections.abc import Callable
import math
from typing import Any

import torch


def damped_rank1_gauss_newton_candidate(
    *,
    parameters: torch.Tensor,
    basis: torch.Tensor,
    adapt_logits_fn: Callable[[torch.Tensor], torch.Tensor],
    adapt_labels: torch.Tensor,
    adapt_mask: torch.Tensor | None = None,
    damping: float,
    max_step_norm: float,
    gradient_epsilon: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """只依据 Adapt Pilot 构造一个 rank-1 阻尼 Gauss–Newton 候选。

    ``parameters`` 是待更新参数的展平向量，``basis`` 是同形状的固定子空间
    方向。``adapt_logits_fn`` 必须只用合法 Adapt 接收数据计算 logits；本函数
    不接收 Reward、Data 标签或隐藏信道状态，也不实施候选验收、提交或回滚。

    在系数坐标 ``z`` 上计算平均 BCE 梯度和 logistic Fisher 曲率：

    ``g = mean((sigmoid(logit) - label) * dlogit/dz)``
    ``F = mean(sigmoid(logit) * (1-sigmoid(logit)) * (dlogit/dz)^2)``
    ``delta_z = -g / (F + damping)``

    最后将系数步长裁剪到 ``max_step_norm``。返回的候选与诊断均为停止梯度的
    张量/标量；该模块不构造外层元梯度。
    """

    if not isinstance(parameters, torch.Tensor) or not parameters.is_floating_point():
        raise TypeError("parameters 必须是浮点张量。")
    if parameters.ndim != 1 or parameters.numel() == 0:
        raise ValueError("parameters 必须是非空一维向量。")
    if not isinstance(basis, torch.Tensor) or not basis.is_floating_point():
        raise TypeError("basis 必须是浮点张量。")
    if basis.shape != parameters.shape:
        raise ValueError("basis 与 parameters 必须具有相同的一维形状。")
    if not callable(adapt_logits_fn):
        raise TypeError("adapt_logits_fn 必须是可调用对象。")
    if not math_is_finite_positive(damping):
        raise ValueError("damping 必须是正的有限数。")
    if not math_is_finite_positive(max_step_norm):
        raise ValueError("max_step_norm 必须是正的有限数。")
    if not math_is_finite_nonnegative(gradient_epsilon):
        raise ValueError("gradient_epsilon 必须是非负有限数。")
    if not torch.isfinite(parameters).all() or not torch.isfinite(basis).all():
        raise FloatingPointError("parameters 或 basis 含有非有限值。")

    device = parameters.device
    work_dtype = torch.float64 if parameters.dtype == torch.float64 else torch.float32
    origin = parameters.detach().to(dtype=work_dtype)
    direction = basis.detach().to(device=device, dtype=work_dtype)
    direction_norm = torch.linalg.vector_norm(direction)
    if not torch.isfinite(direction_norm) or float(direction_norm) <= 1e-12:
        raise ValueError("basis 必须具有非零有限范数。")
    direction = direction / direction_norm

    labels = torch.as_tensor(adapt_labels, device=device, dtype=work_dtype).reshape(-1)
    if not torch.isfinite(labels).all():
        raise FloatingPointError("Adapt Pilot 标签含有非有限值。")
    if bool(((labels < 0) | (labels > 1)).any()):
        raise ValueError("Adapt Pilot 标签必须位于[0, 1]。")

    def logits_at(coefficient: torch.Tensor) -> torch.Tensor:
        candidate_parameters = origin + direction * coefficient.reshape(())
        logits = adapt_logits_fn(candidate_parameters)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("adapt_logits_fn 必须返回张量。")
        return logits.reshape(-1).to(device=device, dtype=work_dtype)

    with torch.enable_grad():
        coefficient = torch.zeros((), device=device, dtype=work_dtype, requires_grad=True)
        logits = logits_at(coefficient)
        if logits.shape != labels.shape:
            raise ValueError("Adapt logits 与 Adapt 标签元素数量必须一致。")
        if adapt_mask is None:
            mask = torch.ones_like(labels, dtype=torch.bool)
        else:
            mask = torch.as_tensor(adapt_mask, device=device, dtype=torch.bool).reshape(-1)
            if mask.shape != labels.shape:
                raise ValueError("adapt_mask 与 Adapt 标签元素数量必须一致。")

        count = int(mask.sum().detach().cpu())
        unchanged = origin.to(dtype=parameters.dtype).detach().clone()
        if count == 0:
            return unchanged, {
                "no_op": True,
                "no_op_reason": "empty_adapt_mask",
                "adapt_count": 0,
                "basis_norm": float(direction_norm.detach().cpu()),
                "gradient": 0.0,
                "fisher": 0.0,
                "damping": float(damping),
                "raw_step_norm": 0.0,
                "step_norm": 0.0,
            }

        selected_logits = logits[mask]
        selected_labels = labels[mask]
        if not torch.isfinite(selected_logits).all():
            raise FloatingPointError("Adapt Pilot logits 含有非有限值。")

        # Efficient SDPA 当前不支持该图的双反向 JVP；使用向量化 VJP
        # 一次性构造标量系数对可见 Adapt logits 的方向导数。
        jacobian = torch.autograd.functional.jacobian(
            logits_at,
            coefficient,
            create_graph=False,
            strict=False,
            vectorize=True,
        )
        jacobian = jacobian.reshape(-1)
        selected_jacobian = jacobian[mask].detach()
        probabilities = torch.sigmoid(selected_logits.detach())
        gradient = torch.mean((probabilities - selected_labels) * selected_jacobian)
        fisher = torch.mean(probabilities * (1.0 - probabilities) * selected_jacobian.square())

    if not torch.isfinite(jacobian).all() or not torch.isfinite(gradient) or not torch.isfinite(fisher):
        raise FloatingPointError("Adapt Pilot Jacobian、梯度或 Fisher 曲率含有非有限值。")
    gradient_value = float(gradient.detach().cpu())
    fisher_value = float(fisher.detach().cpu())
    raw_step = -gradient / (fisher + float(damping))
    raw_norm = torch.abs(raw_step)
    step = raw_step * (float(max_step_norm) / raw_norm.clamp_min(1e-12)).clamp(max=1.0)
    step_norm = torch.abs(step)

    if float(torch.abs(gradient).detach().cpu()) <= gradient_epsilon:
        step = torch.zeros_like(step)
        step_norm = torch.zeros_like(step_norm)
        no_op_reason = "zero_adapt_gradient"
    else:
        no_op_reason = ""

    candidate = (origin + direction * step).to(dtype=parameters.dtype).detach()
    if not torch.isfinite(candidate).all():
        raise FloatingPointError("Gauss–Newton 候选包含非有限值。")
    return candidate, {
        "no_op": bool(no_op_reason),
        "no_op_reason": no_op_reason,
        "adapt_count": count,
        "basis_norm": float(direction_norm.detach().cpu()),
        "gradient": gradient_value,
        "fisher": fisher_value,
        "damping": float(damping),
        "raw_step_norm": float(raw_norm.detach().cpu()),
        "step_norm": float(step_norm.detach().cpu()),
    }


def differentiable_damped_rank1_gauss_newton_candidate(
    *,
    parameters: torch.Tensor,
    basis: torch.Tensor,
    adapt_logits_fn: Callable[[torch.Tensor], torch.Tensor],
    adapt_labels: torch.Tensor,
    adapt_mask: torch.Tensor | None = None,
    damping: float,
    max_step_norm: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """构造可对初始参数和 rank-1 basis 反传的 Adapt-only GN 候选。

    该接口与推理函数分离，不对输入或候选执行 ``detach``。它使用
    ``create_graph=True`` 计算 ``dlogits/dz``，因此外层目标可以通过梯度/Fisher
    更新反传到 ``parameters`` 与 ``basis``。候选步幅仍按单步范数裁剪。

    此函数本身只保证纯系数更新的高阶可微性。若 ``adapt_logits_fn`` 含有
    Flash/efficient attention、自定义 CUDA 算子或其他不支持双反向的算子，
    调用方需在此训练分支选择支持二阶导数的 backend；线上推理可继续用原
    ``damped_rank1_gauss_newton_candidate``，不要求保留外层计算图。
    """

    if not isinstance(parameters, torch.Tensor) or not parameters.is_floating_point():
        raise TypeError("parameters 必须是浮点张量。")
    if parameters.ndim != 1 or parameters.numel() == 0:
        raise ValueError("parameters 必须是非空一维向量。")
    if not isinstance(basis, torch.Tensor) or not basis.is_floating_point():
        raise TypeError("basis 必须是浮点张量。")
    if basis.shape != parameters.shape:
        raise ValueError("basis 与 parameters 必须具有相同的一维形状。")
    if not callable(adapt_logits_fn):
        raise TypeError("adapt_logits_fn 必须是可调用对象。")
    if not math_is_finite_positive(damping):
        raise ValueError("damping 必须是正的有限数。")
    if not math_is_finite_positive(max_step_norm):
        raise ValueError("max_step_norm 必须是正的有限数。")
    if not torch.isfinite(parameters).all() or not torch.isfinite(basis).all():
        raise FloatingPointError("parameters 或 basis 含有非有限值。")

    device = parameters.device
    work_dtype = torch.float64 if parameters.dtype == torch.float64 else torch.float32
    labels = torch.as_tensor(adapt_labels, device=device, dtype=work_dtype).reshape(-1)
    if not torch.isfinite(labels).all():
        raise FloatingPointError("Adapt Pilot 标签含有非有限值。")
    if bool(((labels < 0) | (labels > 1)).any()):
        raise ValueError("Adapt Pilot 标签必须位于[0, 1]。")
    if adapt_mask is None:
        mask = torch.ones_like(labels, dtype=torch.bool)
    else:
        mask = torch.as_tensor(adapt_mask, device=device, dtype=torch.bool).reshape(-1)
        if mask.shape != labels.shape:
            raise ValueError("adapt_mask 与 Adapt 标签元素数量必须一致。")
    if not bool(mask.any()):
        unchanged = parameters.to(dtype=work_dtype)
        zero = unchanged.new_zeros(())
        return unchanged, {"gradient": zero, "fisher": zero, "step": zero, "step_norm": zero}

    with torch.enable_grad():
        origin = parameters.to(dtype=work_dtype)
        raw_direction = basis.to(device=device, dtype=work_dtype)
        direction_norm = torch.linalg.vector_norm(raw_direction)
        if not torch.isfinite(direction_norm) or float(direction_norm.detach()) <= 1e-12:
            raise ValueError("basis 必须具有非零有限范数。")
        direction = raw_direction / direction_norm
        coefficient = torch.zeros((), device=device, dtype=work_dtype, requires_grad=True)

        def logits_at(value: torch.Tensor) -> torch.Tensor:
            logits = adapt_logits_fn(origin + direction * value.reshape(()))
            if not isinstance(logits, torch.Tensor):
                raise TypeError("adapt_logits_fn 必须返回张量。")
            return logits.reshape(-1).to(device=device, dtype=work_dtype)

        logits = logits_at(coefficient)
        if logits.shape != labels.shape:
            raise ValueError("Adapt logits 与 Adapt 标签元素数量必须一致。")
        selected_logits = logits[mask]
        selected_labels = labels[mask]
        if not torch.isfinite(selected_logits).all():
            raise FloatingPointError("Adapt Pilot logits 含有非有限值。")

        # 保留 Jacobian 对外层参数的图，以便对有限帧内更新做元梯度。
        jacobian = torch.autograd.functional.jacobian(
            logits_at,
            coefficient,
            create_graph=True,
            strict=False,
            vectorize=True,
        ).reshape(-1)
        selected_jacobian = jacobian[mask]
        probabilities = torch.sigmoid(selected_logits)
        gradient = ((probabilities - selected_labels) * selected_jacobian).mean()
        fisher = (probabilities * (1.0 - probabilities) * selected_jacobian.square()).mean()
        step = -gradient / (fisher + float(damping))
        step_norm_before_clip = torch.abs(step)
        clip_scale = (float(max_step_norm) / step_norm_before_clip.clamp_min(1e-12)).clamp(max=1.0)
        step = step * clip_scale
        candidate = origin + direction * step

    if not torch.isfinite(jacobian).all() or not torch.isfinite(gradient).all():
        raise FloatingPointError("Adapt Pilot Jacobian 或梯度含有非有限值。")
    if not torch.isfinite(fisher).all() or not torch.isfinite(candidate).all():
        raise FloatingPointError("Adapt Pilot Fisher 或 Gauss–Newton 候选含有非有限值。")
    return candidate.to(dtype=parameters.dtype), {
        "gradient": gradient,
        "fisher": fisher,
        "step": step,
        "step_norm": torch.abs(step),
    }


def math_is_finite_positive(value: float) -> bool:
    """返回有限正标量校验结果。"""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric > 0.0


def math_is_finite_nonnegative(value: float) -> bool:
    """返回有限非负标量校验结果。"""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric >= 0.0
