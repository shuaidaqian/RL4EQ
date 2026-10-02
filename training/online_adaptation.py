# -*- coding: utf-8 -*-
"""Pilot 驱动的非 PPO 在线神经均衡适配。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from agent.cir_estimator import CIRCondition, condition_from_cir
from agent.unfolded_equalizer import UnfoldedEqualizer


@dataclass(frozen=True)
class OnlineAdaptationResult:
    """一帧 Adapt Pilot 更新的可审计结果。"""

    accepted: bool
    adapt_pilot_count: int
    adapt_loss_before: float
    adapt_loss_after: float
    parameter_delta_norm: float
    data_labels_used_online: bool
    weighted_adapt_loss_before: float = 0.0
    weighted_adapt_loss_after: float = 0.0
    hard_example_weighting: bool = False
    hard_example_temperature: float = 0.5


@dataclass(frozen=True)
class OnlineRLSAdaptationResult:
    """一帧 Pilot 驱动 RLS 残差更新的可审计结果。"""

    accepted: bool
    adapt_pilot_count: int
    adapt_loss_before: float
    adapt_loss_after: float
    pilot_mse_before: float
    pilot_mse_after: float
    parameter_delta_norm: float
    data_labels_used_online: bool
    cumulative_delta_norm: float = 0.0
    max_total_delta_norm: float = 0.0


class PilotResidualRLSAdapter:
    """只用有效前缀 Pilot 递推更新零初始化神经 logit 残差。

    离线均衡器主干和 head 保持冻结。RLS 的回归目标是目标符号 logit
    与冻结离线输出之间的残差，因此该更新器只改变动态挂载的 Adapter。
    """

    def __init__(
        self,
        model: UnfoldedEqualizer,
        forgetting_factor: float = 0.99,
        ridge: float = 1.0,
        warmup_symbols: int = 0,
        target_logit: float = 2.0,
        max_delta_norm: float = 1.0,
        max_total_delta_norm: float | None = None,
    ) -> None:
        self.model = model
        self.forgetting_factor = float(forgetting_factor)
        self.ridge = float(ridge)
        self.warmup_symbols = int(warmup_symbols)
        self.target_logit = float(target_logit)
        self.max_delta_norm = float(max_delta_norm)
        self.max_total_delta_norm = float(
            self.max_delta_norm
            if max_total_delta_norm is None
            else max_total_delta_norm
        )
        if not 0.0 < self.forgetting_factor <= 1.0:
            raise ValueError("forgetting_factor 必须位于 (0, 1]。")
        if self.ridge <= 0.0:
            raise ValueError("ridge 必须为正数。")
        if self.warmup_symbols < 0:
            raise ValueError("warmup_symbols 不能为负数。")
        if (
            self.target_logit <= 0.0
            or self.max_delta_norm <= 0.0
            or self.max_total_delta_norm <= 0.0
        ):
            raise ValueError("target_logit 和参数变化上限必须为正数。")
        self.module = self.model.attach_online_residual_adapter()
        self.anchor_weight = self.module.linear.weight.detach().clone()
        self.anchor_bias = self.module.linear.bias.detach().clone()
        self.parameter_delta_norm = 0.0
        self.cumulative_delta_norm = 0.0
        self.covariance = self._initial_covariance()

    def _initial_covariance(self) -> torch.Tensor:
        dimension = self.model.config.d_model + 1
        parameter = next(self.model.parameters())
        return torch.eye(
            dimension,
            device=parameter.device,
            dtype=torch.float32,
        ) / self.ridge

    def _effective_mask(self, frame, device: torch.device) -> torch.Tensor:
        mask = frame.adapt_mask.to(device=device, dtype=torch.bool).reshape(-1)
        if self.warmup_symbols <= 0:
            return mask
        positions = torch.arange(mask.numel(), device=device)
        return mask & (positions >= self.warmup_symbols)

    def adapt(
        self,
        frame,
        condition: CIRCondition,
        soft_tail: torch.Tensor,
    ) -> OnlineRLSAdaptationResult:
        """使用当前帧有效 Adapt Pilot 更新残差 Adapter。"""

        device = next(self.model.parameters()).device
        mask = self._effective_mask(frame, device)
        pilot_count = int(mask.sum().item())
        if pilot_count == 0:
            return OnlineRLSAdaptationResult(False, 0, 0.0, 0.0, 0.0, 0.0, 0.0, False)

        if hasattr(frame, "receiver_view"):
            receiver_view = frame.receiver_view()
            adapt_symbols = receiver_view.adapt_symbols
        else:
            adapt_symbols = torch.zeros_like(frame.tx_symbols)
            full_mask = frame.adapt_mask.to(torch.bool)
            adapt_symbols[full_mask] = frame.tx_symbols[full_mask]
        rx = receiver_view.rx_symbols if hasattr(frame, "receiver_view") else frame.rx_symbols
        rx = rx.to(device)
        adapt_symbols = adapt_symbols.to(device).to(torch.complex64)
        rx_iq = torch.stack((rx.real, rx.imag), dim=-1).unsqueeze(0).float()
        region_ids = frame.model_region_ids.to(device).unsqueeze(0).long()
        pilot_symbols = adapt_symbols.unsqueeze(0)
        full_adapt_mask = frame.adapt_mask.to(device=device, dtype=torch.bool).reshape(1, -1)
        condition = _condition_to_device(condition, device)
        tail = soft_tail.to(device).to(torch.complex64)
        if tail.ndim == 1:
            tail = tail.unsqueeze(0)

        was_training = self.model.training
        self.model.eval()
        old_weight = self.module.linear.weight.detach().clone()
        old_bias = self.module.linear.bias.detach().clone()
        old_covariance = self.covariance.detach().clone()
        try:
            with torch.no_grad():
                logits_before, _ = self.model(
                    rx_iq,
                    condition,
                    region_ids,
                    tail,
                    adapt_symbols=pilot_symbols,
                    adapt_mask=full_adapt_mask,
                )
                features = self.model.online_residual_features
                if features is None:
                    raise RuntimeError("模型未产生在线残差 Adapter 所需的隐藏特征。")
                features = features[0].float()
                logits_before = logits_before[0].float()
                current_residual = self.module(features).squeeze(-1).float()
                frozen_logits = logits_before - current_residual
                target = (adapt_symbols.real > 0.0).float()
                design = torch.cat(
                    [features[mask], torch.ones(pilot_count, 1, device=device)],
                    dim=1,
                )
                signed_target = 2.0 * target[mask] - 1.0
                # 正确样本保留离线输出，只对错误样本施加有限幅度的纠正目标。
                # 这样 RLS 学习的是当前帧残差，而不是重写已经正确的离线判决。
                correct = frozen_logits[mask] * signed_target > 0.0
                desired_logits = torch.where(
                    correct,
                    frozen_logits[mask],
                    self.target_logit * signed_target,
                )
                target_residual = desired_logits - frozen_logits[mask]
                initial_prediction = design @ torch.cat(
                    [old_weight.reshape(-1).float(), old_bias.reshape(-1).float()]
                )
                mse_before = torch.mean((initial_prediction - target_residual) ** 2)
                parameters = torch.cat(
                    [old_weight.reshape(-1).float(), old_bias.reshape(-1).float()]
                )
                covariance = old_covariance
                for row, target_value in zip(design, target_residual):
                    covariance_row = covariance @ row
                    denominator = self.forgetting_factor + row @ covariance_row
                    gain = covariance_row / denominator.clamp_min(1e-8)
                    error = target_value - row @ parameters
                    parameters = parameters + gain * error
                    covariance = (
                        covariance - torch.outer(gain, row @ covariance)
                    ) / self.forgetting_factor
                covariance = 0.5 * (covariance + covariance.transpose(0, 1))
                if not torch.isfinite(parameters).all() or not torch.isfinite(covariance).all():
                    accepted = False
                else:
                    self.module.linear.weight.copy_(parameters[:-1].reshape_as(old_weight))
                    self.module.linear.bias.copy_(parameters[-1:].reshape_as(old_bias))
                    delta = parameters - torch.cat(
                        [old_weight.reshape(-1).float(), old_bias.reshape(-1).float()]
                    )
                    delta_norm = float(torch.linalg.vector_norm(delta).cpu())
                    anchor = torch.cat(
                        [
                            self.anchor_weight.reshape(-1).float(),
                            self.anchor_bias.reshape(-1).float(),
                        ]
                    )
                    cumulative_delta_norm = float(
                        torch.linalg.vector_norm(parameters - anchor).cpu()
                    )
                    # 零增量不算真实在线参数更新，避免浮点噪声被审计为 PEFT 成功。
                    accepted = bool(
                        1e-12 < delta_norm <= self.max_delta_norm
                        and cumulative_delta_norm <= self.max_total_delta_norm
                    )
                if not accepted:
                    self.module.linear.weight.copy_(old_weight)
                    self.module.linear.bias.copy_(old_bias)
                    self.covariance = old_covariance
                    self.parameter_delta_norm = 0.0
                    delta_norm = 0.0
                    current = torch.cat(
                        [
                            old_weight.reshape(-1).float(),
                            old_bias.reshape(-1).float(),
                        ]
                    )
                    anchor = torch.cat(
                        [
                            self.anchor_weight.reshape(-1).float(),
                            self.anchor_bias.reshape(-1).float(),
                        ]
                    )
                    self.cumulative_delta_norm = float(
                        torch.linalg.vector_norm(current - anchor).cpu()
                    )
                else:
                    self.covariance = covariance
                    self.parameter_delta_norm = delta_norm
                    self.cumulative_delta_norm = cumulative_delta_norm

                logits_after, _ = self.model(
                    rx_iq,
                    condition,
                    region_ids,
                    tail,
                    adapt_symbols=pilot_symbols,
                    adapt_mask=full_adapt_mask,
                )
                logits_after = logits_after[0].float()
                prediction_after = design @ torch.cat(
                    [
                        self.module.linear.weight.reshape(-1).float(),
                        self.module.linear.bias.reshape(-1).float(),
                    ]
                )
                mse_after = torch.mean((prediction_after - target_residual) ** 2)
                loss_before = F.binary_cross_entropy_with_logits(
                    logits_before[mask], target[mask]
                )
                loss_after = F.binary_cross_entropy_with_logits(
                    logits_after[mask], target[mask]
                )
            return OnlineRLSAdaptationResult(
                accepted,
                pilot_count,
                float(loss_before.cpu()),
                float(loss_after.cpu()),
                float(mse_before.cpu()),
                float(mse_after.cpu()),
                float(self.parameter_delta_norm),
                False,
                float(self.cumulative_delta_norm),
                float(self.max_total_delta_norm),
            )
        finally:
            self.model.train(was_training)


class PilotDrivenOnlineAdapter:
    """只用 Adapt Pilot 更新受限 PEFT 参数的在线适配器。

    该类刻意不读取 Reward Pilot 或 Data 标签。动作策略可以由固定规则、
    小型控制器或 PPO 提供，但参数更新本身始终由已知 Adapt Pilot 损失驱动。
    """

    def __init__(
        self,
        model: UnfoldedEqualizer,
        groups: set[str] | None = None,
        learning_rate: float = 1e-4,
        steps: int = 1,
        max_delta_norm: float = 0.5,
        proximal_weight: float = 0.0,
        hard_example_weighting: bool = False,
        hard_example_temperature: float = 0.5,
    ) -> None:
        self.model = model
        self.groups = set(groups or {"head"})
        self.learning_rate = float(learning_rate)
        self.steps = max(1, int(steps))
        self.max_delta_norm = float(max_delta_norm)
        self.proximal_weight = float(proximal_weight)
        self.hard_example_weighting = bool(hard_example_weighting)
        self.hard_example_temperature = float(hard_example_temperature)
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate 必须为正数。")
        if self.max_delta_norm <= 0.0:
            raise ValueError("max_delta_norm 必须为正数。")
        if self.proximal_weight < 0.0:
            raise ValueError("proximal_weight 不能为负数。")
        if self.hard_example_temperature <= 0.0:
            raise ValueError("hard_example_temperature 必须为正数。")

    def adapt(
        self,
        frame,
        condition: CIRCondition,
        soft_tail: torch.Tensor,
        *,
        groups: set[str] | None = None,
        learning_rate: float | None = None,
        steps: int | None = None,
        max_delta_norm: float | None = None,
        proximal_weight: float | None = None,
        hard_example_weighting: bool | None = None,
        hard_example_temperature: float | None = None,
    ) -> OnlineAdaptationResult:
        """使用当前帧 Adapt Pilot 做一次受限在线更新。"""

        selected_groups = set(self.groups if groups is None else groups)
        selected_learning_rate = self.learning_rate if learning_rate is None else float(learning_rate)
        selected_steps = self.steps if steps is None else max(1, int(steps))
        selected_max_delta_norm = self.max_delta_norm if max_delta_norm is None else float(max_delta_norm)
        selected_proximal_weight = self.proximal_weight if proximal_weight is None else float(proximal_weight)
        selected_hard_example_weighting = (
            self.hard_example_weighting
            if hard_example_weighting is None
            else bool(hard_example_weighting)
        )
        selected_hard_example_temperature = (
            self.hard_example_temperature
            if hard_example_temperature is None
            else float(hard_example_temperature)
        )
        if selected_learning_rate <= 0.0 or selected_max_delta_norm <= 0.0:
            raise ValueError("在线动作覆盖的学习率和更新范数上限必须为正数。")
        if selected_proximal_weight < 0.0:
            raise ValueError("在线动作覆盖的 proximal_weight 不能为负数。")
        if selected_hard_example_temperature <= 0.0:
            raise ValueError("在线动作覆盖的 hard_example_temperature 必须为正数。")
        mask = frame.adapt_mask.to(torch.bool)
        pilot_count = int(mask.sum().item())
        if pilot_count == 0:
            return OnlineAdaptationResult(False, 0, 0.0, 0.0, 0.0, False)

        device = next(self.model.parameters()).device
        if hasattr(frame, "receiver_view"):
            receiver_view = frame.receiver_view()
        else:
            # 兼容只提供最小字段的单元测试对象；正式 Frame 始终走 receiver_view。
            adapt_symbols = torch.zeros_like(frame.tx_symbols)
            adapt_symbols[mask] = frame.tx_symbols[mask]
            receiver_view = SimpleNamespace(
                rx_symbols=frame.rx_symbols,
                adapt_symbols=adapt_symbols,
                adapt_mask=frame.adapt_mask,
                model_region_ids=frame.model_region_ids,
            )
        rx = receiver_view.rx_symbols.to(device)
        tx = receiver_view.adapt_symbols.to(device).to(torch.complex64)
        mask = mask.to(device)
        rx_iq = torch.stack((rx.real, rx.imag), dim=-1).unsqueeze(0).float()
        region_ids = frame.model_region_ids.to(device).unsqueeze(0).long()
        adapt_symbols = tx.unsqueeze(0)
        target = (tx.real > 0.0).float()
        tail = soft_tail.to(device).to(torch.complex64)
        if tail.ndim == 1:
            tail = tail.unsqueeze(0)
        condition = _condition_to_device(condition, device)

        snapshot = _snapshot_groups(self.model, selected_groups)
        was_training = self.model.training
        self.model.train()
        self.model.set_trainable_groups(selected_groups)
        trainable = self.model.trainable_parameters()
        if not trainable:
            self.model.set_trainable_groups(set())
            self.model.train(was_training)
            return OnlineAdaptationResult(False, pilot_count, 0.0, 0.0, 0.0, False)

        optimizer = torch.optim.SGD(trainable, lr=selected_learning_rate)
        trainable_items = [(name, parameter) for name, parameter in self.model.named_parameters() if name in snapshot]
        try:
            with torch.no_grad():
                before_logits, _ = self.model(
                    rx_iq,
                    condition,
                    region_ids,
                    tail,
                    adapt_symbols=adapt_symbols,
                    adapt_mask=mask.unsqueeze(0),
                )
                before_selected = before_logits[0, mask]
                target_selected = target[mask]
                loss_before = F.binary_cross_entropy_with_logits(
                    before_selected, target_selected
                )
                weighted_loss_before = _weighted_adapt_loss(
                    before_selected,
                    target_selected,
                    selected_hard_example_weighting,
                    selected_hard_example_temperature,
                )
            for _ in range(selected_steps):
                logits, _ = self.model(
                    rx_iq,
                    condition,
                    region_ids,
                    tail,
                    adapt_symbols=adapt_symbols,
                    adapt_mask=mask.unsqueeze(0),
                )
                selected_logits = logits[0, mask]
                loss = _weighted_adapt_loss(
                    selected_logits,
                    target[mask],
                    selected_hard_example_weighting,
                    selected_hard_example_temperature,
                )
                if selected_proximal_weight > 0.0:
                    loss = loss + selected_proximal_weight * _normalized_proximal_penalty(
                        trainable_items,
                        snapshot,
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
            with torch.no_grad():
                after_logits, _ = self.model(
                    rx_iq,
                    condition,
                    region_ids,
                    tail,
                    adapt_symbols=adapt_symbols,
                    adapt_mask=mask.unsqueeze(0),
                )
                after_selected = after_logits[0, mask]
                target_selected = target[mask]
                loss_after = F.binary_cross_entropy_with_logits(
                    after_selected, target_selected
                )
                weighted_loss_after = _weighted_adapt_loss(
                    after_selected,
                    target_selected,
                    selected_hard_example_weighting,
                    selected_hard_example_temperature,
                )
            delta_norm = _delta_norm(self.model, snapshot)
            accepted = (
                bool(torch.isfinite(loss_after).item())
                and 1e-12 < delta_norm <= selected_max_delta_norm
            )
            if not accepted:
                _restore_groups(self.model, snapshot)
                delta_norm = 0.0
            return OnlineAdaptationResult(
                accepted,
                pilot_count,
                float(loss_before.cpu()),
                float(loss_after.cpu()),
                float(delta_norm),
                False,
                float(weighted_loss_before.cpu()),
                float(weighted_loss_after.cpu()),
                selected_hard_example_weighting,
                selected_hard_example_temperature,
            )
        finally:
            self.model.eval()
            self.model.set_trainable_groups(set())
            self.model.train(was_training)


def _normalized_proximal_penalty(
    parameter_items: list[tuple[str, torch.Tensor]],
    snapshot: dict[str, torch.Tensor],
) -> torch.Tensor:
    """计算相对本次更新起点的归一化参数距离。"""

    total = None
    element_count = 0
    for name, parameter in parameter_items:
        reference = snapshot.get(name)
        if reference is None:
            continue
        delta = parameter - reference.to(device=parameter.device, dtype=parameter.dtype)
        contribution = torch.sum(delta * delta)
        total = contribution if total is None else total + contribution
        element_count += int(parameter.numel())
    if total is None:
        return torch.zeros((), dtype=torch.float32)
    return total / max(1, element_count)


def hard_example_weights(
    logits: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """返回边界样本更高、且有界的 Adapt Pilot 权重。

    logits 绝对值越小表示模型越不确定，权重越接近 2；高置信度样本的
    权重逐渐接近 1。权重对 logits 停止梯度，避免训练目标通过权重本身
    产生额外的梯度路径。
    """

    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError("temperature 必须为正数。")
    return 1.0 + torch.exp(-logits.detach().abs() / temperature)


def _weighted_adapt_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    enabled: bool,
    temperature: float,
) -> torch.Tensor:
    """计算原始或 hard-example 加权的 Adapt Pilot BCE。"""

    per_example = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )
    if not enabled:
        return per_example.mean()
    return (per_example * hard_example_weights(logits, temperature)).mean()


def run_pilot_driven_online(
    config_path: str | Path,
    frames: int,
    num_seeds: int,
    output_dir: str | Path,
    delays: list[int] | None = None,
    snrs: list[float] | None = None,
    pilot_total: int = 128,
    pilot_layout: str = "prefix",
    pretrained: str | Path | None = None,
    cir_update_mode: str = "fixed",
    cir_update_alpha: float = 0.2,
    state_split: str | None = None,
    scheduler: str = "fixed",
    device: str = "cpu",
) -> dict:
    """运行正式的 Pilot 驱动在线适配入口。"""

    if cir_update_mode not in {"fixed", "pilot_sparse", "decision_directed"}:
        raise ValueError(f"未知 CIR 更新模式：{cir_update_mode}")
    if scheduler not in {"fixed", "bandit"}:
        raise ValueError(f"未知在线调度器：{scheduler}")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    selected_delays = delays or [int(value) for value in config.get("main_delays", [20, 30, 40])]
    selected_snrs = snrs or [float(value) for value in config.get("main_snrs", [0, 5, 10, 15])]
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    pretrained_path = Path(pretrained) if pretrained is not None else None
    model_config = _load_model_config(config, pretrained_path)
    rows: list[dict] = []
    effective_channel = None
    from baseline.traditional_equalizers import estimate_phase_residual_vector
    from agent.cir_estimator import pilot_sparse_cir_update
    from env.comm_env import CommunicationEnvironment, ReceiverState
    from env.experiment_config import build_comm_env_config, effective_channel_metadata
    from training.meta_training import estimate_acquisition_cir_for_profile
    from training.windowed_discrete_ppo import _masked_bce
    from agent.safe_contextual_bandit import SafeContextualBandit, SafeUpdateAction

    for delay in selected_delays:
        for snr_db in selected_snrs:
            for seed in range(int(num_seeds)):
                model = _build_model(model_config, pretrained_path, device)
                online_groups = {
                    str(group)
                    for group in config.get(
                        "online_adaptation_groups",
                        ["head", "conditioner_film"],
                    )
                }
                if "logit_affine" in online_groups:
                    model.attach_online_logit_affine_adapter()
                if "input_affine" in online_groups:
                    model.attach_online_input_affine_adapter()
                if "input_trend" in online_groups:
                    model.attach_online_input_trend_adapter()
                if "input_fir" in online_groups:
                    model.attach_online_input_fir_adapter()
                if "logit_fir" in online_groups:
                    model.attach_online_logit_fir_adapter()
                adapter = PilotDrivenOnlineAdapter(
                    model,
                    groups=online_groups,
                    learning_rate=float(config.get("online_adaptation_learning_rate", 1e-4)),
                    steps=int(config.get("online_adaptation_steps", 1)),
                    max_delta_norm=float(config.get("online_adaptation_max_delta_norm", 0.5)),
                    proximal_weight=float(config.get("online_adaptation_proximal_weight", 0.0)),
                    hard_example_weighting=bool(config.get("online_hard_example_weighting", False)),
                    hard_example_temperature=float(
                        config.get("online_hard_example_temperature", 0.5)
                    ),
                )
                bandit = SafeContextualBandit(seed=90_000 + int(seed)) if scheduler == "bandit" else None
                allowed_bandit_actions = (
                    {
                        action.name
                        for action in bandit.actions
                        if action.groups.issubset(adapter.groups)
                    }
                    if bandit is not None
                    else set()
                )
                active_action = None
                hold_remaining = 0
                previous_reward_gain = 0.0
                rollback_count = 0
                env_config = build_comm_env_config(
                    config,
                    level="B",
                    snr_db=float(snr_db),
                    seed=70_000 + int(seed),
                    max_delay=int(delay),
                    total_pilot=int(pilot_total),
                    pilot_layout=str(pilot_layout),
                    state_split=state_split,
                )
                effective_channel = effective_channel or effective_channel_metadata(env_config)
                env = CommunicationEnvironment(env_config)
                start = env.reset_episode()
                if str(config.get("impairment_profile", "clean")) == "clean":
                    cir = estimate_acquisition_cir_for_profile(
                        start.acquisition,
                        int(delay),
                        "clean",
                    ).to(device).to(torch.complex64)
                    acquisition_cfo = 0.0
                else:
                    from baseline.traditional_equalizers import estimate_acquisition_cir_with_cfo

                    cir, acquisition_cfo = estimate_acquisition_cir_with_cfo(
                        start.acquisition,
                        int(delay),
                        cfo_limit=float(config.get("profile_prior", {}).get("acquisition_cfo_limit", 0.004)),
                    )
                    cir = cir.to(device).to(torch.complex64)
                receiver_state = ReceiverState(start.initial_soft_tail.to(device).to(torch.complex64))
                previous_parameter_delta_norm = 0.0
                consecutive_rejections = 0
                for frame_index in range(1, int(frames) + 1):
                    frame = _frame_to_device(env.next_frame(), device)
                    cir_before_update = cir.detach().clone()
                    if cir_update_mode == "pilot_sparse":
                        cir = pilot_sparse_cir_update(
                            frame,
                            cir,
                            receiver_state.soft_tail,
                            max_paths=24,
                            alpha=float(cir_update_alpha),
                            cfo_hint=float(acquisition_cfo),
                        ).to(device)
                    phase_features = estimate_phase_residual_vector(
                        frame.receiver_view(),
                        cir,
                        receiver_state.soft_tail,
                        blocks=4,
                        cfo_hint=acquisition_cfo,
                    )
                    cir_drift = float(
                        torch.linalg.vector_norm(cir - cir_before_update).detach().cpu()
                        / torch.linalg.vector_norm(cir_before_update).clamp_min(1e-8).detach().cpu()
                    )
                    phase_slope = float(
                        torch.diff(phase_features.reshape(-1)).abs().mean().detach().cpu()
                        if phase_features.numel() > 1
                        else torch.zeros((), device=device)
                    )
                    condition = condition_from_cir(cir, float(snr_db), phase_features=phase_features)
                    tail = receiver_state.soft_tail.unsqueeze(0).to(torch.complex64)
                    rx_iq = torch.stack((frame.rx_symbols.real, frame.rx_symbols.imag), dim=-1).unsqueeze(0).float()
                    region_ids = frame.model_region_ids.unsqueeze(0).long()
                    adapt_symbols = frame.receiver_view().adapt_symbols.unsqueeze(0).to(torch.complex64)
                    adapt_mask = frame.adapt_mask.unsqueeze(0).bool()
                    with torch.no_grad():
                        before_logits, _ = model(
                            rx_iq, condition, region_ids, tail,
                            adapt_symbols=adapt_symbols, adapt_mask=adapt_mask,
                        )
                    before = before_logits.squeeze(0)
                    adapt_loss_before = _masked_bce(before, frame.bits, frame.adapt_mask)
                    context = {
                        "adapt_loss": float(adapt_loss_before.detach().cpu()),
                        "pilot_confidence": float(condition.confidence.mean().detach().cpu()),
                        "residual_cfo": float(acquisition_cfo),
                        "phase_slope": phase_slope,
                        "cir_drift": cir_drift,
                        "snr_db": float(snr_db),
                        "reward_trend": float(previous_reward_gain),
                        "rollback_rate": float(rollback_count / max(1, frame_index - 1)),
                        "consecutive_rejections": float(consecutive_rejections),
                        "parameter_delta_norm": float(previous_parameter_delta_norm),
                    }
                    if bandit is None:
                        action = SafeUpdateAction(
                            "fixed",
                            frozenset(adapter.groups),
                            1.0,
                            1.0,
                            1,
                            0.0,
                        )
                    elif hold_remaining <= 0 or active_action is None:
                        action = bandit.select(context, allowed_names=allowed_bandit_actions)
                        active_action = action
                        hold_remaining = int(action.hold_frames)
                    else:
                        action = active_action
                    hold_remaining = max(0, hold_remaining - 1)
                    update_applied = action.name != "skip" and bool(action.groups)
                    snapshot = model.peft.snapshot(set(action.groups)) if update_applied else {}
                    adaptation = (
                        adapter.adapt(
                            frame,
                            condition,
                            tail,
                            groups=set(action.groups),
                            learning_rate=adapter.learning_rate * float(action.learning_rate_scale),
                            steps=adapter.steps,
                            max_delta_norm=adapter.max_delta_norm * max(0.01, float(action.max_delta_scale)),
                        )
                        if update_applied
                        else None
                    )
                    with torch.no_grad():
                        after_logits, _ = model(
                            rx_iq, condition, region_ids, tail,
                            adapt_symbols=adapt_symbols, adapt_mask=adapt_mask,
                        )
                    after = after_logits.squeeze(0)
                    reward_before = _masked_bce(before, frame.bits, frame.reward_mask)
                    reward_after = _masked_bce(after, frame.bits, frame.reward_mask)
                    raw_reward_gain = float(reward_before.detach().cpu() - reward_after.detach().cpu())
                    accepted = bool(
                        not update_applied
                        or (adaptation is not None and adaptation.accepted and raw_reward_gain >= 0.0)
                    )
                    parameter_delta_norm = float(
                        adaptation.parameter_delta_norm
                        if adaptation is not None and accepted
                        else 0.0
                    )
                    update_cost = float(config.get("online_bandit_update_cost", 0.001)) * parameter_delta_norm
                    action_cost = float(config.get("online_bandit_action_cost", 0.000001)) if update_applied else 0.0
                    rollback_penalty = (
                        float(config.get("online_bandit_rollback_penalty", 0.01))
                        if update_applied and not accepted
                        else 0.0
                    )
                    reward_gain = raw_reward_gain - update_cost - action_cost - rollback_penalty
                    if not accepted:
                        model.peft.restore(snapshot)
                        final = before
                        rollback_count += 1
                    else:
                        final = after
                    if bandit is not None:
                        bandit.update(action.name, context, reward_gain, accepted)
                    previous_reward_gain = reward_gain if accepted else -abs(reward_gain)
                    consecutive_rejections = consecutive_rejections + 1 if not accepted else 0
                    previous_parameter_delta_norm = parameter_delta_norm
                    tail_len = receiver_state.soft_tail.numel()
                    detected_tail = torch.complex(
                        torch.tanh(final[-tail_len:] / 2.0),
                        torch.zeros_like(final[-tail_len:]),
                    )
                    tail_alpha = float(config.get("tail_update_alpha", 0.5))
                    receiver_state.update_tail(
                        (1.0 - tail_alpha) * receiver_state.soft_tail + tail_alpha * detected_tail
                    )
                    if cir_update_mode == "decision_directed":
                        from agent.cir_estimator import decision_directed_cir_update

                        cir = decision_directed_cir_update(
                            frame,
                            final.detach(),
                            int(delay),
                            cir,
                            alpha=float(cir_update_alpha),
                        ).to(device)
                    rows.append(
                        {
                            "method": "Pilot-Driven Online Adaptation",
                            "level": "B",
                            "delay": int(delay),
                            "snr_db": float(snr_db),
                            "pilot_total": int(pilot_total),
                            "pilot_layout": str(pilot_layout),
                            "seed": int(seed),
                            "frame": int(frame_index),
                            "ber_data": _ber(final[frame.data_mask], frame.bits[frame.data_mask]),
                            "ber_reward_pilot": _ber(final[frame.reward_mask], frame.bits[frame.reward_mask]),
                            "ber_adapt_pilot": _ber(final[frame.adapt_mask], frame.bits[frame.adapt_mask]),
                            "reward_pilot_loss_before": float(reward_before.detach().cpu()),
                            "reward_pilot_loss_after": float(reward_after.detach().cpu()),
                            "reward_gain_raw": float(raw_reward_gain),
                            "bandit_reward": float(reward_gain),
                            "bandit_update_cost": float(update_cost),
                            "bandit_action_cost": float(action_cost),
                            "bandit_rollback_penalty": float(rollback_penalty),
                            "adapt_pilot_count": int(
                                adaptation.adapt_pilot_count
                                if adaptation is not None
                                else frame.adapt_mask.sum().item()
                            ),
                            "adapt_loss_before": float(
                                adaptation.adapt_loss_before
                                if adaptation is not None
                                else adapt_loss_before.detach().cpu()
                            ),
                            "adapt_loss_after": float(
                                adaptation.adapt_loss_after
                                if adaptation is not None
                                else adapt_loss_before.detach().cpu()
                            ),
                            "weighted_adapt_loss_before": float(
                                adaptation.weighted_adapt_loss_before
                                if adaptation is not None
                                else adapt_loss_before.detach().cpu()
                            ),
                            "weighted_adapt_loss_after": float(
                                adaptation.weighted_adapt_loss_after
                                if adaptation is not None
                                else adapt_loss_before.detach().cpu()
                            ),
                            "hard_example_weighting": bool(adapter.hard_example_weighting),
                            "hard_example_temperature": float(adapter.hard_example_temperature),
                            "online_update_source": "adapt_pilot_only",
                            "adaptation_accepted": bool(accepted),
                            "update_applied": bool(update_applied),
                            "scheduler": scheduler,
                            "action": action.name,
                            "action_hold_frames": int(action.hold_frames),
                            "tail_update_alpha": tail_alpha,
                            "cir_update_mode": str(cir_update_mode),
                            "cir_update_alpha": float(cir_update_alpha),
                            "state_split": state_split,
                            "state_instance": env.state_metadata(),
                            "parameter_delta_norm": parameter_delta_norm,
                            "bandit_context": dict(context),
                            "reward_data_labels_used_online": False,
                            "data_labels_used_online": False,
                            "uses_neural_network": True,
                            "uses_rl": scheduler == "bandit",
                            "pretrained_loaded": pretrained_path is not None,
                        }
                    )
    payload = {
        "schema_version": "pilot-driven-online-adaptation-v1",
        "pretrained_loaded": pretrained_path is not None,
        "effective_channel": effective_channel,
        "state_split": state_split,
        "scheduler": scheduler,
        "rows": rows,
        "mean_ber_data": float(sum(row["ber_data"] for row in rows) / max(1, len(rows))),
    }
    with (target / "frame_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (target / "online_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def _load_model_config(config: dict, pretrained_path: Path | None):
    if pretrained_path is None:
        from agent.unfolded_equalizer import UnfoldedConfig

        return UnfoldedConfig.from_dict(config.get("model", {}))
    payload = json.loads((pretrained_path.parent / "model_config.json").read_text(encoding="utf-8"))
    from agent.unfolded_equalizer import UnfoldedConfig

    return UnfoldedConfig.from_dict(payload.get("model", payload))


def _build_model(model_config, pretrained_path: Path | None, device: str) -> UnfoldedEqualizer:
    model = UnfoldedEqualizer(model_config).to(device)
    if pretrained_path is not None:
        payload = torch.load(pretrained_path, map_location=device, weights_only=False)
        state_dict = payload.get("model_state_dict", payload.get("state_dict"))
        if state_dict is None:
            raise KeyError("checkpoint 必须包含 model_state_dict 或 state_dict。")
        model.load_state_dict(state_dict, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _frame_to_device(frame, device: str):
    return replace(
        frame,
        bits=frame.bits.to(device),
        tx_symbols=frame.tx_symbols.to(device),
        rx_symbols=frame.rx_symbols.to(device),
        adapt_mask=frame.adapt_mask.to(device),
        reward_mask=frame.reward_mask.to(device),
        data_mask=frame.data_mask.to(device),
        model_region_ids=frame.model_region_ids.to(device),
        tail_symbols=frame.tail_symbols.to(device) if frame.tail_symbols is not None else None,
        true_cir=frame.true_cir.to(device) if frame.true_cir is not None else None,
    )


def _ber(logits: torch.Tensor, bits: torch.Tensor) -> float:
    if logits.numel() == 0:
        return 0.0
    return float((logits >= 0).ne(bits.bool()).float().mean().detach().cpu())


def _condition_to_device(condition: CIRCondition, device: torch.device) -> CIRCondition:
    return CIRCondition(
        complex_cir=condition.complex_cir.to(device),
        support_probability=condition.support_probability.to(device),
        noise_variance=condition.noise_variance.to(device),
        confidence=condition.confidence.to(device),
        latent_residual=condition.latent_residual.to(device),
    )


def _snapshot_groups(model: UnfoldedEqualizer, groups: set[str]) -> dict[str, torch.Tensor]:
    resolved = model.peft.resolve(groups)
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if getattr(parameter, "_peft_group", None) in resolved
    }


def _restore_groups(model: UnfoldedEqualizer, snapshot: dict[str, torch.Tensor]) -> None:
    lookup = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in snapshot.items():
            lookup[name].copy_(value.to(lookup[name].device))


def _delta_norm(model: UnfoldedEqualizer, snapshot: dict[str, torch.Tensor]) -> float:
    lookup = dict(model.named_parameters())
    total = torch.zeros(())
    for name, value in snapshot.items():
        total = total + (lookup[name].detach().cpu() - value.cpu()).float().pow(2).sum()
    return float(torch.sqrt(total).item())
