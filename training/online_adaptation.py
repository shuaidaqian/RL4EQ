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
    ) -> None:
        self.model = model
        self.groups = set(groups or {"head"})
        self.learning_rate = float(learning_rate)
        self.steps = max(1, int(steps))
        self.max_delta_norm = float(max_delta_norm)
        self.proximal_weight = float(proximal_weight)
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate 必须为正数。")
        if self.max_delta_norm <= 0.0:
            raise ValueError("max_delta_norm 必须为正数。")
        if self.proximal_weight < 0.0:
            raise ValueError("proximal_weight 不能为负数。")

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
    ) -> OnlineAdaptationResult:
        """使用当前帧 Adapt Pilot 做一次受限在线更新。"""

        selected_groups = set(self.groups if groups is None else groups)
        selected_learning_rate = self.learning_rate if learning_rate is None else float(learning_rate)
        selected_steps = self.steps if steps is None else max(1, int(steps))
        selected_max_delta_norm = self.max_delta_norm if max_delta_norm is None else float(max_delta_norm)
        selected_proximal_weight = self.proximal_weight if proximal_weight is None else float(proximal_weight)
        if selected_learning_rate <= 0.0 or selected_max_delta_norm <= 0.0:
            raise ValueError("在线动作覆盖的学习率和更新范数上限必须为正数。")
        if selected_proximal_weight < 0.0:
            raise ValueError("在线动作覆盖的 proximal_weight 不能为负数。")
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
                loss_before = F.binary_cross_entropy_with_logits(
                    before_logits[0, mask], target[mask]
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
                loss = F.binary_cross_entropy_with_logits(logits[0, mask], target[mask])
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
                loss_after = F.binary_cross_entropy_with_logits(
                    after_logits[0, mask], target[mask]
                )
            delta_norm = _delta_norm(self.model, snapshot)
            accepted = bool(torch.isfinite(loss_after).item()) and delta_norm <= selected_max_delta_norm
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
                adapter = PilotDrivenOnlineAdapter(
                    model,
                    groups=online_groups,
                    learning_rate=float(config.get("online_adaptation_learning_rate", 1e-4)),
                    steps=int(config.get("online_adaptation_steps", 1)),
                    max_delta_norm=float(config.get("online_adaptation_max_delta_norm", 0.5)),
                    proximal_weight=float(config.get("online_adaptation_proximal_weight", 0.0)),
                )
                bandit = SafeContextualBandit(seed=90_000 + int(seed)) if scheduler == "bandit" else None
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
                        action = bandit.select(context)
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
                    action_cost = float(config.get("online_bandit_action_cost", 0.0001)) if update_applied else 0.0
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
