# -*- coding: utf-8 -*-
"""离散安全动作 + 窗口级 reward 的在线 PPO runner。"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from agent.cir_estimator import CIRCondition, condition_from_cir
from agent.cir_estimator import decision_directed_cir_update
from agent.discrete_safe_policy import (
    DiscreteSafeAction,
    DiscreteSafePolicy,
    initialize_safe_discrete_policy_prior,
    safe_modulation_actions,
)
from agent.rl_modulator import ModulationObservationEncoder
from agent.unfolded_equalizer import UnfoldedEqualizer
from baseline.block_equalizers import bit_error_rate
from baseline.traditional_equalizers import estimate_phase_residual_vector
from env.comm_env import CommEnvConfig, CommunicationEnvironment, ReceiverState
from training.meta_training import estimate_acquisition_cir_for_profile
from training.rl_modulated_online import _build_equalizer, _load_model_config


@dataclass(frozen=True)
class WindowReward:
    reward: float
    reward_loss_improvement: float
    reward_ber_improvement: float
    data_labels_used_online: bool


class WindowRewardAccumulator:
    """聚合一个窗口内的 Reward Pilot 指标。

    Data BER 参数只写入诊断字段，不参与 reward 计算。
    """

    def __init__(self, action_delta_norm: float):
        self.action_delta_norm = float(action_delta_norm)
        self.loss_improvements: list[float] = []
        self.ber_improvements: list[float] = []
        self.data_ber_before: list[float] = []
        self.data_ber_after: list[float] = []

    def add_frame(
        self,
        *,
        reward_loss_before: float,
        reward_loss_after: float,
        reward_ber_before: float,
        reward_ber_after: float,
        data_ber_before: float,
        data_ber_after: float,
    ) -> None:
        self.loss_improvements.append(float(reward_loss_before) - float(reward_loss_after))
        self.ber_improvements.append(float(reward_ber_before) - float(reward_ber_after))
        self.data_ber_before.append(float(data_ber_before))
        self.data_ber_after.append(float(data_ber_after))

    def finalize(self) -> WindowReward:
        loss_delta = _mean(self.loss_improvements)
        ber_delta = _mean(self.ber_improvements)
        reward = ber_delta + loss_delta - 0.005 * self.action_delta_norm
        return WindowReward(
            reward=float(reward),
            reward_loss_improvement=float(loss_delta),
            reward_ber_improvement=float(ber_delta),
            data_labels_used_online=False,
        )


@dataclass
class WindowedDiscreteOnlineState:
    cir: torch.Tensor
    receiver_state: ReceiverState
    model: UnfoldedEqualizer
    policy: DiscreteSafePolicy
    optimizer: torch.optim.Optimizer
    encoder: ModulationObservationEncoder
    actions: list[DiscreteSafeAction]
    hidden: torch.Tensor
    window_size: int
    previous_window_reward: float
    last_action_delta_norm: float
    rollout: list[dict]
    cir_update_mode: str = "fixed"
    cir_update_alpha: float = 0.2
    current_action: DiscreteSafeAction | None = None
    current_observation: torch.Tensor | None = None
    current_hidden: torch.Tensor | None = None
    current_log_prob: torch.Tensor | None = None
    current_value: torch.Tensor | None = None
    current_accumulator: WindowRewardAccumulator | None = None
    current_window_peft_snapshot: dict[str, torch.Tensor] | None = None
    current_window_cir_snapshot: torch.Tensor | None = None
    current_window_tail_snapshot: torch.Tensor | None = None
    frames_in_window: int = 0


def run_windowed_discrete_online(
    config_path: str | Path,
    frames: int,
    num_seeds: int,
    output_dir: str | Path,
    delays: list[int] | None = None,
    snrs: list[float] | None = None,
    pilot_total: int = 128,
    pilot_layout: str = "prefix",
    window_size: int = 8,
    update_interval: int = 32,
    pretrained: str | Path | None = None,
    policy_path: str | Path | None = None,
    policy_required: bool = False,
    cir_update_mode: str = "fixed",
    cir_update_alpha: float = 0.2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    if cir_update_mode not in {"fixed", "decision_directed"}:
        raise ValueError(f"未知 CIR 更新模式：{cir_update_mode}")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    selected_delays = delays or [int(value) for value in config.get("main_delays", [20, 30, 40])]
    selected_snrs = snrs or [float(value) for value in config.get("main_snrs", [0, 5, 10, 15])]
    pretrained_path = Path(pretrained) if pretrained is not None else None
    model_config = _load_model_config(config, pretrained_path)
    rows = []
    last_policy: DiscreteSafePolicy | None = None
    last_actions: list[DiscreteSafeAction] = []
    for delay in selected_delays:
        for snr_db in selected_snrs:
            for seed in range(int(num_seeds)):
                torch.manual_seed(71_000 + int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(71_000 + int(seed))
                model = _build_equalizer(model_config, pretrained_path, device)
                actions = safe_modulation_actions(len(model.blocks), device=device)
                encoder = ModulationObservationEncoder()
                policy = DiscreteSafePolicy(len(encoder.FIELDS), len(actions)).to(device)
                initialize_safe_discrete_policy_prior(policy, actions)
                policy_loaded = _load_discrete_policy_if_available(
                    policy,
                    Path(policy_path) if policy_path else None,
                    device,
                    policy_required,
                    expected_action_names=[action.name for action in actions],
                )
                optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-5)
                last_policy = policy
                last_actions = actions
                env = CommunicationEnvironment(
                    CommEnvConfig(
                        level="B",
                        max_delay=int(delay),
                        snr_db=float(snr_db),
                        rho=float(config.get("rho", 0.99)),
                        total_pilot=int(pilot_total),
                        layout=str(pilot_layout),
                        seed=70_000 + int(seed),
                        impairment_profile=str(config.get("impairment_profile", "clean")),
                    )
                )
                start = env.reset_episode()
                state = WindowedDiscreteOnlineState(
                    cir=estimate_acquisition_cir_for_profile(
                        start.acquisition,
                        int(delay),
                        str(config.get("impairment_profile", "clean")),
                    ).to(device),
                    receiver_state=ReceiverState(start.initial_soft_tail.to(device)),
                    model=model,
                    policy=policy,
                    optimizer=optimizer,
                    encoder=encoder,
                    actions=actions,
                    hidden=policy.initial_hidden(batch_size=1, device=torch.device(device)),
                    window_size=int(window_size),
                    previous_window_reward=0.0,
                    last_action_delta_norm=0.0,
                    rollout=[],
                    cir_update_mode=str(cir_update_mode),
                    cir_update_alpha=float(cir_update_alpha),
                )
                for frame_index in range(1, int(frames) + 1):
                    frame = _frame_to_device(env.next_frame(), device)
                    phase_features = estimate_phase_residual_vector(
                        frame.receiver_view(),
                        state.cir,
                        state.receiver_state.soft_tail,
                        blocks=4,
                    )
                    row = run_windowed_discrete_frame(
                        state,
                        frame,
                        condition_from_cir(state.cir, float(snr_db), phase_features=phase_features),
                        float(snr_db),
                        frame_index,
                        update_interval,
                    )
                    row.update(
                        {
                            "level": "B",
                            "delay": int(delay),
                            "pilot_total": int(pilot_total),
                            "pilot_layout": str(pilot_layout),
                            "seed": int(seed),
                            "pretrained_loaded": pretrained_path is not None,
                            "policy_loaded": bool(policy_loaded),
                        }
                    )
                    rows.append(row)
    _write_rows(target, rows)
    payload = {
        "schema_version": "windowed-discrete-ppo-v1",
        "pretrained_loaded": pretrained_path is not None,
        "rows": rows,
        "mean_ber_data": float(sum(row["ber_data"] for row in rows) / max(1, len(rows))),
    }
    (target / "online_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if last_policy is not None:
        torch.save(
            {
                "schema_version": "windowed-discrete-policy-v1",
                "state_dict": last_policy.state_dict(),
                "action_names": [action.name for action in last_actions],
            },
            target / "policy.pt",
        )
    return payload


def run_windowed_discrete_frame(
    state: WindowedDiscreteOnlineState,
    frame,
    condition: CIRCondition,
    snr_db: float,
    frame_index: int,
    update_interval: int,
) -> dict:
    rx_iq = torch.stack((frame.rx_symbols.real, frame.rx_symbols.imag), dim=-1).unsqueeze(0).float()
    region_ids = frame.model_region_ids.unsqueeze(0).long()
    tail = state.receiver_state.soft_tail.unsqueeze(0).to(torch.complex64)
    identity = state.actions[0]
    base_logits, _ = state.model(rx_iq, condition, region_ids, tail, modulation=identity.modulation)
    base = base_logits.squeeze(0)
    if state.current_action is None or state.frames_in_window <= 0:
        observation = state.encoder(_policy_view(frame, float(snr_db), base, state.previous_window_reward, state.last_action_delta_norm)).tensor.unsqueeze(0)
        hidden_in = state.hidden.detach()
        sample = state.policy.sample(observation, hidden_in, state.actions)
        state.hidden = sample.hidden
        state.current_action = identity if sample.action.name == "rollback_identity" else sample.action
        state.current_observation = observation.detach()
        state.current_hidden = hidden_in.detach()
        state.current_log_prob = sample.log_prob.detach()
        state.current_value = sample.value.detach()
        state.current_accumulator = WindowRewardAccumulator(state.current_action.delta_norm)
        state.current_window_peft_snapshot = (
            _snapshot_peft_groups(state.model, state.current_action.peft_groups)
            if state.current_action.peft_groups
            else None
        )
        state.current_window_cir_snapshot = state.cir.detach().clone()
        state.current_window_tail_snapshot = state.receiver_state.soft_tail.detach().clone()
        state.frames_in_window = 0
    sampled_action = state.current_action
    applied_action = sampled_action
    if state.current_window_cir_snapshot is None:
        state.current_window_cir_snapshot = state.cir.detach().clone()
    if state.current_window_tail_snapshot is None:
        state.current_window_tail_snapshot = state.receiver_state.soft_tail.detach().clone()
    selected_cir_alpha = (
        float(sampled_action.cir_alpha)
        if getattr(sampled_action, "cir_alpha", None) is not None
        else float(getattr(state, "cir_update_alpha", 0.2))
    )
    peft_delta_norm = 0.0
    if sampled_action.peft_groups:
        peft_delta_norm = _apply_adapt_peft_update(
            state.model,
            frame,
            condition,
            tail,
            groups=sampled_action.peft_groups,
            lr=float(sampled_action.peft_lr),
            steps=int(sampled_action.peft_steps),
        )
        if state.current_accumulator is not None:
            state.current_accumulator.action_delta_norm += float(peft_delta_norm)
    logits_after, _ = state.model(rx_iq, condition, region_ids, tail, modulation=sampled_action.modulation)
    after = logits_after.squeeze(0)
    reward_before_loss = _masked_bce(base, frame.bits, frame.reward_mask)
    reward_after_loss = _masked_bce(after, frame.bits, frame.reward_mask)
    reward_before_ber = bit_error_rate(base[frame.reward_mask], frame.bits[frame.reward_mask])
    reward_after_ber = bit_error_rate(after[frame.reward_mask], frame.bits[frame.reward_mask])
    data_before_ber = bit_error_rate(base[frame.data_mask], frame.bits[frame.data_mask])
    data_after_ber = bit_error_rate(after[frame.data_mask], frame.bits[frame.data_mask])
    assert state.current_accumulator is not None
    state.current_accumulator.add_frame(
        reward_loss_before=float(reward_before_loss.detach().cpu()),
        reward_loss_after=float(reward_after_loss.detach().cpu()),
        reward_ber_before=float(reward_before_ber),
        reward_ber_after=float(reward_after_ber),
        data_ber_before=float(data_before_ber),
        data_ber_after=float(data_after_ber),
    )
    tail_len = state.receiver_state.soft_tail.numel()
    state.receiver_state.update_tail(torch.complex(torch.tanh(after[-tail_len:] / 2.0), torch.zeros_like(after[-tail_len:])))
    if state.cir_update_mode == "decision_directed":
        state.cir = decision_directed_cir_update(
            frame,
            after.detach(),
            int(tail_len),
            state.cir,
            alpha=selected_cir_alpha,
        ).to(state.cir.device)
    else:
        selected_cir_alpha = float(getattr(state, "cir_update_alpha", 0.0))
    state.frames_in_window += 1
    window_closed = state.frames_in_window >= max(1, int(state.window_size))
    window_reward: WindowReward | None = None
    policy_loss = None
    window_rollback = False
    if window_closed:
        window_reward = state.current_accumulator.finalize()
        if window_reward.reward <= 0.0 and sampled_action.name != "identity":
            if state.current_window_peft_snapshot is not None:
                _restore_snapshot(state.model, state.current_window_peft_snapshot)
            if state.current_window_cir_snapshot is not None:
                state.cir = state.current_window_cir_snapshot.to(state.cir.device).clone()
            if state.current_window_tail_snapshot is not None:
                state.receiver_state.update_tail(state.current_window_tail_snapshot.to(state.receiver_state.soft_tail.device))
            window_rollback = True
        state.rollout.append(
            {
                "observation": state.current_observation,
                "hidden": state.current_hidden,
                "action_index": torch.tensor([sampled_action.index], device=after.device),
                "old_log_prob": state.current_log_prob,
                "value": state.current_value,
                "reward": float(window_reward.reward),
            }
        )
        state.previous_window_reward = float(window_reward.reward)
        state.last_action_delta_norm = float(sampled_action.delta_norm)
        if int(update_interval) > 0 and int(frame_index) % int(update_interval) == 0:
            policy_loss = _ppo_update(state.policy, state.optimizer, state.rollout)
            state.rollout.clear()
        state.current_action = None
        state.current_observation = None
        state.current_hidden = None
        state.current_log_prob = None
        state.current_value = None
        state.current_accumulator = None
        state.current_window_peft_snapshot = None
        state.current_window_cir_snapshot = None
        state.current_window_tail_snapshot = None
        state.frames_in_window = 0
    accepted = not bool(window_rollback)
    return {
        "method": "RL-Modulated Neural Block Equalizer",
        "snr_db": float(snr_db),
        "frame": int(frame_index),
        "ber_data": float(data_after_ber),
        "ber_reward_pilot": float(reward_after_ber),
        "ber_adapt_pilot": bit_error_rate(after[frame.adapt_mask], frame.bits[frame.adapt_mask]),
        "offline_identity_ber_data": float(data_before_ber),
        "reward": float(window_reward.reward if window_reward else 0.0),
        "reward_pilot_loss_before": float(reward_before_loss.detach().cpu()),
        "reward_pilot_loss_after": float(reward_after_loss.detach().cpu()),
        "reward_pilot_ber_before": float(reward_before_ber),
        "reward_pilot_ber_after": float(reward_after_ber),
        "policy_learning": "windowed_discrete_safe_ppo",
        "cir_update_mode": str(state.cir_update_mode),
        "cir_update_alpha": float(getattr(state, "cir_update_alpha", 0.0)),
        "selected_cir_update_alpha": float(selected_cir_alpha),
        "cir_update_uses_data_labels": False,
        "receiver_action_type": _receiver_action_type(sampled_action),
        "policy_loss": policy_loss,
        "policy_action_name": sampled_action.name,
        "applied_action_name": applied_action.name,
        "safe_action_accepted": bool(accepted),
        "window_reward_rollback": bool(window_rollback),
        "policy_window_closed": bool(window_closed),
        "window_size": int(state.window_size),
        "action_delta_norm": float(sampled_action.delta_norm),
        "modulation_delta_norm": float(applied_action.delta_norm),
        "peft_delta_norm": float(peft_delta_norm),
        "data_labels_used_online": False,
        "adapt_params": int(state.policy.parameter_count() if policy_loss is not None else 0),
        "adapt_steps": int(1 if policy_loss is not None else 0),
        "parameter_delta_norm": float(applied_action.delta_norm + peft_delta_norm),
    }


def _receiver_action_type(action: DiscreteSafeAction) -> str:
    if getattr(action, "cir_alpha", None) is not None:
        return "cir_alpha"
    if action.peft_groups:
        return "peft"
    if action.name == "rollback_identity":
        return "rollback"
    return "identity"


def _ppo_update(policy: DiscreteSafePolicy, optimizer: torch.optim.Optimizer, rollout: list[dict]) -> float | None:
    if not rollout:
        return None
    device = rollout[0]["value"].device
    rewards = torch.tensor([item["reward"] for item in rollout], dtype=torch.float32, device=device)
    returns = []
    running = torch.zeros((), device=device)
    for reward in reversed(rewards):
        running = reward + 0.95 * running
        returns.append(running)
    returns = torch.stack(list(reversed(returns)))
    values_old = torch.cat([item["value"].flatten().float() for item in rollout]).detach()
    advantages = returns - values_old
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    last_loss = torch.zeros((), device=device)
    for _ in range(2):
        losses = []
        for index, item in enumerate(rollout):
            log_prob, value, entropy = policy.evaluate_action(item["observation"], item["hidden"], item["action_index"])
            ratio = torch.exp(log_prob.flatten()[0] - item["old_log_prob"].flatten()[0])
            clipped = torch.clamp(ratio, 0.8, 1.2) * advantages[index]
            policy_loss = -torch.minimum(ratio * advantages[index], clipped)
            value_loss = 0.5 * (value.flatten()[0] - returns[index]).pow(2)
            losses.append(policy_loss + value_loss - 0.001 * entropy.flatten()[0])
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        last_loss = loss.detach()
    return float(last_loss.cpu())


def should_accept_reward_guard(
    *,
    reward_loss_before: float,
    reward_loss_after: float,
    reward_ber_before: float,
    reward_ber_after: float,
    allow_loss_only: bool = False,
    peft_delta_norm: float = 0.0,
) -> bool:
    """只用 Reward Pilot 判断候选动作是否允许作用到接收机状态。"""

    ber_delta = float(reward_ber_before) - float(reward_ber_after)
    if ber_delta > 1e-12:
        return True
    if ber_delta < -1e-12:
        return False
    loss_delta = float(reward_loss_before) - float(reward_loss_after)
    safe_loss_delta = loss_delta - 0.005 * float(peft_delta_norm)
    return bool(allow_loss_only and safe_loss_delta > 1e-6)


def _apply_adapt_peft_update(
    model: UnfoldedEqualizer,
    frame,
    condition: CIRCondition,
    tail: torch.Tensor,
    *,
    groups: set[str],
    lr: float,
    steps: int,
) -> float:
    """只用 Adapt Pilot 对 PEFT 参数做候选更新，返回参数变化范数。"""

    before = _snapshot_peft_groups(model, groups)
    model.train()
    model.set_trainable_groups(set(groups))
    trainable = model.trainable_parameters()
    if not trainable:
        model.eval()
        model.set_trainable_groups(set())
        return 0.0
    optimizer = torch.optim.AdamW(trainable, lr=float(lr))
    for _ in range(max(1, int(steps))):
        rx_iq = torch.stack((frame.rx_symbols.real, frame.rx_symbols.imag), dim=-1).unsqueeze(0).float()
        region_ids = frame.model_region_ids.unsqueeze(0).long()
        logits, _ = model(rx_iq, condition, region_ids, tail)
        loss = _masked_bce(logits.squeeze(0), frame.bits, frame.adapt_mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
    model.eval()
    model.set_trainable_groups(set())
    delta_sq = torch.zeros(())
    for name, parameter in model.named_parameters():
        if name in before:
            delta_sq = delta_sq + (parameter.detach().cpu() - before[name].cpu()).float().pow(2).sum()
    return float(torch.sqrt(delta_sq).item())


def _snapshot_peft_groups(model: UnfoldedEqualizer, groups: set[str]) -> dict[str, torch.Tensor]:
    resolved = model.peft.resolve(set(groups))
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if getattr(parameter, "_peft_group", None) in resolved
    }


def _restore_snapshot(model: UnfoldedEqualizer, snapshot: dict[str, torch.Tensor]) -> None:
    lookup = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in snapshot.items():
            lookup[name].copy_(value.to(lookup[name].device))


def _load_discrete_policy_if_available(
    policy: DiscreteSafePolicy,
    policy_path: Path | None,
    device: str,
    required: bool,
    expected_action_names: list[str] | None = None,
) -> bool:
    if policy_path is None:
        if required:
            raise FileNotFoundError("已显式要求加载 policy，但 policy_path 为空。")
        return False
    checkpoint = torch.load(policy_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("policy checkpoint 必须是字典。")
    stored_actions = checkpoint.get("action_names")
    if expected_action_names is not None and stored_actions is not None and list(stored_actions) != list(expected_action_names):
        message = "动作表不兼容，跳过旧 policy checkpoint。"
        if required:
            raise ValueError(message)
        return False
    state_dict = checkpoint.get("state_dict")
    if not state_dict:
        raise ValueError("policy checkpoint 缺少非空 state_dict。")
    try:
        policy.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        message = f"动作表不兼容，无法加载 policy checkpoint：{exc}"
        if required:
            raise ValueError(message) from exc
        return False
    return True


def _policy_view(frame, snr_db: float, logits: torch.Tensor, previous_reward: float, last_delta_norm: float):
    view = frame.receiver_view()
    return SimpleNamespace(
        rx_symbols=view.rx_symbols,
        adapt_symbols=view.adapt_symbols,
        adapt_mask=view.adapt_mask,
        model_region_ids=view.model_region_ids,
        noise_variance=torch.tensor(10.0 ** (-float(snr_db) / 10.0), device=logits.device),
        confidence=torch.sigmoid(torch.abs(logits)).mean(),
        previous_reward=torch.tensor(previous_reward, device=logits.device),
        last_modulation_delta_norm=torch.tensor(last_delta_norm, device=logits.device),
    )


def _masked_bce(logits: torch.Tensor, bits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if int(mask.sum().item()) == 0:
        return torch.zeros((), device=logits.device)
    return F.binary_cross_entropy_with_logits(logits[mask].float(), bits[mask].float())


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


def _write_rows(target: Path, rows: list[dict]) -> None:
    with (target / "frame_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))
