# -*- coding: utf-8 -*-
"""离散安全动作 + 窗口级 reward 的在线 PPO runner。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from agent.cir_estimator import (
    CIRCondition,
    condition_from_cir,
    decision_directed_cir_update,
    pilot_sparse_cir_update,
)
from agent.discrete_safe_policy import (
    DiscreteSafeAction,
    DiscreteSafePolicy,
    initialize_safe_discrete_policy_prior,
    safe_modulation_actions,
)
from agent.pilot_state import PilotStateEmbedding, build_pilot_state_summary
from agent.rl_modulator import ModulationObservationEncoder
from agent.unfolded_equalizer import UnfoldedEqualizer
from baseline.block_equalizers import bit_error_rate
from baseline.traditional_equalizers import (
    estimate_phase_residual_features,
    estimate_phase_residual_vector,
)
from env.comm_env import CommunicationEnvironment, ReceiverState
from env.experiment_config import (
    build_comm_env_config,
    effective_channel_metadata,
    validate_model_dimensions,
)
from training.meta_training import estimate_acquisition_cir_for_profile
from training.online_adaptation import _pilot_signal_reconstruction_loss
from training.rl_modulated_online import _build_equalizer, _load_model_config


@dataclass(frozen=True)
class WindowReward:
    reward: float
    reward_loss_improvement: float
    reward_ber_improvement: float
    data_labels_used_online: bool


class WindowRewardAccumulator:
    """聚合一个窗口内的 Reward Pilot 指标。

    窗口 reward 完全由 Reward Pilot 指标构成。历史调用中的 Data BER
    参数保留为兼容字段，但会被明确忽略，不参与任何在线控制。
    """

    def __init__(self, action_delta_norm: float):
        self.action_delta_norm = float(action_delta_norm)
        self.loss_improvements: list[float] = []
        self.ber_improvements: list[float] = []

    def add_frame(
        self,
        *,
        reward_loss_before: float,
        reward_loss_after: float,
        reward_ber_before: float,
        reward_ber_after: float,
        data_ber_before: float | None = None,
        data_ber_after: float | None = None,
    ) -> None:
        # 仅为旧测试和旧调用保留这两个参数；Data 标签不能进入在线 reward。
        del data_ber_before, data_ber_after
        self.loss_improvements.append(float(reward_loss_before) - float(reward_loss_after))
        self.ber_improvements.append(float(reward_ber_before) - float(reward_ber_after))

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
    policy_generator: torch.Generator | None = None
    cir_update_mode: str = "fixed"
    cir_update_alpha: float = 0.2
    current_action: DiscreteSafeAction | None = None
    current_observation: torch.Tensor | None = None
    current_pilot_state_embedding: torch.Tensor | None = None
    current_hidden: torch.Tensor | None = None
    current_log_prob: torch.Tensor | None = None
    current_value: torch.Tensor | None = None
    current_accumulator: WindowRewardAccumulator | None = None
    current_window_peft_snapshot: dict[str, torch.Tensor] | None = None
    trajectory_peft_snapshot: dict[str, torch.Tensor] = field(default_factory=dict)
    current_window_cir_snapshot: torch.Tensor | None = None
    current_window_tail_snapshot: torch.Tensor | None = None
    frames_in_window: int = 0
    tail_refinement_passes: int = 0
    tail_update_alpha: float = 1.0


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
    selected_delays = delays or [int(value) for value in config.get("main_delays", [20, 30, 40])]
    selected_snrs = snrs or [float(value) for value in config.get("main_snrs", [0, 5, 10, 15])]
    env_configs = {
        (int(delay), float(snr_db), int(seed)): build_comm_env_config(
            config,
            level="B",
            snr_db=float(snr_db),
            seed=70_000 + int(seed),
            max_delay=int(delay),
            total_pilot=int(pilot_total),
            pilot_layout=str(pilot_layout),
        )
        for delay in selected_delays
        for snr_db in selected_snrs
        for seed in range(int(num_seeds))
    }
    if not env_configs:
        raise ValueError("在线实验至少需要一个 delay/SNR/seed 配置。")
    effective_channel = effective_channel_metadata(next(iter(env_configs.values())))
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    pretrained_path = Path(pretrained) if pretrained is not None else None
    model_config = _load_model_config(config, pretrained_path)
    if config.get("channel_profile") in {"eme_measurement_v1", "eme_long_memory_v2"}:
        validate_model_dimensions(model_config, next(iter(env_configs.values())))
    rows = []
    last_policy: DiscreteSafePolicy | None = None
    last_actions: list[DiscreteSafeAction] = []
    last_observation_fields: tuple[str, ...] = ()
    for delay in selected_delays:
        for snr_db in selected_snrs:
            for seed in range(int(num_seeds)):
                torch.manual_seed(71_000 + int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(71_000 + int(seed))
                model = _build_equalizer(model_config, pretrained_path, device)
                model.attach_online_phase_trend_adapter()
                actions = safe_modulation_actions(
                    len(model.blocks),
                    device=device,
                    action_names=config.get("rl_action_names"),
                    peft_max_episode_delta_norm=config.get("rl_peft_max_episode_delta_norm"),
                )
                encoder = ModulationObservationEncoder(include_pilot_state=True)
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
                last_observation_fields = encoder.FIELDS
                env_config = env_configs[(int(delay), float(snr_db), int(seed))]
                env = CommunicationEnvironment(env_config)
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
                    policy_generator=torch.Generator(
                        device=torch.device(device)
                    ).manual_seed(72_000 + int(seed)),
                    window_size=int(window_size),
                    previous_window_reward=0.0,
                    last_action_delta_norm=0.0,
                    rollout=[],
                    cir_update_mode=str(cir_update_mode),
                    cir_update_alpha=float(cir_update_alpha),
                    tail_refinement_passes=int(config.get("tail_refinement_passes", 0)),
                    tail_update_alpha=float(config.get("tail_update_alpha", 1.0)),
                )
                for frame_index in range(1, int(frames) + 1):
                    frame = _frame_to_device(env.next_frame(), device)
                    phase_features = estimate_phase_residual_vector(
                        _adapt_receiver_view(frame),
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
                            "level": str(env_config.level),
                            "delay": int(env_config.max_delay),
                            "pilot_total": int(env_config.total_pilot),
                            "pilot_layout": str(env_config.layout),
                            "profile_name": str(env_config.profile_name),
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
        "effective_channel": effective_channel,
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
                "observation_fields": list(last_observation_fields),
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
    adapt_view = _adapt_receiver_view(frame)
    adapt_only_mask = adapt_view.adapt_mask.to(dtype=torch.bool)
    adapt_only_rx_iq = torch.stack(
        (adapt_view.rx_symbols.real, adapt_view.rx_symbols.imag), dim=-1
    ).unsqueeze(0).float()
    region_ids = frame.model_region_ids.unsqueeze(0).long()
    adapt_symbols = adapt_view.adapt_symbols.unsqueeze(0).to(torch.complex64)
    adapt_mask = adapt_view.adapt_mask.unsqueeze(0).bool()
    adapt_labels = (adapt_view.adapt_symbols.real > 0.0).to(frame.bits.dtype)
    tail = state.receiver_state.soft_tail.unsqueeze(0).to(torch.complex64)
    identity = state.actions[0]
    if state.current_action is None or state.frames_in_window <= 0:
        # PPO 的动作 observation 只由 Adapt Pilot 构成；Reward Pilot 留给动作后的验收。
        policy_logits = refine_logits_with_tail(
            state.model,
            adapt_only_rx_iq,
            condition,
            region_ids,
            tail,
            modulation=identity.modulation,
            passes=int(state.tail_refinement_passes),
            adapt_symbols=adapt_symbols,
            adapt_mask=adapt_mask,
        ).squeeze(0)
        pilot_state_embedding = None
        if bool(getattr(state.encoder, "include_pilot_state", False)):
            pilot_state_embedding = _build_adapt_pilot_state_embedding(
                frame,
                state.cir,
                tail.squeeze(0),
                condition,
                policy_logits,
            )
        observation = state.encoder(
            _policy_view(
                frame,
                float(snr_db),
                policy_logits,
                state.previous_window_reward,
                state.last_action_delta_norm,
                pilot_state_embedding=pilot_state_embedding,
            )
        ).tensor.unsqueeze(0)
        state.current_pilot_state_embedding = (
            None
            if pilot_state_embedding is None
            else pilot_state_embedding.detach().clone()
        )
        hidden_in = state.hidden.detach()
        if state.policy_generator is None:
            sample = state.policy.sample(observation, hidden_in, state.actions)
        else:
            sample = state.policy.sample(
                observation,
                hidden_in,
                state.actions,
                generator=state.policy_generator,
            )
        state.hidden = sample.hidden
        state.current_action = sample.action
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
    # 先完成动作选择，再构造 Reward Pilot 视图；该留出 Pilot 只用于验收与回滚。
    reward_eval_mask = (frame.adapt_mask | frame.reward_mask).bool()
    reward_eval_rx_symbols = torch.where(
        reward_eval_mask,
        frame.rx_symbols,
        torch.zeros_like(frame.rx_symbols),
    )
    reward_eval_rx_iq = torch.stack(
        (reward_eval_rx_symbols.real, reward_eval_rx_symbols.imag), dim=-1
    ).unsqueeze(0).float()
    reward_view = _pilot_receiver_view(frame)
    reward_labels = (reward_view.reward_symbols.real > 0.0).to(frame.bits.dtype)
    base_logits = refine_logits_with_tail(
        state.model,
        reward_eval_rx_iq,
        condition,
        region_ids,
        tail,
        modulation=identity.modulation,
        passes=int(state.tail_refinement_passes),
        adapt_symbols=adapt_symbols,
        adapt_mask=adapt_mask,
    )
    base = base_logits.squeeze(0)
    sampled_action = state.current_action
    applied_action = identity if sampled_action.name == "rollback_identity" else sampled_action
    if state.current_window_cir_snapshot is None:
        state.current_window_cir_snapshot = state.cir.detach().clone()
    if state.current_window_tail_snapshot is None:
        state.current_window_tail_snapshot = state.receiver_state.soft_tail.detach().clone()
    frame_peft_snapshot = (
        _snapshot_peft_groups(state.model, set(sampled_action.peft_groups))
        if sampled_action.peft_groups
        else {}
    )
    if sampled_action.peft_max_episode_delta_norm is not None:
        for name, value in frame_peft_snapshot.items():
            state.trajectory_peft_snapshot.setdefault(name, value.detach().clone())
    selected_cir_alpha = (
        float(sampled_action.cir_alpha)
        if getattr(sampled_action, "cir_alpha", None) is not None
        else float(getattr(state, "cir_update_alpha", 0.2))
    )
    peft_delta_norm = 0.0
    window_peft_delta_norm = 0.0
    trajectory_peft_delta_norm = (
        _snapshot_delta_norm(state.model, state.trajectory_peft_snapshot)
        if state.trajectory_peft_snapshot
        else 0.0
    )
    if sampled_action.peft_groups:
        _apply_adapt_peft_update(
            state.model,
            frame,
            condition,
            tail,
            groups=sampled_action.peft_groups,
            lr=float(sampled_action.peft_lr),
            steps=int(sampled_action.peft_steps),
            objective=str(sampled_action.peft_objective),
            max_delta_norm=sampled_action.peft_max_delta_norm,
        )
        if sampled_action.peft_max_delta_norm is not None:
            if state.current_window_peft_snapshot is not None:
                _clip_peft_delta_to_snapshot(
                    state.model,
                    state.current_window_peft_snapshot,
                    float(sampled_action.peft_max_delta_norm),
                )
        if sampled_action.peft_max_episode_delta_norm is not None:
            _clip_peft_delta_to_snapshot(
                state.model,
                state.trajectory_peft_snapshot,
                float(sampled_action.peft_max_episode_delta_norm),
            )
        peft_delta_norm = _snapshot_delta_norm(state.model, frame_peft_snapshot)
        if state.current_window_peft_snapshot is not None:
            window_peft_delta_norm = _snapshot_delta_norm(
                state.model,
                state.current_window_peft_snapshot,
            )
        if state.trajectory_peft_snapshot:
            trajectory_peft_delta_norm = _snapshot_delta_norm(
                state.model,
                state.trajectory_peft_snapshot,
            )
        if state.current_accumulator is not None:
            state.current_accumulator.action_delta_norm += float(peft_delta_norm)
    logits_after = refine_logits_with_tail(
        state.model,
        reward_eval_rx_iq,
        condition,
        region_ids,
        tail,
        modulation=sampled_action.modulation,
        passes=int(state.tail_refinement_passes),
        adapt_symbols=adapt_symbols,
        adapt_mask=adapt_mask,
    )
    after = logits_after.squeeze(0)
    reward_before_loss = _masked_bce(base, reward_labels, frame.reward_mask)
    reward_after_loss = _masked_bce(after, reward_labels, frame.reward_mask)
    reward_before_ber = bit_error_rate(base[frame.reward_mask], reward_labels[frame.reward_mask])
    reward_after_ber = bit_error_rate(after[frame.reward_mask], reward_labels[frame.reward_mask])
    peft_frame_accepted = True
    if sampled_action.peft_groups and peft_delta_norm > 0.0:
        peft_frame_accepted = should_accept_reward_guard(
            reward_loss_before=float(reward_before_loss.detach().cpu()),
            reward_loss_after=float(reward_after_loss.detach().cpu()),
            reward_ber_before=float(reward_before_ber),
            reward_ber_after=float(reward_after_ber),
            allow_loss_only=True,
            peft_delta_norm=float(peft_delta_norm),
        )
        if not peft_frame_accepted:
            # 先用当前帧 Reward Pilot 验收 PEFT，再让 Data 波形进入最终评估。
            # 失败时本帧立即恢复参数。
            _restore_snapshot(state.model, frame_peft_snapshot)
            after = base
    assert state.current_accumulator is not None
    state.current_accumulator.add_frame(
        reward_loss_before=float(reward_before_loss.detach().cpu()),
        reward_loss_after=float(reward_after_loss.detach().cpu()),
        reward_ber_before=float(reward_before_ber),
        reward_ber_after=float(reward_after_ber),
    )
    tail_len = state.receiver_state.soft_tail.numel()
    if state.cir_update_mode == "decision_directed":
        state_modulation = (
            applied_action.modulation
            if peft_frame_accepted
            else identity.modulation
        )
        adapt_state_logits = refine_logits_with_tail(
            state.model,
            adapt_only_rx_iq,
            condition,
            region_ids,
            tail,
            modulation=state_modulation,
            passes=int(state.tail_refinement_passes),
            adapt_symbols=adapt_symbols,
            adapt_mask=adapt_mask,
        ).squeeze(0)
        state.cir = decision_directed_cir_update(
            frame,
            adapt_state_logits.detach(),
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
        if window_reward.reward <= 0.0 and applied_action.name != "identity":
            if state.current_window_peft_snapshot is not None:
                _restore_snapshot(state.model, state.current_window_peft_snapshot)
            if state.current_window_cir_snapshot is not None:
                state.cir = state.current_window_cir_snapshot.to(state.cir.device).clone()
            # soft tail 是由 Pilot 可见输出逐帧维护的接收机状态；动作回滚只撤销
            # 本窗口的 PEFT/CIR 参数，不能把已经观测到的 Pilot 状态倒退。
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
    accepted = bool(peft_frame_accepted and not window_rollback)
    # 接收端 tail 只从 Adapt Pilot 产生；Reward Pilot 只参与动作后的验收。
    tail_visible_mask = adapt_only_mask[-tail_len:]
    selected_tail_alpha = (
        float(sampled_action.tail_alpha)
        if getattr(sampled_action, "tail_alpha", None) is not None
        else float(getattr(state, "tail_update_alpha", 1.0))
    )
    selected_tail_alpha = min(1.0, max(0.0, selected_tail_alpha))
    if bool(tail_visible_mask.any()):
        tail_modulation = (
            identity.modulation
            if window_rollback or not peft_frame_accepted
            else applied_action.modulation
        )
        adapt_tail_logits = refine_logits_with_tail(
            state.model,
            adapt_only_rx_iq,
            condition,
            region_ids,
            state.receiver_state.soft_tail.unsqueeze(0).to(torch.complex64),
            modulation=tail_modulation,
            passes=int(state.tail_refinement_passes),
            adapt_symbols=adapt_symbols,
            adapt_mask=adapt_mask,
        ).squeeze(0)
        detected_tail = torch.complex(
            torch.tanh(adapt_tail_logits[-tail_len:] / 2.0),
            torch.zeros_like(adapt_tail_logits[-tail_len:]),
        )
        previous_tail = state.receiver_state.soft_tail
        next_tail = (1.0 - selected_tail_alpha) * previous_tail + selected_tail_alpha * detected_tail
        state.receiver_state.update_tail(
            torch.where(tail_visible_mask, next_tail, previous_tail)
        )
    # 所有策略、PEFT 更新、Pilot reward 和窗口回滚完成后，才允许完整波形
    # 进入前向；以下两次输出只用于事后 Data 指标，不反馈给在线控制。
    full_rx_iq = torch.stack(
        (frame.rx_symbols.real, frame.rx_symbols.imag), dim=-1
    ).unsqueeze(0).float()
    current_peft_snapshot = (
        _snapshot_peft_groups(state.model, set(sampled_action.peft_groups))
        if sampled_action.peft_groups
        else {}
    )
    if sampled_action.peft_groups:
        _restore_snapshot(state.model, frame_peft_snapshot)
    data_evaluation_tail = state.receiver_state.soft_tail.unsqueeze(0).to(torch.complex64)
    with torch.no_grad():
        full_before_logits = refine_logits_with_tail(
            state.model,
            full_rx_iq,
            condition,
            region_ids,
            data_evaluation_tail,
            modulation=identity.modulation,
            passes=int(state.tail_refinement_passes),
            adapt_symbols=adapt_symbols,
            adapt_mask=adapt_mask,
        )
    if sampled_action.peft_groups:
        _restore_snapshot(state.model, current_peft_snapshot)
    final_modulation = identity.modulation if window_rollback else sampled_action.modulation
    with torch.no_grad():
        full_after_logits = refine_logits_with_tail(
            state.model,
            full_rx_iq,
            condition,
            region_ids,
            data_evaluation_tail,
            modulation=final_modulation,
            passes=int(state.tail_refinement_passes),
            adapt_symbols=adapt_symbols,
            adapt_mask=adapt_mask,
        )
    full_before = full_before_logits.squeeze(0)
    full_after = full_after_logits.squeeze(0)
    data_before_ber = bit_error_rate(
        full_before[frame.data_mask], frame.bits[frame.data_mask]
    )
    data_after_ber = bit_error_rate(
        full_after[frame.data_mask], frame.bits[frame.data_mask]
    )
    data_before_logits = full_before[frame.data_mask]
    data_after_logits = full_after[frame.data_mask]
    data_bce_before = _masked_bce(full_before, frame.bits, frame.data_mask)
    data_bce_after = _masked_bce(full_after, frame.bits, frame.data_mask)
    data_logits_abs_delta = torch.mean(torch.abs(data_after_logits - data_before_logits))
    data_logits_direction_change = (
        (data_before_logits >= 0.0) != (data_after_logits >= 0.0)
    ).to(torch.float32).mean()
    data_soft_output_delta = torch.mean(
        torch.abs(torch.sigmoid(data_after_logits) - torch.sigmoid(data_before_logits))
    )
    return {
        "method": "RL-Modulated Neural Block Equalizer",
        "snr_db": float(snr_db),
        "frame": int(frame_index),
        "ber_data": float(data_after_ber),
        # 这是本帧动作前的同一接收状态对照，不是独立 Frozen 方法结果。
        "pre_update_ber_data": float(data_before_ber),
        "ber_reward_pilot": float(reward_after_ber),
        "ber_adapt_pilot": bit_error_rate(after[frame.adapt_mask], adapt_labels[frame.adapt_mask]),
        "data_bce_before_eval_only": float(data_bce_before.detach().cpu()),
        "data_bce_after_eval_only": float(data_bce_after.detach().cpu()),
        "data_bce_improvement_eval_only": float(
            (data_bce_before - data_bce_after).detach().cpu()
        ),
        "data_logits_abs_delta_mean_eval_only": float(data_logits_abs_delta.detach().cpu()),
        "data_logits_direction_change_rate_eval_only": float(
            data_logits_direction_change.detach().cpu()
        ),
        "data_soft_output_delta_mean_eval_only": float(data_soft_output_delta.detach().cpu()),
        "reward": float(window_reward.reward if window_reward else 0.0),
        "reward_pilot_loss_before": float(reward_before_loss.detach().cpu()),
        "reward_pilot_loss_after": float(reward_after_loss.detach().cpu()),
        "reward_pilot_ber_before": float(reward_before_ber),
        "reward_pilot_ber_after": float(reward_after_ber),
        "policy_learning": "windowed_discrete_safe_ppo",
        "cir_update_mode": str(state.cir_update_mode),
        "cir_update_alpha": float(getattr(state, "cir_update_alpha", 0.0)),
        "selected_cir_update_alpha": float(selected_cir_alpha),
        "selected_tail_update_alpha": float(selected_tail_alpha),
        "cir_update_uses_data_labels": False,
        "receiver_action_type": _receiver_action_type(sampled_action),
        "policy_loss": policy_loss,
        "policy_action_name": sampled_action.name,
        "applied_action_name": applied_action.name,
        "safe_action_accepted": bool(accepted),
        "peft_frame_accepted": bool(peft_frame_accepted),
        "window_reward_rollback": bool(window_rollback),
        "policy_window_closed": bool(window_closed),
        "window_size": int(state.window_size),
        "action_delta_norm": float(sampled_action.delta_norm),
        "modulation_delta_norm": float(applied_action.delta_norm),
        "peft_delta_norm": float(peft_delta_norm),
        "peft_max_delta_norm": (
            None if sampled_action.peft_max_delta_norm is None
            else float(sampled_action.peft_max_delta_norm)
        ),
        "peft_max_episode_delta_norm": (
            None if sampled_action.peft_max_episode_delta_norm is None
            else float(sampled_action.peft_max_episode_delta_norm)
        ),
        "trajectory_peft_delta_norm_before_reward": float(trajectory_peft_delta_norm),
        "window_peft_delta_norm_before_reward": float(window_peft_delta_norm),
        "peft_objective": str(sampled_action.peft_objective),
        "data_labels_used_online": False,
        "data_waveform_used_online": False,
        "policy_observation_source": "adapt_pilot_only",
        "reward_pilot_waveform_used_for_policy_observation": False,
        "pilot_state_embedding_used": bool(
            getattr(state.encoder, "include_pilot_state", False)
        ),
        "pilot_state_embedding": (
            None
            if state.current_pilot_state_embedding is None
            else state.current_pilot_state_embedding.detach().cpu().tolist()
        ),
        "policy_observation_fields": list(getattr(state.encoder, "FIELDS", ())),
        "online_waveform_visibility": "adapt_pilot_for_update_reward_pilot_for_guard",
        "adapt_params": int(state.policy.parameter_count() if policy_loss is not None else 0),
        "adapt_steps": int(1 if policy_loss is not None else 0),
        "parameter_delta_norm": float(applied_action.delta_norm + peft_delta_norm),
    }


def _receiver_action_type(action: DiscreteSafeAction) -> str:
    if getattr(action, "tail_alpha", None) is not None:
        return "tail_alpha"
    if getattr(action, "cir_alpha", None) is not None:
        return "cir_alpha"
    if action.peft_groups:
        return "peft"
    if action.name == "rollback_identity":
        return "rollback"
    return "identity"


def refine_logits_with_tail(
    model: UnfoldedEqualizer,
    rx_iq: torch.Tensor,
    condition: CIRCondition,
    region_ids: torch.Tensor,
    soft_tail: torch.Tensor,
    *,
    modulation,
    passes: int,
    adapt_symbols: torch.Tensor | None = None,
    adapt_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """用当前帧无标签输出做有限次跨帧 soft-tail fixed-point refinement。"""

    logits, _ = _forward_model(
        model,
        rx_iq,
        condition,
        region_ids,
        soft_tail,
        modulation=modulation,
        adapt_symbols=adapt_symbols,
        adapt_mask=adapt_mask,
    )
    tail_length = int(soft_tail.shape[-1])
    for _ in range(max(0, int(passes))):
        tail_logits = logits[..., -tail_length:]
        refined_tail = torch.complex(
            torch.tanh(tail_logits / 2.0),
            torch.zeros_like(tail_logits),
        )
        logits, _ = _forward_model(
            model,
            rx_iq,
            condition,
            region_ids,
            refined_tail,
            modulation=modulation,
            adapt_symbols=adapt_symbols,
            adapt_mask=adapt_mask,
        )
    return logits


def _forward_model(
    model,
    rx_iq: torch.Tensor,
    condition: CIRCondition,
    region_ids: torch.Tensor,
    soft_tail: torch.Tensor,
    *,
    modulation,
    adapt_symbols: torch.Tensor | None,
    adapt_mask: torch.Tensor | None,
):
    """兼容旧诊断桩，并向新模型传递 Adapt Pilot 条件。"""

    model_config = getattr(model, "config", None)
    if bool(getattr(model_config, "pilot_conditioned", False)):
        return model(
            rx_iq,
            condition,
            region_ids,
            soft_tail,
            modulation=modulation,
            adapt_symbols=adapt_symbols,
            adapt_mask=adapt_mask,
        )
    return model(
        rx_iq,
        condition,
        region_ids,
        soft_tail,
        modulation=modulation,
    )


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
    objective: str = "bce",
    max_delta_norm: float | None = None,
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
    # CIR、状态置信度与 soft-tail 是已估计的在线输入；梯度只更新指定 PEFT 参数，
    # 不跨帧反传到接收机状态或复用已经释放的计算图。
    condition = replace(
        condition,
        complex_cir=condition.complex_cir.detach(),
        support_probability=condition.support_probability.detach(),
        noise_variance=condition.noise_variance.detach(),
        confidence=condition.confidence.detach(),
        latent_residual=condition.latent_residual.detach(),
    )
    tail = tail.detach()
    optimizer = torch.optim.AdamW(trainable, lr=float(lr))
    for _ in range(max(1, int(steps))):
        pilot_mask = frame.adapt_mask.to(device=frame.rx_symbols.device, dtype=torch.bool)
        pilot_rx = torch.where(pilot_mask, frame.rx_symbols, torch.zeros_like(frame.rx_symbols))
        device = pilot_rx.device
        pilot_view = _adapt_receiver_view(frame)
        if objective == "pilot_signal_reconstruction":
            if groups != {"phase_trend"}:
                raise ValueError(
                    "Pilot 信号重构目标只允许更新 phase_trend 参数组。"
                )
            loss = _pilot_signal_reconstruction_loss(
                model,
                condition,
                pilot_view.adapt_symbols.to(device),
                pilot_rx,
                frame.adapt_mask.to(device),
                tail,
            )
        elif objective == "bce":
            rx_iq = torch.stack((pilot_rx.real, pilot_rx.imag), dim=-1).unsqueeze(0).float()
            region_ids = frame.model_region_ids.unsqueeze(0).long()
            logits, _ = model(
                rx_iq,
                condition,
                region_ids,
                tail,
                adapt_symbols=pilot_view.adapt_symbols.to(device).unsqueeze(0).to(torch.complex64),
                adapt_mask=frame.adapt_mask.to(device).unsqueeze(0).bool(),
            )
            adapt_labels = (pilot_view.adapt_symbols.real > 0.0).to(frame.bits.dtype)
            loss = _masked_bce(logits.squeeze(0), adapt_labels, frame.adapt_mask)
        else:
            raise ValueError(f"不支持的在线 PEFT 目标：{objective!r}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
    if max_delta_norm is not None:
        _clip_peft_delta_to_snapshot(model, before, float(max_delta_norm))
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


def _snapshot_delta_norm(model: UnfoldedEqualizer, snapshot: dict[str, torch.Tensor]) -> float:
    """计算指定 PEFT 参数相对快照的总 L2 偏移。"""

    parameters = dict(model.named_parameters())
    if set(snapshot) - set(parameters):
        raise ValueError("PEFT 参数快照与当前模型不匹配。")
    delta_sq = torch.zeros((), dtype=torch.float32)
    for name, value in snapshot.items():
        delta_sq = delta_sq + (
            parameters[name].detach().float() - value.to(parameters[name].device).float()
        ).pow(2).sum().cpu()
    return float(torch.sqrt(delta_sq).item())


def _clip_peft_delta_to_snapshot(
    model: UnfoldedEqualizer,
    snapshot: dict[str, torch.Tensor],
    max_delta_norm: float,
) -> float:
    """把单步或窗口累计 PEFT 偏移投影到预设 L2 信任区域内。"""

    cap = float(max_delta_norm)
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("PEFT delta norm 上限必须为正有限数。")
    current_norm = _snapshot_delta_norm(model, snapshot)
    if current_norm > cap:
        scale = cap / max(current_norm, 1e-12)
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name, value in snapshot.items():
                parameter = parameters[name]
                reference = value.to(parameter.device, dtype=parameter.dtype)
                parameter.copy_(reference + scale * (parameter - reference))
        current_norm = _snapshot_delta_norm(model, snapshot)
    return current_norm


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
    state_dict = dict(state_dict)
    input_weight_name = "encoder.0.weight"
    target_state = policy.state_dict()
    stored_input_weight = state_dict.get(input_weight_name)
    target_input_weight = target_state.get(input_weight_name)
    if (
        stored_input_weight is not None
        and target_input_weight is not None
        and stored_input_weight.ndim == 2
        and target_input_weight.ndim == 2
        and stored_input_weight.shape[0] == target_input_weight.shape[0]
        and stored_input_weight.shape[1] < target_input_weight.shape[1]
    ):
        # 旧策略只认识基础 observation；复制原列，新增状态列从零权重开始，
        # 因此加载瞬间旧策略行为不变，在线 PPO 再学习新状态特征。
        padded_weight = torch.zeros_like(target_input_weight)
        padded_weight[:, : stored_input_weight.shape[1]] = stored_input_weight.to(
            device=padded_weight.device, dtype=padded_weight.dtype
        )
        state_dict[input_weight_name] = padded_weight
    try:
        policy.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        message = f"动作表不兼容，无法加载 policy checkpoint：{exc}"
        if required:
            raise ValueError(message) from exc
        return False
    return True


def _policy_view(
    frame,
    snr_db: float,
    logits: torch.Tensor,
    previous_reward: float,
    last_delta_norm: float,
    pilot_state_embedding: torch.Tensor | None = None,
):
    view = _adapt_receiver_view(frame)
    adapt_only_mask = view.adapt_mask.to(dtype=torch.bool)
    adapt_only_rx = torch.where(
        adapt_only_mask,
        view.rx_symbols,
        torch.zeros_like(view.rx_symbols),
    )
    return SimpleNamespace(
        # 当前动作只看 Adapt Pilot；Reward Pilot 在动作完成后用于验收。
        rx_symbols=adapt_only_rx,
        adapt_symbols=view.adapt_symbols,
        adapt_mask=view.adapt_mask,
        model_region_ids=view.model_region_ids,
        noise_variance=torch.tensor(10.0 ** (-float(snr_db) / 10.0), device=logits.device),
        confidence=torch.sigmoid(torch.abs(logits)).mean(),
        previous_reward=torch.tensor(previous_reward, device=logits.device),
        last_modulation_delta_norm=torch.tensor(last_delta_norm, device=logits.device),
        pilot_state_embedding=pilot_state_embedding,
    )


def _build_adapt_pilot_state_embedding(
    frame,
    cir: torch.Tensor,
    soft_tail: torch.Tensor,
    condition: CIRCondition,
    adapt_logits: torch.Tensor,
) -> torch.Tensor:
    """把当前 Adapt Pilot 的信道与 phase 诊断压成有界状态向量。"""

    latent = condition.latent_residual.reshape(-1)
    phase0, cfo = estimate_phase_residual_features(
        _adapt_receiver_view(frame), cir, soft_tail
    )
    candidate_cir = pilot_sparse_cir_update(
        frame,
        cir.detach(),
        soft_tail.detach(),
        max_paths=24,
        alpha=1.0,
        cfo_hint=cfo,
    ).to(cir.device)
    adapt_mask = frame.adapt_mask.to(device=adapt_logits.device, dtype=torch.bool)
    confidence = torch.sigmoid(torch.abs(adapt_logits[adapt_mask])).mean()
    residual_indices = torch.arange(3, min(latent.numel(), 16), 4, device=latent.device)
    reconstruction_error = (
        latent[residual_indices].mean()
        if residual_indices.numel() > 0
        else torch.zeros((), dtype=torch.float32, device=cir.device)
    )
    summary = build_pilot_state_summary(
        cir=candidate_cir,
        reference_cir=cir,
        phase0=phase0,
        cfo_cycles_per_symbol=cfo,
        noise_variance=condition.noise_variance,
        confidence=confidence,
        reconstruction_error=reconstruction_error,
    )
    return PilotStateEmbedding()(summary).to(device=adapt_logits.device)


def _adapt_receiver_view(frame):
    """返回严格只含 Adapt Pilot 的接收端视图。

    动作 observation、PEFT 梯度和接收状态递推只能读取此视图。
    """

    tx_symbols = getattr(frame, "tx_symbols", torch.zeros_like(frame.rx_symbols))
    adapt_mask = frame.adapt_mask.to(dtype=torch.bool)
    return SimpleNamespace(
        rx_symbols=torch.where(
            adapt_mask,
            frame.rx_symbols,
            torch.zeros_like(frame.rx_symbols),
        ),
        adapt_symbols=torch.where(
            adapt_mask,
            tx_symbols,
            torch.zeros_like(tx_symbols),
        ),
        adapt_mask=adapt_mask.clone(),
        model_region_ids=frame.model_region_ids.clone(),
    )


def _pilot_receiver_view(frame):
    """返回 Adapt/Reward Pilot 视图，只供动作后的 Reward 验收使用。

    真实环境使用 ``Frame.receiver_view``。轻量诊断桩可能只有基础字段，
    此时按相同的 mask 构造留出验收视图；Data 位置始终置零。
    """

    receiver_view = getattr(frame, "receiver_view", None)
    if callable(receiver_view):
        view = receiver_view()
        visible_mask = view.adapt_mask | (view.reward_symbols != 0)
        return replace(
            view,
            rx_symbols=torch.where(
                visible_mask,
                view.rx_symbols,
                torch.zeros_like(view.rx_symbols),
            ),
        )
    tx_symbols = getattr(frame, "tx_symbols", torch.zeros_like(frame.rx_symbols))
    adapt_mask = frame.adapt_mask.to(dtype=torch.bool)
    reward_mask = frame.reward_mask.to(dtype=torch.bool)
    return SimpleNamespace(
        rx_symbols=torch.where(
            adapt_mask | reward_mask,
            frame.rx_symbols,
            torch.zeros_like(frame.rx_symbols),
        ),
        adapt_symbols=torch.where(adapt_mask, tx_symbols, torch.zeros_like(tx_symbols)),
        adapt_mask=adapt_mask.clone(),
        reward_symbols=torch.where(reward_mask, tx_symbols, torch.zeros_like(tx_symbols)),
        model_region_ids=frame.model_region_ids.clone(),
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
