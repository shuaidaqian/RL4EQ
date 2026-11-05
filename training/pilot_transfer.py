# -*- coding: utf-8 -*-
"""真实信道上的离线 Pilot 更新后 Data 泛化训练。

内循环默认复现窗口 PPO 的 Adapt-only AdamW 更新；实验入口可显式选择
整帧接收波形，损失标签仍仅来自 Adapt。外循环仅在离线阶段
使用 Data 标签。临时 phase_trend 参数不写入基础 checkpoint，保证 Frozen
与在线方法仍能通过现有 strict-load 入口共享完全相同的基础模型。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call
from torch.nn.attention import SDPBackend, sdpa_kernel

from agent.cir_estimator import CIRCondition, condition_from_cir
from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from baseline.traditional_equalizers import estimate_acquisition_cir_with_cfo, estimate_phase_residual_vector
from env.comm_env import CommunicationEnvironment
from env.experiment_config import build_comm_env_config, validate_model_dimensions
from env.frame_structure import Frame
from training.checkpointing import save_checkpoint
from training.online_adaptation import _pilot_signal_reconstruction_loss, _frame_lookahead_model_kwargs
from training.windowed_discrete_ppo import _adapt_receiver_view


@dataclass(frozen=True)
class TransferContext:
    """条件和历史状态只由 acquisition 与 Adapt Pilot 生成。"""

    frame: Frame
    condition: CIRCondition
    tail: torch.Tensor
    # 仅供离线外层评分；模型与 Adapt 目标不会读取这些未来上下文。
    offline_post_contexts: tuple["TransferContext", ...] = ()
    # 仅用于离线复现共同状态内循环；不把历史 Data/Reward 标签带入更新。
    offline_adapt_contexts: tuple["TransferContext", ...] = ()
    # 当前接收上下文使用的 acquisition CFO 估计；不含仿真真实 CFO。
    cfo: float = 0.0


def receiver_inputs(settings):
    """区分接收波形范围与监督标签范围；历史配置保持原输入协议。"""
    inner = settings.get("inner_receiver_input", "adapt_only")
    reward = settings.get("reward_receiver_input", "pilot_only")
    if inner not in {"adapt_only", "full_rx"} or reward not in {"pilot_only", "full_rx"}:
        raise ValueError("不支持的内循环或 Reward 接收输入协议。")
    if inner == "full_rx" and settings.get("inner_objective", "pilot_signal_reconstruction") not in {"bce", "channel_state_signal", "channel_state_map", "channel_state_shared_map"}:
        raise ValueError("整帧接收输入只支持 Adapt BCE 或神经信道状态信号目标。")
    return inner, reward


def receiver_tasks(settings):
    """任务描述真实观测窗；接收输入描述该窗内是否还要遮罩波形。"""
    inner = settings.get("inner_task", "full_frame")
    reward = settings.get("reward_task", "full_frame")
    if inner not in {"full_frame", "prefix_boundary"} or reward not in {"full_frame", "prefix_boundary"}:
        raise ValueError("不支持的内循环或 Reward 接收任务。")
    inputs = receiver_inputs(settings)
    if (inner == "prefix_boundary" and inputs[0] != "full_rx") or (reward == "prefix_boundary" and inputs[1] != "full_rx"):
        raise ValueError("真实短窗任务必须使用窗内 full_rx 输入，不能附加零填充遮罩。")
    return inner, reward


def prefix_context(context, end):
    """同步缩短逐符号字段；物理算子由实际输入长度构造。"""
    frame = context.frame
    if end <= context.tail.numel() or end > frame.rx_symbols.numel():
        raise ValueError("短窗必须长于信道记忆且不能超出已接收帧。")
    fields = ("bits", "tx_symbols", "rx_symbols", "adapt_mask", "reward_mask", "data_mask", "model_region_ids")
    short = replace(frame, **{name: getattr(frame, name)[:end] for name in fields},
                    tail_symbols=None, true_cir=None, adapt_block_lengths=[int(frame.adapt_mask[:end].sum())],
                    lookahead_rx_symbols=None, lookahead_adapt_symbols=None, lookahead_frame_index=None)
    return replace(context, frame=short)


def task_context(context, task, *, role):
    """截窗只依据已知 Pilot 位置，评分起点排除前记忆区。"""
    if task == "full_frame":
        return context, 0
    if task != "prefix_boundary" or role not in {"adapt", "reward"}:
        raise ValueError("不支持的接收任务或 Pilot 角色。")
    frame = context.frame
    mask = frame.adapt_mask if role == "adapt" else frame.reward_mask
    positions = torch.nonzero(mask, as_tuple=False).flatten()
    if not positions.numel():
        raise ValueError("短窗任务缺少目标 Pilot。")
    end = int(positions.max()) + 1
    if bool(frame.data_mask[:end].any()) or (role == "adapt" and bool(frame.reward_mask[:end].any())):
        raise ValueError("短窗任务要求 Adapt/Reward 按前缀顺序放置。")
    return prefix_context(context, end), context.tail.numel() if role == "adapt" else 0


def frame_to_device(frame: Frame, device: torch.device | str) -> Frame:
    """统一移动所有张量字段，不把隐藏标签传入模型。"""

    from dataclasses import fields, replace

    return replace(frame, **{
        field.name: getattr(frame, field.name).to(device)
        for field in fields(frame) if isinstance(getattr(frame, field.name), torch.Tensor)
    })


def context_logits(model, context: TransferContext, parameters=None, *, visible="data"):
    """显式选择接收输入；模型只获得 Adapt 已知符号，不获得 Data 标签。"""

    frame = context.frame
    view = _adapt_receiver_view(frame)
    if visible == "adapt":
        rx = view.rx_symbols
    elif visible == "pilot":
        mask = frame.adapt_mask | frame.reward_mask
        rx = torch.where(mask, frame.rx_symbols, torch.zeros_like(frame.rx_symbols))
    elif visible in {"data", "full_rx"}:
        rx = frame.rx_symbols
    else:
        raise ValueError("visible 必须为 adapt、pilot、full_rx 或兼容名称 data。")
    args = (
        torch.stack((rx.real, rx.imag), dim=-1).unsqueeze(0).float(),
        context.condition, frame.model_region_ids.unsqueeze(0).long(), context.tail,
    )
    kwargs = {
        "adapt_symbols": view.adapt_symbols.unsqueeze(0),
        "adapt_mask": frame.adapt_mask.unsqueeze(0),
        **_frame_lookahead_model_kwargs(frame),
    }
    if model.config.enable_channel_state_branch:
        kwargs["frame_index"] = frame.frame_index
    if parameters is None:
        logits, _ = model(*args, **kwargs)
    else:
        logits, _ = functional_call(model, parameters, args, kwargs)
    return logits.squeeze(0)


class ChannelStateSignalObjective(nn.Module):
    """Adapt 构造神经状态，Reward 仅可作为动作后的独立信号验收目标。"""

    def __init__(self, model, role="adapt", score_start=0, include_prior=False):
        super().__init__()
        if role not in {"adapt", "reward"}:
            raise ValueError("信道信号损失角色必须为 Adapt 或 Reward。")
        self.model, self.role, self.score_start = model, role, int(score_start)
        if include_prior and role != "adapt":
            raise ValueError("先验惩罚只属于 Adapt 更新目标，不作为 Reward 标签拟合。")
        self.include_prior = bool(include_prior)

    def forward(self, context):
        frame = context.frame
        adapt = _adapt_receiver_view(frame)
        known_mask = frame.adapt_mask.clone()
        symbols = adapt.adapt_symbols
        score_mask = known_mask.clone()
        if self.role == "reward":
            known_mask = known_mask | frame.reward_mask
            symbols = symbols + frame.receiver_view().reward_symbols
            score_mask = frame.reward_mask
        score_mask = score_mask & (torch.arange(score_mask.numel(), device=score_mask.device) >= self.score_start)
        rx = torch.where(known_mask, frame.rx_symbols, torch.zeros_like(frame.rx_symbols))
        iq = torch.stack((rx.real, rx.imag), dim=-1).unsqueeze(0).float()
        return self.model.channel_state_signal_loss(iq, context.condition,
            adapt.adapt_symbols.unsqueeze(0), frame.adapt_mask.unsqueeze(0),
            symbols.unsqueeze(0), known_mask.unsqueeze(0), score_mask.unsqueeze(0), frame.frame_index,
            include_prior=self.include_prior)


def channel_state_reward_loss(model, context, parameters=None):
    objective = ChannelStateSignalObjective(model, role="reward")
    if parameters is None:
        return objective(context)
    return functional_call(objective, {"model." + n: p for n, p in parameters.items()}, (context,))


class _AdaptObjective(nn.Module):
    """通过 functional_call 对相位信号重构目标替换候选参数。"""

    def __init__(self, model, score_start=0, objective="pilot_signal_reconstruction", receiver_input="adapt_only"):
        super().__init__()
        self.model = model
        self.score_start = int(score_start)
        self.objective = objective
        if receiver_input not in {"adapt_only", "full_rx"}:
            raise ValueError("接收输入必须为 adapt_only 或 full_rx。")
        if objective not in {"bce", "channel_state_signal", "channel_state_map"} and receiver_input != "adapt_only":
            raise ValueError("整帧接收输入只支持 Adapt BCE 或神经信道状态信号目标。")
        self.receiver_input = receiver_input

    def forward(self, context):
        if self.objective in {"channel_state_signal", "channel_state_map"}:
            return ChannelStateSignalObjective(self.model, score_start=self.score_start,
                include_prior=self.objective == "channel_state_map")(context)
        view = _adapt_receiver_view(context.frame)
        positions = torch.arange(view.adapt_mask.numel(), device=view.adapt_mask.device)
        score_mask = view.adapt_mask & (positions >= self.score_start)
        if not bool(score_mask.any()):
            raise ValueError("信号重构评分窗没有有效 Adapt Pilot。")
        if self.objective == "bce":
            logits = context_logits(self.model, context,
                                    visible="full_rx" if self.receiver_input == "full_rx" else "adapt")
            labels = (view.adapt_symbols.real > 0).float()
            return F.binary_cross_entropy_with_logits(logits[score_mask].float(), labels[score_mask])
        return _pilot_signal_reconstruction_loss(
            self.model, context.condition, view.adapt_symbols, view.rx_symbols,
            score_mask,
            (torch.zeros_like(context.tail) if self.score_start >= self.model.config.max_delay
             else context.tail.detach()),
        )


def differentiable_adapt(model, contexts, *, learning_rate, steps, max_delta_norm,
                         create_graph=False, initial_parameters=None, score_start=0,
                         groups=None, objective="pilot_signal_reconstruction", receiver_input="adapt_only",
                         optimizer_state=None, return_optimizer_state=False):
    """复现 AdamW、梯度裁剪和投影，损失不读取 Reward/Data 标签。

    外层训练保留更新算子的计算图；在线/诊断关闭高阶图。未使用的状态映射
    参数保持原值，与无 state embedding 的正式动作语义一致。
    """

    if not contexts or steps < 1:
        raise ValueError("内循环需要非空上下文和正更新步数。")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("内循环学习率必须为正有限数。")
    if not math.isfinite(max_delta_norm) or max_delta_norm <= 0:
        raise ValueError("更新半径必须为正有限数。")
    selected_groups = {"phase_trend"} if groups is None else set(groups)
    if objective not in {"pilot_signal_reconstruction", "bce", "channel_state_signal", "channel_state_map", "channel_state_shared_map"}:
        raise ValueError("内循环目标必须为 pilot_signal_reconstruction 或 bce。")
    if receiver_input not in {"adapt_only", "full_rx"}:
        raise ValueError("接收输入必须为 adapt_only 或 full_rx。")
    if objective not in {"bce", "channel_state_signal", "channel_state_map", "channel_state_shared_map"} and receiver_input != "adapt_only":
        raise ValueError("整帧接收输入只支持 Adapt BCE 或神经信道状态信号目标。")
    if objective == "pilot_signal_reconstruction" and selected_groups != {"phase_trend"}:
        raise ValueError("相位信号重构只允许更新 phase_trend 参数组。")
    if objective in {"channel_state_signal", "channel_state_map", "channel_state_shared_map"}:
        allowed = ({"channel_state_head"}, {"channel_state_memory"})
        if (selected_groups not in allowed or model.channel_state is None or
                (selected_groups == {"channel_state_memory"} and model.channel_state.memory is None)):
            raise ValueError("信道状态信号重构只允许更新已启用的神经 channel_state_head 或 channel_state_memory。")
    parameters = dict(model.named_parameters())
    if initial_parameters is not None:
        parameters.update(initial_parameters)
    names = [name for name, _ in model.peft.named_group_parameters(selected_groups)]
    if not names:
        raise ValueError("指定参数组没有可更新的 PEFT 参数。")
    fast = dict(parameters)
    for name in names:
        if not fast[name].requires_grad:
            fast[name] = fast[name].detach().clone().requires_grad_(True)
    before = {name: fast[name] for name in names}
    optimizer_state = optimizer_state or {}
    step_offset = int(optimizer_state.get("steps", 0))
    moments = {name: value for name, value in optimizer_state.get("moments", {}).items()}
    variances = {name: value for name, value in optimizer_state.get("variances", {}).items()}
    if objective == "channel_state_shared_map":
        from training.shared_pilot_state import SharedNeuralPilotObjective
        if selected_groups != {"channel_state_memory"}:
            raise ValueError("共同状态只支持受限神经复数记忆内循环。")
        objective_module = SharedNeuralPilotObjective(model, contexts)
    else:
        objective_module = _AdaptObjective(model, score_start, objective, receiver_input)
    losses = []
    for step in range(1, int(steps) + 1):
        if objective == "channel_state_shared_map":
            loss = functional_call(objective_module, {"model." + name: value for name, value in fast.items()}, ())
        else:
            loss = torch.stack([
                functional_call(objective_module, {"model." + name: value for name, value in fast.items()}, (context,))
                for context in contexts
            ]).mean()
        gradients = torch.autograd.grad(
            loss, [fast[name] for name in names], create_graph=create_graph, allow_unused=True,
        )
        used = [(name, grad) for name, grad in zip(names, gradients) if grad is not None]
        norm = torch.sqrt(sum(grad.float().square().sum() for _, grad in used) + 1e-24)
        clip = (1.0 / (norm + 1e-6)).clamp(max=1.0)
        for name, grad in used:
            grad = grad * clip
            moments[name] = 0.9 * moments.get(name, torch.zeros_like(grad)) + 0.1 * grad
            variances[name] = 0.999 * variances.get(name, torch.zeros_like(grad)) + 0.001 * grad.square()
            optimizer_step = step_offset + step
            corrected_mean = moments[name] / (1.0 - 0.9 ** optimizer_step)
            corrected_variance = variances[name] / (1.0 - 0.999 ** optimizer_step)
            # 避免零方差点的高阶导数出现 0 * inf；不影响浮点精度下的 AdamW 一致性。
            denominator = corrected_variance.clamp_min(1e-24).sqrt() + 1e-8
            fast[name] = fast[name] * (1.0 - learning_rate * 0.01) - learning_rate * corrected_mean / denominator
        losses.append(float(loss.detach().cpu()))
    norm = torch.sqrt(sum((fast[name] - before[name]).float().square().sum() for name in names) + 1e-24)
    scale = (max_delta_norm / norm).clamp(max=1.0)
    for name in names:
        fast[name] = before[name] + scale * (fast[name] - before[name])
    result = {
        "adapt_losses": losses,
        "delta_norm": float(norm.detach().clamp(max=max_delta_norm).cpu()),
    }
    if return_optimizer_state:
        result["_optimizer_state"] = {
            "steps": step_offset + int(steps),
            "moments": moments,
            "variances": variances,
        }
    return fast, result


def make_episode(config, model, *, seed, snr, frames, state_split, device):
    """真实 Level B acquisition；禁止真实 CIR 或真实 tail 注入接收机。"""

    env_config = build_comm_env_config(
        config, level="B", snr_db=float(snr), seed=int(seed),
        max_delay=model.config.max_delay, total_pilot=int(config["pilot_total"]),
        reward_pilot_total=int(config["reward_pilot_total"]), pilot_layout="prefix", state_split=state_split,
    )
    validate_model_dimensions(model.config, env_config)
    env = CommunicationEnvironment(env_config)
    start = env.reset_episode()
    cir, cfo = estimate_acquisition_cir_with_cfo(
        start.acquisition, model.config.max_delay,
        cfo_limit=float(config.get("acquisition_cfo_limit", 0.004)),
    )
    return env, cir.to(device), float(cfo), start.initial_soft_tail.to(device), [
        frame_to_device(env.next_frame(lookahead_len=int(config.get("receiver_lookahead_len", 0))), device)
        for _ in range(frames)
    ]


def make_context(frame, cir, cfo, tail, snr):
    view = _adapt_receiver_view(frame)
    phase = estimate_phase_residual_vector(view, cir, tail, blocks=4, cfo_hint=cfo)
    condition = condition_from_cir(cir, float(snr), cfo_residual=cfo, phase_features=phase)
    return TransferContext(frame, condition, tail.detach(), cfo=float(cfo))


def next_pilot_tail(model, context, *, alpha, parameters=None):
    """历史递推只观察 Adapt Pilot，与正式 PPO 的信息边界一致。"""

    visible = context.frame.adapt_mask[-context.tail.numel():].bool()
    if not bool(visible.any()):
        return context.tail.detach().clone()
    with torch.no_grad():
        logits = context_logits(model, context, parameters, visible="adapt")
        values = torch.tanh(logits[-context.tail.numel():] / 2.0)
        detected = torch.complex(values, torch.zeros_like(values))
        updated = (1.0 - alpha) * context.tail + alpha * detected
        return torch.where(visible, updated, context.tail).detach()


def differentiable_online_trajectory(model, contexts, *, learning_rate, steps,
                                     max_delta_norm, create_graph, groups,
                                     objective, receiver_input, reward_receiver_input,
                                     reward_task, reward_objective, snr_db,
                                     inner_task="full_frame", tail_alpha=0.5,
                                     soft_gate_rule="conservative_and",
                                     persist_optimizer_state=False,
                                     update_rule="adam",
                                     rank1_basis=None,
                                     rank1_damping=0.01,
                                     rank1_step_scale=1.0,
                                     history_frames=1, initial_history=(),
                                     trend_temperature=0.01,
                                     soft_gate_bce_temperature=0.01,
                                     soft_gate_ber_temperature=0.02):
    """按在线顺序展开多帧Adapt更新、Reward守门、参数和soft-tail持久状态。"""

    if len(contexts) < 2:
        raise ValueError("持续轨迹训练至少需要两个连续数据帧。")
    if any(contexts[index + 1].frame.frame_index != contexts[index].frame.frame_index + 1
           for index in range(len(contexts) - 1)):
        raise ValueError("持续轨迹上下文必须来自连续帧，才能计算帧龄趋势。")
    reference_cir = contexts[0].condition.complex_cir.detach()
    reference_cfo = float(contexts[0].cfo)
    if any(not torch.equal(item.condition.complex_cir.detach(), reference_cir)
           or float(item.cfo) != reference_cfo for item in contexts[1:]):
        raise ValueError("持续轨迹必须共享同一会话的 acquisition CIR 与 CFO 估计。")
    if reward_objective != "bce":
        raise ValueError("持续轨迹内循环首版仅支持pilot BCE Reward守门。")
    if soft_gate_rule not in {"conservative_and", "hard_guard_or"}:
        raise ValueError("不支持的持续轨迹门控梯度代理。")
    if min(trend_temperature, soft_gate_bce_temperature, soft_gate_ber_temperature) <= 0:
        raise ValueError("持续轨迹训练的平滑温度必须为正数。")
    if not 0.0 <= tail_alpha <= 1.0:
        raise ValueError("soft-tail 更新系数必须位于[0, 1]。")
    if int(history_frames) < 1:
        raise ValueError("Adapt历史长度必须为正整数。")
    if update_rule not in {"adam", "rank1_gn"}:
        raise ValueError("trajectory update_rule 必须为 adam 或 rank1_gn。")
    selected_groups = set(groups)
    named = dict(model.named_parameters())
    names = [name for name, _ in model.peft.named_group_parameters(selected_groups)]
    if not names:
        raise ValueError("持续轨迹指定的PEFT参数组为空。")
    if update_rule == "rank1_gn":
        if selected_groups != {"head"} or rank1_basis is None:
            raise ValueError("rank1_gn 仅支持 head 参数组且必须提供可学习basis。")
        if objective != "bce" or receiver_input != "adapt_only":
            raise ValueError("rank1_gn候选仅支持Adapt-only BCE。")
        if persist_optimizer_state:
            raise ValueError("rank1_gn 是无优化器状态的标量GN更新。")
        if rank1_damping <= 0:
            raise ValueError("rank1_gn damping 必须为正数。")
    origin = {name: named[name] for name in names}
    fast = dict(named)
    rank1_direction = (rank1_basis / rank1_basis.norm().clamp_min(1e-12)
                       if update_rule == "rank1_gn" else None)
    rank1_alpha = torch.zeros((), device=rank1_basis.device, dtype=rank1_basis.dtype,
                              requires_grad=create_graph) if update_rule == "rank1_gn" else None
    def rank1_parameters_at(alpha):
        vector = torch.cat([origin[name].reshape(-1) for name in names]) + rank1_direction * alpha
        updated = dict(named)
        offset = 0
        for name in names:
            count = origin[name].numel()
            updated[name] = vector[offset:offset + count].reshape_as(origin[name])
            offset += count
        return updated
    cir = reference_cir
    cfo = reference_cfo
    tail = contexts[0].tail.detach()
    active_optimizer_state = None
    history = list(initial_history)[-(int(history_frames) - 1):] if int(history_frames) > 1 else []
    if len(history) > int(history_frames) - 1:
        raise ValueError("初始Adapt历史长度超过配置窗口。")
    if history:
        history_contexts = history + [contexts[0]]
        if (history[-1].frame.frame_index + 1 != contexts[0].frame.frame_index
                or any(history_contexts[index + 1].frame.frame_index != history_contexts[index].frame.frame_index + 1
                       for index in range(len(history_contexts) - 1))):
            raise ValueError("初始Adapt历史必须连续并紧邻持续轨迹起点。")
        if any(not torch.equal(item.condition.complex_cir.detach(), reference_cir)
               or float(item.cfo) != reference_cfo for item in history):
            raise ValueError("初始Adapt历史必须来自同一acquisition CIR/CFO会话。")
    base_all_losses, frozen_data_losses, online_data_losses = [], [], []
    step_records = []
    from training.windowed_discrete_ppo import should_accept_reward_guard

    for index, raw_context in enumerate(contexts):
        context = make_context(raw_context.frame, cir, cfo, tail, float(snr_db))
        frame = context.frame
        frozen_logits = context_logits(model, context, named)
        base_all_losses.append(F.binary_cross_entropy_with_logits(
            frozen_logits.float(), frame.bits.float()))
        frozen_data_losses.append(F.binary_cross_entropy_with_logits(
            frozen_logits[frame.data_mask].float(), frame.bits[frame.data_mask].float()))

        inner_context, score_start = task_context(context, inner_task, role="adapt")
        update_contexts = [task_context(item, inner_task, role="adapt")[0] for item in history] + [inner_context]
        if update_rule == "rank1_gn":
            before = rank1_parameters_at(rank1_alpha)
            candidate, candidate_alpha, update = rank1_gauss_newton_adapt(
                model, update_contexts, parameters=named, names=names, basis=rank1_basis,
                coefficient=rank1_alpha, steps=steps, learning_rate=rank1_step_scale,
                damping=rank1_damping, max_step_norm=max_delta_norm,
                max_total_norm=max_delta_norm, create_graph=create_graph)
            candidate_optimizer_state = None
        else:
            before = fast
            candidate, update = differentiable_adapt(
                model, update_contexts, learning_rate=learning_rate, steps=steps,
                max_delta_norm=max_delta_norm, create_graph=create_graph,
                initial_parameters=before, groups=selected_groups,
                objective=objective, receiver_input=receiver_input, score_start=score_start,
                optimizer_state=active_optimizer_state if persist_optimizer_state else None,
                return_optimizer_state=persist_optimizer_state)
            candidate_alpha = None
            candidate_optimizer_state = update.pop("_optimizer_state", None)
        step_delta = torch.sqrt(sum((candidate[name] - before[name]).float().square().sum()
                                    for name in names) + 1e-24)
        cumulative = torch.sqrt(sum((candidate[name] - origin[name]).float().square().sum()
                                    for name in names) + 1e-24)
        total_scale = (max_delta_norm / cumulative).clamp(max=1.0)
        for name in names:
            candidate[name] = origin[name] + (candidate[name] - origin[name]) * total_scale
        if update_rule == "rank1_gn":
            projected_delta = torch.cat([(candidate[name] - origin[name]).reshape(-1) for name in names])
            # 以真正通过总半径投影的神经参数重建坐标，保证alpha状态与fast参数完全一致。
            candidate_alpha = torch.dot(rank1_direction, projected_delta)
            candidate = rank1_parameters_at(candidate_alpha)
        step_delta = torch.sqrt(sum((candidate[name] - before[name]).float().square().sum()
                                    for name in names) + 1e-24)

        reward_context, _ = task_context(context, reward_task, role="reward")
        reward_visible = "full_rx" if reward_receiver_input == "full_rx" else "pilot"
        logits_before = context_logits(model, reward_context, before, visible=reward_visible)
        logits_candidate = context_logits(model, reward_context, candidate, visible=reward_visible)
        reward_frame = reward_context.frame
        reward_view = reward_frame.receiver_view()
        reward_mask = reward_frame.reward_mask.bool()
        reward_labels = (reward_view.reward_symbols.real > 0).to(logits_before.dtype)
        loss_before = F.binary_cross_entropy_with_logits(logits_before[reward_mask].float(),
                                                          reward_labels[reward_mask].float())
        loss_candidate = F.binary_cross_entropy_with_logits(logits_candidate[reward_mask].float(),
                                                             reward_labels[reward_mask].float())
        ber_before = (((logits_before[reward_mask] >= 0) != reward_labels[reward_mask].bool())
                      .float().mean())
        ber_candidate = (((logits_candidate[reward_mask] >= 0) != reward_labels[reward_mask].bool())
                         .float().mean())
        hard_accept = bool(should_accept_reward_guard(
            reward_loss_before=float(loss_before.detach().cpu()),
            reward_loss_after=float(loss_candidate.detach().cpu()),
            reward_ber_before=float(ber_before.detach().cpu()),
            reward_ber_after=float(ber_candidate.detach().cpu()),
            allow_loss_only=True, peft_delta_norm=float(step_delta.detach().cpu())))

        signed = 2.0 * reward_labels[reward_mask] - 1.0
        soft_error_before = torch.sigmoid(-signed * logits_before[reward_mask] / soft_gate_ber_temperature).mean()
        soft_error_candidate = torch.sigmoid(-signed * logits_candidate[reward_mask] / soft_gate_ber_temperature).mean()
        soft_bce_accept = torch.sigmoid((loss_before - loss_candidate - 0.005 * step_delta)
                                        / soft_gate_bce_temperature)
        soft_ber_delta = soft_error_before - soft_error_candidate
        soft_ber_accept = torch.sigmoid(soft_ber_delta / soft_gate_ber_temperature)
        if soft_gate_rule == "conservative_and":
            # 首轮保守代理：只在两种平滑反馈都支持更新时传梯度。
            soft_accept = soft_bce_accept * soft_ber_accept
        else:
            # 线上规则是 BER 改善，或 BER 持平时 BCE 改善；用软 BER 平局概率近似第二分支。
            soft_ber_tie = torch.exp(-soft_ber_delta.abs() / soft_gate_ber_temperature)
            soft_accept = soft_ber_accept + (1.0 - soft_ber_accept) * soft_ber_tie * soft_bce_accept
        gate = soft_accept + (1.0 if hard_accept else 0.0) - soft_accept.detach()
        fast = dict(before)
        if update_rule == "rank1_gn":
            rank1_alpha = rank1_alpha + gate * (candidate_alpha - rank1_alpha)
            fast = rank1_parameters_at(rank1_alpha)
        else:
            for name in names:
                fast[name] = before[name] + gate * (candidate[name] - before[name])
        if persist_optimizer_state:
            previous_moments = {} if active_optimizer_state is None else active_optimizer_state["moments"]
            previous_variances = {} if active_optimizer_state is None else active_optimizer_state["variances"]
            active_optimizer_state = {
                "steps": ((0 if active_optimizer_state is None else active_optimizer_state["steps"])
                          + (int(steps) if hard_accept else 0)),
                "moments": {}, "variances": {},
            }
            for name, value in candidate_optimizer_state["moments"].items():
                old = previous_moments.get(name, torch.zeros_like(value))
                active_optimizer_state["moments"][name] = old + gate * (value - old)
            for name, value in candidate_optimizer_state["variances"].items():
                old = previous_variances.get(name, torch.zeros_like(value))
                active_optimizer_state["variances"][name] = old + gate * (value - old)

        online_logits = context_logits(model, context, fast)
        online_data_losses.append(F.binary_cross_entropy_with_logits(
            online_logits[frame.data_mask].float(), frame.bits[frame.data_mask].float()))
        tail = next_pilot_tail(model, context, alpha=tail_alpha, parameters=fast)
        if int(history_frames) > 1:
            from training.shared_pilot_state import adapt_only_history_context
            history.append(adapt_only_history_context(frame, context.condition, context.tail))
            history = history[-(int(history_frames) - 1):]
        step_records.append({
            "frame_index": int(frame.frame_index), "reward_hard_accepted": hard_accept,
            "reward_loss_before": float(loss_before.detach().cpu()),
            "reward_loss_candidate": float(loss_candidate.detach().cpu()),
            "reward_ber_before": float(ber_before.detach().cpu()),
            "reward_ber_candidate": float(ber_candidate.detach().cpu()),
            "step_delta_norm": float(step_delta.detach().clamp(max=max_delta_norm).cpu()),
            "cumulative_delta_norm": float(cumulative.detach().clamp(max=max_delta_norm).cpu()),
            "adapt_delta_norm": update["delta_norm"],
            "adapt_history_frame_indices": [item.frame.frame_index for item in update_contexts],
            "soft_gate_accept_probability": float(soft_accept.detach().cpu()),
            "soft_gate_rule": soft_gate_rule,
            "update_rule": update_rule,
            "rank1_coefficient": (float(rank1_alpha.detach().cpu()) if update_rule == "rank1_gn" else None),
            "optimizer_state_steps": 0 if active_optimizer_state is None else active_optimizer_state["steps"],
            "data_labels_used_for_transition": False,
        })

    frozen_data = torch.stack(frozen_data_losses)
    online_data = torch.stack(online_data_losses)
    gains = frozen_data - online_data
    split = max(1, len(gains) // 2)
    trend_delta = gains[-split:].mean() - gains[:split].mean()
    trend_loss = trend_temperature * F.softplus(-trend_delta / trend_temperature)
    return fast, {
        "base_all_loss": torch.stack(base_all_losses).mean(),
        "frozen_data_loss": frozen_data.mean(),
        "online_data_loss": online_data.mean(),
        "frozen_data_windows": [frozen_data[:split].mean(), frozen_data[-split:].mean()],
        "online_data_windows": [online_data[:split].mean(), online_data[-split:].mean()],
        "gain_windows": [gains[:split].mean(), gains[-split:].mean()],
        "trend_delta": trend_delta, "trend_loss": trend_loss,
        "steps": step_records,
    }


def rank1_gauss_newton_adapt(model, contexts, *, parameters, names, basis, coefficient,
                             steps, learning_rate, damping, max_step_norm,
                             max_total_norm, create_graph=True):
    """在单位范数神经参数方向上做一维 Pilot logistic-GN 更新。

    参数和候选仅依赖 Adapt Pilot。外层可对 basis 反传；Data 标签不进入此函数。
    """
    if not contexts or not names or steps < 1:
        raise ValueError("rank-1 GN 需要非空 Adapt 上下文、参数名和正步数。")
    if damping <= 0 or learning_rate <= 0 or max_step_norm <= 0 or max_total_norm <= 0:
        raise ValueError("rank-1 GN 阻尼、步长和半径必须为正数。")
    origin = torch.cat([parameters[name].reshape(-1) for name in names])
    if basis.ndim != 1 or basis.numel() != origin.numel():
        raise ValueError("rank-1 basis 维度必须等于指定神经参数总维度。")
    direction = basis / basis.norm().clamp_min(1e-12)
    alpha = coefficient

    def logits_at(value):
        vector = origin + direction * value
        updated = dict(parameters)
        offset = 0
        for name in names:
            count = parameters[name].numel()
            updated[name] = vector[offset:offset + count].reshape_as(parameters[name])
            offset += count
        outputs = []
        for context in contexts:
            frame = context.frame
            outputs.append(context_logits(model, context, updated, visible="adapt")[frame.adapt_mask])
        return torch.cat(outputs)

    labels = torch.cat([
        (context.frame.receiver_view().adapt_symbols.real > 0)[context.frame.adapt_mask].to(origin.dtype)
        for context in contexts
    ])
    for _ in range(int(steps)):
        logits, jacobian = torch.autograd.functional.jvp(
            logits_at, (alpha,), (torch.ones_like(alpha),), create_graph=create_graph,
        )
        probability = torch.sigmoid(logits)
        gradient = ((probability - labels) * jacobian).mean()
        fisher = (probability * (1.0 - probability) * jacobian.square()).mean()
        delta = -float(learning_rate) * gradient / (fisher + float(damping))
        delta = delta.clamp(min=-float(max_step_norm), max=float(max_step_norm))
        alpha = alpha + delta
        alpha = alpha.clamp(min=-float(max_total_norm), max=float(max_total_norm))

    vector = origin + direction * alpha
    updated = dict(parameters)
    offset = 0
    for name in names:
        count = parameters[name].numel()
        updated[name] = vector[offset:offset + count].reshape_as(parameters[name])
        offset += count
    return updated, alpha, {
        "delta_norm": float((alpha - coefficient).detach().abs().cpu()),
        "coefficient": alpha,
        "data_labels_used": False,
    }


class PilotTransferTrainer:
    """同一真实信道采样下比较普通监督与更新后 Data 外层训练。"""

    def __init__(self, config, device, *, objective=None):
        self.config = config
        self.settings = dict(config["pilot_transfer"])
        offsets = self.settings.get("post_target_frame_offsets", [0])
        if (not isinstance(offsets, list) or not offsets or offsets[0] != 0
                or any(type(value) is not int or value < 0 for value in offsets)
                or offsets != sorted(set(offsets))):
            raise ValueError("离线外层帧偏移必须为从零开始的递增非负整数列表。")
        self.post_target_frame_offsets = offsets
        self.inner_receiver_input, self.reward_receiver_input = receiver_inputs(self.settings)
        self.inner_history_frames = int(self.settings.get("inner_history_frames", 1))
        if self.inner_history_frames < 1:
            raise ValueError("离线Adapt历史长度必须为正整数。")
        self.inner_task, self.reward_task = receiver_tasks(self.settings)
        if self.settings.get("inner_objective") == "channel_state_shared_map" and self.inner_task != "full_frame":
            raise ValueError("共同状态内循环要求完整帧的合法 Adapt 方程。")
        self.objective = objective or self.settings.get("objective", "post_adapt")
        self.trajectory_update_rule = self.settings.get("trajectory_update_rule", "adam")
        if self.trajectory_update_rule not in {"adam", "rank1_gn"}:
            raise ValueError("trajectory_update_rule 必须为 adam 或 rank1_gn。")
        if self.objective not in {"baseline", "post_adapt", "trajectory"}:
            raise ValueError("训练目标必须为 baseline、post_adapt 或 trajectory。")
        if self.objective == "trajectory" and (
                len(self.post_target_frame_offsets) < 2
                or self.post_target_frame_offsets != list(range(len(self.post_target_frame_offsets)))):
            raise ValueError("trajectory 目标要求连续帧偏移 [0, 1, ..., T-1]，且至少两帧。")
        if self.settings.get("trajectory_persist_optimizer_state", False) and self.objective != "trajectory":
            raise ValueError("跨帧优化器状态只允许用于 trajectory 外层目标。")
        if self.trajectory_update_rule == "rank1_gn":
            if self.objective != "trajectory":
                raise ValueError("rank1_gn 只用于多帧 trajectory 外层目标。")
            if (set(self.settings.get("inner_groups", ["phase_trend"])) != {"head"}
                    or self.settings.get("inner_objective", "pilot_signal_reconstruction") != "bce"
                    or self.inner_receiver_input != "adapt_only"
                    or self.settings.get("trajectory_persist_optimizer_state", False)):
                raise ValueError("rank1_gn 要求 head、Adapt-only BCE，且不持久化优化器状态。")
        if config.get("pilot_layout") != "prefix" or config.get("main_level") != "B":
            raise ValueError("适配训练只支持 Level B 前缀 Pilot。")
        if config.get("offline_state_split") != "offline_train":
            raise ValueError("训练必须使用 offline_train，不得使用 heldout_edge。")
        torch.manual_seed(int(self.settings.get("seed", 31000)))
        self.device = torch.device(device)
        self.model = UnfoldedEqualizer(UnfoldedConfig.from_dict(config["model"])).to(self.device)
        self.model.attach_online_phase_trend_adapter()
        self.model.eval()
        base_parameters = []
        self.rank1_basis_raw = None
        if self.trajectory_update_rule == "rank1_gn":
            head = dict(self.model.peft.named_group_parameters({"head"}))
            dimension = sum(parameter.numel() for parameter in head.values())
            generator = torch.Generator(device=self.device).manual_seed(
                int(self.settings.get("trajectory_rank1_basis_seed", self.settings.get("seed", 31000))))
            initial_basis = torch.randn(dimension, generator=generator, device=self.device)
            initial_basis = initial_basis / initial_basis.norm().clamp_min(1e-12)
            self.rank1_basis_raw = nn.Parameter(initial_basis)
        outer_groups = self.settings.get("outer_groups")
        outer_names = None if outer_groups is None else {name for name, _ in self.model.peft.named_group_parameters(outer_groups)}
        if outer_names is not None and not outer_names:
            raise ValueError("离线 outer_groups 没有可训练参数。")
        for name, parameter in self.model.named_parameters():
            temporary = name.startswith("online_phase_trend_adapter.")
            trainable = (self.trajectory_update_rule != "rank1_gn" and not temporary
                         and (outer_names is None or name in outer_names))
            parameter.requires_grad_(trainable)
            if trainable:
                base_parameters.append(parameter)
        if self.rank1_basis_raw is not None:
            base_parameters = [self.rank1_basis_raw]
        self.optimizer = torch.optim.AdamW(base_parameters, lr=float(self.settings.get("outer_learning_rate", 1e-5)))
        self.history = []
        self.source_checkpoint = None
        self.initialized_modules = []
        self.sample_index = 0
        self.sample_cache_hits = 0
        self.sample_cache_misses = 0
        self.last_sample_cache_key = None

    def load_resume(self, path):
        """明确失败优于静默从随机初始化开始训练。"""

        if path is None:
            raise ValueError("首轮配对实验必须指定同一 --resume checkpoint。")
        source = Path(path)
        payload = torch.load(source, map_location=self.device, weights_only=False)
        state = payload.get("state_dict", payload.get("model_state_dict"))
        if state is None:
            raise ValueError("离线 checkpoint 缺少 state_dict。")
        base_state = self.model.state_dict()
        expected = {key for key in base_state if not key.startswith("online_phase_trend_adapter.")}
        if set(state) != expected:
            source_config = UnfoldedConfig.from_dict(payload.get("model_config", {})).to_dict()
            target_config = self.model.config.to_dict()
            candidates = [(module, flag) for module, flag in (
                ("symbol_residual", "enable_symbol_residual_branch"), ("channel_state", "enable_channel_state_branch"),
                ("channel_state_memory", "enable_channel_state_memory"))
                if not source_config[flag] and target_config[flag]]
            if len(candidates) != 1:
                raise ValueError("基础 checkpoint 参数结构不一致，禁止非严格加载。")
            module, flag = candidates[0]
            if module == "channel_state_memory":
                new_keys = {key for key in expected if key.startswith(("channel_state.memory.", "channel_state.memory_scale."))}
                zero_module = self.model.channel_state.memory
            else:
                new_keys = {key for key in expected if key.startswith(module + ".")}
                zero_module = getattr(self.model, module).head
            permitted = (self.settings.get("initialize_" + module + "_from_base", False)
                         and bool(new_keys) and set(state) == expected - new_keys)
            source_config[flag] = target_config[flag]
            if not permitted or source_config != target_config:
                raise ValueError("基础 checkpoint 参数结构不一致，禁止非严格加载。")
            if any(bool(parameter.detach().ne(0).any()) for parameter in zero_module.parameters()):
                raise ValueError("新增神经残差 head 必须零初始化，才能从基础 checkpoint 迁移。")
            self.initialized_modules = [module]
        base_state.update(state)
        self.model.load_state_dict(base_state, strict=True)
        self.source_checkpoint = {"path": str(source.resolve()), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        return True

    def _sample(self):
        self.last_sample_cache_key = None
        snrs = self.config["main_snrs"]
        snr = float(snrs[self.sample_index % len(snrs)])
        seed = int(self.settings.get("seed", 31000)) + self.sample_index
        self.sample_index += 1
        choices = self.settings.get("history_frame_choices")
        if choices is not None and (not choices or any(int(value) < 0 for value in choices)):
            raise ValueError("离线历史帧选项必须非空且非负。")
        history_frames = int(choices[(self.sample_index - 1) % len(choices)] if choices is not None
                             else self.settings.get("history_frames", 1))
        if self.settings.get("offline_sample_cache_dir"):
            return self._cached_sample(seed=seed, snr=snr, history_frames=history_frames)
        env, cir, cfo, tail, frames = make_episode(
            self.config, self.model, seed=seed, snr=snr,
            frames=history_frames + self.post_target_frame_offsets[-1] + 1,
            state_split="offline_train", device=self.device,
        )
        for frame in frames[:history_frames]:
            if bool(frame.adapt_mask[-tail.numel():].any()):
                context = make_context(frame, cir, cfo, tail, snr)
                tail = next_pilot_tail(self.model, context, alpha=float(self.config.get("tail_update_alpha", 0.5)))
        context = make_context(frames[history_frames], cir, cfo, tail, snr)
        if self.inner_history_frames > 1:
            from training.shared_pilot_state import adapt_only_history_context
            past = tuple(adapt_only_history_context(frame, make_context(frame, cir, cfo, tail, snr).condition, tail)
                         for frame in frames[max(0, history_frames-self.inner_history_frames+1):history_frames])
            context = replace(context, offline_adapt_contexts=past)
        if len(self.post_target_frame_offsets) > 1:
            # 未来条件只允许由各自 acquisition/Adapt 生成，不沿用隐藏发送符号。
            # 当前 prefix 不覆盖帧尾；未来窗口若覆盖帧尾，需要逐帧合法递推，不能静默复用。
            if any(bool(frame.adapt_mask[-tail.numel():].any()) for frame in frames[history_frames:]):
                raise ValueError("跨帧离线目标目前要求前缀不覆盖帧尾。")
            future = tuple(make_context(frames[history_frames + offset], cir, cfo, tail, snr)
                           for offset in self.post_target_frame_offsets[1:])
            context = replace(context, offline_post_contexts=future)
        return context, env.state_metadata(), snr, seed

    def _cached_sample(self, *, seed, snr, history_frames):
        """只缓存离线监督样本，A/B 复用相同长轨迹，不缓存模型输出或决策。"""
        protocol = {"schema_version": "offline-transfer-sample-v1", "config": self.config,
                    "seed": seed, "snr_db": snr, "history_frames": history_frames,
                    "offsets": self.post_target_frame_offsets}
        digest = hashlib.sha256(json.dumps(protocol, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        self.last_sample_cache_key = digest
        root = Path(self.settings["offline_sample_cache_dir"])
        path = root / (digest + ".pt")
        indices = [history_frames + offset for offset in self.post_target_frame_offsets]
        adapt_indices = list(range(max(0, history_frames-self.inner_history_frames+1), history_frames))
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if (payload.get("protocol") != protocol or not payload.get("offline_supervision_only")
                    or [frame.frame_index for frame in payload.get("frames", [])] != indices
                    or [frame.frame_index for frame in payload.get("adapt_history_frames", [])] != adapt_indices):
                raise ValueError("离线样本缓存协议或帧龄不匹配，禁止静默复用。")
            self.sample_cache_hits += 1
        else:
            env, cir, cfo, tail, frames = make_episode(self.config, self.model, seed=seed, snr=snr,
                frames=indices[-1] + 1, state_split="offline_train", device="cpu")
            if any(bool(frame.adapt_mask[-tail.numel():].any()) for frame in frames):
                raise ValueError("离线长轨迹缓存目前只支持前缀不覆盖帧尾。")
            payload = {"protocol": protocol, "offline_supervision_only": True,
                       "cir": cir.detach().cpu(), "cfo": float(cfo), "tail": tail.detach().cpu(),
                       "channel_state": env.state_metadata(),
                       "frames": [frame_to_device(frames[index], "cpu") for index in indices],
                       "adapt_history_frames": [frame_to_device(frames[index], "cpu") for index in adapt_indices]}
            root.mkdir(parents=True, exist_ok=True)
            # 多个离线准备进程可重复生成同一确定性样本；使用独立临时文件原子发布。
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            torch.save(payload, temporary)
            temporary.replace(path)
            self.sample_cache_misses += 1
        contexts = [make_context(frame_to_device(frame, self.device), payload["cir"].to(self.device),
                    payload["cfo"], payload["tail"].to(self.device), snr) for frame in payload["frames"]]
        context = replace(contexts[0], offline_post_contexts=tuple(contexts[1:]))
        if adapt_indices:
            from training.shared_pilot_state import adapt_only_history_context
            past = []
            for frame in payload["adapt_history_frames"]:
                item = make_context(frame_to_device(frame, self.device), payload["cir"].to(self.device),
                                    payload["cfo"], payload["tail"].to(self.device), snr)
                past.append(adapt_only_history_context(item.frame, item.condition, item.tail))
            context = replace(context, offline_adapt_contexts=tuple(past))
        return context, payload["channel_state"], snr, seed

    def normalized_rank1_basis(self):
        """返回训练中的单位范数rank-1方向；普通trajectory没有此状态。"""
        if self.rank1_basis_raw is None:
            return None
        return self.rank1_basis_raw / self.rank1_basis_raw.norm().clamp_min(1e-12)

    def train(self, steps, batch_size=1, accumulation_steps=1, use_amp=False):
        # CUDA 的高效 attention backward 不支持本任务所需的二阶导数；两分支
        # 统一使用 math 后端，避免内循环外层反传失败或训练数值路径不同。
        with sdpa_kernel(SDPBackend.MATH):
            return self._train(steps, batch_size, accumulation_steps, use_amp)

    def _train(self, steps, batch_size=1, accumulation_steps=1, use_amp=False):
        """以样本逐个反传控制显存；精确内循环不使用半精度。"""

        if use_amp:
            raise ValueError("可微 AdamW 内循环暂只支持 float32，请去掉 --amp。")
        if min(steps, batch_size, accumulation_steps) < 1:
            raise ValueError("steps、batch_size 与 accumulation_steps 必须为正。")
        started = time.monotonic()
        for step in range(int(steps)):
            self.optimizer.zero_grad(set_to_none=True)
            for _ in range(int(batch_size) * int(accumulation_steps)):
                context, channel_state, snr, seed = self._sample()
                frame = context.frame
                targets = (context,) + context.offline_post_contexts
                # A/B 使用完全相同的当前/未来离线标签暴露，区别只在是否优化更新后目标。
                base_terms, before_terms = [], []
                for target_context in targets:
                    before_logits = context_logits(self.model, target_context)
                    target_frame = target_context.frame
                    base_terms.append(F.binary_cross_entropy_with_logits(before_logits.float(), target_frame.bits.float()))
                    before_terms.append(F.binary_cross_entropy_with_logits(before_logits[target_frame.data_mask].float(),
                                        target_frame.bits[target_frame.data_mask].float()))
                base_loss, before_data = torch.stack(base_terms).mean(), torch.stack(before_terms).mean()
                inner_context, score_start = task_context(context, self.inner_task, role="adapt")
                trajectory_stats = None
                if self.objective == "trajectory":
                    fast, trajectory_stats = differentiable_online_trajectory(
                        self.model, targets,
                        learning_rate=float(self.settings["inner_learning_rate"]),
                        steps=int(self.settings["inner_steps"]),
                        max_delta_norm=float(self.settings["max_delta_norm"]),
                        create_graph=True,
                        groups=self.settings.get("inner_groups", ["phase_trend"]),
                        objective=self.settings.get("inner_objective", "pilot_signal_reconstruction"),
                        receiver_input=self.inner_receiver_input,
                        reward_receiver_input=self.reward_receiver_input,
                        reward_task=self.reward_task,
                        reward_objective=self.settings.get("reward_objective", "bce"),
                        snr_db=snr,
                        inner_task=self.inner_task,
                        tail_alpha=float(self.config.get("tail_update_alpha", 0.5)),
                        soft_gate_rule=self.settings.get("trajectory_soft_gate_rule", "conservative_and"),
                        persist_optimizer_state=bool(self.settings.get("trajectory_persist_optimizer_state", False)),
                        update_rule=self.trajectory_update_rule,
                        rank1_basis=self.rank1_basis_raw,
                        rank1_damping=float(self.settings.get("trajectory_rank1_damping", 0.01)),
                        rank1_step_scale=float(self.settings.get("trajectory_rank1_step_scale", 1.0)),
                        history_frames=self.inner_history_frames,
                        initial_history=context.offline_adapt_contexts,
                        trend_temperature=float(self.settings.get("trajectory_trend_temperature", 0.01)),
                        soft_gate_bce_temperature=float(self.settings.get("trajectory_soft_gate_bce_temperature", 0.01)),
                        soft_gate_ber_temperature=float(self.settings.get("trajectory_soft_gate_ber_temperature", 0.02)),
                    )
                    post_loss = trajectory_stats["online_data_loss"]
                    loss = (float(self.settings.get("base_weight", 1.0)) * base_loss
                            + float(self.settings.get("post_weight", 1.0)) * post_loss
                            + float(self.settings.get("trajectory_trend_weight", 1.0))
                            * trajectory_stats["trend_loss"])
                    update = {"adapt_losses": [], "delta_norm": max(
                        item["cumulative_delta_norm"] for item in trajectory_stats["steps"])}
                else:
                    fast, update = differentiable_adapt(
                        self.model, list(context.offline_adapt_contexts) + [inner_context], learning_rate=float(self.settings["inner_learning_rate"]),
                        steps=int(self.settings["inner_steps"]), max_delta_norm=float(self.settings["max_delta_norm"]),
                        create_graph=self.objective == "post_adapt",
                        groups=self.settings.get("inner_groups", ["phase_trend"]),
                        objective=self.settings.get("inner_objective", "pilot_signal_reconstruction"),
                        receiver_input=self.inner_receiver_input,
                        score_start=score_start,
                    )
                    post_terms = []
                    for target_context in targets:
                        after_logits = context_logits(self.model, target_context, fast)
                        target_frame = target_context.frame
                        post_terms.append(F.binary_cross_entropy_with_logits(after_logits[target_frame.data_mask].float(),
                                          target_frame.bits[target_frame.data_mask].float()))
                    post_loss = torch.stack(post_terms).mean()
                    loss = base_loss if self.objective == "baseline" else (
                        float(self.settings.get("base_weight", 1.0)) * base_loss
                        + float(self.settings.get("post_weight", 1.0)) * post_loss
                    )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("离线外层损失不是有限值，停止训练，避免导出损坏 checkpoint。")
                (loss / (batch_size * accumulation_steps)).backward()
                self.history.append({
                    "step": step + 1, "seed": seed, "snr_db": snr,
                    "frame_index": frame.frame_index,
                    "post_target_frame_indices": [target.frame.frame_index for target in targets],
                    "inner_adapt_frame_indices": [item.frame.frame_index for item in context.offline_adapt_contexts] + [frame.frame_index],
                    "inner_history_adapt_rx_sha256": [hashlib.sha256(item.frame.rx_symbols.detach().cpu().numpy().tobytes()).hexdigest()
                                                       for item in context.offline_adapt_contexts],
                    "offline_sample_cache_key": self.last_sample_cache_key,
                    "offline_sample_input_sha256": hashlib.sha256(frame.rx_symbols.detach().cpu().numpy().tobytes()).hexdigest(),
                    "lookahead_len": int(frame.lookahead_rx_symbols.numel()) if frame.lookahead_rx_symbols is not None else 0,
                    "offline_sample_lookahead_sha256": (
                        hashlib.sha256(frame.lookahead_rx_symbols.detach().cpu().numpy().tobytes()).hexdigest()
                        if frame.lookahead_rx_symbols is not None else None),
                    "loss": float(loss.detach().cpu()), "base_bce": float(base_loss.detach().cpu()),
                    "data_bce_before": float(before_data.detach().cpu()), "data_bce_after": float(post_loss.detach().cpu()),
                    "channel_state": channel_state, **update,
                })
                if trajectory_stats is not None:
                    self.history[-1].update(
                        trajectory_trend_delta=float(trajectory_stats["trend_delta"].detach().cpu()),
                        trajectory_trend_loss=float(trajectory_stats["trend_loss"].detach().cpu()),
                        trajectory_frozen_data_windows=[float(item.detach().cpu()) for item in trajectory_stats["frozen_data_windows"]],
                        trajectory_online_data_windows=[float(item.detach().cpu()) for item in trajectory_stats["online_data_windows"]],
                        trajectory_gain_windows=[float(item.detach().cpu()) for item in trajectory_stats["gain_windows"]],
                        trajectory_steps=trajectory_stats["steps"],
                    )
            torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], 1.0, error_if_nonfinite=True)
            self.optimizer.step()
            print(json.dumps({"objective": self.objective, **{k: self.history[-1][k] for k in (
                "step", "snr_db", "loss", "data_bce_before", "data_bce_after",
            )}, "elapsed_seconds": round(time.monotonic() - started, 2)}, ensure_ascii=False), flush=True)
        return {
            "stage": "pilot_transfer", "objective": self.objective,
            "linear_operator_backend": os.environ.get("RL4EQ_LINEAR_OPERATOR_BACKEND", "direct"),
            "attention_backend": "math",
            "source_checkpoint": self.source_checkpoint, "history": self.history,
            "initialized_modules": self.initialized_modules,
            "outer_trainable_parameter_count": sum(parameter.numel() for parameter in self.optimizer.param_groups[0]["params"]),
            "elapsed_seconds": time.monotonic() - started,
            "inner_uses_data_labels": False,
            "receiver_lookahead_len": int(self.config.get("receiver_lookahead_len", 0)),
            "inner_uses_data_waveform": self.inner_receiver_input == "full_rx" and self.inner_task == "full_frame"
                                         and self.settings.get("inner_objective") == "bce",
            "inner_receiver_input": self.inner_receiver_input,
            "inner_history_frames": self.inner_history_frames,
            "reward_receiver_input": self.reward_receiver_input,
            "inner_task": self.inner_task, "reward_task": self.reward_task,
            "outer_uses_offline_data_labels": True,
            "post_adapt_data_objective_enabled": self.objective in {"post_adapt", "trajectory"},
            "post_target_frame_offsets": self.post_target_frame_offsets,
            "trajectory_soft_gate_rule": self.settings.get("trajectory_soft_gate_rule", "conservative_and"),
            "trajectory_persist_optimizer_state": bool(self.settings.get("trajectory_persist_optimizer_state", False)),
            "trajectory_update_rule": self.trajectory_update_rule,
            "trajectory_rank1_damping": float(self.settings.get("trajectory_rank1_damping", 0.01)),
            "trajectory_rank1_step_scale": float(self.settings.get("trajectory_rank1_step_scale", 1.0)),
            "trajectory_rank1_basis": (self.normalized_rank1_basis().detach().cpu().tolist()
                                       if self.rank1_basis_raw is not None else None),
            "trajectory_rank1_basis_source": ({
                "type": "seeded_unit_gaussian_trainable_initialization",
                "seed": int(self.settings.get("trajectory_rank1_basis_seed", self.settings.get("seed", 31000))),
                "base_checkpoint_sha256": (self.source_checkpoint or {}).get("sha256"),
                "status": "learned_basis_in_training; basis and source are embedded in checkpoint metrics; evaluator basis loading is pending",
            } if self.rank1_basis_raw is not None else None),
            "offline_sample_cache_hits": self.sample_cache_hits,
            "offline_sample_cache_misses": self.sample_cache_misses,
            "offline_base_loss": "bce_all",
            "reward_pilot_used_for_post_adapt_loss": False,
            "checkpoint_selection": "fixed_training_budget_no_test_selection",
            "settings": self.settings,
        }

    def save(self, save_dir, metrics):
        target = Path(save_dir)
        target.mkdir(parents=True, exist_ok=True)
        state = {name: value.detach().cpu() for name, value in self.model.state_dict().items()
                 if not name.startswith("online_phase_trend_adapter.")}
        payload = {"schema_version": "unfolded-eq-v1", "model_config": self.model.config.to_dict(),
                   "state_dict": state, "metrics": metrics, "selected_by": "fixed_training_budget"}
        save_checkpoint(target / "model_final.pt", payload)
        # 与现有 CLI 兼容；此处 best 只是固定预算终点，不声称经过验证集选优。
        save_checkpoint(target / "model_best.pt", payload)
        (target / "model_config.json").write_text(json.dumps(self.model.config.to_dict(), indent=2), encoding="utf-8")
        (target / "pretrain_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
