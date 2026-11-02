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
from training.online_adaptation import _pilot_signal_reconstruction_loss
from training.windowed_discrete_ppo import _adapt_receiver_view


@dataclass(frozen=True)
class TransferContext:
    """条件和历史状态只由 acquisition 与 Adapt Pilot 生成。"""

    frame: Frame
    condition: CIRCondition
    tail: torch.Tensor
    # 仅供离线外层评分；模型与 Adapt 目标不会读取这些未来上下文。
    offline_post_contexts: tuple["TransferContext", ...] = ()


def receiver_inputs(settings):
    """区分接收波形范围与监督标签范围；历史配置保持原输入协议。"""
    inner = settings.get("inner_receiver_input", "adapt_only")
    reward = settings.get("reward_receiver_input", "pilot_only")
    if inner not in {"adapt_only", "full_rx"} or reward not in {"pilot_only", "full_rx"}:
        raise ValueError("不支持的内循环或 Reward 接收输入协议。")
    if inner == "full_rx" and settings.get("inner_objective", "pilot_signal_reconstruction") not in {"bce", "channel_state_signal", "channel_state_map"}:
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
                    tail_symbols=None, true_cir=None, adapt_block_lengths=[int(frame.adapt_mask[:end].sum())])
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
                         groups=None, objective="pilot_signal_reconstruction", receiver_input="adapt_only"):
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
    if objective not in {"pilot_signal_reconstruction", "bce", "channel_state_signal", "channel_state_map"}:
        raise ValueError("内循环目标必须为 pilot_signal_reconstruction 或 bce。")
    if receiver_input not in {"adapt_only", "full_rx"}:
        raise ValueError("接收输入必须为 adapt_only 或 full_rx。")
    if objective not in {"bce", "channel_state_signal", "channel_state_map"} and receiver_input != "adapt_only":
        raise ValueError("整帧接收输入只支持 Adapt BCE 或神经信道状态信号目标。")
    if objective == "pilot_signal_reconstruction" and selected_groups != {"phase_trend"}:
        raise ValueError("相位信号重构只允许更新 phase_trend 参数组。")
    if objective in {"channel_state_signal", "channel_state_map"} and (selected_groups != {"channel_state_head"} or model.channel_state is None):
        raise ValueError("信道状态信号重构只允许更新已启用的神经 channel_state_head。")
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
    moments, variances = {}, {}
    objective_module = _AdaptObjective(model, score_start, objective, receiver_input)
    losses = []
    for step in range(1, int(steps) + 1):
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
            corrected_mean = moments[name] / (1.0 - 0.9 ** step)
            corrected_variance = variances[name] / (1.0 - 0.999 ** step)
            # 避免零方差点的高阶导数出现 0 * inf；不影响浮点精度下的 AdamW 一致性。
            denominator = corrected_variance.clamp_min(1e-24).sqrt() + 1e-8
            fast[name] = fast[name] * (1.0 - learning_rate * 0.01) - learning_rate * corrected_mean / denominator
        losses.append(float(loss.detach().cpu()))
    norm = torch.sqrt(sum((fast[name] - before[name]).float().square().sum() for name in names) + 1e-24)
    scale = (max_delta_norm / norm).clamp(max=1.0)
    for name in names:
        fast[name] = before[name] + scale * (fast[name] - before[name])
    return fast, {"adapt_losses": losses, "delta_norm": float(norm.detach().clamp(max=max_delta_norm).cpu())}


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
        frame_to_device(env.next_frame(), device) for _ in range(frames)
    ]


def make_context(frame, cir, cfo, tail, snr):
    view = _adapt_receiver_view(frame)
    phase = estimate_phase_residual_vector(view, cir, tail, blocks=4, cfo_hint=cfo)
    condition = condition_from_cir(cir, float(snr), cfo_residual=cfo, phase_features=phase)
    return TransferContext(frame, condition, tail.detach())


def next_pilot_tail(model, context, *, alpha):
    """历史递推只观察 Adapt Pilot，与正式 PPO 的信息边界一致。"""

    visible = context.frame.adapt_mask[-context.tail.numel():].bool()
    if not bool(visible.any()):
        return context.tail.detach().clone()
    with torch.no_grad():
        logits = context_logits(model, context, visible="adapt")
        values = torch.tanh(logits[-context.tail.numel():] / 2.0)
        detected = torch.complex(values, torch.zeros_like(values))
        updated = (1.0 - alpha) * context.tail + alpha * detected
        return torch.where(visible, updated, context.tail).detach()


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
        self.inner_task, self.reward_task = receiver_tasks(self.settings)
        self.objective = objective or self.settings.get("objective", "post_adapt")
        if self.objective not in {"baseline", "post_adapt"}:
            raise ValueError("训练目标必须为 baseline 或 post_adapt。")
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
        outer_groups = self.settings.get("outer_groups")
        outer_names = None if outer_groups is None else {name for name, _ in self.model.peft.named_group_parameters(outer_groups)}
        if outer_names is not None and not outer_names:
            raise ValueError("离线 outer_groups 没有可训练参数。")
        for name, parameter in self.model.named_parameters():
            temporary = name.startswith("online_phase_trend_adapter.")
            trainable = not temporary and (outer_names is None or name in outer_names)
            parameter.requires_grad_(trainable)
            if trainable:
                base_parameters.append(parameter)
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
                ("symbol_residual", "enable_symbol_residual_branch"), ("channel_state", "enable_channel_state_branch"))
                if not source_config[flag] and target_config[flag]]
            if len(candidates) != 1:
                raise ValueError("基础 checkpoint 参数结构不一致，禁止非严格加载。")
            module, flag = candidates[0]
            new_keys = {key for key in expected if key.startswith(module + ".")}
            permitted = (self.settings.get("initialize_" + module + "_from_base", False)
                         and bool(new_keys) and set(state) == expected - new_keys)
            source_config[flag] = target_config[flag]
            if not permitted or source_config != target_config:
                raise ValueError("基础 checkpoint 参数结构不一致，禁止非严格加载。")
            if any(bool(parameter.detach().ne(0).any()) for parameter in getattr(self.model, module).head.parameters()):
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
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if (payload.get("protocol") != protocol or not payload.get("offline_supervision_only")
                    or [frame.frame_index for frame in payload.get("frames", [])] != indices):
                raise ValueError("离线样本缓存协议或帧龄不匹配，禁止静默复用。")
            self.sample_cache_hits += 1
        else:
            env, cir, cfo, tail, frames = make_episode(self.config, self.model, seed=seed, snr=snr,
                frames=indices[-1] + 1, state_split="offline_train", device=self.device)
            if any(bool(frame.adapt_mask[-tail.numel():].any()) for frame in frames):
                raise ValueError("离线长轨迹缓存目前只支持前缀不覆盖帧尾。")
            payload = {"protocol": protocol, "offline_supervision_only": True,
                       "cir": cir.detach().cpu(), "cfo": float(cfo), "tail": tail.detach().cpu(),
                       "channel_state": env.state_metadata(),
                       "frames": [frame_to_device(frames[index], "cpu") for index in indices]}
            root.mkdir(parents=True, exist_ok=True)
            # 多个离线准备进程可重复生成同一确定性样本；使用独立临时文件原子发布。
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            torch.save(payload, temporary)
            temporary.replace(path)
            self.sample_cache_misses += 1
        contexts = [make_context(frame_to_device(frame, self.device), payload["cir"].to(self.device),
                    payload["cfo"], payload["tail"].to(self.device), snr) for frame in payload["frames"]]
        context = replace(contexts[0], offline_post_contexts=tuple(contexts[1:]))
        return context, payload["channel_state"], snr, seed

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
                fast, update = differentiable_adapt(
                    self.model, [inner_context], learning_rate=float(self.settings["inner_learning_rate"]),
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
                    "offline_sample_cache_key": self.last_sample_cache_key,
                    "offline_sample_input_sha256": hashlib.sha256(frame.rx_symbols.detach().cpu().numpy().tobytes()).hexdigest(),
                    "loss": float(loss.detach().cpu()), "base_bce": float(base_loss.detach().cpu()),
                    "data_bce_before": float(before_data.detach().cpu()), "data_bce_after": float(post_loss.detach().cpu()),
                    "channel_state": channel_state, **update,
                })
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
            "inner_uses_data_waveform": self.inner_receiver_input == "full_rx" and self.inner_task == "full_frame"
                                         and self.settings.get("inner_objective") == "bce",
            "inner_receiver_input": self.inner_receiver_input,
            "reward_receiver_input": self.reward_receiver_input,
            "inner_task": self.inner_task, "reward_task": self.reward_task,
            "outer_uses_offline_data_labels": True,
            "post_adapt_data_objective_enabled": self.objective == "post_adapt",
            "post_target_frame_offsets": self.post_target_frame_offsets,
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
