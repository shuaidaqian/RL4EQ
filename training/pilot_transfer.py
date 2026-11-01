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
from dataclasses import dataclass
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


def receiver_inputs(settings):
    """区分接收波形范围与监督标签范围；历史配置保持原输入协议。"""
    inner = settings.get("inner_receiver_input", "adapt_only")
    reward = settings.get("reward_receiver_input", "pilot_only")
    if inner not in {"adapt_only", "full_rx"} or reward not in {"pilot_only", "full_rx"}:
        raise ValueError("不支持的内循环或 Reward 接收输入协议。")
    if inner == "full_rx" and settings.get("inner_objective", "pilot_signal_reconstruction") != "bce":
        raise ValueError("整帧接收输入目前只支持 Adapt BCE。")
    return inner, reward


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
    if parameters is None:
        logits, _ = model(*args, **kwargs)
    else:
        logits, _ = functional_call(model, parameters, args, kwargs)
    return logits.squeeze(0)


class _AdaptObjective(nn.Module):
    """通过 functional_call 对相位信号重构目标替换候选参数。"""

    def __init__(self, model, score_start=0, objective="pilot_signal_reconstruction", receiver_input="adapt_only"):
        super().__init__()
        self.model = model
        self.score_start = int(score_start)
        self.objective = objective
        if receiver_input not in {"adapt_only", "full_rx"}:
            raise ValueError("接收输入必须为 adapt_only 或 full_rx。")
        if objective != "bce" and receiver_input != "adapt_only":
            raise ValueError("整帧接收输入实验目前只支持 Adapt BCE。")
        self.receiver_input = receiver_input

    def forward(self, context):
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
    if objective not in {"pilot_signal_reconstruction", "bce"}:
        raise ValueError("内循环目标必须为 pilot_signal_reconstruction 或 bce。")
    if receiver_input not in {"adapt_only", "full_rx"}:
        raise ValueError("接收输入必须为 adapt_only 或 full_rx。")
    if objective != "bce" and receiver_input != "adapt_only":
        raise ValueError("整帧接收输入实验目前只支持 Adapt BCE。")
    if objective == "pilot_signal_reconstruction" and selected_groups != {"phase_trend"}:
        raise ValueError("相位信号重构只允许更新 phase_trend 参数组。")
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
        self.inner_receiver_input, self.reward_receiver_input = receiver_inputs(self.settings)
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
            new_keys = {key for key in expected if key.startswith("symbol_residual.")}
            source_config = UnfoldedConfig.from_dict(payload.get("model_config", {})).to_dict()
            target_config = self.model.config.to_dict()
            permitted = (self.settings.get("initialize_symbol_residual_from_base", False)
                         and bool(new_keys) and set(state) == expected - new_keys
                         and not source_config["enable_symbol_residual_branch"]
                         and target_config["enable_symbol_residual_branch"])
            source_config["enable_symbol_residual_branch"] = target_config["enable_symbol_residual_branch"]
            if not permitted or source_config != target_config:
                raise ValueError("基础 checkpoint 参数结构不一致，禁止非严格加载。")
            if any(bool(parameter.detach().ne(0).any()) for parameter in self.model.symbol_residual.head.parameters()):
                raise ValueError("新增符号残差 head 必须零初始化，才能从基础 checkpoint 迁移。")
            self.initialized_modules = ["symbol_residual"]
        base_state.update(state)
        self.model.load_state_dict(base_state, strict=True)
        self.source_checkpoint = {"path": str(source.resolve()), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        return True

    def _sample(self):
        snrs = self.config["main_snrs"]
        snr = float(snrs[self.sample_index % len(snrs)])
        seed = int(self.settings.get("seed", 31000)) + self.sample_index
        self.sample_index += 1
        history_frames = int(self.settings.get("history_frames", 1))
        env, cir, cfo, tail, frames = make_episode(
            self.config, self.model, seed=seed, snr=snr, frames=history_frames + 1,
            state_split="offline_train", device=self.device,
        )
        for frame in frames[:-1]:
            context = make_context(frame, cir, cfo, tail, snr)
            tail = next_pilot_tail(self.model, context, alpha=float(self.config.get("tail_update_alpha", 0.5)))
        return make_context(frames[-1], cir, cfo, tail, snr), env.state_metadata(), snr, seed

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
                before_logits = context_logits(self.model, context)
                frame = context.frame
                base_loss = F.binary_cross_entropy_with_logits(before_logits.float(), frame.bits.float())
                before_data = F.binary_cross_entropy_with_logits(before_logits[frame.data_mask].float(), frame.bits[frame.data_mask].float())
                fast, update = differentiable_adapt(
                    self.model, [context], learning_rate=float(self.settings["inner_learning_rate"]),
                    steps=int(self.settings["inner_steps"]), max_delta_norm=float(self.settings["max_delta_norm"]),
                    create_graph=self.objective == "post_adapt",
                    groups=self.settings.get("inner_groups", ["phase_trend"]),
                    objective=self.settings.get("inner_objective", "pilot_signal_reconstruction"),
                    receiver_input=self.inner_receiver_input,
                )
                after_logits = context_logits(self.model, context, fast)
                post_loss = F.binary_cross_entropy_with_logits(after_logits[frame.data_mask].float(), frame.bits[frame.data_mask].float())
                loss = base_loss if self.objective == "baseline" else (
                    float(self.settings.get("base_weight", 1.0)) * base_loss
                    + float(self.settings.get("post_weight", 1.0)) * post_loss
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("离线外层损失不是有限值，停止训练，避免导出损坏 checkpoint。")
                (loss / (batch_size * accumulation_steps)).backward()
                self.history.append({
                    "step": step + 1, "seed": seed, "snr_db": snr,
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
            "inner_uses_data_labels": False, "inner_uses_data_waveform": self.inner_receiver_input == "full_rx",
            "inner_receiver_input": self.inner_receiver_input,
            "reward_receiver_input": self.reward_receiver_input,
            "outer_uses_offline_data_labels": True,
            "post_adapt_data_objective_enabled": self.objective == "post_adapt",
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
