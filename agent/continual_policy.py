# -*- coding: utf-8 -*-
"""部署期间持续更新 PPO 的结构化 observation 与分层策略。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical, Normal


MODES = ("skip", "update-channel", "update-equalizer", "joint-update", "detector-refine", "rollback")
PARAMETER_GROUPS = (
    "conditioner_film",
    "adapter",
    "attention_lora",
    "ffn_lora",
    "adapter_lora",
    "conditioner_peft",
)
STEP_CHOICES = (1, 2, 4)
ITERATION_CHOICES = (2, 4, 6, 8)


@dataclass(frozen=True)
class Observation:
    tensor: torch.Tensor
    fields: tuple[str, ...]


@dataclass(frozen=True)
class HierarchicalAction:
    mode: str
    parameter_group: str
    steps: int
    detector_iterations: int
    learning_rate: float
    proximal_weight: float
    reconstruction_weight: float
    damping: float
    cir_trust: float
    mode_index: int = -1
    group_index: int = -1
    steps_index: int = -1
    iteration_index: int = -1
    continuous_raw: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)


class ObservationEncoder:
    """把接收端可见状态压缩为固定长度向量。

    当前 Reward/Data 标签、Data BER 不属于可见状态，即使 view 对象上存在这些
    调试属性，本编码器也不会读取。
    """

    FIELDS = (
        "rx_power_mean",
        "rx_power_std",
        "adapt_fraction",
        "adapt_rx_mean",
        "adapt_rx_std",
        "cir_energy",
        "cir_delay_mean",
        "cir_tail_ratio",
        "support_mean",
        "noise_variance",
        "confidence",
        "previous_reward",
        "last_parameter_delta_norm",
        "region_adapt_fraction",
        "rx_real_mean",
        "rx_imag_mean",
        "rx_real_std",
        "rx_imag_std",
        "adapt_symbol_mean",
        "adapt_symbol_std",
        "long_echo_energy_ratio",
        "max_delay_hat",
        "snr_hat",
        "reserved_0",
        "reserved_1",
        "reserved_2",
        "reserved_3",
        "reserved_4",
        "reserved_5",
        "reserved_6",
        "reserved_7",
        "reserved_8",
    )

    def __call__(self, view: Any) -> Observation:
        rx = _as_complex(getattr(view, "rx_symbols")).flatten()
        adapt_symbols = _as_complex(getattr(view, "adapt_symbols")).flatten()
        adapt_mask = getattr(view, "adapt_mask").bool().flatten()
        region_ids = getattr(view, "model_region_ids").long().flatten()
        cir = _as_complex(getattr(view, "complex_cir", torch.zeros(41, dtype=torch.complex64))).flatten()
        support = torch.as_tensor(getattr(view, "support_probability", torch.zeros_like(cir.real))).float().flatten()
        noise_variance = _scalar(getattr(view, "noise_variance", torch.tensor(1.0)))
        confidence = _scalar(getattr(view, "confidence", torch.tensor(0.0)))
        previous_reward = _scalar(getattr(view, "previous_reward", torch.tensor(0.0)))
        delta_norm = _scalar(getattr(view, "last_parameter_delta_norm", torch.tensor(0.0)))
        rx_power = rx.abs().pow(2)
        adapt_rx = rx[adapt_mask] if bool(adapt_mask.any()) else rx[:1]
        cir_power = cir.abs().pow(2)
        delays = torch.arange(cir.numel(), dtype=torch.float32, device=cir_power.device)
        cir_energy = cir_power.sum().clamp_min(1e-8)
        tail_energy = cir_power[1:].sum() if cir_power.numel() > 1 else torch.zeros_like(cir_energy)
        features = torch.stack(
            [
                rx_power.mean(),
                rx_power.std(unbiased=False),
                adapt_mask.float().mean(),
                adapt_rx.real.mean(),
                adapt_rx.abs().std(unbiased=False),
                cir_energy,
                (delays * cir_power).sum() / cir_energy,
                tail_energy / cir_energy,
                support.mean(),
                noise_variance,
                confidence,
                previous_reward,
                delta_norm,
                (region_ids == 1).float().mean(),
                rx.real.mean(),
                rx.imag.mean(),
                rx.real.std(unbiased=False),
                rx.imag.std(unbiased=False),
                adapt_symbols.real[adapt_mask].mean() if bool(adapt_mask.any()) else torch.zeros((), device=rx.device),
                adapt_symbols.real[adapt_mask].std(unbiased=False) if bool(adapt_mask.any()) else torch.zeros((), device=rx.device),
                tail_energy / cir_energy,
                torch.argmax(cir_power).to(torch.float32),
                1.0 / noise_variance.clamp_min(1e-8),
                *[torch.zeros((), device=rx.device) for _ in range(9)],
            ]
        ).to(torch.float32)
        return Observation(tensor=features, fields=self.FIELDS)


class ContinualPolicy(nn.Module):
    """小型 recurrent mixed-action Actor-Critic。"""

    def __init__(self, observation_dim: int = 32, hidden_size: int = 128, ablation: str = "none"):
        super().__init__()
        if ablation not in {"none", "no_gru", "no_detector_control"}:
            raise ValueError(f"未知 PPO 消融：{ablation}")
        self.ablation = ablation
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(nn.Linear(observation_dim, hidden_size), nn.Tanh())
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=False)
        self.no_gru_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.mode_head = nn.Linear(hidden_size, len(MODES))
        self.group_head = nn.Linear(hidden_size, len(PARAMETER_GROUPS))
        self.steps_head = nn.Linear(hidden_size, len(STEP_CHOICES))
        self.iter_head = nn.Linear(hidden_size, len(ITERATION_CHOICES))
        self.continuous_mean = nn.Linear(hidden_size, 5)
        self.continuous_log_std = nn.Parameter(torch.full((5,), -1.5))
        self.value_head = nn.Linear(hidden_size, 1)

    def sample(self, observation: torch.Tensor, hidden: torch.Tensor) -> tuple[HierarchicalAction, torch.Tensor, torch.Tensor, torch.Tensor]:
        mode_dist, group_dist, steps_dist, iter_dist, continuous_dist, state, next_hidden = self._dists(observation, hidden)
        mode_idx = mode_dist.sample()
        group_idx = group_dist.sample()
        steps_idx = steps_dist.sample()
        if self.ablation == "no_detector_control":
            iter_idx = torch.full_like(mode_idx, ITERATION_CHOICES.index(4))
            iter_log_prob = torch.zeros_like(mode_dist.log_prob(mode_idx))
        else:
            iter_idx = iter_dist.sample()
            iter_log_prob = iter_dist.log_prob(iter_idx)
        raw = continuous_dist.rsample()
        action = self._build_action(mode_idx, group_idx, steps_idx, iter_idx, raw)
        log_prob = (
            mode_dist.log_prob(mode_idx)
            + group_dist.log_prob(group_idx)
            + steps_dist.log_prob(steps_idx)
            + iter_log_prob
            + continuous_dist.log_prob(raw).sum(dim=-1)
        )
        value = self.value_head(state).squeeze(-1)
        return action, log_prob, value, next_hidden

    def evaluate_action(self, observation: torch.Tensor, hidden: torch.Tensor, action: HierarchicalAction) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """重新计算给定动作的 log_prob/value，用于 clipped PPO 更新。"""

        mode_dist, group_dist, steps_dist, iter_dist, continuous_dist, state, _ = self._dists(observation, hidden)
        device = state.device
        mode_idx = torch.tensor([_resolve_index(action.mode_index, MODES, action.mode)], dtype=torch.long, device=device)
        group_idx = torch.tensor([_resolve_index(action.group_index, PARAMETER_GROUPS, action.parameter_group)], dtype=torch.long, device=device)
        steps_idx = torch.tensor([_resolve_index(action.steps_index, STEP_CHOICES, action.steps)], dtype=torch.long, device=device)
        iter_idx = torch.tensor([_resolve_index(action.iteration_index, ITERATION_CHOICES, action.detector_iterations)], dtype=torch.long, device=device)
        raw = torch.tensor([action.continuous_raw], dtype=torch.float32, device=device)
        iter_log_prob = (
            torch.zeros_like(mode_dist.log_prob(mode_idx))
            if self.ablation == "no_detector_control"
            else iter_dist.log_prob(iter_idx)
        )
        log_prob = (
            mode_dist.log_prob(mode_idx)
            + group_dist.log_prob(group_idx)
            + steps_dist.log_prob(steps_idx)
            + iter_log_prob
            + continuous_dist.log_prob(raw).sum(dim=-1)
        )
        value = self.value_head(state).squeeze(-1)
        entropy = mode_dist.entropy() + group_dist.entropy() + steps_dist.entropy() + iter_dist.entropy()
        return log_prob, value, entropy

    def _dists(self, observation: torch.Tensor, hidden: torch.Tensor):
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        encoded = self.encoder(observation.float())
        if self.ablation == "no_gru":
            core = self.no_gru_mlp(encoded).unsqueeze(0)
            next_hidden = hidden
        else:
            core, next_hidden = self.gru(encoded.unsqueeze(0), hidden)
        state = core.squeeze(0)
        return (
            Categorical(logits=self.mode_head(state)),
            Categorical(logits=self.group_head(state)),
            Categorical(logits=self.steps_head(state)),
            Categorical(logits=self.iter_head(state)),
            Normal(self.continuous_mean(state), self.continuous_log_std.exp().expand_as(self.continuous_mean(state))),
            state,
            next_hidden,
        )

    def _build_action(self, mode_idx: torch.Tensor, group_idx: torch.Tensor, steps_idx: torch.Tensor, iter_idx: torch.Tensor, raw: torch.Tensor) -> HierarchicalAction:
        squashed = torch.sigmoid(raw)
        mode = MODES[int(mode_idx[0].item())]
        parameter_group = PARAMETER_GROUPS[int(group_idx[0].item())]
        if mode in {"skip", "rollback", "detector-refine"}:
            parameter_group = "conditioner_film"
        return HierarchicalAction(
            mode=mode,
            parameter_group=parameter_group,
            steps=STEP_CHOICES[int(steps_idx[0].item())],
            detector_iterations=ITERATION_CHOICES[int(iter_idx[0].item())],
            learning_rate=float(_scale(squashed[0, 0], 1e-6, 1e-3).detach().cpu()),
            proximal_weight=float(_scale(squashed[0, 1], 1e-6, 1e-2).detach().cpu()),
            reconstruction_weight=float(squashed[0, 2].detach().cpu()),
            damping=float(_scale(squashed[0, 3], 0.0, 0.95).detach().cpu()),
            cir_trust=float(squashed[0, 4].detach().cpu()),
            mode_index=int(mode_idx[0].item()),
            group_index=int(group_idx[0].item()),
            steps_index=int(steps_idx[0].item()),
            iteration_index=int(iter_idx[0].item()),
            continuous_raw=tuple(float(value) for value in raw[0].detach().cpu().tolist()),
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _as_complex(value: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if torch.is_complex(tensor):
        return tensor.to(torch.complex64)
    if tensor.shape[-1] == 2:
        return torch.complex(tensor[..., 0].float(), tensor[..., 1].float())
    return torch.complex(tensor.float(), torch.zeros_like(tensor.float()))


def _scalar(value: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value).float().flatten()[0]


def _scale(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return low + (high - low) * value


def _resolve_index(index: int, choices: tuple, value) -> int:
    if index >= 0:
        return int(index)
    return int(choices.index(value))
