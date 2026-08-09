# -*- coding: utf-8 -*-
"""连续动作 PPO 调制策略，用于在线调制神经均衡器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.distributions import Normal

from agent.modulation import ModulationConfig, ModulationState


@dataclass(frozen=True)
class ModulationObservation:
    tensor: torch.Tensor
    fields: tuple[str, ...]


@dataclass(frozen=True)
class ContinuousModulationAction:
    state: ModulationState
    raw: torch.Tensor


class ModulationObservationEncoder:
    """把在线可观测量编码为 PPO observation。

    字段白名单显式排除 Reward/Data 标签。Reward Pilot 标签只在动作执行后
    计算 reward；Data 标签只在仿真评估中使用。
    """

    FIELDS = (
        "rx_power_mean",
        "rx_power_std",
        "adapt_fraction",
        "adapt_rx_mean",
        "adapt_rx_std",
        "noise_variance",
        "confidence",
        "previous_reward",
        "last_modulation_delta_norm",
        "rx_real_mean",
        "rx_imag_mean",
        "rx_real_std",
        "rx_imag_std",
        "adapt_symbol_mean",
        "adapt_symbol_std",
        "reserved_0",
    )

    def __call__(self, view: Any) -> ModulationObservation:
        rx = _as_complex(getattr(view, "rx_symbols")).flatten()
        device = rx.device
        adapt_symbols = _as_complex(getattr(view, "adapt_symbols")).flatten()
        adapt_mask = getattr(view, "adapt_mask").bool().flatten()
        rx_power = rx.abs().pow(2)
        adapt_rx = rx[adapt_mask] if bool(adapt_mask.any()) else rx[:1]
        adapt_tx = adapt_symbols[adapt_mask] if adapt_symbols.numel() == rx.numel() and bool(adapt_mask.any()) else adapt_symbols[:1]
        features = torch.stack(
            (
                rx_power.mean(),
                rx_power.std(unbiased=False),
                adapt_mask.float().mean(),
                adapt_rx.real.mean(),
                adapt_rx.abs().std(unbiased=False),
                _scalar(getattr(view, "noise_variance", torch.tensor(1.0))).to(device),
                _scalar(getattr(view, "confidence", torch.tensor(0.0))).to(device),
                _scalar(getattr(view, "previous_reward", torch.tensor(0.0))).to(device),
                _scalar(getattr(view, "last_modulation_delta_norm", torch.tensor(0.0))).to(device),
                rx.real.mean(),
                rx.imag.mean(),
                rx.real.std(unbiased=False),
                rx.imag.std(unbiased=False),
                adapt_tx.real.mean(),
                adapt_tx.real.std(unbiased=False),
                torch.zeros((), device=device),
            )
        ).to(torch.float32)
        return ModulationObservation(features, self.FIELDS)


class ContinuousModulationPolicy(nn.Module):
    """低维连续调制 PPO actor-critic。"""

    def __init__(self, observation_dim: int, modulation_config: ModulationConfig, hidden_size: int = 128):
        super().__init__()
        self.modulation_config = modulation_config
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(nn.Linear(observation_dim, hidden_size), nn.Tanh())
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=False)
        self.mean = nn.Linear(hidden_size, modulation_config.action_dim)
        self.log_std = nn.Parameter(torch.full((modulation_config.action_dim,), -1.0))
        self.value_head = nn.Linear(hidden_size, 1)

    def initial_hidden(self, batch_size: int = 1, device: torch.device | None = None) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_size, device=device or torch.device("cpu"))

    def sample(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[ContinuousModulationAction, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, state, next_hidden = self._dist(observation, hidden)
        raw = dist.rsample()
        action = ContinuousModulationAction(ModulationState.from_raw(raw[0], self.modulation_config), raw.detach())
        log_prob = dist.log_prob(raw).sum(dim=-1)
        value = self.value_head(state).squeeze(-1)
        return action, log_prob, value, next_hidden

    def evaluate_action(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor,
        action: ContinuousModulationAction,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, state, _ = self._dist(observation, hidden)
        raw = action.raw.to(state.device)
        if raw.dim() == 1:
            raw = raw.unsqueeze(0)
        log_prob = dist.log_prob(raw).sum(dim=-1)
        value = self.value_head(state).squeeze(-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, value, entropy

    def _dist(self, observation: torch.Tensor, hidden: torch.Tensor):
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        encoded = self.encoder(observation.float())
        core, next_hidden = self.gru(encoded.unsqueeze(0), hidden)
        state = core.squeeze(0)
        mean = self.mean(state)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std), state, next_hidden

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def initialize_identity_policy_prior(policy: ContinuousModulationPolicy) -> None:
    """让未训练 PPO 初始动作尽量等价于 identity modulation。

    这避免部署 smoke 中随机连续动作破坏离线神经均衡器。探索仍可通过训练阶段
    调整 log_std 或权重实现，但正式比较入口必须从安全接收机开始。
    """

    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
        policy.log_std.fill_(-4.0)
        bias = torch.zeros(policy.modulation_config.action_dim)
        cursor = policy.modulation_config.num_adapter_gates
        bias[cursor] = 0.0
        cursor += 1 + policy.modulation_config.num_lora_scales
        bias[cursor] = torch.logit(torch.tensor((1.0 - 0.5) / 1.5))
        bias[cursor + 1] = 0.0
        bias[cursor + 2] = -8.0
        policy.mean.bias.copy_(bias.to(policy.mean.bias.device))


def _as_complex(value: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if torch.is_complex(tensor):
        return tensor.to(torch.complex64)
    if tensor.shape[-1] == 2:
        return torch.complex(tensor[..., 0].float(), tensor[..., 1].float())
    return torch.complex(tensor.float(), torch.zeros_like(tensor.float()))


def _scalar(value: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value).float().flatten()[0]
