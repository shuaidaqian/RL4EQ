# -*- coding: utf-8 -*-
"""离散安全动作 PPO 策略。

动作空间只包含围绕 identity 的小幅 modulation；PPO 学的是在窗口级 Pilot
反馈下选择哪一个安全动作，而不是直接判决 Data bit。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from agent.modulation import ModulationConfig, ModulationState


@dataclass(frozen=True)
class DiscreteSafeAction:
    index: int
    name: str
    modulation: ModulationState
    delta_norm: float
    peft_groups: set[str] | None = None
    peft_lr: float = 0.0
    peft_steps: int = 0


@dataclass(frozen=True)
class DiscretePolicySample:
    action: DiscreteSafeAction
    log_prob: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    hidden: torch.Tensor


def safe_modulation_actions(num_blocks: int, device: torch.device | str = "cpu") -> list[DiscreteSafeAction]:
    """返回固定顺序的离散安全动作表。"""

    config = ModulationConfig(num_adapter_gates=int(num_blocks), num_lora_scales=int(num_blocks))
    identity = ModulationState.identity(config, device=device)
    identity_vector = identity.to_vector()
    raw_actions: list[tuple[str, ModulationState, set[str] | None, float, int]] = [
        ("identity", identity, None, 0.0, 0),
        ("peft_head_light", identity, {"head"}, 1e-4, 1),
        ("peft_head_fast", identity, {"head"}, 5e-4, 1),
        ("peft_adapter_lora_conservative", identity, {"adapter", "attention_lora", "ffn_lora"}, 5e-5, 1),
        ("peft_adapter_lora_light", identity, {"adapter", "attention_lora", "ffn_lora"}, 1e-4, 1),
        ("peft_adapter_lora_head_light", identity, {"adapter_lora"}, 1e-4, 1),
        ("rollback_identity", identity, None, 0.0, 0),
    ]
    actions = []
    for index, (name, modulation, peft_groups, peft_lr, peft_steps) in enumerate(raw_actions):
        delta = float(torch.norm(modulation.to_vector() - identity_vector).detach().cpu())
        actions.append(
            DiscreteSafeAction(
                index=index,
                name=name,
                modulation=modulation,
                delta_norm=delta,
                peft_groups=peft_groups,
                peft_lr=float(peft_lr),
                peft_steps=int(peft_steps),
            )
        )
    return actions


def initialize_safe_discrete_policy_prior(policy: "DiscreteSafePolicy", actions: list[DiscreteSafeAction] | None = None) -> None:
    """为未训练策略设置安全探索先验。

    先验不使用 Data 标签；它只把探索质量集中到诊断后保留的安全 PEFT 动作上，
    避免初期 rollout 大量浪费在 rollback 或完全无动作上。
    """

    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
        policy.logits.bias.fill_(0.0)
        if actions is None:
            policy.logits.bias[0] = 1.0
            policy.logits.bias[-1] = 0.5
            return
        for action in actions:
            if action.name == "identity":
                policy.logits.bias[action.index] = 0.3
            elif action.name == "peft_head_fast":
                policy.logits.bias[action.index] = 2.0
            elif action.name == "peft_adapter_lora_conservative":
                policy.logits.bias[action.index] = 0.8
            elif action.name in {"peft_head_light", "peft_adapter_lora_light", "peft_adapter_lora_head_light"}:
                policy.logits.bias[action.index] = 0.0
            elif action.name == "rollback_identity":
                policy.logits.bias[action.index] = -0.5


class DiscreteSafePolicy(nn.Module):
    """窗口级离散安全动作 actor-critic。"""

    def __init__(self, observation_dim: int, action_count: int, hidden_size: int = 128):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.encoder = nn.Sequential(nn.Linear(observation_dim, hidden_size), nn.Tanh())
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=False)
        self.logits = nn.Linear(hidden_size, int(action_count))
        self.value_head = nn.Linear(hidden_size, 1)

    def initial_hidden(self, batch_size: int = 1, device: torch.device | None = None) -> torch.Tensor:
        return torch.zeros(1, int(batch_size), self.hidden_size, device=device or torch.device("cpu"))

    def sample(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor,
        actions: list[DiscreteSafeAction],
    ) -> DiscretePolicySample:
        dist, state, next_hidden = self._dist(observation, hidden)
        index = dist.sample()
        action = actions[int(index.flatten()[0].item())]
        return DiscretePolicySample(
            action=action,
            log_prob=dist.log_prob(index),
            value=self.value_head(state).squeeze(-1),
            entropy=dist.entropy(),
            hidden=next_hidden,
        )

    def evaluate_action(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor,
        action_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, state, _ = self._dist(observation, hidden)
        index = action_index.to(state.device).long().reshape(-1)
        return dist.log_prob(index), self.value_head(state).squeeze(-1), dist.entropy()

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _dist(self, observation: torch.Tensor, hidden: torch.Tensor):
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        encoded = self.encoder(observation.float())
        core, next_hidden = self.gru(encoded.unsqueeze(0), hidden)
        state = core.squeeze(0)
        return Categorical(logits=self.logits(state)), state, next_hidden


def _state_from_values(
    config: ModulationConfig,
    device: torch.device | str,
    *,
    adapter: float = 1.0,
    film: float = 0.0,
    lora: float = 1.0,
    temperature: float = 1.0,
    bias: float = 0.0,
) -> ModulationState:
    vector = torch.cat(
        (
            torch.full((config.num_adapter_gates,), float(adapter), device=device),
            torch.tensor([float(film)], device=device),
            torch.full((config.num_lora_scales,), float(lora), device=device),
            torch.tensor([float(temperature), float(bias), 0.0], device=device),
        )
    )
    return ModulationState.from_vector(vector, config)
