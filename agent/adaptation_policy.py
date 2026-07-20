# -*- coding: utf-8 -*-
"""帧级离散 PPO 参数高效适配策略。"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


@dataclass(frozen=True)
class AdaptationAction:
    name: str
    learning_rate: float = 0.0
    steps: int = 0
    train_adapters: bool = False
    train_output: bool = False
    rollback: bool = False


ACTION_TABLE = (
    AdaptationAction("skip"),
    AdaptationAction("head-light", 1e-4, 1, False, True),
    AdaptationAction("adapters-light", 1e-4, 1, True, False),
    AdaptationAction("adapters+head-light", 1e-4, 1, True, True),
    AdaptationAction("adapters+head-fast", 5e-4, 1, True, True),
    AdaptationAction("adapters+head-deep", 1e-4, 3, True, True),
    AdaptationAction("rollback-last-good", rollback=True),
)


class PPOPolicy(nn.Module):
    """小型 MLP Actor-Critic，动作粒度为一帧一次。"""

    def __init__(self, observation_dim: int, action_count: int, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_count)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(observations.float())
        return self.actor(encoded), self.critic(encoded).squeeze(-1)

    def sample_action(
        self, observation: torch.Tensor, deterministic: bool = False
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(observation)
        distribution = Categorical(logits=logits)
        action = torch.argmax(logits) if deterministic else distribution.sample()
        return int(action.item()), distribution.log_prob(action), value

    def evaluate_actions(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(observations)
        distribution = Categorical(logits=logits)
        return distribution.log_prob(actions.long()), values, distribution.entropy()

    def ppo_update(
        self,
        optimizer: torch.optim.Optimizer,
        observations: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        epochs: int = 4,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ) -> dict[str, float]:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
        last_loss = torch.tensor(0.0, device=observations.device)
        for _ in range(int(epochs)):
            log_probs, values, entropy = self.evaluate_actions(observations, actions)
            ratio = torch.exp(log_probs - old_log_probs.detach())
            clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
            policy_loss = -torch.minimum(ratio * advantages, clipped * advantages).mean()
            value_loss = F.mse_loss(values, returns)
            last_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()
            optimizer.zero_grad()
            last_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
        return {"ppo_loss": float(last_loss.detach().item())}

