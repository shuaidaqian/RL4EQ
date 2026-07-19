# -*- coding: utf-8 -*-
"""轻量离散 PPO 策略，用于选择在线更新动作。"""

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscretePolicyValueNet(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy = nn.Linear(hidden, num_actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        x = self.shared(obs)
        return self.policy(x), self.value(x).squeeze(-1)


@dataclass
class PPOTransition:
    obs: torch.Tensor
    action: int
    log_prob: torch.Tensor
    reward: float
    value: torch.Tensor


@dataclass
class PPORollout:
    items: List[PPOTransition] = field(default_factory=list)

    def add(self, item: PPOTransition) -> None:
        self.items.append(item)

    def clear(self) -> None:
        self.items.clear()


class DiscretePPOPolicy:
    """单步帧级 PPO，reward 来自 pilot loss 改善。"""

    def __init__(self, obs_dim: int, num_actions: int, device: torch.device, lr: float = 3e-4):
        self.device = device
        self.net = DiscretePolicyValueNet(obs_dim, num_actions).to(device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr)
        self.rollout = PPORollout()

    def act(self, obs: torch.Tensor):
        obs_b = obs.unsqueeze(0).to(self.device)
        logits, value = self.net(obs_b)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), log_prob.squeeze(0), value.squeeze(0)

    def remember(self, obs: torch.Tensor, action: int, log_prob: torch.Tensor, reward: float, value: torch.Tensor) -> None:
        self.rollout.add(PPOTransition(obs.detach(), action, log_prob.detach(), reward, value.detach()))

    def pretrain_imitation(self, observations: torch.Tensor, labels: torch.Tensor, epochs: int = 25) -> float:
        """用 data-oracle action 标签做监督预训练，只训练策略头。"""
        if observations.numel() == 0:
            return 0.0
        obs = observations.to(self.device)
        y = labels.to(self.device).long()
        last_loss = 0.0
        for _ in range(max(1, epochs)):
            logits, values = self.net(obs)
            loss = F.cross_entropy(logits, y) + 0.01 * values.square().mean()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
            last_loss = float(loss.item())
        return last_loss

    def update(self, gamma: float = 0.95, clip_eps: float = 0.2, epochs: int = 3) -> float:
        if not self.rollout.items:
            return 0.0

        obs = torch.stack([x.obs for x in self.rollout.items]).to(self.device)
        actions = torch.tensor([x.action for x in self.rollout.items], dtype=torch.long, device=self.device)
        old_log_probs = torch.stack([x.log_prob for x in self.rollout.items]).to(self.device)
        rewards = torch.tensor([x.reward for x in self.rollout.items], dtype=torch.float32, device=self.device)

        returns = []
        ret = torch.tensor(0.0, device=self.device)
        for reward in reversed(rewards):
            ret = reward + gamma * ret
            returns.append(ret)
        returns = torch.stack(list(reversed(returns)))
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-6)

        total_loss = 0.0
        for _ in range(epochs):
            logits, values = self.net(obs)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            adv = returns - values.detach()
            policy_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv).mean()
            value_loss = F.mse_loss(values, returns)
            entropy = dist.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
            total_loss += float(loss.item())

        self.rollout.clear()
        return total_loss / max(epochs, 1)
