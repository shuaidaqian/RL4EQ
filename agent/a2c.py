# -*- coding: utf-8 -*-
"""
A2C-KPG (Known-only Policy Gradient) 算法 — Transformer 版

核心思想: 只对已知位 (训练 + 导频) 计算策略梯度,
数据位通过共享网络参数间接学习.

损失函数:
  L = policy_coef * L_policy(已知位)
    + value_coef * L_value(全局)
    - entropy_coef * H(全局)
    + sup_weight * L_BCE(已知位)

三阶段训练策略:
  SUP (1-10):  纯监督学习
  MIX (11-30): 混合模式 (策略梯度 + BCE)
  RL (31+):    纯强化学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from agent.actor_critic import ActorCritic, TransformerConfig


@dataclass
class A2CConfig:
    gamma: float = 0.97
    actor_lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    sup_weight: float = 1.0
    policy_coef: float = 1.0
    transformer_config: TransformerConfig = field(default_factory=TransformerConfig)


class A2C:
    """A2C-KPG 算法 (Transformer 版)。支持多帧缓冲更新。"""

    def __init__(self, config: A2CConfig, device):
        self.config = config
        self.device = device
        self.ac = ActorCritic(config.transformer_config).to(device)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=config.actor_lr)
        self.buffer = []

    @torch.no_grad()
    def act(self, state, context=None, deterministic=True):
        self.ac.train(not deterministic)
        return self.ac.sample_action(state, context)

    def store_transition(self, state, action, log_prob, reward, value,
                         done, bit_type, true_bit):
        self.buffer.append(dict(
            state=state.cpu(), action=action.item(), log_prob=log_prob.item(),
            reward=reward, value=value.item(), done=float(done),
            bit_type=bit_type, true_bit=true_bit,
        ))

    def clear_buffer(self):
        self.buffer = []

    def update_from_trajectory(self, policy_coef=1.0, sup_weight=1.0):
        if len(self.buffer) < 5:
            return {"error": "轨迹过短", "traj_len": len(self.buffer)}

        states = torch.stack([b["state"] for b in self.buffer]).to(self.device)
        actions = torch.tensor([b["action"] for b in self.buffer],
                               device=self.device).float().unsqueeze(-1)
        rewards = torch.tensor([b["reward"] for b in self.buffer], device=self.device)
        dones = torch.tensor([b["done"] for b in self.buffer], device=self.device)
        bit_types = [b["bit_type"] for b in self.buffer]
        true_bits = torch.tensor([b["true_bit"] for b in self.buffer],
                                 device=self.device, dtype=torch.float32).unsqueeze(-1)

        T = states.shape[0]
        gamma = self.config.gamma

        # Monte Carlo 回报
        returns = torch.zeros(T, device=self.device)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t].item() + gamma * (1 - dones[t].item()) * G
            returns[t] = G

        # 整帧 Transformer 前向
        self.ac.train()
        log_probs, values, entropy, logits = self.ac.evaluate_actions(states, actions)
        values = values.squeeze(-1)

        # 优势 (归一化)
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        known = torch.tensor(
            [bt in ("train", "pilot") for bt in bit_types],
            device=self.device, dtype=torch.bool,
        )

        c = self.config

        # 已知位策略梯度 (A2C-KPG 核心)
        if known.any():
            policy_loss = -(log_probs[known].squeeze(-1) * advantages[known]).mean()
        else:
            policy_loss = torch.zeros(1, device=self.device)

        value_loss = F.mse_loss(values, returns)
        entropy_loss = entropy.mean()

        sup_loss = (F.binary_cross_entropy_with_logits(logits[known], true_bits[known])
                    if known.any() else torch.zeros(1, device=self.device))

        total = (policy_coef * policy_loss + c.value_coef * value_loss
                 - c.entropy_coef * entropy_loss + sup_weight * sup_loss)

        self.optimizer.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(self.ac.parameters(), c.max_grad_norm)
        self.optimizer.step()
        self.clear_buffer()

        return {
            "total": total.item(), "policy": policy_loss.item(),
            "value": value_loss.item(), "entropy": entropy_loss.item(),
            "supervised": sup_loss.item(), "traj_len": T,
        }


def test_a2c():
    device = torch.device("cpu")
    agent = A2C(A2CConfig(), device)
    params = sum(p.numel() for p in agent.ac.parameters())
    print(f"参数量: {params}")
    ctx = None
    for t in range(40):
        s = torch.randn(45)
        a, lp, v, ctx = agent.act(s, ctx, deterministic=False)
        bt = "train" if t < 20 else "data"
        agent.store_transition(s, a, lp, 1.0 if bt == "train" else 0.0,
                               v, False, bt, 1.0 if a.item() > 0.5 else 0.0)
    m = agent.update_from_trajectory()
    if "error" in m:
        print(f"出错: {m}")
    else:
        print(f"更新: total={m['total']:.4f}, policy={m['policy']:.4f}")
    print("A2C (Transformer) 自测通过。")


if __name__ == "__main__":
    test_a2c()
