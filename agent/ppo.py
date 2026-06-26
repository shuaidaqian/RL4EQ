# -*- coding: utf-8 -*-
"""
PPO 训练模块 — 为通信在线均衡场景定制。

核心功能:
  1. GAE (Generalized Advantage Estimation) — 有效传播稀疏导频奖励
  2. PPO-Clip 损失 — 限制策略更新步长，防止 burst error 破坏策略
  3. 优势归一化 — 平衡导频段和数据段的梯度量级
  4. KL 早停 — 策略变化过快时提前结束 epoch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_gae(
    rewards: torch.Tensor,       # (T,)
    values: torch.Tensor,        # (T,)
    gamma: float = 0.99,
    lam: float = 0.95,
    dones: torch.Tensor = None,
    normalize: bool = True,
) -> tuple:
    """计算 GAE 优势估计。"""
    T = rewards.shape[0]
    if dones is None:
        dones = torch.zeros(T, device=rewards.device)
        dones[-1] = 1.0

    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            delta = rewards[t] - values[t]
            next_gae = 0.0
        else:
            next_val = values[t + 1] * (1 - dones[t])
            delta = rewards[t] + gamma * next_val - values[t]
            next_gae = advantages[t + 1]
        gae = delta + gamma * lam * next_gae * (1 - dones[t])
        advantages[t] = gae

    returns = advantages + values
    if normalize:
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
    return advantages.detach(), returns.detach()


class PPOTrainer:
    """PPO 训练器 — 通信在线均衡场景。"""

    def __init__(
        self,
        actor_critic: nn.Module,
        lr: float = 3e-4,
        gamma: float = 0.95,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ):
        self.actor_critic = actor_critic
        self.optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

    def compute_gae_and_returns(self, rewards, values, dones=None):
        return compute_gae(rewards, values, gamma=self.gamma, lam=self.lam,
                           dones=dones, normalize=True)

    def ppo_update(self, states, actions, old_log_probs, advantages, returns):
        """单次 PPO-Clip 更新。"""
        T = states.shape[0]
        log_probs, values, entropy, _ = self.actor_critic.evaluate_actions(states, actions)
        log_probs = log_probs.squeeze(-1)
        values = values.squeeze(-1)

        ratios = torch.exp(log_probs - old_log_probs.squeeze(-1))
        advantages = advantages.view(-1)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(values.view(-1), returns.view(-1))
        entropy_loss = -entropy.mean()
        total_loss = policy_loss + self.value_coef * value_loss + self.ent_coef * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            approx_kl = ((ratios - 1.0) - log_probs + old_log_probs.squeeze(-1)).mean().item()
            clamped_frac = ((ratios < 1 - self.clip_eps) | (ratios > 1 + self.clip_eps)).float().mean().item()

        return {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
            "approx_kl": approx_kl,
            "clamped_frac": clamped_frac,
        }

    def train_on_trajectory(self, states, actions, log_probs, rewards, values,
                            dones=None, k_epochs=4, kl_threshold=0.01):
        """
        用一帧完整轨迹做多轮 PPO 更新。
        states:    (T, D)
        actions:   (T, 1)
        log_probs: (T, 1)
        rewards:   (T,)
        values:    (T,)
        dones:     (T,)
        """
        advantages, returns = self.compute_gae_and_returns(rewards, values, dones)

        total_epochs = 0
        all_metrics = []
        for _ in range(k_epochs):
            m = self.ppo_update(states, actions, log_probs, advantages, returns)
            total_epochs += 1
            all_metrics.append(m)
            if m["approx_kl"] > kl_threshold:
                break

        agg = {}
        for key in all_metrics[0].keys():
            agg[key] = sum(m[key] for m in all_metrics[:total_epochs]) / total_epochs
        agg["epochs_used"] = total_epochs
        return agg


def test_gae():
    T = 10
    rewards = torch.zeros(T)
    rewards[3] = 1.0
    values = torch.zeros(T)
    adv, ret = compute_gae(rewards, values, gamma=0.9, lam=0.9, normalize=False)
    print(f"GAE 测试: rewards at pos 3 =1, adv[2]={adv[2]:.3f}")
    assert adv[2].abs() > 0
    print("GAE 测试通过。")


def test_ppo():
    from agent.actor_critic import ActorCritic, TransformerConfig
    ac = ActorCritic(TransformerConfig())
    trainer = PPOTrainer(ac, lr=1e-3, gamma=0.95, lam=0.95)
    T = 32
    states = torch.randn(T, 45)
    logits = torch.randn(T, 1)
    probs = torch.sigmoid(logits)
    d = torch.distributions.Bernoulli(probs=probs)
    actions = d.sample()
    old_lp = d.log_prob(actions)
    known_mask = torch.zeros(T, dtype=torch.bool)
    known_mask[5:10] = True
    rewards = torch.where(known_mask, 1.0, 0.0)
    with torch.no_grad():
        _, values, _, _ = ac.evaluate_actions(states, actions)
    metrics = trainer.train_on_trajectory(states, actions, old_lp, rewards, values.squeeze(-1), k_epochs=3)
    print(f"PPO: total={metrics['total_loss']:.4f} kl={metrics['approx_kl']:.4f} epochs={metrics['epochs_used']}")
    print("PPO 训练器自测通过。")


if __name__ == "__main__":
    test_gae()
    print()
    test_ppo()
