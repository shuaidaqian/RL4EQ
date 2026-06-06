# -*- coding: utf-8 -*-
"""
PPO 训练模块 — 为通信在线均衡场景定制。

核心功能:
  1. GAE (Generalized Advantage Estimation) — 有效传播稀疏导频奖励
  2. PPO-Clip 损失 — 限制策略更新步长，防止 burst error 破坏策略
  3. 优势归一化 — 平衡导频段和数据段的梯度量级
  4. KL 早停 — 策略变化过快时提前结束 epoch
  5. 混合损失 — 导频段可加入交叉熵监督信号

使用方式:
  from agent.ppo import PPOTrainer
  trainer = PPOTrainer(agent.ac, lr=3e-4, gamma=0.95, lam=0.95)
  metrics = trainer.update(trajectory)  # 传入一帧数据
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_gae(
    rewards: torch.Tensor,       # (T,)
    values: torch.Tensor,        # (T,)
    gamma: float = 0.99,
    lam: float = 0.95,
    dones: torch.Tensor = None,  # (T,)，全 0 除最后一帧为 1
    normalize: bool = True,
) -> tuple:
    """
    计算 GAE 优势估计。

    通信场景特殊处理:
    - 数据段 r=0，优势完全靠 Critic 自举 + GAE 传播导频信号
    - 归一化防止导频段优势量级淹没问题

    返回:
        advantages: GAE 优势 (T,)
        returns:    折扣回报 G_t = A_t + V(s_t) (T,)
    """
    T = rewards.shape[0]
    if dones is None:
        dones = torch.zeros(T, device=rewards.device)
        dones[-1] = 1.0

    advantages = torch.zeros_like(rewards)
    gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            # 帧末尾，V(s_{T+1})=0
            delta = rewards[t] - values[t]
            next_gae = 0.0
        else:
            # 注意: 如果 done=1 (帧末), 则 V(s_{t+1}) 应视为 0
            next_val = values[t + 1] * (1 - dones[t])
            delta = rewards[t] + gamma * next_val - values[t]
            next_gae = advantages[t + 1]

        gae = delta + gamma * lam * next_gae * (1 - dones[t])
        advantages[t] = gae

    returns = advantages + values

    # 优势归一化 — 通信场景强烈建议开启
    if normalize:
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

    return advantages.detach(), returns.detach()


class PPOTrainer:
    """
    PPO 训练器 — 通信在线均衡场景定制版。

    用法:
        trainer = PPOTrainer(actor_critic, lr=3e-4)
        for _ in range(K_epochs):
            loss = trainer.ppo_update(states, actions, old_log_probs,
                                       advantages, returns)
    """

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
        """包装 GAE 计算函数。"""
        return compute_gae(
            rewards, values,
            gamma=self.gamma, lam=self.lam,
            dones=dones, normalize=True,
        )

    def ppo_update(
        self,
        states: torch.Tensor,        # (T, D)
        actions: torch.Tensor,       # (T, 1)
        old_log_probs: torch.Tensor, # (T, 1)
        advantages: torch.Tensor,    # (T,)
        returns: torch.Tensor,       # (T,)
        known_mask: torch.Tensor = None,  # (T,) bool, 可选
    ) -> dict:
        """
        单次 PPO-Clip 更新。

        返回:
            metrics dict: policy_loss, value_loss, entropy, approx_kl, clamped_frac
        """
        T = states.shape[0]
        device = states.device

        # 当前策略的输出
        log_probs, values, entropy, logits = self.actor_critic.evaluate_actions(
            states, actions
        )
        log_probs = log_probs.squeeze(-1)  # (T,)
        values = values.squeeze(-1)         # (T,)

        # PPO ratio
        ratios = torch.exp(log_probs - old_log_probs.squeeze(-1))

        # PPO-Clip 策略损失
        advantages = advantages.view(-1)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 价值损失
        value_loss = F.mse_loss(values, returns)

        # 熵正则
        entropy_loss = -entropy.mean()

        total_loss = policy_loss + self.value_coef * value_loss + self.ent_coef * entropy_loss

        # 可选: 已知位加入交叉熵监督 (如果提供了 known_mask)
        if known_mask is not None and known_mask.any():
            # 用 logits 计算 BCE (只在已知位)
            bce_loss = F.binary_cross_entropy_with_logits(
                logits.squeeze(-1)[known_mask],
                actions.squeeze(-1)[known_mask].detach(),
            )
            total_loss = total_loss + bce_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # 统计
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

    def train_on_trajectory(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor = None,
        known_mask: torch.Tensor = None,
        k_epochs: int = 4,
        kl_threshold: float = 0.01,
    ) -> dict:
        """
        用一帧完整轨迹做多轮 PPO 更新。

        参数:
            states:    (T, D) 每步状态
            actions:   (T, 1) 采样的比特动作
            log_probs: (T, 1) 采样时的 logπ(a|s)
            rewards:   (T,)   每步奖励
            values:    (T,)   Critic 输出 V(s_t)
            dones:     (T,)   帧结束标志
            known_mask: (T,) bool 已知位 mask
            k_epochs:  PPO 数据复用轮数
            kl_threshold: KL 早停阈值

        返回:
            metrics: 聚合后的训练指标
        """
        T = states.shape[0]

        # GAE 计算
        advantages, returns = self.compute_gae_and_returns(rewards, values, dones)

        # 多轮 PPO 更新
        total_epochs = 0
        all_metrics = []
        for epoch in range(k_epochs):
            m = self.ppo_update(
                states, actions, log_probs,
                advantages, returns,
                known_mask=known_mask,
            )
            total_epochs += 1
            all_metrics.append(m)

            # KL 早停
            if m["approx_kl"] > kl_threshold:
                total_epochs = epoch + 1
                break

        # 聚合指标
        agg = {}
        for key in all_metrics[0].keys():
            values_list = [m[key] for m in all_metrics[:total_epochs]]
            agg[key] = sum(values_list) / len(values_list)

        agg["epochs_used"] = total_epochs
        return agg


def test_gae():
    """测试 GAE 计算。"""
    T = 10
    rewards = torch.zeros(T)
    rewards[3] = 1.0   # 导频位置奖励
    values = torch.zeros(T)
    adv, ret = compute_gae(rewards, values, gamma=0.9, lam=0.9, normalize=False)
    print(f"GAE 测试: rewards at pos 3 =1")
    print(f"优势值: {[f'{a:.3f}' for a in adv.tolist()]}")
    # 导频附近的步应有非零优势
    assert adv[2].abs() > 0  # 导频前一步应受传播影响
    print("GAE 测试通过。")
    print(f"  形状: adv={list(adv.shape)} ret={list(ret.shape)}")


def test_ppo():
    """测试 PPO 训练器。"""
    from agent.actor_critic import ActorCritic, TransformerConfig
    ac = ActorCritic(TransformerConfig())
    trainer = PPOTrainer(ac, lr=1e-3, gamma=0.95, lam=0.95)

    # 模拟一帧数据
    T = 32
    states = torch.randn(T, 45)
    logits = torch.randn(T, 1)
    probs = torch.sigmoid(logits)
    dist = torch.distributions.Bernoulli(probs=probs)
    actions = dist.sample()
    old_log_probs = dist.log_prob(actions)

    known_mask = torch.zeros(T, dtype=torch.bool)
    known_mask[5:10] = True
    rewards = torch.where(known_mask, 1.0, 0.0)
    with torch.no_grad():
        _, values, _, _ = ac.evaluate_actions(states, actions)

    metrics = trainer.train_on_trajectory(
        states, actions, old_log_probs,
        rewards, values.squeeze(-1),
        known_mask=known_mask,
        k_epochs=3,
    )
    print(f"PPO 更新: total_loss={metrics['total_loss']:.4f} "
          f"policy={metrics['policy_loss']:.4f} "
          f"kl={metrics['approx_kl']:.4f} "
          f"epochs={metrics['epochs_used']}")
    print("PPO 训练器自测通过。")


if __name__ == "__main__":
    test_gae()
    print()
    test_ppo()
