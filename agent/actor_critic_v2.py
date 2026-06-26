# -*- coding: utf-8 -*-
"""
基于 Transformer 编码器的 Actor-Critic 网络 — 用于在线信道均衡

用 Transformer 替代 LSTM 的核心考量:
  - 自注意力机制可以捕获长距离 ISI 依赖 (最多 15 个符号周期)
  - 并行计算: 整帧 512 个状态一次前向 (LSTM 需要 512 步)
  - 因果掩码保证在线推理的时序一致性

网络架构:
  输入 s_t (45 维) -> Linear(45->64) + LayerNorm + 位置编码
    -> TransformerEncoder (2 层, 4 头, d_model=64, FFN=128, 因果掩码)
    -> Actor Head: Linear(64->64, ReLU) -> Linear(64->1) -> Sigmoid
    -> Critic Head: Linear(64->64, ReLU) -> Linear(64->1)

推理: 滑动窗口维护最近 W=128 个状态
训练: 整帧 512 状态一次前向 (利用并行计算)
"""

import torch
import torch.nn as nn
import torch.distributions as dist
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    state_dim: int = 45
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.0
    max_len: int = 512
    window_size: int = 128
    cond_dim: int = 8



class FiLMLayer(nn.Module):
    def __init__(self,d_model,cond_dim):
        super().__init__()
        self.gamma=nn.Linear(cond_dim,d_model)
        self.beta=nn.Linear(cond_dim,d_model)
    def forward(self,x,cond):
        g=self.gamma(cond).unsqueeze(1)
        b=self.beta(cond).unsqueeze(1)
        return g*x+b

class ActorCritic(nn.Module):
    """基于 Transformer 编码器的 Actor-Critic 网络。

    核心方法:
      forward(states, mask): 整帧前向, 返回 (probs, values, logits)
      evaluate_actions(states, actions): 批量评估动作
      sample_action(state, context): 单步推理采样
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.window_size = config.window_size

        self.input_proj = nn.Sequential(
            nn.Linear(config.state_dim, config.d_model),
            nn.LayerNorm(config.d_model),
        )
        self.pos_encoding = nn.Embedding(config.max_len, config.d_model)

        self.cond_net = nn.Sequential(
            nn.Linear(config.state_dim, config.cond_dim * 4),
            nn.ReLU(),
            nn.Linear(config.cond_dim * 4, config.cond_dim),
            nn.Tanh(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward, dropout=config.dropout,
            activation="relu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        self.film_layers = nn.ModuleList([
            FiLMLayer(config.d_model, config.cond_dim)
            for _ in range(config.n_layers)
        ])

        self.actor_fc = nn.Sequential(
            nn.Linear(config.d_model, 64), nn.ReLU(), nn.Linear(64, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(config.d_model, 64), nn.ReLU(), nn.Linear(64, 1),
        )
        self._causal_mask_cache = {}

    def extract_condition(self, states):
        B,T,D = states.shape
        avg_state = states.mean(dim=1)
        return self.cond_net(avg_state)

    def _get_causal_mask(self, T: int, device):
        """因果掩码: 确保 t 时刻只能看到前 t 个状态。"""
        if T not in self._causal_mask_cache:
            mask = torch.triu(torch.ones(T, T, device=device) * float("-inf"), diagonal=1)
            self._causal_mask_cache[T] = mask
        return self._causal_mask_cache[T].to(device)

    def forward(self, states, mask=None):
        B, T, D = states.shape
        x = self.input_proj(states)
        pos = torch.arange(T, device=states.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_encoding(pos)
        if mask is None:
            mask = self._get_causal_mask(T, states.device)
        x = self.transformer(x, mask=mask)
        cond = self.extract_condition(states)
        for film in self.film_layers:
            x = film(x, cond)
        logits = self.actor_fc(x)
        values = self.critic(x)
        return torch.sigmoid(logits), values, logits

    def evaluate_actions(self, states, actions):
        T = states.shape[0]
        mask = self._get_causal_mask(T, states.device)
        probs, values, logits = self.forward(states.unsqueeze(0), mask=mask)
        probs = probs.squeeze(0)
        values = values.squeeze(0)
        logits = logits.squeeze(0)
        p = probs.clamp(1e-6, 1 - 1e-6)
        b = dist.Bernoulli(probs=p)
        return b.log_prob(actions), values, b.entropy(), logits

    def sample_action(self, state, context=None):
        """单时间步推理, 维护滑动上下文窗口。"""
        dev = state.device
        s = state.unsqueeze(0).unsqueeze(0)
        if context is None:
            context = s
        else:
            context = torch.cat([context, s], dim=1)
            if context.shape[1] > self.window_size:
                context = context[:, -self.window_size:]
        T = context.shape[1]
        mask = self._get_causal_mask(T, dev)
        probs, values, _ = self.forward(context, mask=mask)
        prob = probs[0, -1:]
        value = values[0, -1:]
        p = prob.clamp(1e-6, 1 - 1e-6)
        b = dist.Bernoulli(probs=p)
        action = b.sample() if self.training else (p > 0.5).float()
        return action, b.log_prob(action), value, context

    def _init_hidden(self, *args, **kwargs):
        return None


def test_transformer_ac():
    cfg = TransformerConfig()
    ac = ActorCritic(cfg)
    params = sum(p.numel() for p in ac.parameters())
    print(f"参数量: {params}")
    states = torch.randn(1, 10, 45)
    probs, values, logits = ac.forward(states)
    print(f"前向: probs={list(probs.shape)}, values={list(values.shape)}")
    T_s = torch.randn(20, 45)
    T_a = torch.randint(0, 2, (20, 1)).float()
    lp, v, ent, lg = ac.evaluate_actions(T_s, T_a)
    print(f"评估: lp={list(lp.shape)}, v={list(v.shape)}")
    ctx = None
    for t in range(20):
        a, lp, v, ctx = ac.sample_action(torch.randn(45), ctx)
    print(f"20 步后 context: {list(ctx.shape)}")
    print("Transformer AC 自测通过。")


if __name__ == "__main__":
    test_transformer_ac()
