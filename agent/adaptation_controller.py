# -*- coding: utf-8 -*-
"""在线参数高效更新控制器。"""

from dataclasses import dataclass
from time import perf_counter
from typing import Dict

import torch
import torch.nn.functional as F

from env.frame_structure import FrameConfig


@dataclass(frozen=True)
class AdaptationStrategy:
    name: str
    lr: float
    steps: int
    train_adapter: bool = True
    train_output: bool = True
    train_sync: bool = False


@dataclass
class AdaptationResult:
    ber_data: float
    ber_pilot: float
    pilot_loss: float
    reward: float
    adapt_params: int
    adapt_steps: int
    latency_ms: float


def make_strategy_table():
    """离散动作表：PPO 输出 action_id，再映射为实际更新策略。"""
    return [
        AdaptationStrategy("skip", lr=0.0, steps=0, train_adapter=False, train_output=False),
        AdaptationStrategy("head-slow", lr=1e-4, steps=1, train_adapter=False, train_output=True),
        AdaptationStrategy("head-fast", lr=5e-4, steps=1, train_adapter=False, train_output=True),
        AdaptationStrategy("adapter-slow", lr=1e-4, steps=1, train_adapter=True, train_output=False, train_sync=True),
        AdaptationStrategy("adapter-fast", lr=5e-4, steps=1, train_adapter=True, train_output=False, train_sync=True),
        AdaptationStrategy("both-slow", lr=1e-4, steps=1, train_adapter=True, train_output=True, train_sync=True),
        AdaptationStrategy("both-fast", lr=5e-4, steps=1, train_adapter=True, train_output=True, train_sync=True),
        AdaptationStrategy("both-deep", lr=5e-4, steps=3, train_adapter=True, train_output=True, train_sync=True),
    ]


class AdaptationController:
    """根据 action 对少量参数执行 pilot loss 更新。"""

    def __init__(self, model, frame_config: FrameConfig, device: torch.device):
        self.model = model
        self.frame_config = frame_config
        self.device = device
        self.pilot_mask = torch.tensor(
            [frame_config.bit_type(t) == "pilot" for t in range(frame_config.frame_len)],
            dtype=torch.bool,
            device=device,
        )
        self.data_mask = torch.tensor(
            [frame_config.bit_type(t) == "data" for t in range(frame_config.frame_len)],
            dtype=torch.bool,
            device=device,
        )

    def build_observation(self, env, history: Dict[str, float] | None = None) -> torch.Tensor:
        """构建低维 RL 状态：只使用在线可观测 pilot 统计量。"""
        history = history or {}
        states = self._states(env)
        bits = env.get_true_bits().to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits, probs = self.model(states)
            pilot_logits = logits[0, self.pilot_mask]
            pilot_bits = bits[self.pilot_mask]
            loss = F.binary_cross_entropy_with_logits(pilot_logits, pilot_bits)
            preds = (probs[0] > 0.5).float()
            pilot_ber = (preds[self.pilot_mask] != pilot_bits).float().mean()
            conf = (probs[0, self.pilot_mask] - 0.5).abs() * 2.0
        obs = torch.tensor([
            float(loss.item()),
            float(pilot_ber.item()),
            float(conf.mean().item()),
            float(conf.std(unbiased=False).item()),
            float(getattr(env.channel, "snr_db", 0.0)) / 30.0,
            float(history.get("loss_ema", loss.item())),
            float(history.get("ber_ema", pilot_ber.item())),
            float(history.get("last_reward", 0.0)),
        ], dtype=torch.float32, device=self.device)
        return obs

    def adapt_frame(self, env, strategy: AdaptationStrategy) -> AdaptationResult:
        states = self._states(env)
        bits = env.get_true_bits().to(self.device)

        self.model.set_trainable_targets(strategy.train_adapter, strategy.train_output, strategy.train_sync)
        before_loss = self._pilot_loss(states, bits).detach()
        start = perf_counter()

        if strategy.steps > 0 and self.model.trainable_parameter_count() > 0:
            optimizer = torch.optim.AdamW(self.model.trainable_parameters(), lr=strategy.lr)
            self.model.train()
            for _ in range(strategy.steps):
                optimizer.zero_grad()
                loss = self._pilot_loss(states, bits)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.model.trainable_parameters()), 1.0)
                optimizer.step()

        latency_ms = (perf_counter() - start) * 1000.0
        self.model.eval()
        with torch.no_grad():
            logits, probs = self.model(states)
            after_loss = F.binary_cross_entropy_with_logits(
                logits[0, self.pilot_mask],
                bits[self.pilot_mask],
            )
            preds = (probs[0] > 0.5).float()
            ber_pilot = (preds[self.pilot_mask] != bits[self.pilot_mask]).float().mean().item()
            ber_data = (preds[self.data_mask] != bits[self.data_mask]).float().mean().item()

        reward = float((before_loss - after_loss).item() - 0.01 * strategy.steps)
        return AdaptationResult(
            ber_data=ber_data,
            ber_pilot=ber_pilot,
            pilot_loss=float(after_loss.item()),
            reward=reward,
            adapt_params=self.model.trainable_parameter_count(),
            adapt_steps=strategy.steps,
            latency_ms=latency_ms,
        )

    def _pilot_loss(self, states: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(states)
        return F.binary_cross_entropy_with_logits(logits[0, self.pilot_mask], bits[self.pilot_mask])

    def _states(self, env) -> torch.Tensor:
        return env.get_all_states().unsqueeze(0).to(self.device)
