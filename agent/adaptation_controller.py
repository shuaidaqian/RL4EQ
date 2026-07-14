# -*- coding: utf-8 -*-
"""在线参数高效更新控制器。"""

from dataclasses import dataclass
from time import perf_counter
from typing import Dict

import torch
import torch.nn.functional as F

from env.frame_structure import FrameConfig


OBS_DIM = 14


@dataclass(frozen=True)
class AdaptationStrategy:
    name: str
    lr: float
    steps: int
    train_adapter: bool = True
    train_output: bool = True
    train_sync: bool = False
    regularization: float = 0.02
    pseudo_label_gate: str = "off"
    pseudo_weight: float = 0.0


@dataclass
class AdaptationResult:
    ber_data: float
    ber_pilot: float
    pilot_loss: float
    reward: float
    adapt_params: int
    adapt_steps: int
    latency_ms: float
    parameter_delta_norm: float = 0.0


def make_strategy_table():
    """离散动作表：PPO 输出 action_id，再映射为实际在线更新策略。"""
    return [
        AdaptationStrategy("skip", lr=0.0, steps=0, train_adapter=False, train_output=False, regularization=0.0),
        AdaptationStrategy("head-slow", lr=1e-4, steps=1, train_adapter=False, train_output=True, regularization=0.03),
        AdaptationStrategy("head-fast", lr=5e-4, steps=1, train_adapter=False, train_output=True, regularization=0.05),
        AdaptationStrategy("adapter-slow", lr=1e-4, steps=1, train_adapter=True, train_output=False, train_sync=True, regularization=0.03),
        AdaptationStrategy("adapter-fast", lr=5e-4, steps=1, train_adapter=True, train_output=False, train_sync=True, regularization=0.06),
        AdaptationStrategy("both-slow", lr=1e-4, steps=1, train_adapter=True, train_output=True, train_sync=True, regularization=0.04),
        AdaptationStrategy("both-fast", lr=5e-4, steps=1, train_adapter=True, train_output=True, train_sync=True, regularization=0.07),
        AdaptationStrategy("both-deep", lr=5e-4, steps=3, train_adapter=True, train_output=True, train_sync=True, regularization=0.10),
        AdaptationStrategy("pseudo-both-fast", lr=3e-4, steps=1, train_adapter=True, train_output=True, train_sync=True, regularization=0.08, pseudo_label_gate="agree-high", pseudo_weight=0.35),
        AdaptationStrategy("pseudo-both-deep", lr=3e-4, steps=3, train_adapter=True, train_output=True, train_sync=True, regularization=0.12, pseudo_label_gate="agree-high", pseudo_weight=0.35),
    ]


class AdaptationController:
    """根据 action 对少量参数执行 pilot-only 更新。"""

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
        """构建 RL 状态，只使用 pilot 和历史可观测量。"""
        history = history or {}
        states = self._states(env)
        bits = env.get_true_bits().to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits, probs = self.model(states)
            pilot_logits = logits[0, self.pilot_mask]
            pilot_probs = probs[0, self.pilot_mask]
            pilot_bits = bits[self.pilot_mask]
            loss = F.binary_cross_entropy_with_logits(pilot_logits, pilot_bits)
            preds = (pilot_probs > 0.5).float()
            pilot_ber = (preds != pilot_bits).float().mean()
            conf = (pilot_probs - 0.5).abs() * 2.0
            residual = pilot_probs - pilot_bits
            residual_autocorr = self._lag1_autocorr(residual)
            error_flags = (preds != pilot_bits).float()
            burst_len = self._max_burst_length(error_flags) / max(float(error_flags.numel()), 1.0)
            q10, q50, q90 = torch.quantile(conf, torch.tensor([0.1, 0.5, 0.9], device=self.device))

        return torch.tensor([
            float(loss.item()),
            float(pilot_ber.item()),
            float(conf.mean().item()),
            float(conf.std(unbiased=False).item()),
            float(q10.item()),
            float(q50.item()),
            float(q90.item()),
            float(burst_len),
            float(residual_autocorr),
            float(getattr(env.channel, "snr_db", 0.0)) / 30.0,
            float(history.get("loss_ema", loss.item())),
            float(history.get("ber_ema", pilot_ber.item())),
            float(history.get("last_reward", 0.0)),
            float(history.get("last_latency_ms", 0.0)) / 100.0,
        ], dtype=torch.float32, device=self.device)

    def adapt_frame(
        self,
        env,
        strategy: AdaptationStrategy,
        pseudo_bits: torch.Tensor | None = None,
        pseudo_mask: torch.Tensor | None = None,
    ) -> AdaptationResult:
        states = self._states(env)
        bits = env.get_true_bits().to(self.device)

        self.model.set_trainable_targets(strategy.train_adapter, strategy.train_output, strategy.train_sync)
        before_stats = self._pilot_stats(states, bits)
        before_params = [param.detach().clone() for param in self.model.trainable_parameters()]
        start = perf_counter()

        if strategy.steps > 0 and self.model.trainable_parameter_count() > 0:
            optimizer = torch.optim.AdamW(self.model.trainable_parameters(), lr=strategy.lr)
            self.model.train()
            for _ in range(strategy.steps):
                optimizer.zero_grad()
                loss = self._adapt_loss(states, bits, strategy, pseudo_bits, pseudo_mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.model.trainable_parameters()), 1.0)
                optimizer.step()

        latency_ms = (perf_counter() - start) * 1000.0
        self.model.eval()
        with torch.no_grad():
            after_stats = self._pilot_stats(states, bits)
            _, probs = self.model(states)
            preds = (probs[0] > 0.5).float()
            ber_data = (preds[self.data_mask] != bits[self.data_mask]).float().mean().item()

        delta_norm = self._parameter_delta_norm(before_params)
        reward = float(
            (before_stats["loss"] - after_stats["loss"])
            + 0.5 * (before_stats["ber"] - after_stats["ber"])
            + 0.05 * (after_stats["conf_mean"] - before_stats["conf_mean"])
            - strategy.regularization * delta_norm
            - 0.01 * strategy.steps
            - 0.0001 * latency_ms
            - 0.02 * after_stats["residual_autocorr_abs"]
        )
        return AdaptationResult(
            ber_data=ber_data,
            ber_pilot=float(after_stats["ber"]),
            pilot_loss=float(after_stats["loss"]),
            reward=reward,
            adapt_params=self.model.trainable_parameter_count(),
            adapt_steps=strategy.steps,
            latency_ms=latency_ms,
            parameter_delta_norm=delta_norm,
        )

    def _pilot_loss(self, states: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(states)
        return F.binary_cross_entropy_with_logits(logits[0, self.pilot_mask], bits[self.pilot_mask])

    def _adapt_loss(
        self,
        states: torch.Tensor,
        bits: torch.Tensor,
        strategy: AdaptationStrategy,
        pseudo_bits: torch.Tensor | None,
        pseudo_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        logits, _ = self.model(states)
        loss = F.binary_cross_entropy_with_logits(logits[0, self.pilot_mask], bits[self.pilot_mask])
        if (
            strategy.pseudo_label_gate != "off"
            and strategy.pseudo_weight > 0.0
            and pseudo_bits is not None
            and pseudo_mask is not None
            and bool(pseudo_mask.any().item())
        ):
            loss = loss + strategy.pseudo_weight * F.binary_cross_entropy_with_logits(
                logits[0, pseudo_mask],
                pseudo_bits[pseudo_mask].to(logits.device),
            )
        return loss

    def _pilot_stats(self, states: torch.Tensor, bits: torch.Tensor) -> Dict[str, float]:
        logits, probs = self.model(states)
        pilot_logits = logits[0, self.pilot_mask]
        pilot_probs = probs[0, self.pilot_mask]
        pilot_bits = bits[self.pilot_mask]
        loss = F.binary_cross_entropy_with_logits(pilot_logits, pilot_bits)
        preds = (pilot_probs > 0.5).float()
        ber = (preds != pilot_bits).float().mean()
        conf = (pilot_probs - 0.5).abs() * 2.0
        residual = pilot_probs - pilot_bits
        residual_autocorr = self._lag1_autocorr(residual)
        return {
            "loss": float(loss.detach().item()),
            "ber": float(ber.detach().item()),
            "conf_mean": float(conf.detach().mean().item()),
            "residual_autocorr_abs": abs(float(residual_autocorr)),
        }

    def _parameter_delta_norm(self, before_params) -> float:
        after_params = list(self.model.trainable_parameters())
        if not before_params or not after_params:
            return 0.0
        delta_sq = torch.tensor(0.0, device=self.device)
        base_sq = torch.tensor(0.0, device=self.device)
        for before, after in zip(before_params, after_params):
            before = before.to(after.device)
            delta_sq = delta_sq + torch.sum((after.detach() - before) ** 2)
            base_sq = base_sq + torch.sum(before ** 2)
        return float(torch.sqrt(delta_sq / base_sq.clamp_min(1e-12)).item())

    def _lag1_autocorr(self, values: torch.Tensor) -> float:
        if values.numel() < 2:
            return 0.0
        x0 = values[:-1] - values[:-1].mean()
        x1 = values[1:] - values[1:].mean()
        denom = torch.sqrt(torch.sum(x0 ** 2) * torch.sum(x1 ** 2)).clamp_min(1e-12)
        return float((torch.sum(x0 * x1) / denom).detach().item())

    def _max_burst_length(self, flags: torch.Tensor) -> float:
        max_run = 0.0
        cur = 0.0
        for item in flags.detach().cpu().tolist():
            if item > 0.5:
                cur += 1.0
                max_run = max(max_run, cur)
            else:
                cur = 0.0
        return max_run

    def _states(self, env) -> torch.Tensor:
        return env.get_all_states().unsqueeze(0).to(self.device)
