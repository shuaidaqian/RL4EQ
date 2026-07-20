# -*- coding: utf-8 -*-
"""只依赖 Pilot 标签的参数高效在线适配控制器。"""

from dataclasses import dataclass
import math
import time

import torch
import torch.nn.functional as F

from agent.adaptation_policy import AdaptationAction
from agent.neural_equalizer import ExtremeDelayEqualizer
from env.comm_env import ReceivedFrame


OBS_DIM = 10


@dataclass
class AdaptationResult:
    action_name: str
    reward: float
    adapt_pilot_loss: float
    reward_pilot_loss_before: float
    reward_pilot_loss_after: float
    ber_adapt_pilot: float
    ber_reward_pilot: float
    adapt_params: int
    adapt_steps: int
    latency_ms: float
    parameter_delta_norm: float


class AdaptationController:
    """执行 Adapter/输出头更新，Data 标签不参与 observation 与 reward。"""

    def __init__(self, model: ExtremeDelayEqualizer, device: str | torch.device = "cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.loss_ema = 0.0
        self.last_reward = 0.0
        self.last_parameter_delta_norm = 0.0
        self._episode_anchor: dict[str, torch.Tensor] | None = None
        self._last_good: dict[str, torch.Tensor] | None = None

    def start_episode(self) -> None:
        self._episode_anchor = self.model.capture_peft_state()
        self._last_good = {name: value.clone() for name, value in self._episode_anchor.items()}
        self.loss_ema = 0.0
        self.last_reward = 0.0
        self.last_parameter_delta_norm = 0.0

    def end_episode(self) -> None:
        if self._episode_anchor is not None:
            self.model.restore_peft_state(self._episode_anchor)
        self._episode_anchor = None
        self._last_good = None

    def _model_inputs(self, received: ReceivedFrame) -> tuple[torch.Tensor, ...]:
        frame = received.frame
        return (
            received.rx_symbols.unsqueeze(0).to(self.device),
            frame.region_ids.unsqueeze(0).to(self.device),
            frame.adapt_pilot_symbols.unsqueeze(0).to(self.device),
            frame.adapt_pilot_mask.unsqueeze(0).to(self.device),
        )

    def _forward_outputs(
        self, received: ReceivedFrame, grad: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            logits, probabilities = self.model(*self._model_inputs(received))
        return logits[0], probabilities[0]

    def _adapt_metrics(self, received: ReceivedFrame, grad: bool = False) -> dict[str, torch.Tensor]:
        logits, probabilities = self._forward_outputs(received, grad=grad)
        frame = received.frame
        bits = frame.bits.to(self.device)
        adapt_mask = frame.adapt_pilot_mask.to(self.device)
        adapt_loss = F.binary_cross_entropy_with_logits(logits[adapt_mask], bits[adapt_mask])
        predictions = (probabilities >= 0.5).float()
        adapt_ber = (predictions[adapt_mask] != bits[adapt_mask]).float().mean()
        confidence = (probabilities[adapt_mask] - 0.5).abs() * 2.0
        return {
            "loss": adapt_loss,
            "ber": adapt_ber,
            "confidence_mean": confidence.mean(),
            "confidence_std": confidence.std(unbiased=False),
        }

    def _reward_metrics(self, received: ReceivedFrame) -> dict[str, torch.Tensor]:
        logits, probabilities = self._forward_outputs(received, grad=False)
        with torch.no_grad():
            bits = received.frame.bits.to(self.device)
            reward_mask = received.frame.reward_pilot_mask.to(self.device)
            reward_loss = F.binary_cross_entropy_with_logits(logits[reward_mask], bits[reward_mask])
            predictions = (probabilities >= 0.5).float()
            reward_ber = (predictions[reward_mask] != bits[reward_mask]).float().mean()
        return {
            "loss": reward_loss,
            "ber": reward_ber,
        }

    @staticmethod
    def _channel_observables(received: ReceivedFrame) -> tuple[float, float, float]:
        mask = received.frame.adapt_pilot_mask
        rx = torch.complex(received.rx_symbols[mask, 0], received.rx_symbols[mask, 1])
        tx = received.frame.adapt_pilot_symbols[mask].to(torch.complex64)
        gain = (rx * tx.conj()).sum() / tx.abs().square().sum().clamp_min(1e-8)
        residual = rx - gain * tx
        signal_power = (gain * tx).abs().square().mean().clamp_min(1e-8)
        noise_power = residual.abs().square().mean().clamp_min(1e-8)
        snr_hat = float(10.0 * torch.log10(signal_power / noise_power).item())

        max_lag = min(int(received.max_delay_symbols), int(tx.numel()) - 1)
        correlations = []
        for lag in range(max_lag + 1):
            correlations.append((rx[lag:] * tx[: tx.numel() - lag].conj()).abs().mean())
        powers = torch.stack(correlations).square()
        threshold = powers.max() * 0.1
        significant = torch.nonzero(powers >= threshold, as_tuple=False).flatten()
        delay_hat = float(significant.max().item()) if significant.numel() else 0.0
        split = max(1, (max_lag + 1) // 2)
        echo_ratio = float(powers[split:].sum().div(powers.sum().clamp_min(1e-8)).item())
        return snr_hat, delay_hat, echo_ratio

    def build_observation(self, received: ReceivedFrame) -> torch.Tensor:
        metrics = self._adapt_metrics(received, grad=False)
        snr_hat, delay_hat, echo_ratio = self._channel_observables(received)
        loss_value = float(metrics["loss"].item())
        self.loss_ema = loss_value if self.loss_ema == 0.0 else 0.9 * self.loss_ema + 0.1 * loss_value
        observation = torch.tensor(
            [
                loss_value,
                float(metrics["ber"].item()),
                float(metrics["confidence_mean"].item()),
                float(metrics["confidence_std"].item()),
                max(-2.0, min(2.0, snr_hat / 20.0)),
                delay_hat / max(float(received.max_delay_symbols), 1.0),
                echo_ratio,
                self.loss_ema,
                self.last_reward,
                min(math.log1p(max(self.last_parameter_delta_norm, 0.0)) / 5.0, 2.0),
            ],
            dtype=torch.float32,
            device=self.device,
        )
        return observation

    @staticmethod
    def _delta_norm(
        before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]
    ) -> float:
        total = 0.0
        for name in before:
            difference = after[name].float() - before[name].float()
            total += float(difference.square().sum().item())
        return math.sqrt(total)

    def adapt_frame(self, received: ReceivedFrame, action: AdaptationAction) -> AdaptationResult:
        if self._episode_anchor is None:
            self.start_episode()
        start = time.perf_counter()
        before_reward_metrics = self._reward_metrics(received)
        before_state = self.model.capture_peft_state()
        adapt_params = 0

        if action.rollback:
            if self._last_good is not None:
                self.model.restore_peft_state(self._last_good)
        elif action.steps > 0:
            self.model.set_trainable_targets(action.train_adapters, action.train_output)
            parameters = list(self.model.trainable_parameters())
            adapt_params = sum(parameter.numel() for parameter in parameters)
            optimizer = torch.optim.Adam(parameters, lr=action.learning_rate)
            for _ in range(action.steps):
                metrics = self._adapt_metrics(received, grad=True)
                optimizer.zero_grad()
                metrics["loss"].backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()

        after_adapt_metrics = self._adapt_metrics(received, grad=False)
        after_reward_metrics = self._reward_metrics(received)
        after_state = self.model.capture_peft_state()
        delta_norm = self._delta_norm(before_state, after_state)
        reward = (
            float(before_reward_metrics["loss"].item())
            - float(after_reward_metrics["loss"].item())
            - 0.002 * action.steps
            - 1e-6 * adapt_params
            - 0.001 * math.log1p(delta_norm)
        )
        finite = math.isfinite(reward) and all(
            torch.isfinite(value).all().item() for value in after_state.values()
        )
        if not finite:
            if self._last_good is not None:
                self.model.restore_peft_state(self._last_good)
            reward = -1.0
            delta_norm = 0.0
            after_adapt_metrics = self._adapt_metrics(received, grad=False)
            after_reward_metrics = self._reward_metrics(received)
        elif float(after_reward_metrics["loss"].item()) < float(
            before_reward_metrics["loss"].item()
        ):
            self._last_good = self.model.capture_peft_state()

        self.last_reward = reward
        self.last_parameter_delta_norm = delta_norm
        latency_ms = (time.perf_counter() - start) * 1000.0
        return AdaptationResult(
            action_name=action.name,
            reward=reward,
            adapt_pilot_loss=float(after_adapt_metrics["loss"].item()),
            reward_pilot_loss_before=float(before_reward_metrics["loss"].item()),
            reward_pilot_loss_after=float(after_reward_metrics["loss"].item()),
            ber_adapt_pilot=float(after_adapt_metrics["ber"].item()),
            ber_reward_pilot=float(after_reward_metrics["ber"].item()),
            adapt_params=adapt_params,
            adapt_steps=int(action.steps),
            latency_ms=latency_ms,
            parameter_delta_norm=delta_norm,
        )
