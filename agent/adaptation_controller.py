# -*- coding: utf-8 -*-
"""分层动作执行、安全回滚与 Reward Pilot shadow reward。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch

from agent.continual_policy import HierarchicalAction
from agent.peft import PEFTSnapshot
from agent.unfolded_equalizer import UnfoldedEqualizer


@dataclass(frozen=True)
class AdaptationResult:
    reward: float
    rollback: bool
    adapt_steps: int
    adapt_params: int
    parameter_delta_norm: float
    detector_iterations: int


def compute_reward(loss_before, loss_after, shadow_loss, beta, eps=1e-8):
    """只基于 Reward Pilot loss 与 episode shadow loss 计算在线 reward。"""

    before = torch.as_tensor(loss_before).float()
    after = torch.as_tensor(loss_after).float()
    shadow = torch.as_tensor(shadow_loss).float()
    immediate = torch.log((before + eps) / (after + eps))
    cumulative = torch.log((shadow + eps) / (after + eps))
    return float((immediate + beta * cumulative).detach().cpu())


class AdaptationController:
    """执行 PPO 分层动作，并维护 episode 内 PEFT 持续状态。"""

    SAFE_GROUP = {"conditioner_peft"}

    def __init__(self, model: UnfoldedEqualizer, beta: float = 0.5, max_delta_norm: float = 10.0):
        self.model = model
        self.beta = beta
        self.max_delta_norm = max_delta_norm
        self._initial_checkpoint: dict[str, torch.Tensor] | None = None
        self._episode_start: PEFTSnapshot | None = None
        self._last_safe: PEFTSnapshot | None = None
        self._seed: int | None = None

    def reset_episode(self, seed: int, checkpoint: Mapping[str, torch.Tensor]) -> None:
        self._seed = seed
        self._initial_checkpoint = {name: tensor.detach().clone() for name, tensor in checkpoint.items()}
        self.model.load_state_dict(self._initial_checkpoint, strict=True)
        self.model.set_trainable_groups(set())
        self._episode_start = self.model.peft.snapshot(self.SAFE_GROUP)
        self._last_safe = self.model.peft.snapshot(self.SAFE_GROUP)

    def execute(
        self,
        action: HierarchicalAction,
        reward_loss_before: torch.Tensor,
        reward_loss_after: torch.Tensor,
        shadow_loss: torch.Tensor,
    ) -> AdaptationResult:
        if self._last_safe is None or self._episode_start is None:
            raise RuntimeError("必须先调用 reset_episode()。")
        if action.mode == "rollback":
            self.model.peft.restore(self._last_safe)
            return AdaptationResult(0.0, True, 0, 0, self._delta_norm(), action.detector_iterations)
        if not self._action_is_finite(action):
            return self._hard_rollback(action.detector_iterations)
        before = self.model.peft.snapshot(self.SAFE_GROUP)
        try:
            adapt_params = self._apply_action(action)
            if not self._parameters_are_finite():
                self.model.peft.restore(before)
                return self._hard_rollback(action.detector_iterations)
            delta = self._delta_norm()
            if delta > self.max_delta_norm:
                self.model.peft.restore(before)
                return self._hard_rollback(action.detector_iterations)
            reward = compute_reward(reward_loss_before, reward_loss_after, shadow_loss, self.beta)
            if not math.isfinite(reward):
                self.model.peft.restore(before)
                return self._hard_rollback(action.detector_iterations)
            self._last_safe = self.model.peft.snapshot(self.SAFE_GROUP)
            return AdaptationResult(
                reward=reward,
                rollback=False,
                adapt_steps=0 if action.mode in {"skip", "detector-refine"} else action.steps,
                adapt_params=adapt_params,
                parameter_delta_norm=delta,
                detector_iterations=action.detector_iterations,
            )
        finally:
            self.model.set_trainable_groups(set())

    def peft_vector(self) -> torch.Tensor:
        values = [parameter.detach().flatten().cpu() for _, parameter in self.model.peft.named_group_parameters(self.SAFE_GROUP)]
        return torch.cat(values) if values else torch.zeros(0)

    def last_safe_vector(self) -> torch.Tensor:
        if self._last_safe is None:
            return torch.zeros(0)
        return _snapshot_vector(self._last_safe)

    def _apply_action(self, action: HierarchicalAction) -> int:
        if action.mode in {"skip", "detector-refine"}:
            return 0
        groups = {action.parameter_group}
        if action.mode in {"update-channel", "joint-update"}:
            groups.add("conditioner_film")
        self.model.set_trainable_groups(groups)
        params = self.model.trainable_parameters()
        with torch.no_grad():
            for parameter in params:
                parameter.add_(torch.full_like(parameter, float(action.learning_rate)))
        return sum(parameter.numel() for parameter in params)

    def _hard_rollback(self, detector_iterations: int) -> AdaptationResult:
        if self._last_safe is not None:
            self.model.peft.restore(self._last_safe)
        return AdaptationResult(-1.0, True, 0, 0, self._delta_norm(), detector_iterations)

    def _delta_norm(self) -> float:
        if self._episode_start is None:
            return 0.0
        return self.model.peft.delta_norm(self._episode_start)

    def _parameters_are_finite(self) -> bool:
        return all(torch.isfinite(parameter.detach()).all().item() for parameter in self.model.parameters())

    @staticmethod
    def _action_is_finite(action: HierarchicalAction) -> bool:
        values = [
            action.learning_rate,
            action.proximal_weight,
            action.reconstruction_weight,
            action.damping,
            action.cir_trust,
        ]
        return all(math.isfinite(float(value)) for value in values)


def _snapshot_vector(snapshot: PEFTSnapshot) -> torch.Tensor:
    values = [tensor.detach().flatten().cpu() for tensor in snapshot.tensors.values()]
    return torch.cat(values) if values else torch.zeros(0)
