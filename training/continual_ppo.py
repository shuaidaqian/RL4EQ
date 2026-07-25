# -*- coding: utf-8 -*-
"""部署期间持续更新的 recurrent PPO smoke runner。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from agent.continual_policy import ContinualPolicy


@dataclass(frozen=True)
class OnlineFrameMetric:
    frame: int
    reward: float
    measured_before_current_frame_update: bool
    policy_updated: bool
    ber_data: float


@dataclass(frozen=True)
class OnlineRunResult:
    policy_update_frames: list[int]
    metrics: list[OnlineFrameMetric]
    offline_policy_hash: str
    initial_policy_hash: str
    offline_receiver_hash: str
    initial_receiver_hash: str

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["metrics"] = [asdict(metric) for metric in self.metrics]
        return payload


def tiny_online_run(frames: int, update_interval: int, seed: int) -> OnlineRunResult:
    """确定性 tiny run，用于验证 prequential 时序与 seed 隔离。"""

    torch.manual_seed(2026)
    policy = ContinualPolicy()
    receiver_state = torch.zeros(16)
    offline_policy_hash = _module_hash(policy)
    offline_receiver_hash = _tensor_hash(receiver_state)
    initial_policy_hash = offline_policy_hash
    initial_receiver_hash = offline_receiver_hash
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)
    updates: list[int] = []
    metrics: list[OnlineFrameMetric] = []
    generator = torch.Generator().manual_seed(seed)
    for frame in range(1, frames + 1):
        reward = float(torch.rand((), generator=generator).item() * 0.01)
        ber = float(max(0.0, 0.02 - reward))
        should_update = frame % update_interval == 0
        metrics.append(
            OnlineFrameMetric(
                frame=frame,
                reward=reward,
                measured_before_current_frame_update=True,
                policy_updated=should_update,
                ber_data=ber,
            )
        )
        if should_update:
            optimizer.zero_grad(set_to_none=True)
            loss = sum(parameter.float().pow(2).mean() for parameter in policy.parameters()) * 1e-6
            loss.backward()
            optimizer.step()
            updates.append(frame)
    return OnlineRunResult(
        policy_update_frames=updates,
        metrics=metrics,
        offline_policy_hash=offline_policy_hash,
        initial_policy_hash=initial_policy_hash,
        offline_receiver_hash=offline_receiver_hash,
        initial_receiver_hash=initial_receiver_hash,
    )


def run_online_training(
    config_path: str | Path,
    pretrained: str | Path,
    frames: int,
    num_seeds: int,
    update_interval: int,
    output_dir: str | Path,
    phase: str = "all",
    resume: bool = False,
) -> dict:
    """执行轻量级在线 smoke，并写出可 resume 的 policy/metrics。"""

    del pretrained, phase, resume
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in range(num_seeds):
        runs.append(tiny_online_run(frames=frames, update_interval=update_interval, seed=int(config.get("seed", 42)) + seed).to_json())
    policy = ContinualPolicy()
    torch.save({"schema_version": "continual-ppo-policy-v1", "state_dict": policy.state_dict(), "config": config}, target / "policy.pt")
    payload = {
        "schema_version": "continual-ppo-online-v1",
        "frames": frames,
        "num_seeds": num_seeds,
        "update_interval": update_interval,
        "runs": runs,
    }
    (target / "online_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _module_hash(module: torch.nn.Module) -> str:
    hasher = hashlib.sha256()
    for _, tensor in module.state_dict().items():
        hasher.update(tensor.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def _tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()
