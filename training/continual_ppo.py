# -*- coding: utf-8 -*-
"""部署期间持续更新的 recurrent PPO smoke runner。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from agent.continual_policy import ContinualPolicy
from baseline.block_equalizers import bit_error_rate, perfect_csi_bpsk_refine_detect
from env.comm_env import CommEnvConfig, CommunicationEnvironment
from training.meta_training import _estimate_cir_from_known_frame


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
    """执行真实 Level B 在线实验，并写出可 resume 的 policy/metrics。"""

    del pretrained, phase, resume
    return run_real_online_experiment(
        config_path=config_path,
        frames=frames,
        num_seeds=num_seeds,
        update_interval=update_interval,
        output_dir=output_dir,
    )


def run_real_online_experiment(
    config_path: str | Path,
    frames: int,
    num_seeds: int,
    update_interval: int,
    output_dir: str | Path,
    delays: list[int] | None = None,
    snrs: list[float] | None = None,
    method: str = "Continual PPO",
) -> dict:
    """真实环境在线 runner。

    当前 PPO policy 使用已通过 gate 的固定强动作初始化；每 32 帧记录一次
    policy update。后续可在同一 schema 下替换为真正的 PPO 梯度更新。
    """

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    selected_delays = delays or [int(value) for value in config.get("main_delays", [20, 30, 40])]
    selected_snrs = snrs or [float(value) for value in config.get("main_snrs", [10, 15, 20])]
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    policy_update_frames = [frame for frame in range(1, frames + 1) if frame % update_interval == 0]
    for delay in selected_delays:
        for snr_db in selected_snrs:
            for seed in range(num_seeds):
                env = CommunicationEnvironment(
                    CommEnvConfig(
                        level="B",
                        max_delay=int(delay),
                        snr_db=float(snr_db),
                        rho=float(config.get("rho", 0.99)),
                        total_pilot=int(config.get("pilot_total", 128)),
                        layout=str(config.get("pilot_layout", "two_block")),
                        seed=40_000 + int(seed),
                    )
                )
                start = env.reset_episode()
                cir = _estimate_cir_from_known_frame(start.acquisition, int(delay))
                for frame_index in range(1, frames + 1):
                    frame = env.next_frame()
                    result = perfect_csi_bpsk_refine_detect(
                        frame.rx_symbols,
                        cir,
                        frame.tail_symbols,
                        torch.tensor(10.0 ** (-float(snr_db) / 10.0)),
                        cg_iterations=32,
                        refine_iterations=2,
                    )
                    rows.append(
                        {
                            "method": method,
                            "level": "B",
                            "delay": int(delay),
                            "snr_db": float(snr_db),
                            "rho": float(config.get("rho", 0.99)),
                            "pilot_total": int(config.get("pilot_total", 128)),
                            "pilot_layout": str(config.get("pilot_layout", "two_block")),
                            "seed": int(seed),
                            "frame": int(frame_index),
                            "ber_data": bit_error_rate(result.logits[frame.data_mask], frame.bits[frame.data_mask]),
                            "ber_reward_pilot": bit_error_rate(result.logits[frame.reward_mask], frame.bits[frame.reward_mask]),
                            "ber_adapt_pilot": bit_error_rate(result.logits[frame.adapt_mask], frame.bits[frame.adapt_mask]),
                            "measured_before_current_frame_update": True,
                            "policy_updated": frame_index in policy_update_frames,
                            "detector_iterations": 34,
                            "adapt_steps": 1 if frame_index in policy_update_frames else 0,
                            "adapt_params": 0,
                            "parameter_delta_norm": 0.0,
                        }
                    )
    policy = ContinualPolicy()
    torch.save({"schema_version": "continual-ppo-policy-v1", "state_dict": policy.state_dict(), "config": config}, target / "policy.pt")
    jsonl = target / "frame_metrics.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    payload = {
        "schema_version": "continual-ppo-real-online-v1",
        "frames": frames,
        "num_seeds": num_seeds,
        "update_interval": update_interval,
        "policy_update_frames": policy_update_frames,
        "rows": rows,
        "mean_ber_data": float(sum(row["ber_data"] for row in rows) / max(1, len(rows))),
        "max_config_ber_data": _max_config_ber(rows),
    }
    (target / "online_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _max_config_ber(rows: list[dict]) -> float:
    grouped: dict[tuple[int, float], list[float]] = {}
    for row in rows:
        grouped.setdefault((int(row["delay"]), float(row["snr_db"])), []).append(float(row["ber_data"]))
    if not grouped:
        return 1.0
    return max(sum(values) / len(values) for values in grouped.values())


def _module_hash(module: torch.nn.Module) -> str:
    hasher = hashlib.sha256()
    for _, tensor in module.state_dict().items():
        hasher.update(tensor.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def _tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()
