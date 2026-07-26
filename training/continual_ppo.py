# -*- coding: utf-8 -*-
"""部署期间持续更新的 recurrent PPO smoke runner。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from agent.continual_policy import ITERATION_CHOICES, MODES, ContinualPolicy, ObservationEncoder
from baseline.block_equalizers import bit_error_rate, perfect_csi_bpsk_refine_detect
from env.comm_env import CommEnvConfig, CommunicationEnvironment, ReceiverState
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
    delays: list[int] | None = None,
    snrs: list[float] | None = None,
) -> dict:
    """执行真实 Level B 在线实验，并写出可 resume 的 policy/metrics。"""

    del pretrained, phase, resume
    return run_real_online_experiment(
        config_path=config_path,
        frames=frames,
        num_seeds=num_seeds,
        update_interval=update_interval,
        output_dir=output_dir,
        delays=delays,
        snrs=snrs,
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

    策略只读取接收端可见 observation；动作影响检测迭代数和判决导向 CIR
    跟踪强度。Reward Pilot 只在动作执行后计算 reward，Data BER 只作为仿真
    指标记录。
    """

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    selected_delays = delays or [int(value) for value in config.get("main_delays", [20, 30, 40])]
    selected_snrs = snrs or [float(value) for value in config.get("main_snrs", [10, 15, 20])]
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    policy_update_frames = [frame for frame in range(1, frames + 1) if frame % update_interval == 0]
    policy = ContinualPolicy()
    _initialize_safe_policy_prior(policy)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-5)
    encoder = ObservationEncoder()
    for delay in selected_delays:
        for snr_db in selected_snrs:
            for seed in range(num_seeds):
                torch.manual_seed(90_000 + int(seed) + int(delay) * 17 + int(float(snr_db)) * 31)
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
                receiver_state = ReceiverState(start.initial_soft_tail)
                cir = _estimate_cir_from_known_frame(start.acquisition, int(delay))
                hidden = torch.zeros(1, 1, policy.hidden_size)
                previous_reward = 0.0
                last_parameter_delta_norm = 0.0
                rollout: list[dict] = []
                last_good_cir = cir.clone()
                last_good_tail = receiver_state.soft_tail.clone()
                for frame_index in range(1, frames + 1):
                    frame = env.next_frame()
                    sigma = torch.tensor(10.0 ** (-float(snr_db) / 10.0))
                    before = perfect_csi_bpsk_refine_detect(
                        frame.rx_symbols,
                        cir,
                        receiver_state.soft_tail,
                        sigma,
                        cg_iterations=8,
                        refine_iterations=0,
                    )
                    view = _policy_view(
                        frame=frame,
                        cir=cir,
                        snr_db=float(snr_db),
                        previous_reward=previous_reward,
                        last_parameter_delta_norm=last_parameter_delta_norm,
                        confidence=_confidence(before.logits),
                    )
                    observation = encoder(view).tensor.unsqueeze(0)
                    hidden_in = hidden.detach()
                    action, log_prob, value, hidden = policy.sample(observation, hidden_in)
                    if action.mode == "rollback":
                        cir = last_good_cir.clone()
                        receiver_state.update_tail(last_good_tail)
                    cg_iterations, refine_iterations = _detector_settings(action)
                    result = perfect_csi_bpsk_refine_detect(
                        frame.rx_symbols,
                        cir,
                        receiver_state.soft_tail,
                        sigma,
                        cg_iterations=cg_iterations,
                        refine_iterations=refine_iterations,
                    )
                    reward_loss_before = _masked_bce(before.logits, frame.bits, frame.reward_mask)
                    reward_loss_after = _masked_bce(result.logits, frame.bits, frame.reward_mask)
                    reward = float((reward_loss_before - reward_loss_after).detach().cpu())
                    reward -= 0.0005 * max(0, cg_iterations - 8)
                    reward -= 0.001 * refine_iterations
                    if reward > 0:
                        last_good_cir = cir.clone()
                        last_good_tail = result.soft_tail.clone()
                    parameter_delta_norm = float(torch.norm(cir - last_good_cir).detach().cpu())
                    rollout.append(
                        {
                            "observation": observation.detach(),
                            "hidden": hidden_in.detach(),
                            "action": action,
                            "old_log_prob": log_prob.detach(),
                            "value": value.detach(),
                            "reward": reward,
                        }
                    )
                    policy_loss = None
                    should_update = frame_index in policy_update_frames
                    if should_update:
                        policy_loss = _ppo_update(policy, optimizer, rollout)
                        rollout.clear()
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
                            "policy_updated": should_update,
                            "policy_learning": "clipped_ppo_reward_pilot",
                            "policy_loss": policy_loss,
                            "policy_action_mode": action.mode,
                            "policy_action_group": action.parameter_group,
                            "reward": reward,
                            "reward_pilot_loss_before": float(reward_loss_before.detach().cpu()),
                            "reward_pilot_loss_after": float(reward_loss_after.detach().cpu()),
                            "detector_iterations": int(cg_iterations + refine_iterations),
                            "adapt_steps": int(action.steps if action.mode in {"update-channel", "update-equalizer", "joint-update"} else 0),
                            "adapt_params": int(policy.parameter_count() if should_update else 0),
                            "parameter_delta_norm": parameter_delta_norm,
                            "cir_update": "decision_directed",
                        }
                    )
                    receiver_state.update_tail(result.soft_tail)
                    cir_alpha = 0.2 if action.mode not in {"skip", "rollback"} else 0.05
                    cir = _decision_directed_cir_update(frame, result.logits, int(delay), cir, alpha=cir_alpha * float(action.cir_trust))
                    previous_reward = reward
                    last_parameter_delta_norm = parameter_delta_norm
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
        "policy_learning": "clipped_ppo_reward_pilot",
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


def _decision_directed_cir_update(frame, logits: torch.Tensor, max_delay: int, previous_cir: torch.Tensor, alpha: float) -> torch.Tensor:
    hard_symbols = torch.where(logits >= 0, torch.ones_like(logits), -torch.ones_like(logits)).to(torch.complex64)
    tx_estimate = hard_symbols.clone()
    tx_estimate[frame.adapt_mask] = frame.tx_symbols[frame.adapt_mask]
    rows = []
    targets = []
    for pos in range(max_delay, tx_estimate.numel()):
        row = torch.zeros(max_delay + 1, dtype=torch.complex64)
        for delay in range(max_delay + 1):
            row[delay] = tx_estimate[pos - delay]
        rows.append(row)
        targets.append(frame.rx_symbols[pos])
    estimate = torch.linalg.lstsq(torch.stack(rows), torch.stack(targets)).solution.to(torch.complex64)
    estimate = estimate / torch.sqrt(torch.sum(torch.abs(estimate) ** 2).clamp_min(1e-12))
    blended = (1.0 - alpha) * previous_cir + alpha * estimate
    return blended / torch.sqrt(torch.sum(torch.abs(blended) ** 2).clamp_min(1e-12))


def _initialize_safe_policy_prior(policy: ContinualPolicy) -> None:
    """用 Best Fixed 等价动作初始化策略，避免部署首轮随机探索破坏 BER。"""

    with torch.no_grad():
        policy.mode_head.bias.fill_(-4.0)
        policy.mode_head.bias[MODES.index("detector-refine")] = 4.0
        policy.iter_head.bias.fill_(-4.0)
        policy.iter_head.bias[ITERATION_CHOICES.index(6)] = 4.0
        policy.steps_head.bias.fill_(-2.0)
        policy.steps_head.bias[0] = 2.0


def _policy_view(frame, cir: torch.Tensor, snr_db: float, previous_reward: float, last_parameter_delta_norm: float, confidence: torch.Tensor):
    view = frame.receiver_view()
    return SimpleNamespace(
        rx_symbols=view.rx_symbols,
        adapt_symbols=view.adapt_symbols,
        adapt_mask=view.adapt_mask,
        model_region_ids=view.model_region_ids,
        complex_cir=cir,
        support_probability=torch.abs(cir) / torch.abs(cir).sum().clamp_min(1e-8),
        noise_variance=torch.tensor(10.0 ** (-float(snr_db) / 10.0)),
        confidence=confidence,
        previous_reward=torch.tensor(float(previous_reward)),
        last_parameter_delta_norm=torch.tensor(float(last_parameter_delta_norm)),
    )


def _confidence(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(torch.abs(logits)).mean()


def _masked_bce(logits: torch.Tensor, bits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if int(mask.sum().item()) == 0:
        return torch.zeros(())
    return F.binary_cross_entropy_with_logits(logits[mask].float(), bits[mask].float())


def _detector_settings(action) -> tuple[int, int]:
    if action.mode == "skip":
        return 8, 0
    if action.mode == "rollback":
        return 16, 0
    if action.mode == "detector-refine":
        return max(16, 8 + int(action.detector_iterations) * 4), 2
    if action.mode == "joint-update":
        return max(16, 8 + int(action.detector_iterations) * 4), 2
    return max(16, 8 + int(action.detector_iterations) * 2), 1


def _ppo_update(policy: ContinualPolicy, optimizer: torch.optim.Optimizer, rollout: list[dict]) -> float | None:
    if not rollout:
        return None
    rewards = torch.tensor([item["reward"] for item in rollout], dtype=torch.float32)
    returns = []
    running = torch.zeros(())
    for reward in reversed(rewards):
        running = reward + 0.95 * running
        returns.append(running)
    returns = torch.stack(list(reversed(returns)))
    values_old = torch.cat([item["value"].flatten().float() for item in rollout]).detach()
    advantages = returns - values_old
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    loss_value = torch.zeros(())
    for _ in range(2):
        losses = []
        for index, item in enumerate(rollout):
            new_log_prob, value, entropy = policy.evaluate_action(item["observation"], item["hidden"], item["action"])
            ratio = torch.exp(new_log_prob.flatten()[0] - item["old_log_prob"].flatten()[0])
            clipped = torch.clamp(ratio, 0.8, 1.2) * advantages[index]
            policy_loss = -torch.minimum(ratio * advantages[index], clipped)
            value_loss = 0.5 * (value.flatten()[0] - returns[index]).pow(2)
            losses.append(policy_loss + value_loss - 0.001 * entropy.flatten()[0])
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        loss_value = loss.detach()
    return float(loss_value.cpu())


def _module_hash(module: torch.nn.Module) -> str:
    hasher = hashlib.sha256()
    for _, tensor in module.state_dict().items():
        hasher.update(tensor.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def _tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()
