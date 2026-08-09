# -*- coding: utf-8 -*-
"""RL 连续调制神经块均衡器的在线 runner。"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from agent.cir_estimator import CIRCondition, condition_from_cir
from agent.modulation import ModulationConfig, ModulationState
from agent.rl_modulator import ContinuousModulationPolicy, ModulationObservationEncoder, initialize_identity_policy_prior
from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from baseline.block_equalizers import bit_error_rate
from env.comm_env import CommEnvConfig, CommunicationEnvironment, ReceiverState
from training.meta_training import _estimate_cir_from_known_frame


@dataclass
class RLModulatedOnlineState:
    cir: torch.Tensor
    receiver_state: ReceiverState
    model: UnfoldedEqualizer
    policy: ContinuousModulationPolicy
    optimizer: torch.optim.Optimizer
    encoder: ModulationObservationEncoder
    hidden: torch.Tensor
    modulation: ModulationState
    previous_reward: float
    last_modulation_delta_norm: float
    rollout: list[dict]


def run_rl_modulated_online(
    config_path: str | Path,
    frames: int,
    num_seeds: int,
    output_dir: str | Path,
    delays: list[int] | None = None,
    snrs: list[float] | None = None,
    pilot_total: int = 128,
    pilot_layout: str = "prefix",
    update_interval: int = 32,
    pretrained: str | Path | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """运行 proposed 方法。

    线上 observation、动作选择和 PPO 更新不读取 Data 标签；Data BER 只作为
    仿真指标写入结果。
    """

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    selected_delays = delays or [int(value) for value in config.get("main_delays", [20, 30, 40])]
    selected_snrs = snrs or [float(value) for value in config.get("snrs_main", config.get("main_snrs", [0, 5, 10, 15]))]
    pretrained_path = Path(pretrained) if pretrained is not None else None
    model_config = _load_model_config(config, pretrained_path)
    rows = []
    last_policy: ContinuousModulationPolicy | None = None
    last_modulation_config: ModulationConfig | None = None
    for delay in selected_delays:
        for snr_db in selected_snrs:
            for seed in range(int(num_seeds)):
                model = _build_equalizer(model_config, pretrained_path, device)
                modulation_config = ModulationConfig(num_adapter_gates=len(model.blocks), num_lora_scales=len(model.blocks))
                encoder = ModulationObservationEncoder()
                policy = ContinuousModulationPolicy(len(encoder.FIELDS), modulation_config).to(device)
                initialize_identity_policy_prior(policy)
                optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-5)
                last_policy = policy
                last_modulation_config = modulation_config
                env = CommunicationEnvironment(
                    CommEnvConfig(
                        level="B",
                        max_delay=int(delay),
                        snr_db=float(snr_db),
                        rho=float(config.get("rho", 0.99)),
                        total_pilot=int(pilot_total),
                        layout=str(pilot_layout),
                        seed=70_000 + int(seed),
                    )
                )
                start = env.reset_episode()
                cir = _estimate_cir_from_known_frame(start.acquisition, int(delay)).to(device)
                state = RLModulatedOnlineState(
                    cir=cir,
                    receiver_state=ReceiverState(start.initial_soft_tail.to(device)),
                    model=model,
                    policy=policy,
                    optimizer=optimizer,
                    encoder=encoder,
                    hidden=policy.initial_hidden(batch_size=1, device=torch.device(device)),
                    modulation=ModulationState.identity(modulation_config, device=torch.device(device)),
                    previous_reward=0.0,
                    last_modulation_delta_norm=0.0,
                    rollout=[],
                )
                for frame_index in range(1, int(frames) + 1):
                    frame = _frame_to_device(env.next_frame(), device)
                    condition = _condition_to_device(condition_from_cir(state.cir, float(snr_db)), device)
                    row = run_rl_modulated_frame(state, frame, condition, float(snr_db), frame_index, update_interval)
                    row.update(
                        {
                            "level": "B",
                            "delay": int(delay),
                            "pilot_total": int(pilot_total),
                            "pilot_layout": str(pilot_layout),
                            "seed": int(seed),
                            "pretrained_loaded": pretrained_path is not None,
                        }
                    )
                    rows.append(row)
    with (target / "frame_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    payload = {
        "schema_version": "rl-modulated-online-v1",
        "pretrained_loaded": pretrained_path is not None,
        "rows": rows,
        "mean_ber_data": float(sum(row["ber_data"] for row in rows) / max(1, len(rows))),
    }
    (target / "online_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if last_policy is not None:
        torch.save(
            {
                "schema_version": "continuous-modulation-policy-v1",
                "state_dict": last_policy.state_dict(),
                "modulation_config": {
                    "num_adapter_gates": int(last_modulation_config.num_adapter_gates) if last_modulation_config else 0,
                    "num_lora_scales": int(last_modulation_config.num_lora_scales) if last_modulation_config else 0,
                },
            },
            target / "policy.pt",
        )
    return payload


def run_rl_modulated_frame(
    state: RLModulatedOnlineState,
    frame,
    condition: CIRCondition,
    snr_db: float,
    frame_index: int,
    update_interval: int,
) -> dict:
    rx_iq = torch.stack((frame.rx_symbols.real, frame.rx_symbols.imag), dim=-1).unsqueeze(0).float()
    region_ids = frame.model_region_ids.unsqueeze(0).long()
    tail = state.receiver_state.soft_tail.unsqueeze(0).to(torch.complex64)
    logits_before, _ = state.model(rx_iq, condition, region_ids, tail, modulation=state.modulation)
    before = logits_before.squeeze(0)
    reward_before = _masked_bce(before, frame.bits, frame.reward_mask)
    view = _policy_view(frame, float(snr_db), before, state.previous_reward, state.last_modulation_delta_norm)
    observation = state.encoder(view).tensor.unsqueeze(0)
    hidden_in = state.hidden.detach()
    action, log_prob, value, state.hidden = state.policy.sample(observation, hidden_in)
    candidate = action.state
    logits_after, _ = state.model(rx_iq, condition, region_ids, tail, modulation=candidate)
    after = logits_after.squeeze(0)
    reward_after = _masked_bce(after, frame.bits, frame.reward_mask)
    delta_norm = float(torch.norm(candidate.to_vector() - state.modulation.to_vector()).detach().cpu())
    reward = float((reward_before - reward_after).detach().cpu()) - 0.001 * delta_norm
    state.rollout.append(
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
    if int(update_interval) > 0 and int(frame_index) % int(update_interval) == 0:
        policy_loss = _ppo_update(state.policy, state.optimizer, state.rollout)
        state.rollout.clear()
    if torch.isfinite(after).all():
        state.modulation = candidate
    tail_len = state.receiver_state.soft_tail.numel()
    next_tail_real = torch.tanh(after[-tail_len:] / 2.0)
    state.receiver_state.update_tail(torch.complex(next_tail_real, torch.zeros_like(next_tail_real)))
    state.previous_reward = reward
    state.last_modulation_delta_norm = delta_norm
    return {
        "method": "RL-Modulated Neural Block Equalizer",
        "snr_db": float(snr_db),
        "frame": int(frame_index),
        "ber_data": bit_error_rate(after[frame.data_mask], frame.bits[frame.data_mask]),
        "ber_reward_pilot": bit_error_rate(after[frame.reward_mask], frame.bits[frame.reward_mask]),
        "ber_adapt_pilot": bit_error_rate(after[frame.adapt_mask], frame.bits[frame.adapt_mask]),
        "reward": reward,
        "reward_pilot_loss_before": float(reward_before.detach().cpu()),
        "reward_pilot_loss_after": float(reward_after.detach().cpu()),
        "policy_learning": "continuous_modulation_ppo",
        "policy_loss": policy_loss,
        "modulation_delta_norm": delta_norm,
        "data_labels_used_online": False,
        "adapt_params": int(state.policy.parameter_count() if policy_loss is not None else 0),
        "adapt_steps": int(1 if policy_loss is not None else 0),
        "parameter_delta_norm": delta_norm,
    }


def _load_model_config(config: dict, pretrained_path: Path | None) -> UnfoldedConfig:
    if pretrained_path is None:
        return UnfoldedConfig.from_dict(config.get("model", {}))
    payload_path = pretrained_path.parent / "model_config.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    model_payload = payload.get("model", payload)
    return UnfoldedConfig.from_dict(model_payload)


def _build_equalizer(model_config: UnfoldedConfig, pretrained_path: Path | None, device: str) -> UnfoldedEqualizer:
    model = UnfoldedEqualizer(model_config).to(device)
    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
        if state_dict is None:
            raise KeyError("checkpoint 必须包含 model_state_dict 或 state_dict。")
        model.load_state_dict(state_dict, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _condition_to_device(condition: CIRCondition, device: str) -> CIRCondition:
    return CIRCondition(
        complex_cir=condition.complex_cir.to(device),
        support_probability=condition.support_probability.to(device),
        noise_variance=condition.noise_variance.to(device),
        confidence=condition.confidence.to(device),
        latent_residual=condition.latent_residual.to(device),
    )


def _frame_to_device(frame, device: str):
    return replace(
        frame,
        bits=frame.bits.to(device),
        tx_symbols=frame.tx_symbols.to(device),
        rx_symbols=frame.rx_symbols.to(device),
        adapt_mask=frame.adapt_mask.to(device),
        reward_mask=frame.reward_mask.to(device),
        data_mask=frame.data_mask.to(device),
        model_region_ids=frame.model_region_ids.to(device),
        tail_symbols=frame.tail_symbols.to(device) if frame.tail_symbols is not None else None,
        true_cir=frame.true_cir.to(device) if frame.true_cir is not None else None,
    )


def _policy_view(frame, snr_db: float, logits: torch.Tensor, previous_reward: float, last_delta_norm: float):
    view = frame.receiver_view()
    return SimpleNamespace(
        rx_symbols=view.rx_symbols,
        adapt_symbols=view.adapt_symbols,
        adapt_mask=view.adapt_mask,
        model_region_ids=view.model_region_ids,
        noise_variance=torch.tensor(10.0 ** (-float(snr_db) / 10.0), device=logits.device),
        confidence=torch.sigmoid(torch.abs(logits)).mean(),
        previous_reward=torch.tensor(previous_reward, device=logits.device),
        last_modulation_delta_norm=torch.tensor(last_delta_norm, device=logits.device),
    )


def _masked_bce(logits: torch.Tensor, bits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if int(mask.sum().item()) == 0:
        return torch.zeros((), device=logits.device)
    return F.binary_cross_entropy_with_logits(logits[mask].float(), bits[mask].float())


def _ppo_update(policy: ContinuousModulationPolicy, optimizer: torch.optim.Optimizer, rollout: list[dict]) -> float | None:
    if not rollout:
        return None
    device = rollout[0]["value"].device
    rewards = torch.tensor([item["reward"] for item in rollout], dtype=torch.float32, device=device)
    returns = []
    running = torch.zeros((), device=device)
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
