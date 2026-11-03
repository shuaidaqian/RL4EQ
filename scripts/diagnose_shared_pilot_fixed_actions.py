# -*- coding: utf-8 -*-
"""同波形的固定安全动作消融；复用窗口守门，不学习策略，不作为RL结果。"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.discrete_safe_policy import DiscretePolicySample, safe_modulation_actions
from agent.rl_modulator import ModulationObservationEncoder
from env.comm_env import ReceiverState
from evaluation.pilot_transfer import _load_branch, paired_interval
from training.pilot_transfer import frame_to_device, make_context, make_episode
from training.windowed_discrete_ppo import WindowedDiscreteOnlineState, run_windowed_discrete_frame


class FixedSafePolicy(nn.Module):
    """精确选择指定既有安全动作，不以近似饱和概率冒充固定策略。"""

    def __init__(self, action_name, device):
        super().__init__()
        self.action_name = action_name
        self.anchor = nn.Parameter(torch.zeros((), device=device))

    def sample(self, observation, hidden, actions, generator=None):
        selected = next(action for action in actions if action.name == self.action_name)
        probabilities = torch.zeros(1, len(actions), device=observation.device)
        probabilities[:, selected.index] = 1.
        zero = self.anchor.reshape(1) * 0
        return DiscretePolicySample(selected, probabilities, zero, zero, zero, hidden)


def fixed_state(model, cir, tail, config, action_name, device):
    names = ["identity", "peft_memory_map_small", "peft_memory_map_medium"]
    actions = safe_modulation_actions(len(model.blocks), device=device, action_names=names,
        peft_max_episode_delta_norm=float(config["rl_peft_max_episode_delta_norm"]), include_channel_state_memory=True)
    policy = FixedSafePolicy(action_name, device)
    return WindowedDiscreteOnlineState(cir=cir, receiver_state=ReceiverState(tail.clone()), model=model,
        policy=policy, optimizer=torch.optim.AdamW(policy.parameters()),
        encoder=ModulationObservationEncoder(include_pilot_state=False), actions=actions,
        hidden=torch.zeros(1, 1, device=device), window_size=int(config["ppo_window_size"]),
        previous_window_reward=0., last_action_delta_norm=0., rollout=[],
        reward_receiver_input="full_rx", adapt_receiver_input="full_rx",
        tail_update_alpha=float(config.get("tail_update_alpha", 1.)),
        tail_refinement_passes=int(config.get("tail_refinement_passes", 0)),
        channel_state_history_frames=int(config["rl_channel_state_history_frames"]))


def run(config, checkpoint, seeds, snr, frames, device, output, paired_rl):
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    raw = target / "frame_metrics.jsonl"
    if raw.exists() or (target / "summary.json").exists():
        raise ValueError("诊断输出已存在，禁止覆盖。")
    if frames < 3 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("固定动作诊断需要至少3帧和独立seed。")
    reference = [json.loads(line) for line in Path(paired_rl).read_text(encoding="utf-8").splitlines()]
    fingerprints = {(row["environment_seed"], row["frame"]): (row["rx_sha256"], row["lookahead_rx_sha256"]) for row in reference
                    if row["method"] == "Frozen Offline NN" and row["snr_db"] == snr}
    frozen_bers = {(row["environment_seed"], row["frame"]): row["ber_data"] for row in reference
                   if row["method"] == "Frozen Offline NN" and row["snr_db"] == snr}
    if set(fingerprints) != {(seed, frame) for seed in seeds for frame in range(1, frames+1)}:
        raise ValueError("配对RL日志未完整覆盖预先指定的同一会话。")
    arms = ("identity", "peft_memory_map_small", "peft_memory_map_medium")
    records = []
    started = time.perf_counter()
    with raw.open("x", encoding="utf-8") as handle:
        for seed in seeds:
            model, _ = _load_branch(Path(checkpoint), config, device)
            _, cir, cfo, tail, episode = make_episode(config, model, seed=seed, snr=snr,
                frames=frames, state_split="heldout_edge", device="cpu")
            states = {}
            for arm in arms:
                branch, _ = _load_branch(Path(checkpoint), config, device)
                states[arm] = fixed_state(branch, cir.to(device), tail.to(device), config, arm, device)
            # 远大于会话长度的更新间隔保证不执行PPO；窗口关闭、守门与回滚仍原样执行。
            update_interval = (frames + int(config["ppo_window_size"])) * 2
            for index, original in enumerate(episode, 1):
                fingerprint = hashlib.sha256(original.rx_symbols.numpy().tobytes()).hexdigest()
                future_hash = hashlib.sha256(original.lookahead_rx_symbols.numpy().tobytes()).hexdigest()
                if (fingerprint, future_hash) != fingerprints[(seed, index)]:
                    raise ValueError("固定动作与已完成RL的当前或后缀波形不同。")
                frame = frame_to_device(original, device)
                for arm, state in states.items():
                    context = make_context(frame, cir.to(device), cfo, state.receiver_state.soft_tail, snr)
                    row = run_windowed_discrete_frame(state, frame, context.condition, snr, index, update_interval)
                    if row["policy_loss"] is not None or row["policy_action_name"] != arm:
                        raise AssertionError("固定动作诊断不得学习策略或切换动作。")
                    row.update(seed=seed, snr_db=snr, arm=arm, frame=index, rx_sha256=fingerprint,
                        lookahead_rx_sha256=future_hash, uses_rl=False, policy_learning_enabled=False,
                        diagnostic_only=True, main_aggregation_allowed=False)
                    records.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            print(json.dumps({"seed": seed, "frames_complete": frames}), flush=True)
    summary = {}
    if any(row["ber_data"] != frozen_bers[(row["seed"], row["frame"])]
           for row in records if row["arm"] == "identity"):
        raise AssertionError("固定identity与原比较的独立Frozen不一致。")
    for arm in arms:
        rows = [r for r in records if r["arm"] == arm]
        gains, differences = [], []
        for seed in seeds:
            active = [r for r in rows if r["seed"] == seed]
            frozen = [r for r in records if r["arm"] == "identity" and r["seed"] == seed]
            values = [left["ber_data"] - right["ber_data"] for left, right in zip(frozen, active)]
            gains.append(float(np.mean(values)))
            third = frames // 3
            differences.append(float(np.mean(values[-third:]) - np.mean(values[:third])))
        summary[arm] = {"ber_data": float(np.mean([r["ber_data"] for r in rows])),
            "gain": paired_interval(gains), "late_minus_early_gain": paired_interval(differences),
            "window_rollbacks": sum(r["window_reward_rollback"] for r in rows),
            "accepted_frames": sum(r["peft_frame_accepted"] for r in rows),
            "max_retained_norm": max(r["trajectory_peft_delta_norm_after_reward"] for r in rows)}
    result = {"schema_version": "shared-pilot-fixed-actions-v1", "diagnostic_only": True,
        "main_aggregation_allowed": False, "uses_rl": False, "policy_learning_enabled": False,
        "seeds": seeds, "snr_db": snr, "frames": frames, "arms": summary,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "paired_rl_raw_sha256": hashlib.sha256(Path(paired_rl).read_bytes()).hexdigest(),
        "fixed_identity_matches_original_frozen": True,
        "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(), "config": config,
        "wall_seconds": time.perf_counter()-started,
        "interpretation": "既有同波形上的固定动作归因诊断，不是新的独立确认，不作RL或主比较结论。"}
    (target / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed-start", type=int, default=110090)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--snr", type=float, default=10.)
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--paired-rl", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(json.loads(Path(args.config).read_text(encoding="utf-8")), args.checkpoint,
        list(range(args.seed_start, args.seed_start+args.num_seeds)), args.snr, args.frames,
        args.device, args.output, args.paired_rl)
