# -*- coding: utf-8 -*-
"""受限神经记忆的共同状态消融；不提交参数，不执行 PPO，不作为主结果。"""

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.joint_pilot_state import SharedNeuralPilotObjective
from evaluation.pilot_transfer import _load_branch, paired_interval
from scripts.diagnose_cross_frame_pilot_history import history_candidate
from training.pilot_transfer import channel_state_reward_loss, context_logits, frame_to_device, make_context, make_episode
from training.windowed_discrete_ppo import should_accept_reward_guard


def shared_candidate(model, contexts, settings, *, prior_copies=1):
    """优化相同神经记忆，AdamW/裁剪/半径/当前 Reward 守门沿用既定协议。"""
    candidate = copy.deepcopy(model)
    # 深拷贝不会保留 Parameter 的 _peft_group 属性，显式限定副本的可训练张量。
    for value in candidate.parameters():
        value.requires_grad_(False)
    parameter = candidate.channel_state.memory.weight
    parameter.requires_grad_(True)
    initial = parameter.detach().clone()
    optimizer = torch.optim.AdamW([parameter], lr=float(settings["inner_learning_rate"]))
    objective = SharedNeuralPilotObjective(candidate, contexts, prior_copies=prior_copies)
    losses = []
    for _ in range(int(settings["inner_steps"])):
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([parameter], 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        delta = parameter - initial
        radius = float(settings["max_delta_norm"])
        parameter.copy_(initial + delta * (radius / delta.norm().clamp_min(1e-12)).clamp(max=1.))
        current = contexts[-1]
        before, after = context_logits(model, current), context_logits(candidate, current)
        view = current.frame.receiver_view()
        mask, labels = current.frame.reward_mask, view.reward_symbols.real > 0
        ber_before = float(((before[mask] >= 0) != labels[mask]).float().mean())
        ber_after = float(((after[mask] >= 0) != labels[mask]).float().mean())
        loss_before = float(channel_state_reward_loss(model, current))
        loss_after = float(channel_state_reward_loss(candidate, current))
        norm = float((parameter - initial).norm())
        accepted = should_accept_reward_guard(reward_loss_before=loss_before, reward_loss_after=loss_after,
            reward_ber_before=ber_before, reward_ber_after=ber_after, allow_loss_only=True, peft_delta_norm=norm)
    return {"adapt_losses": losses, "delta_norm": norm, "accepted": accepted,
            "reward_loss_before": loss_before, "reward_loss_after": loss_after,
            "reward_ber_before": ber_before, "reward_ber_after": ber_after,
            "shared_current_state": True, "prior_copies": prior_copies}, before, after


def run(config, checkpoint, seeds, snrs, age, frames, window, device, output):
    if window < 2 or age < window - 1 or frames < 1:
        raise ValueError("窗口至少两帧，历史必须足够且评分帧数为正。")
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    if (target / "summary.json").exists() or (target / "frame_metrics.jsonl").exists():
        raise ValueError("诊断输出已存在，禁止覆盖。")
    model, payload = _load_branch(Path(checkpoint), config, device)
    training = payload.get("metrics", {})
    if set(seeds).intersection(item["seed"] for item in training.get("history", [])):
        raise ValueError("诊断 seed 与训练重叠。")
    if int(training.get("receiver_lookahead_len", 0)) != int(config.get("receiver_lookahead_len", 0)):
        raise ValueError("训练与诊断接收窗口不一致。")
    settings = config["pilot_transfer"]
    original = {name: value.clone() for name, value in model.state_dict().items()}
    arms = ("single", "old_mean", "shared_repeated", "shared_once")
    records = []
    raw = target / "frame_metrics.jsonl"
    started = time.perf_counter()
    with raw.open("x", encoding="utf-8") as handle:
        for snr in snrs:
            for seed in seeds:
                _, cir, cfo, tail, episode = make_episode(config, model, seed=seed, snr=snr,
                    frames=age + frames, state_split="heldout_edge", device="cpu")
                contexts = [make_context(frame_to_device(frame, device), cir.to(device), cfo, tail.to(device), snr)
                            for frame in episode[age-window+1:]]
                for local in range(frames):
                    end = window + local
                    history = contexts[end-window:end]
                    candidates = [history_candidate(model, history[-1:], settings),
                                  history_candidate(model, history, settings),
                                  shared_candidate(model, history, settings, prior_copies=window),
                                  shared_candidate(model, history, settings)]
                    # 全部四臂控制完成后才首次读取当前 Data 标签，不按 Data 选择候选。
                    frame = history[-1].frame
                    mask = frame.data_mask
                    for arm, (control, before, after) in zip(arms, candidates):
                        selected = after if control["accepted"] else before
                        ber = lambda logits: float(((logits[mask] >= 0) != frame.bits[mask]).float().mean())
                        row = {"snr_db": snr, "seed": seed, "frame": local,
                            "channel_frame_index": frame.frame_index, "arm": arm,
                            "ber_frozen": ber(before), "ber_candidate": ber(after), "ber_guarded": ber(selected),
                            "data_labels_used_for_control": False, "persistent": False,
                            "rx_sha256": hashlib.sha256(frame.rx_symbols.cpu().numpy().tobytes()).hexdigest(),
                            **control}
                        records.append(row)
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                print(json.dumps({"snr_db": snr, "seed": seed, "frames_complete": frames}), flush=True)
    if any(not torch.equal(value, original[name]) for name, value in model.state_dict().items()):
        raise AssertionError("候选不得修改源模型。")
    summaries = []
    for snr in snrs:
        by_arm, gains = {}, {}
        for arm in arms:
            rows = [r for r in records if r["snr_db"] == snr and r["arm"] == arm]
            gains[arm] = [float(np.mean([r["ber_frozen"]-r["ber_guarded"] for r in rows if r["seed"] == seed])) for seed in seeds]
            by_arm[arm] = {key: float(np.mean([r[key] for r in rows])) for key in
                ("ber_frozen", "ber_candidate", "ber_guarded", "reward_loss_before", "reward_loss_after", "delta_norm")}
            by_arm[arm].update(gain=paired_interval(gains[arm]), accepted=sum(r["accepted"] for r in rows), frames=len(rows))
        summaries.append({"snr_db": snr, "arms": by_arm,
            "shared_once_minus_single_gain": paired_interval([b-a for a,b in zip(gains["single"],gains["shared_once"])]),
            "once_minus_repeated_gain": paired_interval([b-a for a,b in zip(gains["shared_repeated"],gains["shared_once"])])})
    result = {"schema_version": "shared-neural-pilot-diagnostic-v1", "diagnostic_only": True,
        "main_aggregation_allowed": False, "config": config, "seeds": seeds, "snrs": snrs,
        "age": age, "frames": frames, "window": window, "records": len(records), "per_snr": summaries,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(), "model_unchanged": True,
        "wall_seconds": time.perf_counter() - started,
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
            ("evaluation/joint_pilot_state.py", "scripts/diagnose_shared_neural_pilot.py")},
        "interpretation": "同神经记忆的状态/先验消融，非持续更新，非RL，不得当作正式成功。"}
    (target / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed-start", type=int, default=110080)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--snrs", nargs="+", type=float, default=[0,5,10,15])
    parser.add_argument("--age", type=int, default=1023)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(json.loads(Path(args.config).read_text(encoding="utf-8")), args.checkpoint,
        list(range(args.seed_start, args.seed_start + args.num_seeds)), args.snrs,
        args.age, args.frames, args.window, args.device, args.output)
