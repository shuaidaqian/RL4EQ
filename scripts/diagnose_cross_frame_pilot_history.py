# -*- coding: utf-8 -*-
"""只读比较单帧和多帧 Adapt 方程；不是正式 RL 或持续跟踪结果。"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.neural_channel_state import known_pilot_equations
from evaluation.pilot_transfer import _load_branch, paired_interval
from training.pilot_transfer import (
    channel_state_reward_loss, context_logits, differentiable_adapt, frame_to_device,
    make_context, make_episode,
)
from training.windowed_discrete_ppo import should_accept_reward_guard


def adapt_design_rank(contexts):
    """秩只由合法 Adapt 符号决定，不读取隐藏发送符号或真实 CIR。"""
    designs = []
    for context in contexts:
        view = context.frame.receiver_view()
        design, _ = known_pilot_equations(view.rx_symbols, view.adapt_symbols,
            view.adapt_mask, view.adapt_mask, context.condition.complex_cir.numel() - 1)
        designs.append(design)
    joint = torch.cat(designs)
    return {"complex_rows": joint.shape[0], "complex_columns": joint.shape[1],
            "complex_rank": int(torch.linalg.matrix_rank(joint).item())}


def history_candidate(model, contexts, settings):
    """候选与验收仅用 Adapt/Reward；返回的 logits 由调用者作事后 Data 评估。"""
    current = contexts[-1]
    parameters, update = differentiable_adapt(model, contexts,
        learning_rate=float(settings["inner_learning_rate"]), steps=int(settings["inner_steps"]),
        max_delta_norm=float(settings["max_delta_norm"]), groups=settings["inner_groups"],
        objective=settings["inner_objective"], receiver_input=settings["inner_receiver_input"],
        create_graph=False)
    parameters = {name: value.detach() for name, value in parameters.items()}
    with torch.no_grad():
        before = context_logits(model, current)
        candidate = context_logits(model, current, parameters)
        view = current.frame.receiver_view()
        mask = current.frame.reward_mask
        labels = view.reward_symbols.real > 0
        ber_before = float(((before[mask] >= 0) != labels[mask]).float().mean())
        ber_after = float(((candidate[mask] >= 0) != labels[mask]).float().mean())
        if settings.get("reward_objective") == "channel_state_signal":
            loss_before = float(channel_state_reward_loss(model, current))
            loss_after = float(channel_state_reward_loss(model, current, parameters))
        else:
            loss_before = float(torch.nn.functional.binary_cross_entropy_with_logits(before[mask], labels[mask].float()))
            loss_after = float(torch.nn.functional.binary_cross_entropy_with_logits(candidate[mask], labels[mask].float()))
        accepted = should_accept_reward_guard(reward_loss_before=loss_before, reward_loss_after=loss_after,
            reward_ber_before=ber_before, reward_ber_after=ber_after, allow_loss_only=True,
            peft_delta_norm=update["delta_norm"])
    return {**update, **adapt_design_rank(contexts), "accepted": accepted,
            "reward_loss_before": loss_before, "reward_loss_after": loss_after,
            "reward_ber_before": ber_before, "reward_ber_after": ber_after}, before, candidate


def run(config, checkpoint, seeds, snrs, age, frames, windows, device, output):
    if age < max(windows) - 1 or min(frames, *windows) < 1:
        raise ValueError("物理历史必须足够覆盖最长 Adapt 窗口。")
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    raw_path = target / "frame_metrics.jsonl"
    if raw_path.exists():
        raise ValueError("诊断输出已存在，禁止覆盖。")
    model, payload = _load_branch(Path(checkpoint), config, device)
    training = payload.get("metrics", {})
    if set(seeds).intersection(item["seed"] for item in training.get("history", [])):
        raise ValueError("诊断 seed 与离线训练 seed 重叠。")
    if int(training.get("receiver_lookahead_len", 0)) != int(config.get("receiver_lookahead_len", 0)):
        raise ValueError("诊断与训练的接收窗口不匹配。")
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    rows = []
    with raw_path.open("w", encoding="utf-8") as handle:
        for snr in snrs:
            for seed in seeds:
                env, cir, cfo, tail, episode = make_episode(config, model, seed=seed, snr=snr,
                    frames=age + frames, state_split="heldout_edge", device="cpu")
                if any(bool(frame.adapt_mask[-tail.numel():].any()) for frame in episode):
                    raise ValueError("当前诊断仅支持前缀不覆盖帧尾的协议。")
                cir, tail = cir.to(device), tail.to(device)
                start = age - max(windows) + 1
                contexts = [make_context(frame_to_device(frame, device), cir, cfo, tail, snr)
                            for frame in episode[start:]]
                for local in range(frames):
                    end = max(windows) + local
                    current = contexts[end - 1]
                    # 两个候选的控制与验收全部完成之后，才读取当前 Data 标签。
                    candidates = [(window, *history_candidate(model, contexts[end-window:end], config["pilot_transfer"]))
                                  for window in windows]
                    for window, control, before, candidate in candidates:
                        frame = current.frame
                        selected = candidate if control["accepted"] else before
                        mask = frame.data_mask
                        data_ber = lambda logits: float(((logits[mask] >= 0) != frame.bits[mask]).float().mean())
                        row = {"snr_db": snr, "seed": seed, "frame": local,
                               "channel_frame_index": frame.frame_index, "adapt_history_frames": window,
                               "ber_frozen": data_ber(before), "ber_candidate": data_ber(candidate),
                               "ber_guarded": data_ber(selected), "data_labels_used_for_control": False,
                               "input_sha256": hashlib.sha256(frame.rx_symbols.cpu().numpy().tobytes()).hexdigest(),
                               "lookahead_rx_sha256": hashlib.sha256(frame.lookahead_rx_symbols.cpu().numpy().tobytes()).hexdigest(),
                               **control}
                        rows.append(row)
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                print(json.dumps({"snr_db": snr, "seed": seed, "frames_complete": frames}), flush=True)
    assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items())
    summaries = []
    for snr in snrs:
        item = {"snr_db": snr, "windows": {}}
        gains = {}
        for window in windows:
            subset = [row for row in rows if row["snr_db"] == snr and row["adapt_history_frames"] == window]
            gains[window] = [float(np.mean([r["ber_frozen"] - r["ber_guarded"] for r in subset if r["seed"] == seed])) for seed in seeds]
            item["windows"][str(window)] = {"gain": paired_interval(gains[window]),
                "ber_frozen": float(np.mean([r["ber_frozen"] for r in subset])),
                "ber_candidate": float(np.mean([r["ber_candidate"] for r in subset])),
                "ber_guarded": float(np.mean([r["ber_guarded"] for r in subset])),
                "candidate_gain": paired_interval([float(np.mean([r["ber_frozen"] - r["ber_candidate"]
                    for r in subset if r["seed"] == seed])) for seed in seeds]),
                "mean_reward_signal_loss_gain": float(np.mean([r["reward_loss_before"] - r["reward_loss_after"] for r in subset])),
                "accepted": sum(r["accepted"] for r in subset), "frames": len(subset),
                "complex_rows": sorted({r["complex_rows"] for r in subset}),
                "complex_columns": sorted({r["complex_columns"] for r in subset}),
                "complex_rank": sorted({r["complex_rank"] for r in subset})}
        item["last_vs_first_window_gain"] = paired_interval([b-a for a,b in zip(gains[windows[0]],gains[windows[-1]])])
        summaries.append(item)
    result = {"schema_version": "cross-frame-pilot-history-diagnostic-v1", "diagnostic_only": True,
              "main_aggregation_allowed": False, "seeds": seeds, "snrs": snrs, "physical_age": age,
              "frames": frames, "windows": windows, "per_snr": summaries,
              "config": config, "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
              "raw_file_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
              "model_unchanged": True, "persistent": False,
              "adapt_objective_aggregation": "mean_of_per_frame_objectives",
              "phase_coordinates": "per_frame_estimated_phase_coordinates",
              "interpretation": "无参数提交的方程窗口诊断；不是持续在线或RL结果；单帧与历史候选均接受当前Reward验收。"}
    (target / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed-start", type=int, default=110060)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--snrs", nargs="+", type=float, default=[0, 5, 10, 15])
    parser.add_argument("--age", type=int, default=1023)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--windows", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text(encoding="utf-8")), args.checkpoint,
        list(range(args.seed_start, args.seed_start + args.num_seeds)), args.snrs,
        args.age, args.frames, args.windows, args.device, args.output)
