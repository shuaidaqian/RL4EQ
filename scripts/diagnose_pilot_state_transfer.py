# -*- coding: utf-8 -*-
"""用合法 Pilot 方程检验信道状态信息；不是主方法或新传统基线。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pilot_transfer import _load_branch, _metrics, paired_interval
from scripts.diagnose_receiver_input_updates import _reward_metrics
from training.pilot_transfer import context_logits, make_context, make_episode, next_pilot_tail
from training.windowed_discrete_ppo import _adapt_receiver_view, should_accept_reward_guard


@torch.no_grad()
def pilot_equations(model, context):
    """仅使用卷积输入全部为当前 Adapt 的行，不把 soft tail 当作真实历史。"""
    view = _adapt_receiver_view(context.frame)
    memory = context.tail.numel()
    positions = torch.arange(memory, view.adapt_mask.numel(), device=view.adapt_mask.device)
    indices = positions[:, None] - torch.arange(memory + 1, device=positions.device)
    known = view.adapt_mask[positions] & view.adapt_mask[indices].all(dim=1)
    positions, indices = positions[known], indices[known]
    if not positions.numel():
        raise ValueError("当前帧没有完全已知的 Adapt 卷积行。")
    # 只校正 Adapt 视图；相位支路逐位置旋转，不读取隐藏发送符号。
    iq = torch.stack((view.rx_symbols.real, view.rx_symbols.imag), dim=-1).unsqueeze(0).float()
    corrected = model._apply_phase_correction(iq, context.condition).squeeze(0)
    target = torch.complex(corrected[:, 0], corrected[:, 1])[positions]
    return view.adapt_symbols[indices], target


@torch.no_grad()
def posterior_candidate(equations, context, *, acquisition_samples, rho_frame):
    """固定 acquisition 均值、对角过程先验与噪声尺度，不用真实 CIR 选正则。"""
    if not equations or acquisition_samples <= 0 or not 0 < rho_frame <= 1:
        raise ValueError("需要非空方程、正 acquisition 行数和合法相关系数。")
    design = torch.cat([item[0] for item in equations])
    targets = torch.cat([item[1] for item in equations])
    prior = context.condition.complex_cir.reshape(-1)
    noise = context.condition.noise_variance.reshape(()).clamp_min(1e-6)
    acquisition_variance = noise / acquisition_samples
    elapsed_frames = context.frame.frame_index + 1
    if elapsed_frames < 1:
        raise ValueError("仅支持 acquisition 之后的数据帧。")
    drift = 1.0 - float(rho_frame) ** (2 * elapsed_frames)
    variance = acquisition_variance + drift * (prior.abs().square() + acquisition_variance)
    precision = noise / variance.clamp_min(1e-10)
    normal = design.conj().T @ design + torch.diag(precision.to(design.dtype))
    residual = targets - design @ prior
    delta = torch.linalg.solve(normal, design.conj().T @ residual)
    # 保留估计幅度；不能用真 CIR 功率归一化候选。
    return prior + delta, {"equation_rows": design.shape[0], "delta_norm": float(delta.norm()),
                           "prior_variance_mean": float(variance.mean())}


@torch.no_grad()
def state_candidates_and_guards(model, context, history, windows, config):
    """完整生成候选并完成 Reward 验收后，调用方才计算 Data 标签指标。"""
    frozen = context_logits(model, context, visible="full_rx")
    ber0, loss0 = _reward_metrics(frozen, context.frame)
    candidates, decisions = {}, []
    for window in windows:
        cir, info = posterior_candidate(history[-window:], context,
            acquisition_samples=config["frame_len"] - config["max_delay"], rho_frame=config["rho_frame"])
        candidate = replace(context, condition=replace(context.condition, complex_cir=cir.unsqueeze(0)))
        logits = context_logits(model, candidate, visible="full_rx")
        ber1, loss1 = _reward_metrics(logits, context.frame)
        accepted = should_accept_reward_guard(reward_loss_before=loss0, reward_loss_after=loss1,
            reward_ber_before=ber0, reward_ber_after=ber1, allow_loss_only=True, peft_delta_norm=info["delta_norm"])
        candidates[window] = logits
        decisions.append({"window": window, "accepted": accepted,
            "reward_ber_before": ber0, "reward_ber_after": ber1,
            "reward_loss_before": loss0, "reward_loss_after": loss1, **info})
    return frozen, candidates, decisions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--seed-start", type=int, default=75000)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--windows", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    target = Path(args.output)
    if (target.exists() or args.num_seeds < 1 or not args.windows or min(args.windows) < 1
            or args.frames < max(args.windows) or config.get("offline_state_split") != "offline_train"
            or config.get("main_level") != "B" or config.get("pilot_layout") != "prefix"):
        raise ValueError("仅允许新输出、正预算、训练分布、Level B 与前缀 Pilot。")
    model, payload = _load_branch(args.pretrained, config, args.device)
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    if {r["seed"] for r in payload["metrics"]["history"]}.intersection(seeds):
        raise ValueError("诊断 seed 与训练重叠。")
    rows = []
    started = time.monotonic()
    for snr in config["main_snrs"]:
        for seed in seeds:
            _, cir, cfo, tail, frames = make_episode(config, model, seed=seed, snr=snr,
                frames=args.frames, state_split="offline_train", device=args.device)
            equations = []
            for index, frame in enumerate(frames):
                context = make_context(frame, cir, cfo, tail, snr)
                equations.append(pilot_equations(model, context))
                if len(equations) >= max(args.windows):
                    frozen, candidates, decisions = state_candidates_and_guards(
                        model, context, equations, args.windows, config)
                    # 决策已固定；Data 标签只用于事后评分，不生成标签上界。
                    with torch.no_grad():
                        ber0, bce0 = _metrics(frozen, frame, frame.data_mask)
                        for decision in decisions:
                            ber1, bce1 = _metrics(candidates[decision["window"]], frame, frame.data_mask)
                            rows.append({"snr_db": snr, "seed": seed, "frame": index, **decision,
                                "ber_frozen": ber0, "ber_candidate": ber1,
                                "ber_committed": ber1 if decision["accepted"] else ber0,
                                "bce_frozen": bce0, "bce_candidate": bce1,
                                "data_labels_used_for_control": False})
                tail = next_pilot_tail(model, context, alpha=config.get("tail_update_alpha", .5))
        print(json.dumps({"snr_db": snr, "complete": True}), flush=True)
    summary = []
    for snr in config["main_snrs"]:
        for window in args.windows:
            subset = [r for r in rows if r["snr_db"] == snr and r["window"] == window]
            item = {"snr_db": snr, "window": window, "accepted": sum(r["accepted"] for r in subset), "frames": len(subset)}
            for key in ("ber_candidate", "ber_committed", "bce_candidate"):
                before = "bce_frozen" if key.startswith("bce") else "ber_frozen"
                item[key + "_gain"] = paired_interval([
                    np.mean([r[before] - r[key] for r in subset if r["seed"] == seed]) for seed in seeds])
            summary.append(item)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"diagnostic_only": True, "main_aggregation_allowed": False,
        "is_neural_peft_update": False, "is_main_baseline": False, "uses_true_cir": False,
        "data_labels_used_for_control": False, "state_split": "offline_train", "config": config,
        "checkpoint": args.pretrained, "checkpoint_sha256": hashlib.sha256(Path(args.pretrained).read_bytes()).hexdigest(),
        "seeds": seeds, "frames": args.frames, "windows": args.windows,
        "elapsed_seconds": time.monotonic() - started,
        "per_snr": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
