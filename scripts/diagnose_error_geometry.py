# -*- coding: utf-8 -*-
"""用冻结模型定位错误与分支尺度；隐藏标签只参与输出后的离线统计。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.pilot_transfer import _load_branch
from training.pilot_transfer import context_logits, make_context, make_episode, next_pilot_tail


def receiver_outputs(model, context):
    """仅用整帧接收波形与合法条件；不输入 Data 标签或仿真真实 CIR/tail。"""
    if model.config.analytic_logit_skip_scale != 0:
        raise ValueError("本轮分支分解要求 analytic_logit_skip_scale=0。")
    if model.config.enable_symbol_residual_branch:
        raise ValueError("原物理/神经分支分解不适用于新增符号残差模型。")
    rx = context.frame.rx_symbols
    iq = torch.stack((rx.real, rx.imag), -1).unsqueeze(0).float()
    with torch.no_grad():
        logits = context_logits(model, context, visible="full_rx")
        corrected = model._apply_phase_correction(iq, context.condition)
        if model.online_phase_trend_adapter is not None:
            corrected = model.online_phase_trend_adapter(corrected)
        physics = model._physics_warm_start_logits(
            torch.complex(corrected[..., 0], corrected[..., 1]), context.condition, context.tail).squeeze(0)
    return logits, physics


def region_masks(frame, memory):
    """分别记录未知历史影响区、Pilot 内部及 Data 首尾，Data 三段互斥覆盖。"""
    positions = torch.arange(frame.bits.numel(), device=frame.data_mask.device)
    data_positions = torch.nonzero(frame.data_mask, as_tuple=False).flatten()
    if data_positions.numel() < 2 * memory:
        raise ValueError("Data 长度不足以构造互斥的首尾记忆区。")
    first = int(data_positions[0])
    last = int(data_positions[-1]) + 1
    return {"adapt_boundary": frame.adapt_mask & (positions < memory),
            "adapt_interior": frame.adapt_mask & (positions >= memory),
            "reward": frame.reward_mask,
            "data_head": frame.data_mask & (positions < first + memory),
            "data_middle": frame.data_mask & (positions >= first + memory) & (positions < last - memory),
            "data_tail": frame.data_mask & (positions >= last - memory)}


def score_outputs(frame, logits, physics, memory):
    """推理结束后的标签评分；不反传、更新或选择动作。"""
    rows = []
    for name, mask in region_masks(frame, memory).items():
        wrong = (logits[mask] >= 0) != frame.bits[mask]
        physical_wrong = (physics[mask] >= 0) != frame.bits[mask]
        magnitude = logits[mask].abs()
        correction = (logits[mask] - physics[mask]).abs()
        rows.append({"region": name, "bits": int(mask.sum()), "errors": int(wrong.sum()),
                     "physics_errors": int(physical_wrong.sum()),
                     "neural_fixed_errors": int((physical_wrong & ~wrong).sum()),
                     "neural_introduced_errors": int((~physical_wrong & wrong).sum()),
                     "wrong_abs_logit_sum": float(magnitude[wrong].sum()),
                     "wrong_abs_logit_ge_5": int((wrong & (magnitude >= 5)).sum()),
                     "wrong_abs_logit_ge_20": int((wrong & (magnitude >= 20)).sum()),
                     "wrong_physics_clamped": int((wrong & (physics[mask].abs() >= 39.999)).sum()),
                     "neural_abs_sum": float(correction.sum()),
                     "wrong_neural_abs_sum": float(correction[wrong].sum())})
    return rows


def run_diagnostic(config, checkpoint, seeds, frames, device, output):
    target = Path(output)
    if target.exists():
        raise ValueError("诊断输出已存在，请指定新路径。")
    if (config.get("offline_state_split") != "offline_train" or config.get("main_level") != "B"
            or config.get("pilot_layout") != "prefix" or not seeds or len(set(seeds)) != len(seeds) or frames < 1):
        raise ValueError("错误定位仅允许 offline_train、Level B、prefix 和不重复的正样本预算。")
    model, payload = _load_branch(Path(checkpoint), config, device)
    trained = {item["seed"] for item in payload.get("metrics", {}).get("history", []) if "seed" in item}
    if trained.intersection(seeds):
        raise ValueError("诊断 seed 与 checkpoint 训练 seed 重叠。")
    rows = []
    for snr in config["main_snrs"]:
        for seed in seeds:
            _, cir, cfo, tail, received = make_episode(config, model, seed=seed, snr=snr, frames=frames,
                                                     state_split="offline_train", device=device)
            for index, frame in enumerate(received):
                context = make_context(frame, cir, cfo, tail, snr)
                logits, physics = receiver_outputs(model, context)
                scored = score_outputs(frame, logits, physics, model.config.max_delay)
                fingerprint = hashlib.sha256(frame.rx_symbols.detach().cpu().numpy().tobytes()).hexdigest()
                rows.extend({"snr_db": snr, "seed": seed, "frame": index, "input_sha256": fingerprint, **r} for r in scored)
                tail = next_pilot_tail(model, context, alpha=float(config.get("tail_update_alpha", 0.5)))
            print(json.dumps({"snr_db": snr, "seed": seed, "frames_complete": frames}), flush=True)
    summary = []
    for snr in config["main_snrs"]:
        for region in sorted({row["region"] for row in rows}):
            subset = [r for r in rows if r["snr_db"] == snr and r["region"] == region]
            keys = [key for key in subset[0] if key not in {"snr_db", "seed", "frame", "input_sha256", "region"}]
            totals = {key: sum(r[key] for r in subset) for key in keys}
            summary.append({"snr_db": snr, "region": region, **totals,
                            "ber": totals["errors"] / totals["bits"],
                            "mean_wrong_abs_logit": totals["wrong_abs_logit_sum"] / totals["errors"] if totals["errors"] else None,
                            "mean_wrong_neural_abs": totals["wrong_neural_abs_sum"] / totals["errors"] if totals["errors"] else None})
    result = {"schema_version": "error-geometry-v1", "diagnostic_only": True, "main_aggregation_allowed": False,
              "state_split": "offline_train", "config": config, "seeds": list(seeds), "frames": frames,
              "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
              "weights_modified": False, "data_labels_used_for_control": False, "true_cir_or_tail_used": False,
              "per_region": summary, "rows": rows}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--seed-start", type=int, default=45000)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_diagnostic(json.loads(Path(args.config).read_text(encoding="utf-8")), args.pretrained,
                   list(range(args.seed_start, args.seed_start + args.num_seeds)), args.frames, args.device, args.output)


if __name__ == "__main__":
    main()
