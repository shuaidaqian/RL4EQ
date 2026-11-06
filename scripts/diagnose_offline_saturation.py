# -*- coding: utf-8 -*-
"""离线网络饱和/可修复空间开发诊断（只读，不可部署）。

- Frozen：估计 CIR + 前缀 Pilot + 上一帧最后 max_delay 个 logits 的 soft-tail 递推
  （alpha=1.0，与 curriculum 的 online_soft 一致）。
- Oracle head：仅离线诊断，用 Data 真标签对 model.head 做 K 步 Adam 优化，
  衡量给定冻结物理状态时 head 的可修复空间。
- Oracle CIR：仅离线诊断，把条件 CIR 换成该帧真实 CIR 并重新递推 tail，
  衡量 CIR 估计误差的代价。

任何 Data 标签只用于事后评分与离线 oracle，不进入可部署路径。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.cir_estimator import condition_from_cir
from baseline.traditional_equalizers import estimate_phase_residual_vector
from evaluation.pilot_transfer import _load_branch
from training.pilot_transfer import (
    context_logits,
    frame_to_device,
    make_episode,
)
from training.windowed_discrete_ppo import _adapt_receiver_view


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_interval(values: list[float]) -> dict:
    """以独立会话 seed 为统计单位；样本不足时不给区间。"""

    sample = np.asarray(values, dtype=float)
    if not sample.size or not np.isfinite(sample).all():
        raise ValueError("配对统计需要非空有限样本。")
    mean = float(sample.mean())
    if sample.size < 2:
        return {"mean": mean, "ci95": None, "seeds": int(sample.size)}
    from scipy.stats import t

    margin = float(t.ppf(0.975, len(sample) - 1) * sample.std(ddof=1) / np.sqrt(len(sample)))
    return {"mean": mean, "ci95": [mean - margin, mean + margin], "seeds": int(sample.size)}


def _condition(frame, cir: torch.Tensor, cfo: float, tail: torch.Tensor, snr: float):
    """与 make_context 相同的信息边界，只是可替换 CIR 来源。"""

    view = _adapt_receiver_view(frame)
    phase = estimate_phase_residual_vector(view, cir, tail, blocks=4, cfo_hint=cfo)
    return condition_from_cir(cir, float(snr), cfo_residual=float(cfo), phase_features=phase)


def _advance_tail(logits: torch.Tensor, tail_len: int) -> torch.Tensor:
    """上一帧最后 tail_len 个 logits 的 soft 判决，alpha=1.0。"""

    values = torch.tanh(logits[-tail_len:].detach() / 2.0)
    return torch.complex(values, torch.zeros_like(values)).to(torch.complex64)


def _ber(logits: torch.Tensor, frame) -> tuple[float, int, int]:
    mask = frame.data_mask.bool()
    labels = frame.bits[mask].bool()
    wrong = ((logits[mask] >= 0) != labels)
    return float(wrong.float().mean().detach().cpu()), int(wrong.sum().detach().cpu()), int(mask.sum())


def _bce(logits: torch.Tensor, frame) -> float:
    mask = frame.data_mask.bool()
    return float(F.binary_cross_entropy_with_logits(
        logits[mask].float(), frame.bits[mask].float()).detach().cpu())


def collect_session(model, config, *, seed, snr, frames, state_split, device, cir_mode="estimated"):
    """生成一个会话，按 Frozen 递推 tail，并记录每帧上下文与判决。

    cir_mode="true" 只用 frame.true_cir 构造条件，属隔离诊断。
    """

    env, cir, cfo, tail, episode = make_episode(
        config, model, seed=int(seed), snr=float(snr), frames=int(frames),
        state_split=state_split, device="cpu",
    )
    tail = tail.to(device).to(torch.complex64)
    cir = cir.to(device)
    session = []
    for frame in episode:
        frame = frame_to_device(frame, device)
        active_cir = frame.true_cir.to(device).to(torch.complex64) if cir_mode == "true" else cir
        condition = _condition(frame, active_cir, cfo, tail, snr)
        context = type("Ctx", (), {})()
        context.frame = frame
        context.condition = condition
        context.tail = tail.detach().clone()
        context.cfo = float(cfo)
        with torch.no_grad():
            logits = context_logits(model, context, None, visible="data")
            ber, wrong, total = _ber(logits, frame)
            bce = _bce(logits, frame)
        session.append({
            "seed": int(seed), "snr_db": float(snr), "frame_index": len(session),
            "channel_frame_index": int(frame.frame_index),
            "context": context, "logits": logits.detach(),
            "ber": ber, "bce": bce, "wrong": wrong, "total": total,
            "true_cir": frame.true_cir.detach().cpu(),
            "estimated_cir": cir.detach().cpu(),
        })
        tail = _advance_tail(logits, tail.numel())
    return session


def cache_head_features(model, contexts):
    """缓存 head 的冻结输入与固定加性项。

    离线 head 是唯一变化量：logits = additive + scale * head(hidden)。
    hidden 与 additive 由冻结模型和固定上下文决定，因此一次前向即可精确复用。
    """

    captured: dict = {}

    def hook(module, inputs, output):
        captured["hidden"] = inputs[0].detach()

    handle = model.head.register_forward_hook(hook)
    scale = float(model.config.neural_residual_scale)
    frames = []
    with torch.no_grad():
        for ctx in contexts:
            captured.clear()
            logits = context_logits(model, ctx, None, visible="data")
            hidden = captured["hidden"]
            if hidden.ndim != 3 or hidden.shape[0] != 1:
                raise ValueError("head 输入形状必须为 [1, N, d_model]。")
            hidden = hidden[0]
            additive = logits - scale * model.head(hidden).squeeze(-1)
            mask = ctx.frame.data_mask.bool()
            frames.append({
                "hidden": hidden, "additive": additive, "mask": mask,
                "labels": ctx.frame.bits[mask].float(), "logits": logits.detach(),
            })
    handle.remove()
    return frames, scale


def _stack_head_frames(frames):
    hidden = torch.cat([item["hidden"] for item in frames], dim=0)
    additive = torch.cat([item["additive"] for item in frames], dim=0)
    mask = torch.cat([item["mask"] for item in frames], dim=0)
    labels = torch.cat([item["labels"] for item in frames], dim=0)
    return hidden, additive, mask, labels


def _frame_logits_cached(item, weight, bias, scale):
    return item["additive"] + scale * F.linear(item["hidden"], weight, bias).squeeze(-1)


def _head_frame_stats(frames, weight, bias, scale, before_logits):
    total_wrong = fixed_total = harmed_total = 0
    with torch.no_grad():
        for item, before in zip(frames, before_logits):
            logits = _frame_logits_cached(item, weight, bias, scale)
            mask, labels = item["mask"], item["labels"].bool()
            after = logits[mask] >= 0
            before_dec = before[mask] >= 0
            total_wrong += int((after != labels).sum().cpu())
            fixed_total += int((~(before_dec == labels) & (after == labels)).sum().cpu())
            harmed_total += int(((before_dec == labels) & ~(after == labels)).sum().cpu())
        total = int(sum(int(item["mask"].sum()) for item in frames))
    return total_wrong / max(1, total), total_wrong, total, fixed_total, harmed_total


def oracle_head_for_session(model, session, *, k_values, lr, split_mode="in_sample"):
    """单条 Adam 轨迹训练到 max(K)，在 K 检查点评分（精确 head-only 上界）。

    split_mode="heldout" 时只用偶数帧优化，在奇数帧上评分，用于区分表示容量
    与会话内同帧标签过拟合。
    """

    contexts = [row["context"] for row in session]
    frames, scale = cache_head_features(model, contexts)
    if split_mode == "heldout":
        train_frames = [item for index, item in enumerate(frames) if index % 2 == 0]
        eval_frames = [item for index, item in enumerate(frames) if index % 2 == 1]
    elif split_mode == "in_sample":
        train_frames = frames
        eval_frames = frames
    else:
        raise ValueError("head split 必须为 in_sample 或 heldout。")
    before_logits = [item["logits"] for item in eval_frames]
    weight = model.head.weight.detach().clone().requires_grad_(True)
    bias = model.head.bias.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([weight, bias], lr=float(lr))
    hidden, additive, mask, labels = _stack_head_frames(train_frames)
    wanted = sorted({int(value) for value in k_values})
    max_k = wanted[-1]
    before_ber, before_wrong, total, _, _ = _head_frame_stats(
        eval_frames, weight.detach(), bias.detach(), scale, before_logits)
    result = {"ber_before": before_ber, "wrong_before": before_wrong, "total": total,
              "snr_db": session[0]["snr_db"], "seed": session[0]["seed"], "split": split_mode, "K": {}}
    for step in range(1, max_k + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = additive + scale * F.linear(hidden, weight, bias).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits[mask].float(), labels)
        loss.backward()
        optimizer.step()
        if step not in wanted:
            continue
        ber_after, wrong_after, _, fixed_total, harmed_total = _head_frame_stats(
            eval_frames, weight, bias, scale, before_logits)
        result["K"][str(step)] = {
            "ber_after": ber_after,
            "wrong_after": wrong_after,
            "fixed_errors": fixed_total,
            "harmed_errors": harmed_total,
            "net_fixed": fixed_total - harmed_total,
            "loss_last": float(loss.detach().cpu()),
        }
    return result


def _attach_frozen_logits(session):
    for row in session:
        row["context"]._frozen_logits = row["logits"]


def summarize_oracle_head(per_session: list[dict], k_values) -> dict:
    per_snr = {}
    snrs = sorted({row["snr_db"] for row in per_session})
    for snr in snrs:
        subset = [row for row in per_session if row["snr_db"] == snr]
        before = [row["ber_before"] for row in subset]
        block = {"ber_before": paired_interval(before)}
        for k in k_values:
            after = [row["K"][str(k)]["ber_after"] for row in subset]
            delta = [row["K"][str(k)]["ber_after"] - row["ber_before"] for row in subset]
            block[f"K{k}"] = {
                "ber_after": paired_interval(after),
                "delta_after_minus_before": paired_interval(delta),
                "fixed_errors": int(sum(row["K"][str(k)]["fixed_errors"] for row in subset)),
                "harmed_errors": int(sum(row["K"][str(k)]["harmed_errors"] for row in subset)),
                "net_fixed": int(sum(row["K"][str(k)]["net_fixed"] for row in subset)),
                "per_seed_delta": [
                    {"seed": row["seed"], "delta": row["K"][str(k)]["ber_after"] - row["ber_before"],
                     "fixed": row["K"][str(k)]["fixed_errors"],
                     "harmed": row["K"][str(k)]["harmed_errors"]}
                    for row in subset
                ],
            }
        per_snr[snr] = block
    return per_snr


def summarize_frozen(rows: list[dict]) -> dict:
    per_snr = {}
    for snr in sorted({row["snr_db"] for row in rows}):
        subset = [row for row in rows if row["snr_db"] == snr]
        per_seed = []
        for seed in sorted({row["seed"] for row in subset}):
            items = [row for row in subset if row["seed"] == seed]
            wrong = sum(row["wrong"] for row in items)
            total = sum(row["total"] for row in items)
            per_seed.append({"seed": seed, "ber": wrong / max(1, total),
                             "wrong": wrong, "total": total})
        per_snr[snr] = {
            "ber": paired_interval([item["ber"] for item in per_seed]),
            "per_seed": per_seed,
        }
    return per_snr


def summarize_cir(estimated_rows: list[dict], true_rows: list[dict]) -> dict:
    key = lambda row: (row["seed"], row["snr_db"], row["frame_index"])
    true_map = {key(row): row for row in true_rows}
    per_snr = {}
    for snr in sorted({row["snr_db"] for row in estimated_rows}):
        est = [row for row in estimated_rows if row["snr_db"] == snr]
        per_seed = []
        for seed in sorted({row["seed"] for row in est}):
            est_seed = [row for row in est if row["seed"] == seed]
            true_seed = [true_map[key(row)] for row in est_seed]
            est_wrong = sum(row["wrong"] for row in est_seed)
            true_wrong = sum(row["wrong"] for row in true_seed)
            total = sum(row["total"] for row in est_seed)
            per_seed.append({
                "seed": seed,
                "ber_estimated": est_wrong / max(1, total),
                "ber_true": true_wrong / max(1, total),
                "delta_true_minus_estimated": (true_wrong - est_wrong) / max(1, total),
            })
        per_snr[snr] = {
            "ber_estimated": paired_interval([row["ber_estimated"] for row in per_seed]),
            "ber_true": paired_interval([row["ber_true"] for row in per_seed]),
            "delta_true_minus_estimated": paired_interval(
                [row["delta_true_minus_estimated"] for row in per_seed]),
            "per_seed": per_seed,
        }
    return per_snr


def _row_for_json(row: dict) -> dict:
    return {
        "seed": row["seed"], "snr_db": row["snr_db"], "frame_index": row["frame_index"],
        "channel_frame_index": row["channel_frame_index"],
        "ber_frozen": row["ber"], "bce_frozen": row["bce"],
        "wrong": row["wrong"], "total": row["total"],
    }


def run(args) -> None:
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device(args.device)
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and (output_dir / "summary.json").exists() and not args.overwrite:
        raise ValueError("输出已存在，禁止覆盖；请换目录或显式 --overwrite。")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    model, payload = _load_branch(checkpoint, config, device)
    seeds = [int(value) for value in args.seeds]
    snrs = [float(value) for value in args.snrs]
    frames = int(args.frames)
    if frames < 1 or len(set(seeds)) != len(seeds):
        raise ValueError("需要正帧数与不重复 seed。")
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    frozen_rows = []
    head_sessions = []
    true_rows = []
    for snr in snrs:
        for seed in seeds:
            session = collect_session(model, config, seed=seed, snr=snr, frames=frames,
                                      state_split=args.state_split, device=device,
                                      cir_mode="estimated")
            frozen_rows.extend(session)
            _attach_frozen_logits(session)
            if args.mode == "all":
                head_sessions.append(oracle_head_for_session(
                    model, session, k_values=args.head_k, lr=args.head_lr,
                    split_mode=args.head_split))
                true_session = collect_session(model, config, seed=seed, snr=snr, frames=frames,
                                               state_split=args.state_split, device=device,
                                               cir_mode="true")
                true_rows.extend(true_session)
            print(json.dumps({"snr_db": snr, "seed": seed, "done": True}, ensure_ascii=False), flush=True)

    wall = time.perf_counter() - started
    peak_memory = (int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None)
    result = {
        "role": "offline_saturation_development_diagnostic",
        "deployable": False,
        "oracle_uses_data_labels": args.mode == "all",
        "mode": args.mode,
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)},
        "config": str(Path(args.config).resolve()),
        "model_config": payload["model_config"],
        "state_split": args.state_split,
        "seeds": seeds, "snrs_db": snrs, "frames_per_seed_snr": frames,
        "device": str(device), "wall_seconds": wall,
        "peak_cuda_memory_bytes": peak_memory,
        "tail_protocol": "previous_frame_last_max_delay_logits_soft_alpha1.0",
        "frozen_per_snr": summarize_frozen(frozen_rows),
        "frozen_rows": [_row_for_json(row) for row in frozen_rows],
        "command": " ".join(sys.argv),
    }
    if args.mode == "all":
        result["oracle_head"] = {
            "lr": float(args.head_lr),
            "split": args.head_split,
            "k_values": [int(value) for value in args.head_k],
            "per_snr": summarize_oracle_head(head_sessions, [int(value) for value in args.head_k]),
            "per_session": [
                {key: value for key, value in row.items() if key != "contexts"}
                for row in head_sessions
            ],
        }
        result["oracle_cir"] = {
            "per_snr": summarize_cir(frozen_rows, true_rows),
            "true_rows": [_row_for_json(row) for row in true_rows],
        }
    if args.paired_reference:
        reference = json.loads(Path(args.paired_reference).read_text(encoding="utf-8"))
        ref_map = {(row["seed"], row["snr_db"], row["frame_index"]): row
                   for row in reference["frozen_rows"]}
        per_snr = {}
        for snr in snrs:
            subset = [row for row in frozen_rows if row["snr_db"] == snr]
            per_seed = []
            for seed in sorted({row["seed"] for row in subset}):
                items = [row for row in subset if row["seed"] == seed]
                paired = []
                for row in items:
                    ref = ref_map[(row["seed"], row["snr_db"], row["frame_index"])]
                    paired.append((row["wrong"] - ref["wrong"]) / max(1, row["total"]))
                per_seed.append({"seed": seed, "delta_vs_reference": float(np.mean(paired))})
            per_snr[snr] = paired_interval([row["delta_vs_reference"] for row in per_seed])
        result["paired_vs_reference"] = {
            "reference": str(Path(args.paired_reference).resolve()),
            "definition": "ber_current_minus_ber_reference",
            "per_snr": per_snr,
        }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"wall_seconds": wall, "peak_cuda_memory_bytes": peak_memory}, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="离线网络饱和/可修复空间开发诊断")
    parser.add_argument("--config", default="configs/eme_long_memory_v2.json")
    parser.add_argument("--checkpoint", default="D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt")
    parser.add_argument("--seeds", nargs="+", type=int, default=[91001, 91002, 91003, 91004, 91005])
    parser.add_argument("--snrs", nargs="+", type=float, default=[0, 5, 10, 15])
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--state-split", default="offline_train",
                        choices=["offline_train", "heldout_edge"])
    parser.add_argument("--mode", default="all", choices=["frozen", "all"])
    parser.add_argument("--head-k", nargs="+", type=int, default=[1, 4, 16, 64])
    parser.add_argument("--head-lr", type=float, default=1e-2)
    parser.add_argument("--head-split", default="in_sample", choices=["in_sample", "heldout"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--paired-reference", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
