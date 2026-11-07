# -*- coding: utf-8 -*-
"""物理分支的公平传统基线比较。

所有方法共享同一 episode、同一接收波形、同一 Pilot 预算和同一 acquisition；
Frozen/Bayes 使用同一来源 checkpoint 与 dd_soft tail，传统方法使用同一
acquisition CIR 和合法 Pilot/判决反馈，不读取 Reward/Data 标签。

口径：
- Frozen：acquisition 先验 CIR + dd_soft tail + 冻结神经模型；
- Bayes：同一先验 → Pilot/Data 后验 CIR + dd_soft tail + 冻结神经模型；
- 传统：CFO+DD-Phase LMMSE-FIR / DFE-RLS / SC-FDE-MMSE，使用同一 acquisition CIR。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.traditional_equalizers import TraditionalPhaseState, run_traditional_equalizer
from evaluation.pilot_transfer import _load_branch
from training.bayesian_cir import acquisition_prior, add_process_noise, data_posterior, pilot_posterior
from training.pilot_transfer import context_logits, make_context, make_episode, next_pilot_tail

TRADITIONAL_COMPARISON_METHODS = (
    "CFO+DD-Phase LMMSE-FIR",
    "CFO+DD-Phase DFE-RLS",
    "SC-FDE-MMSE",
)
DEFAULT_METHODS = ("frozen", "bayes", *TRADITIONAL_COMPARISON_METHODS)


def _sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def _ber(logits: torch.Tensor, frame, mask_name: str) -> float:
    mask = getattr(frame, f"{mask_name}_mask").bool()
    labels = frame.bits[mask].bool()
    return float(((logits[mask] >= 0) != labels).float().mean())


def _row(frame, method, logits, *, seed, snr, rx_sha256, cir_sha256, uses_neural_network,
         uses_reward_or_data_labels, iterations, pilot_total, pilot_layout, elapsed_ms):
    return {
        "seed": int(seed),
        "snr_db": int(snr),
        "frame_index": int(frame.frame_index),
        "method": method,
        "ber_data": _ber(logits, frame, "data"),
        "ber_adapt": _ber(logits, frame, "adapt"),
        "ber_reward": _ber(logits, frame, "reward"),
        "rx_sha256": rx_sha256,
        "cir_sha256": cir_sha256,
        "uses_neural_network": bool(uses_neural_network),
        "uses_reward_or_data_labels": bool(uses_reward_or_data_labels),
        "iterations": int(iterations),
        "pilot_total": int(pilot_total),
        "pilot_layout": str(pilot_layout),
        "elapsed_ms": float(elapsed_ms),
    }


def run_comparison(config, model, *, seed, snr, frames, methods=DEFAULT_METHODS, device="cpu"):
    """在同一个 episode 上运行 Frozen/Bayes/传统基线并返回逐帧记录。"""

    device = torch.device(device) if isinstance(device, str) else device
    _, _, cfo, tail, received, acquisition = make_episode(
        config, model, seed=seed, snr=snr, frames=frames,
        state_split="offline_train", device=device, return_acquisition=True,
    )
    max_delay = int(config["max_delay"])
    noise_variance = float(10.0 ** (-float(snr) / 10.0))
    prior_mean, prior_covariance = acquisition_prior(
        acquisition, cfo, max_delay, noise_variance, ridge=1e-2,
    )
    frozen_cir = prior_mean.detach().clone()
    bayes_cir = prior_mean.detach().clone()
    bayes_covariance = prior_covariance.detach().clone()
    frozen_tail = tail.clone()
    bayes_tail = tail.clone()
    pilot_total = int(config["pilot_total"])
    pilot_layout = str(config["pilot_layout"])
    neural_iterations = int(model.config.iterations) + int(model.config.physics_warm_start_iterations)
    traditional = {
        method: {
            "tail": tail.clone(),
            "phase": TraditionalPhaseState(
                smoothing=float(config.get("online_phase_tracking_smoothing", 0.5)),
            ),
        }
        for method in TRADITIONAL_COMPARISON_METHODS if method in methods
    }

    rows = []
    for frame in received:
        rx_sha256 = _sha(frame.rx_symbols)
        view = frame.receiver_view()

        if "frozen" in methods:
            context = make_context(frame, frozen_cir, cfo, frozen_tail, snr)
            start = time.perf_counter()
            with torch.no_grad():
                logits = context_logits(model, context, None, visible="data")
            frozen_tail = next_pilot_tail(model, context, alpha=1.0, mode="dd_soft")
            rows.append(_row(
                frame, "frozen", logits, seed=seed, snr=snr, rx_sha256=rx_sha256,
                cir_sha256=_sha(frozen_cir), uses_neural_network=True,
                uses_reward_or_data_labels=False, iterations=neural_iterations,
                pilot_total=pilot_total, pilot_layout=pilot_layout,
                elapsed_ms=(time.perf_counter() - start) * 1000.0,
            ))

        if "bayes" in methods:
            current_bayes_cir = bayes_cir
            context = make_context(frame, current_bayes_cir, cfo, bayes_tail, snr)
            start = time.perf_counter()
            with torch.no_grad():
                logits = context_logits(model, context, None, visible="data")
            latent = context.condition.latent_residual
            phase0 = float(latent[0, 0].detach().cpu()) if latent is not None else 0.0
            cfo_est = float(latent[0, 1].detach().cpu()) if latent is not None else float(cfo)
            bayes_cir, bayes_covariance = pilot_posterior(
                frame, bayes_cir, bayes_covariance, max_delay, noise_variance, phase0, cfo_est,
            )
            bayes_cir, bayes_covariance = data_posterior(
                frame, logits, bayes_cir, bayes_covariance, max_delay, noise_variance, phase0, cfo_est,
            )
            bayes_covariance = add_process_noise(bayes_covariance, 1e-4)
            bayes_tail = next_pilot_tail(model, context, alpha=1.0, mode="dd_soft")
            rows.append(_row(
                frame, "bayes", logits, seed=seed, snr=snr, rx_sha256=rx_sha256,
                cir_sha256=_sha(current_bayes_cir), uses_neural_network=True,
                uses_reward_or_data_labels=False, iterations=neural_iterations,
                pilot_total=pilot_total, pilot_layout=pilot_layout,
                elapsed_ms=(time.perf_counter() - start) * 1000.0,
            ))

        for method in TRADITIONAL_COMPARISON_METHODS:
            if method not in methods:
                continue
            state = traditional[method]
            start = time.perf_counter()
            result = run_traditional_equalizer(
                method, view, frozen_cir, state["tail"], snr, state["phase"],
                residual_cfo_limit=float(config.get("residual_cfo_limit", 0.0012)),
            )
            state["tail"] = result.soft_tail
            rows.append(_row(
                frame, method, result.logits, seed=seed, snr=snr, rx_sha256=rx_sha256,
                cir_sha256=_sha(frozen_cir), uses_neural_network=False,
                uses_reward_or_data_labels=bool(result.extra.get("uses_reward_or_data_labels", True)),
                iterations=result.iterations, pilot_total=pilot_total, pilot_layout=pilot_layout,
                elapsed_ms=(time.perf_counter() - start) * 1000.0,
            ))

    return rows


def _summarize(rows, config, methods, seeds, frames, checkpoint):
    by_snr = {}
    for snr in config["main_snrs"]:
        snr_rows = [row for row in rows if row["snr_db"] == snr]
        method_summary = {}
        for method in methods:
            selected = [row for row in snr_rows if row["method"] == method]
            if not selected:
                continue
            per_seed = {}
            for seed in seeds:
                seed_rows = [row for row in selected if row["seed"] == seed]
                per_seed[str(seed)] = sum(row["ber_data"] for row in seed_rows) / len(seed_rows)
            method_summary[method] = {
                "ber_data": sum(row["ber_data"] for row in selected) / len(selected),
                "ber_adapt": sum(row["ber_adapt"] for row in selected) / len(selected),
                "ber_reward": sum(row["ber_reward"] for row in selected) / len(selected),
                "elapsed_ms": sum(row["elapsed_ms"] for row in selected) / len(selected),
                "iterations": sum(row["iterations"] for row in selected) / len(selected),
                "uses_neural_network": all(row["uses_neural_network"] for row in selected),
                "uses_reward_or_data_labels": any(row["uses_reward_or_data_labels"] for row in selected),
                "per_seed": per_seed,
            }
        paired = {}
        for method in TRADITIONAL_COMPARISON_METHODS:
            if method not in methods or "bayes" not in methods:
                continue
            # 逐 (seed, frame) 配对：Bayes/Frozen 减传统
            bayes_rows = {(row["seed"], row["frame_index"]): row for row in snr_rows if row["method"] == "bayes"}
            frozen_rows = {(row["seed"], row["frame_index"]): row for row in snr_rows if row["method"] == "frozen"}
            trad_rows = {(row["seed"], row["frame_index"]): row for row in snr_rows if row["method"] == method}
            common = sorted(set(bayes_rows) & set(trad_rows) & set(frozen_rows))
            paired[method] = {
                "bayes_minus_traditional": sum(
                    trad_rows[key]["ber_data"] - bayes_rows[key]["ber_data"] for key in common
                ) / max(1, len(common)),
                "frozen_minus_traditional": sum(
                    trad_rows[key]["ber_data"] - frozen_rows[key]["ber_data"] for key in common
                ) / max(1, len(common)),
                "pairs": len(common),
            }
        by_snr[str(snr)] = {"methods": method_summary, "paired": paired}

    fairness = {
        "same_episode_for_all_methods": all(
            len({row["rx_sha256"] for row in rows if row["seed"] == seed and row["snr_db"] == snr
                 and row["frame_index"] == frame}) == 1
            for snr in config["main_snrs"] for seed in seeds for frame in range(frames)
        ),
        "traditional_uses_neural_network": any(
            row["uses_neural_network"] for row in rows if row["method"] in TRADITIONAL_COMPARISON_METHODS
        ),
        "traditional_uses_reward_or_data_labels": any(
            row["uses_reward_or_data_labels"] for row in rows if row["method"] in TRADITIONAL_COMPARISON_METHODS
        ),
        "pilot_total": int(config["pilot_total"]),
        "pilot_layout": str(config["pilot_layout"]),
        "acquisition_cir_shared_by_frozen_and_traditional": all(
            len({row["cir_sha256"] for row in rows if row["seed"] == seed and row["snr_db"] == snr
                 and row["frame_index"] == frame and row["method"] in ("frozen", *TRADITIONAL_COMPARISON_METHODS)}) == 1
            for snr in config["main_snrs"] for seed in seeds for frame in range(frames)
        ),
    }
    return {
        "config": str(config.get("profile_name", config.get("channel_profile"))),
        "checkpoint": str(checkpoint),
        "seeds": list(seeds),
        "frames": int(frames),
        "methods": list(methods),
        "fairness": fairness,
        "per_snr": by_snr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[130001, 130002, 130003])
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model, _ = _load_branch(Path(args.checkpoint), config, device)
    model.eval()

    rows = []
    for snr in config["main_snrs"]:
        for seed in args.seeds:
            rows.extend(run_comparison(
                config, model, seed=seed, snr=snr, frames=args.frames,
                methods=tuple(args.methods), device=device,
            ))
    summary = _summarize(rows, config, tuple(args.methods), args.seeds, args.frames, args.checkpoint)
    summary["rows"] = rows
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for snr, payload in summary["per_snr"].items():
        line = f"snr={snr:>2} " + " ".join(
            f"{method}={payload['methods'][method]['ber_data']:.6f}"
            for method in args.methods if method in payload["methods"]
        )
        print(line, flush=True)


if __name__ == "__main__":
    main()
