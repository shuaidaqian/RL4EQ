# -*- coding: utf-8 -*-
"""0 dB Data 后验变体诊断：soft / oracle / EM / 块相位 / EM+块相位。

两臂共享 acquisition CIR + dd_soft tail；各变体从同一先验出发，逐帧因果更新。
只有 ``oracle`` 变体使用 Data 标签（真实符号错误）构造行噪声；其余变体只用
接收波形、模型 soft 输出与估计 CIR。

同时记录逐帧耗时，作为成本 profile 的粗粒度拆解；Frozen 作为 BER 参考。
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

from evaluation.pilot_transfer import _load_branch
from scripts.diagnose_rt_oracle import _data_posterior as _oracle_data_posterior
from training.bayesian_cir import (
    acquisition_prior,
    add_process_noise,
    data_posterior,
    estimate_data_block_phases,
    pilot_posterior,
)
from training.pilot_transfer import context_logits, make_context, make_episode, next_pilot_tail

VARIANTS = ("soft", "oracle", "em2", "em3", "block_phase", "em_block")


def _sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def _ber(logits: torch.Tensor, frame) -> float:
    mask = frame.data_mask.bool()
    labels = frame.bits[mask].bool()
    return float(((logits[mask] >= 0) != labels).float().mean())


def _nmse(estimate: torch.Tensor, truth: torch.Tensor) -> float:
    est = estimate.reshape(-1).to(torch.complex128)
    tru = truth.reshape(-1).to(torch.complex128)
    scale = torch.vdot(est, tru) / torch.vdot(est, est).real.clamp_min(1e-12)
    residual = tru - scale * est
    return float((residual.abs().square().sum() / tru.abs().square().sum().clamp_min(1e-12)).real)


def _error_lag1(logits: torch.Tensor, frame) -> float:
    mask = frame.data_mask.bool()
    errors = ((logits[mask] >= 0) != frame.bits[mask].bool()).float()
    if errors.numel() < 2:
        return 0.0
    centered = errors - errors.mean()
    denominator = centered.square().mean().clamp_min(1e-12)
    return float((centered[:-1] * centered[1:]).mean() / denominator)


def _em_posterior(
    model, frame, context, tail, cir, covariance, cfo, snr, max_delay,
    noise_variance, phase0, cfo_est, steps, block_phase,
):
    logits = context_logits(model, context, None, visible="data")
    for step in range(int(steps)):
        if block_phase:
            phases = estimate_data_block_phases(
                frame, logits, cir, max_delay, phase0, cfo_est,
            )
            cir, covariance = data_posterior(
                frame, logits, cir, covariance, max_delay, noise_variance,
                phase0, cfo_est, block_phases=phases,
            )
        else:
            cir, covariance = data_posterior(
                frame, logits, cir, covariance, max_delay, noise_variance,
                phase0, cfo_est,
            )
        if step < int(steps) - 1:
            context = make_context(frame, cir, cfo, tail, snr)
            logits = context_logits(model, context, None, visible="data")
    return cir, covariance, context, logits


def run_variant(config, model, *, seed, snr, frames, variants=VARIANTS, device="cpu"):
    """在同一 episode 上运行 Frozen 参考与各 Data 后验变体。"""

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
    frozen_tail = tail.clone()
    states = {
        variant: {
            "cir": prior_mean.detach().clone(),
            "covariance": prior_covariance.detach().clone(),
            "tail": tail.clone(),
        }
        for variant in variants
    }

    rows = []
    for frame in received:
        rx_sha256 = _sha(frame.rx_symbols)
        frozen_context = make_context(frame, frozen_cir, cfo, frozen_tail, snr)
        with torch.no_grad():
            frozen_logits = context_logits(model, frozen_context, None, visible="data")
            frozen_ber = _ber(frozen_logits, frame)
        frozen_tail = next_pilot_tail(model, frozen_context, alpha=1.0, mode="dd_soft")

        for variant in variants:
            state = states[variant]
            start = time.perf_counter()
            context = make_context(frame, state["cir"], cfo, state["tail"], snr)
            with torch.no_grad():
                logits = context_logits(model, context, None, visible="data")
            latent = context.condition.latent_residual
            phase0 = float(latent[0, 0].detach().cpu()) if latent is not None else 0.0
            cfo_est = float(latent[0, 1].detach().cpu()) if latent is not None else float(cfo)
            pilot_cir, pilot_covariance = pilot_posterior(
                frame, state["cir"], state["covariance"], max_delay, noise_variance,
                phase0, cfo_est,
            )

            final_logits = None
            if variant == "soft":
                final_cir, final_covariance = data_posterior(
                    frame, logits, pilot_cir, pilot_covariance, max_delay,
                    noise_variance, phase0, cfo_est,
                )
            elif variant == "oracle":
                final_cir, final_covariance = _oracle_data_posterior(
                    frame, logits, pilot_cir, pilot_covariance, max_delay,
                    noise_variance, phase0, cfo_est, "oracle",
                )
            elif variant in {"em2", "em3"}:
                steps = 2 if variant == "em2" else 3
                final_cir, final_covariance, context, final_logits = _em_posterior(
                    model, frame, context, state["tail"], pilot_cir, pilot_covariance,
                    cfo, snr, max_delay, noise_variance, phase0, cfo_est, steps, False,
                )
            elif variant == "block_phase":
                phases = estimate_data_block_phases(
                    frame, logits, pilot_cir, max_delay, phase0, cfo_est,
                )
                final_cir, final_covariance = data_posterior(
                    frame, logits, pilot_cir, pilot_covariance, max_delay,
                    noise_variance, phase0, cfo_est, block_phases=phases,
                )
            elif variant == "em_block":
                final_cir, final_covariance, context, final_logits = _em_posterior(
                    model, frame, context, state["tail"], pilot_cir, pilot_covariance,
                    cfo, snr, max_delay, noise_variance, phase0, cfo_est, 2, True,
                )
            else:
                raise ValueError(f"未知变体：{variant}")

            final_covariance = add_process_noise(final_covariance, 1e-4)
            if final_logits is None:
                final_context = make_context(frame, final_cir, cfo, state["tail"], snr)
                with torch.no_grad():
                    final_logits = context_logits(model, final_context, None, visible="data")
            else:
                final_context = context

            ber_data = _ber(final_logits, frame)
            nmse = _nmse(
                final_cir, frame.true_cir.to(device=device, dtype=torch.complex64),
            )
            lag1 = _error_lag1(final_logits, frame)
            state["cir"], state["covariance"] = final_cir, final_covariance
            state["tail"] = next_pilot_tail(model, final_context, alpha=1.0, mode="dd_soft")
            rows.append({
                "seed": int(seed), "snr_db": int(snr), "frame_index": int(frame.frame_index),
                "variant": variant, "ber_data": ber_data, "frozen_ber": frozen_ber,
                "gain": frozen_ber - ber_data, "nmse": nmse, "error_lag1": lag1,
                "elapsed_ms": (time.perf_counter() - start) * 1000.0,
                "rx_sha256": rx_sha256,
                "uses_reward_or_data_labels": bool(variant == "oracle"),
            })
    return rows


def _summarize(rows, config, variants, seeds, frames):
    per_snr = {}
    for snr in config["main_snrs"]:
        snr_rows = [row for row in rows if row["snr_db"] == snr]
        variant_summary = {}
        for variant in variants:
            selected = [row for row in snr_rows if row["variant"] == variant]
            if not selected:
                continue
            per_seed = {}
            for seed in seeds:
                seed_rows = [row for row in selected if row["seed"] == seed]
                per_seed[str(seed)] = sum(row["ber_data"] for row in seed_rows) / len(seed_rows)
            window_size = max(1, int(frames) // 4)
            windows = []
            for start in range(0, int(frames), window_size):
                window = [
                    row for row in selected
                    if start <= row["frame_index"] < start + window_size
                ]
                if window:
                    windows.append(sum(row["gain"] for row in window) / len(window))
            if len(windows) >= 2:
                indices = torch.arange(len(windows), dtype=torch.float32)
                centered = indices - indices.mean()
                slope = float(
                    (centered * (torch.tensor(windows) - sum(windows) / len(windows))).sum()
                    / centered.square().sum()
                )
            else:
                slope = 0.0
            variant_summary[variant] = {
                "ber_data": sum(row["ber_data"] for row in selected) / len(selected),
                "frozen_ber": sum(row["frozen_ber"] for row in selected) / len(selected),
                "gain": sum(row["gain"] for row in selected) / len(selected),
                "nmse": sum(row["nmse"] for row in selected) / len(selected),
                "error_lag1": sum(row["error_lag1"] for row in selected) / len(selected),
                "elapsed_ms": sum(row["elapsed_ms"] for row in selected) / len(selected),
                "windows": windows, "first_window": windows[0] if windows else 0.0,
                "last_window": windows[-1] if windows else 0.0, "slope": slope,
                "per_seed": per_seed,
                "uses_reward_or_data_labels": any(
                    row["uses_reward_or_data_labels"] for row in selected
                ),
            }
        per_snr[str(snr)] = variant_summary

    fairness = {
        "same_episode_for_all_variants": all(
            len({
                row["rx_sha256"] for row in rows
                if row["seed"] == seed and row["snr_db"] == snr
                and row["frame_index"] == frame
            }) == 1
            for snr in config["main_snrs"] for seed in seeds for frame in range(int(frames))
        ),
        "non_oracle_uses_reward_or_data_labels": any(
            row["uses_reward_or_data_labels"] for row in rows if row["variant"] != "oracle"
        ),
    }
    return {
        "seeds": list(seeds), "frames": int(frames), "variants": list(variants),
        "fairness": fairness, "per_snr": per_snr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[140001, 140002])
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
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
            rows.extend(run_variant(
                config, model, seed=seed, snr=snr, frames=args.frames,
                variants=tuple(args.variants), device=device,
            ))
    summary = _summarize(rows, config, tuple(args.variants), args.seeds, args.frames)
    summary["rows"] = rows
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for snr, payload in summary["per_snr"].items():
        for variant, row in payload.items():
            print(
                f"snr={snr:>2} {variant:10s} ber={row['ber_data']:.6f} "
                f"gain={row['gain']:+.6f} windows={[round(x, 5) for x in row['windows']]} "
                f"slope={row['slope']:+.6f} nmse={row['nmse']:.4f} "
                f"lag1={row['error_lag1']:+.3f} ms={row['elapsed_ms']:.0f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
