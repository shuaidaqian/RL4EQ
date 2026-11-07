# -*- coding: utf-8 -*-
"""warm-start 迭代数与 AMP 的成本-精度扫描。

固定已确认的 Bayes CIR 逻辑与 tail 复用，只改变：
- ``physics_warm_start_iterations``（8/4/2）；
- 模型前向是否使用 autocast（CUDA fp16 / CPU bf16）。

报告每 SNR 的 Data BER、逐帧总耗时、相对 102.4 ms 预算比、相对
(ws=8, amp=0) 的“不退化”检查，以及 tail 复用与独立 dd_soft 的一致性。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pilot_transfer import _load_branch
from training.bayesian_cir import acquisition_prior, add_process_noise, data_posterior, pilot_posterior
from training.pilot_transfer import (
    context_logits, make_context, make_episode, next_pilot_tail, tail_from_logits,
)

DEFAULT_VARIANTS = ((8, False), (8, True), (4, False), (4, True), (2, False), (2, True))


def _autocast(device: torch.device, enabled: bool):
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=bool(enabled))


def _ber(logits, frame):
    mask = frame.data_mask.bool()
    labels = frame.bits[mask].bool()
    return float(((logits[mask] >= 0) != labels).float().mean())


def _nmse(estimate, truth):
    est = estimate.reshape(-1).to(torch.complex128)
    tru = truth.reshape(-1).to(torch.complex128)
    scale = torch.vdot(est, tru) / torch.vdot(est, est).real.clamp_min(1e-12)
    residual = tru - scale * est
    return float((residual.abs().square().sum() / tru.abs().square().sum().clamp_min(1e-12)).real)


def _run(config, model, *, seed, snr, frames, warm_start, amp, device):
    _, _, cfo, tail, received, acquisition = make_episode(
        config, model, seed=seed, snr=snr, frames=frames,
        state_split="offline_train", device=device, return_acquisition=True,
    )
    max_delay = int(config["max_delay"])
    noise_variance = float(10.0 ** (-float(snr) / 10.0))
    prior_mean, prior_covariance = acquisition_prior(
        acquisition, cfo, max_delay, noise_variance, ridge=1e-2,
    )
    cir = prior_mean.detach().clone()
    covariance = prior_covariance.detach().clone()
    tail_state = tail.clone()
    rows = []
    for frame in received:
        context = make_context(frame, cir, cfo, tail_state, snr)
        start = time.perf_counter()
        with _autocast(device, amp):
            with torch.no_grad():
                logits = context_logits(model, context, None, visible="data").float()
        forward_ms = (time.perf_counter() - start) * 1000.0

        latent = context.condition.latent_residual
        phase0 = float(latent[0, 0].detach().cpu()) if latent is not None else 0.0
        cfo_est = float(latent[0, 1].detach().cpu()) if latent is not None else float(cfo)

        start = time.perf_counter()
        pilot_cir, pilot_covariance = pilot_posterior(
            frame, cir, covariance, max_delay, noise_variance, phase0, cfo_est,
        )
        pilot_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        final_cir, final_covariance = data_posterior(
            frame, logits, pilot_cir, pilot_covariance, max_delay, noise_variance, phase0, cfo_est,
        )
        data_ms = (time.perf_counter() - start) * 1000.0
        final_covariance = add_process_noise(final_covariance, 1e-4)

        final_context = make_context(frame, final_cir, cfo, tail_state, snr)
        with _autocast(device, amp):
            with torch.no_grad():
                final_logits = context_logits(model, final_context, None, visible="data").float()
        ber_data = _ber(final_logits, frame)
        nmse = _nmse(final_cir, frame.true_cir.to(device=device, dtype=torch.complex64))

        start = time.perf_counter()
        reuse_tail = tail_from_logits(final_logits, tail_state, 1.0)
        reuse_ms = (time.perf_counter() - start) * 1000.0
        with _autocast(device, amp):
            with torch.no_grad():
                forward_tail = next_pilot_tail(model, final_context, alpha=1.0, mode="dd_soft")
        forward_tail_ms = (time.perf_counter() - start) * 1000.0 - reuse_ms
        tail_match = bool(torch.allclose(reuse_tail, forward_tail, rtol=0, atol=0))

        cir, covariance, tail_state = final_cir, final_covariance, reuse_tail
        rows.append({
            "seed": int(seed), "snr_db": int(snr), "frame_index": int(frame.frame_index),
            "warm_start": int(warm_start), "amp": bool(amp),
            "ber_data": ber_data, "nmse": nmse,
            "forward_ms": forward_ms, "pilot_ms": pilot_ms, "data_ms": data_ms,
            "tail_forward_ms": forward_tail_ms, "tail_reuse_ms": reuse_ms,
            "total_ms": forward_ms + pilot_ms + data_ms + reuse_ms,
            "tail_match": tail_match,
        })
    return rows


def run_sweep(config, model, *, seeds, snr, frames, variants=DEFAULT_VARIANTS, device="cpu"):
    device = torch.device(device) if isinstance(device, str) else device
    rows = []
    for warm_start, amp in variants:
        model.config = replace(model.config, physics_warm_start_iterations=int(warm_start))
        for seed in seeds:
            rows.extend(_run(
                config, model, seed=seed, snr=snr, frames=frames,
                warm_start=warm_start, amp=amp, device=device,
            ))
    return rows


def _summarize(rows, config, seeds, frames):
    per_snr = {}
    for snr in config["main_snrs"]:
        selected = [row for row in rows if row["snr_db"] == snr]
        variants = {}
        for row in selected:
            label = f"ws{row['warm_start']}_amp{int(row['amp'])}"
            variants.setdefault(label, []).append(row)
        baseline_rows = variants.get("ws8_amp0", [])
        baseline_ber = (
            sum(row["ber_data"] for row in baseline_rows) / len(baseline_rows)
            if baseline_rows else None
        )
        summary = {}
        for label, items in variants.items():
            ber = sum(row["ber_data"] for row in items) / len(items)
            total_ms = sum(row["total_ms"] for row in items) / len(items)
            summary[label] = {
                "ber_data": ber,
                "forward_ms": sum(row["forward_ms"] for row in items) / len(items),
                "total_ms": total_ms,
                "ratio": total_ms / 102.4,
                "no_degradation_vs_ws8_amp0": (baseline_ber is None or ber <= baseline_ber),
                "tail_match_all": all(row["tail_match"] for row in items),
            }
        per_snr[str(snr)] = summary
    return {"seeds": list(seeds), "frames": int(frames), "per_snr": per_snr}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[160201])
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument(
        "--variants", nargs="+", default=[f"{ws}:{int(amp)}" for ws, amp in DEFAULT_VARIANTS],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model, _ = _load_branch(Path(args.checkpoint), config, device)
    model.eval()

    variants = tuple(
        (int(warm), bool(int(amp)))
        for warm, amp in (item.split(":") for item in args.variants)
    )
    rows = []
    for snr in config["main_snrs"]:
        rows.extend(run_sweep(
            config, model, seeds=args.seeds, snr=snr, frames=args.frames,
            variants=variants, device=device,
        ))
    summary = _summarize(rows, config, args.seeds, args.frames)
    summary["rows"] = rows
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for snr, payload in summary["per_snr"].items():
        for label, row in payload.items():
            print(
                f"snr={snr:>2} {label:12s} ber={row['ber_data']:.6f} "
                f"forward={row['forward_ms']:.0f}ms total={row['total_ms']:.0f}ms "
                f"ratio={row['ratio']:.1f}x no_degradation={row['no_degradation_vs_ws8_amp0']} "
                f"tail_match={row['tail_match_all']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
