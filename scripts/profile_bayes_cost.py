# -*- coding: utf-8 -*-
"""Bayes CIR 逐帧成本 profile。

拆解每帧：模型前向、Pilot 后验、Data 后验、soft-tail 独立前向与 tail 复用
（直接用当前帧 logits 的最后 116 个位置递推）。tail 复用与独立前向应完全一致。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pilot_transfer import _load_branch
from training.bayesian_cir import acquisition_prior, add_process_noise, data_posterior, pilot_posterior
from training.pilot_transfer import context_logits, make_context, make_episode, next_pilot_tail


def _row(**values):
    return values


def run_profile(config, model, *, seed, snr, frames, device="cpu"):
    """返回逐帧组件耗时；tail 复用与 next_pilot_tail 的一致性记录在 tail_match。"""

    device = torch.device(device) if isinstance(device, str) else device
    _, _, cfo, tail, received, acquisition = make_episode(
        config, model, seed=seed, snr=snr, frames=frames,
        state_split="offline_train", device=device, return_acquisition=True,
    )
    max_delay = int(config["max_delay"])
    noise_variance = float(10.0 ** (-float(snr) / 10.0))

    start = time.perf_counter()
    prior_mean, prior_covariance = acquisition_prior(
        acquisition, cfo, max_delay, noise_variance, ridge=1e-2,
    )
    acquisition_ms = (time.perf_counter() - start) * 1000.0

    cir = prior_mean.detach().clone()
    covariance = prior_covariance.detach().clone()
    tail_state = tail.clone()
    rows = []
    for frame in received:
        context = make_context(frame, cir, cfo, tail_state, snr)

        start = time.perf_counter()
        with torch.no_grad():
            logits = context_logits(model, context, None, visible="data")
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

        start = time.perf_counter()
        forward_tail = next_pilot_tail(model, context, alpha=1.0, mode="dd_soft")
        tail_forward_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        tail_len = tail_state.numel()
        values = torch.tanh(logits[-tail_len:] / 2.0)
        detected = torch.complex(values, torch.zeros_like(values))
        reuse_tail = ((1.0 - 1.0) * tail_state + 1.0 * detected).detach()
        tail_reuse_ms = (time.perf_counter() - start) * 1000.0
        tail_match = bool(torch.allclose(reuse_tail, forward_tail, rtol=0, atol=0))

        rows.append(_row(
            seed=int(seed), snr_db=int(snr), frame_index=int(frame.frame_index),
            acquisition_ms=acquisition_ms, forward_ms=forward_ms, pilot_ms=pilot_ms,
            data_ms=data_ms, tail_forward_ms=tail_forward_ms, tail_reuse_ms=tail_reuse_ms,
            total_forward_ms=forward_ms + pilot_ms + data_ms + tail_forward_ms,
            total_reuse_ms=forward_ms + pilot_ms + data_ms + tail_reuse_ms,
            tail_match=tail_match,
        ))

        cir, covariance = final_cir, add_process_noise(final_covariance, 1e-4)
        tail_state = reuse_tail
    return rows


def _summarize(rows, config, seeds, frames):
    keys = (
        "acquisition_ms", "forward_ms", "pilot_ms", "data_ms",
        "tail_forward_ms", "tail_reuse_ms", "total_forward_ms", "total_reuse_ms",
    )
    per_snr = {}
    for snr in config["main_snrs"]:
        selected = [row for row in rows if row["snr_db"] == snr]
        summary = {key: sum(row[key] for row in selected) / len(selected) for key in keys}
        summary["tail_match_all"] = all(row["tail_match"] for row in selected)
        summary["frame_budget_ms"] = 102.4
        summary["realtime_ratio"] = summary["total_reuse_ms"] / 102.4
        per_snr[str(snr)] = summary
    return {"seeds": list(seeds), "frames": int(frames), "per_snr": per_snr}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[160001])
    parser.add_argument("--frames", type=int, default=8)
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
            rows.extend(run_profile(
                config, model, seed=seed, snr=snr, frames=args.frames, device=device,
            ))
    summary = _summarize(rows, config, args.seeds, args.frames)
    summary["rows"] = rows
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for snr, row in summary["per_snr"].items():
        print(
            f"snr={snr:>2} forward={row['forward_ms']:.0f}ms pilot={row['pilot_ms']:.1f}ms "
            f"data={row['data_ms']:.1f}ms tail_fwd={row['tail_forward_ms']:.0f}ms "
            f"total_fwd={row['total_forward_ms']:.0f}ms total_reuse={row['total_reuse_ms']:.0f}ms "
            f"ratio={row['realtime_ratio']:.1f}x budget=102.4ms tail_match={row['tail_match_all']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
