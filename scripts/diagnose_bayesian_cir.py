# -*- coding: utf-8 -*-
"""只读诊断：贝叶斯递归 CIR 估计相对 Frozen 的 Data BER 与 CIR NMSE。

Frozen 与 Bayes 共用同一 acquisition 先验、同一 dd_soft tail 规则和同一波形；
Bayes 每帧依次做 Pilot 后验与 Data 后验（Data 只用模型 soft 判决），
后验用于下一帧。真实 CIR 只用于事后 NMSE/oracle，不进入估计。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pilot_transfer import _load_branch
from training.bayesian_cir import acquisition_prior, add_process_noise, data_posterior, pilot_posterior
from training.pilot_transfer import context_logits, make_context, make_episode, tail_from_logits


def _nmse(estimate: torch.Tensor, truth: torch.Tensor) -> float:
    est = estimate.reshape(-1).to(torch.complex128)
    tru = truth.reshape(-1).to(torch.complex128)
    scale = torch.vdot(est, tru) / torch.vdot(est, est).real.clamp_min(1e-12)
    residual = tru - scale * est
    return float((residual.abs().square().sum() / tru.abs().square().sum().clamp_min(1e-12)).real)


def _ber(logits, frame):
    mask = frame.data_mask.bool()
    return float(((logits[mask] >= 0) != frame.bits[mask].bool()).float().mean())


def _run(model, config, seed, snr, frames, process_noise, ridge, device):
    env, cir, cfo, tail, received, acquisition = make_episode(
        config, model, seed=seed, snr=snr, frames=frames,
        state_split="offline_train", device=device, return_acquisition=True,
    )
    max_delay = int(config["max_delay"])
    noise_variance = 10.0 ** (-float(snr) / 10.0)
    prior_mean, prior_cov = acquisition_prior(acquisition, cfo, max_delay, noise_variance, ridge=ridge)
    frozen_cir = prior_mean.clone()
    bayes_cir = prior_mean.clone()
    bayes_cov = prior_cov.clone()
    frozen_tail = tail.clone()
    bayes_tail = tail.clone()
    rows = []
    for frame in received:
        frozen_context = make_context(frame, frozen_cir, cfo, frozen_tail, snr)
        with torch.no_grad():
            frozen_logits = context_logits(model, frozen_context, None, visible="data")
            frozen_ber = _ber(frozen_logits, frame)
        frozen_tail = tail_from_logits(frozen_logits, frozen_tail, 1.0)

        bayes_context = make_context(frame, bayes_cir, cfo, bayes_tail, snr)
        with torch.no_grad():
            bayes_logits = context_logits(model, bayes_context, None, visible="data")
            bayes_ber = _ber(bayes_logits, frame)
        bayes_tail = tail_from_logits(bayes_logits, bayes_tail, 1.0)

        truth = frame.true_cir.to(device=device, dtype=torch.complex64).reshape(-1)
        latent = bayes_context.condition.latent_residual
        phase0 = float(latent[0, 0].detach().cpu()) if latent is not None else 0.0
        cfo_est = float(latent[0, 1].detach().cpu()) if latent is not None else float(cfo)
        rows.append({
            "seed": int(seed), "snr_db": int(snr), "frame_index": int(frame.frame_index),
            "frozen_ber": frozen_ber, "bayes_ber": bayes_ber,
            "frozen_nmse": _nmse(frozen_cir, truth), "bayes_nmse": _nmse(bayes_cir, truth),
        })
        bayes_cir, bayes_cov = pilot_posterior(
            frame, bayes_cir, bayes_cov, max_delay, noise_variance, phase0, cfo_est,
        )
        bayes_cir, bayes_cov = data_posterior(
            frame, bayes_logits, bayes_cir, bayes_cov, max_delay, noise_variance, phase0, cfo_est,
        )
        bayes_cov = add_process_noise(bayes_cov, process_noise)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[95001, 95002])
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--process-noise", type=float, default=1e-4)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model, _ = _load_branch(Path(args.checkpoint), config, device)
    model.eval()

    result = {"config": str(Path(args.config)), "checkpoint": str(Path(args.checkpoint)),
              "seeds": args.seeds, "frames": args.frames, "process_noise": args.process_noise,
              "ridge": args.ridge, "per_snr": {}, "rows": []}
    for snr in config["main_snrs"]:
        rows = [row for seed in args.seeds
                for row in _run(model, config, seed, snr, args.frames, args.process_noise, args.ridge, device)]
        result["rows"].extend(rows)
        n = len(rows)
        frozen = sum(r["frozen_ber"] for r in rows) / n
        bayes = sum(r["bayes_ber"] for r in rows) / n
        result["per_snr"][str(snr)] = {
            "frozen_ber": frozen, "bayes_ber": bayes, "gain": frozen - bayes,
            "frozen_nmse": sum(r["frozen_nmse"] for r in rows) / n,
            "bayes_nmse": sum(r["bayes_nmse"] for r in rows) / n, "frames": n,
        }
        print(f"snr={snr:2d} frozen={frozen:.6f} bayes={bayes:.6f} gain={frozen-bayes:+.6f} "
              f"nmse frozen={result['per_snr'][str(snr)]['frozen_nmse']:.4f} "
              f"bayes={result['per_snr'][str(snr)]['bayes_nmse']:.4f}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
