# -*- coding: utf-8 -*-
"""只读诊断：0 dB 下 Data 行噪声 R_t 的 oracle 上界。

比较三种 Data 后验行噪声：
- frozen：不更新 CIR；
- soft：R_t = sigma^2 + sum_l |h_l|^2 Var_{t-l}（当前启发式）；
- oracle：R_t = sigma^2 + sum_l |h_l|^2 Err_{t-l}，Err 为真实符号错误指示。

真实符号错误只用于隔离 oracle；不进入任何在线估计或部署。
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
from training.bayesian_cir import (
    _phase_correct, _rows, acquisition_prior, add_process_noise, pilot_posterior, posterior_update,
)
from training.pilot_transfer import context_logits, make_context, make_episode, next_pilot_tail


def _nmse(estimate, truth):
    est = estimate.reshape(-1).to(torch.complex128)
    tru = truth.reshape(-1).to(torch.complex128)
    scale = torch.vdot(est, tru) / torch.vdot(est, est).real.clamp_min(1e-12)
    residual = tru - scale * est
    return float((residual.abs().square().sum() / tru.abs().square().sum().clamp_min(1e-12)).real)


def _ber(logits, frame):
    mask = frame.data_mask.bool()
    return float(((logits[mask] >= 0) != frame.bits[mask].bool()).float().mean())


def _data_posterior(frame, logits, prior_mean, prior_cov, max_delay, noise_variance, phase0, cfo, mode):
    device = prior_mean.device
    soft = torch.tanh(logits / 2.0)
    variance = (1.0 - soft.square()).clamp_min(0.0)
    mask = frame.data_mask.to(device=device).bool()
    positions = torch.nonzero(mask, as_tuple=False).flatten()
    positions = positions[positions >= max_delay]
    if positions.numel() < 2:
        return prior_mean, prior_cov
    corrected = _phase_correct(frame.rx_symbols.to(device=device).to(torch.complex64), positions, phase0, cfo)
    real_design = _rows(soft, positions, max_delay)
    design = torch.complex(real_design, torch.zeros_like(real_design))
    energy = prior_mean.abs().square()
    if mode == "soft":
        weights = variance
    elif mode == "oracle":
        weights = ((logits >= 0) != frame.bits.to(device).bool()).to(torch.float32)
    else:
        raise ValueError(mode)
    propagated = torch.zeros_like(weights[positions])
    for delay in range(max_delay + 1):
        propagated = propagated + energy[delay].to(propagated.dtype) * weights[positions - delay]
    row_noise = float(noise_variance) + propagated
    return posterior_update(prior_mean, prior_cov, design, corrected, row_noise)


def _run(model, config, seed, frames, mode, device):
    env, cir, cfo, tail, received, acquisition = make_episode(
        config, model, seed=seed, snr=0, frames=frames,
        state_split="offline_train", device=device, return_acquisition=True,
    )
    max_delay = int(config["max_delay"])
    noise_variance = 1.0
    bayes_cir, bayes_cov = acquisition_prior(acquisition, cfo, max_delay, noise_variance, ridge=1e-2)
    state = bayes_cir if mode != "frozen" else cir
    rows = []
    for frame in received:
        context = make_context(frame, state, cfo, tail, 0)
        with torch.no_grad():
            logits = context_logits(model, context, None, visible="data")
            ber = _ber(logits, frame)
        truth = frame.true_cir.to(device=device, dtype=torch.complex64).reshape(-1)
        rows.append({"seed": int(seed), "mode": mode, "frame_index": int(frame.frame_index),
                     "ber": ber, "nmse": _nmse(state, truth)})
        tail = next_pilot_tail(model, context, alpha=1.0, mode="dd_soft")
        if mode != "frozen":
            latent = context.condition.latent_residual
            phase0 = float(latent[0, 0].detach().cpu()) if latent is not None else 0.0
            cfo_est = float(latent[0, 1].detach().cpu()) if latent is not None else float(cfo)
            bayes_cir, bayes_cov = pilot_posterior(
                frame, bayes_cir, bayes_cov, max_delay, noise_variance, phase0, cfo_est,
            )
            bayes_cir, bayes_cov = _data_posterior(
                frame, logits, bayes_cir, bayes_cov, max_delay, noise_variance, phase0, cfo_est, mode,
            )
            bayes_cov = add_process_noise(bayes_cov, 1e-4)
            state = bayes_cir
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[110101, 110102, 110103])
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model, _ = _load_branch(Path(args.checkpoint), config, device)
    model.eval()

    result = {"config": str(Path(args.config)), "checkpoint": str(Path(args.checkpoint)),
              "seeds": args.seeds, "frames": args.frames, "modes": {}, "rows": []}
    for mode in ["frozen", "soft", "oracle"]:
        rows = [row for seed in args.seeds for row in _run(model, config, seed, args.frames, mode, device)]
        result["rows"].extend(rows)
        result["modes"][mode] = {
            "ber": sum(r["ber"] for r in rows) / len(rows),
            "nmse": sum(r["nmse"] for r in rows) / len(rows),
        }
        print(f"{mode:7s} ber={result['modes'][mode]['ber']:.6f} nmse={result['modes'][mode]['nmse']:.4f}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
