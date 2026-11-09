# -*- coding: utf-8 -*-
"""只读诊断：稀疏贝叶斯学习（SBL）与当前 Bayes CIR 的 CIR NMSE / BER / 成本对照。

四臂共享同一 acquisition 先验、同一 dd_soft tail 规则和同一波形：
- ls：Pilot-only 最小二乘（无先验）；
- lmmse：Pilot-only，acquisition 先验；
- bayes：Pilot + Data soft 后验（当前实现，带过程噪声 Q）；
- sbl：Pilot + Data soft，稀疏先验 + EM 迭代（每帧从当前 CIR 能量初始化）。

Data 标签只用于事后 BER；真实 CIR 只用于事后 NMSE，均不进入估计。
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
from training.bayesian_cir import (
    _phase_correct,
    _rows,
    acquisition_prior,
    add_process_noise,
    data_posterior,
    pilot_posterior,
    posterior_update,
)
from training.pilot_transfer import context_logits, make_context, make_episode, tail_from_logits


def _nmse(estimate: torch.Tensor, truth: torch.Tensor) -> float:
    est = estimate.reshape(-1).to(torch.complex128)
    tru = truth.reshape(-1).to(torch.complex128)
    scale = torch.vdot(est, tru) / torch.vdot(est, est).real.clamp_min(1e-12)
    residual = tru - scale * est
    return float((residual.abs().square().sum() / tru.abs().square().sum().clamp_min(1e-12)).real)


def _ber(logits: torch.Tensor, frame) -> float:
    mask = frame.data_mask.bool()
    return float(((logits[mask] >= 0) != frame.bits[mask].bool()).float().mean())


def _gaussian_posterior(prior_mean, prior_cov, design, observation, row_noise):
    precision = torch.linalg.inv(prior_cov)
    weights = row_noise.reshape(-1).clamp_min(1e-8).reciprocal()
    information = design.conj().transpose(0, 1) @ (weights.unsqueeze(1) * design)
    posterior_precision = precision + information
    rhs = precision @ prior_mean + design.conj().transpose(0, 1) @ (weights * observation)
    mean = torch.linalg.solve(posterior_precision, rhs)
    covariance = torch.linalg.inv(posterior_precision)
    return mean, 0.5 * (covariance + covariance.conj().transpose(0, 1))


def _ls_update(design, observation, ridge: float):
    dimension = design.shape[1]
    identity = torch.eye(dimension, dtype=design.dtype, device=design.device)
    gram = design.conj().transpose(0, 1) @ design + float(ridge) * identity
    return torch.linalg.solve(gram, design.conj().transpose(0, 1) @ observation)


def _sbl_update(design, observation, row_noise, prior_mean, gamma, iterations: int):
    """带先验均值的稀疏残差 EM：E 步高斯后验，M 步更新逐抽头方差。"""

    dimension = design.shape[1]
    dtype = design.dtype
    device = design.device
    gamma = gamma.to(device=device, dtype=torch.float32).clamp_min(1e-8)
    prior_mean = prior_mean.to(dtype=dtype, device=device)
    weights = row_noise.reshape(-1).clamp_min(1e-8).reciprocal().to(dtype)
    information = design.conj().transpose(0, 1) @ (weights.unsqueeze(1) * design)
    data_rhs = design.conj().transpose(0, 1) @ (weights * observation)
    mean = prior_mean.clone()
    covariance = torch.diag(gamma.to(dtype))
    for _ in range(int(iterations)):
        precision = torch.diag(gamma.reciprocal().to(dtype))
        posterior_precision = precision + information
        rhs = precision @ prior_mean + data_rhs
        mean = torch.linalg.solve(posterior_precision, rhs)
        covariance = torch.linalg.inv(posterior_precision)
        gamma = ((mean - prior_mean).abs().square()
                 + torch.diagonal(covariance).real).clamp_min(1e-8)
    return mean, covariance, gamma


def _pilot_observations(frame, max_delay, noise_variance, phase0, cfo):
    device = frame.rx_symbols.device
    mask = frame.adapt_mask.to(device=device).bool()
    positions = torch.nonzero(mask, as_tuple=False).flatten()
    positions = positions[positions >= max_delay]
    symbols = frame.tx_symbols.to(device=device).to(torch.complex64)
    corrected = _phase_correct(
        frame.rx_symbols.to(device=device).to(torch.complex64), positions, phase0, cfo,
    )
    design = _rows(symbols, positions, max_delay)
    row_noise = torch.full((positions.numel(),), float(noise_variance), device=device)
    return design, corrected, row_noise


def _data_observations(frame, logits, prior_mean, max_delay, noise_variance, phase0, cfo):
    device = frame.rx_symbols.device
    soft = torch.tanh(logits / 2.0)
    variance = (1.0 - soft.square()).clamp_min(0.0)
    mask = frame.data_mask.to(device=device).bool()
    positions = torch.nonzero(mask, as_tuple=False).flatten()
    positions = positions[positions >= max_delay]
    corrected = _phase_correct(
        frame.rx_symbols.to(device=device).to(torch.complex64), positions, phase0, cfo,
    )
    real_design = _rows(soft, positions, max_delay)
    design = torch.complex(real_design, torch.zeros_like(real_design))
    cir_energy = prior_mean.abs().square()
    propagated = torch.zeros_like(variance[positions])
    for delay in range(max_delay + 1):
        propagated = propagated + cir_energy[delay].to(propagated.dtype) * variance[positions - delay]
    row_noise = float(noise_variance) + propagated
    return design, corrected, row_noise


def _estimate_phase(context, default_cfo):
    latent = context.condition.latent_residual
    if latent is None:
        return 0.0, float(default_cfo)
    return float(latent[0, 0].detach().cpu()), float(latent[0, 1].detach().cpu())


def _run(model, config, seed, snr, frames, process_noise, ridge, sbl_iterations, device):
    _, _, cfo, tail, received, acquisition = make_episode(
        config, model, seed=seed, snr=snr, frames=frames,
        state_split="offline_train", device=device, return_acquisition=True,
    )
    max_delay = int(config["max_delay"])
    noise_variance = 10.0 ** (-float(snr) / 10.0)
    prior_mean, prior_cov = acquisition_prior(
        acquisition, cfo, max_delay, noise_variance, ridge=ridge,
    )
    arms = {
        name: {"cir": prior_mean.clone(), "tail": tail.clone()}
        for name in ("ls", "lmmse", "bayes", "sbl")
    }
    arms["bayes"]["cov"] = prior_cov.clone()
    arms["sbl"]["prior"] = prior_mean.clone()
    arms["sbl"]["gamma0"] = torch.diagonal(prior_cov).real.clone().clamp_min(1e-8)
    rows = []
    for frame in received:
        truth = frame.true_cir.to(device=device, dtype=torch.complex64).reshape(-1)
        for name, arm in arms.items():
            context = make_context(frame, arm["cir"], cfo, arm["tail"], snr)
            phase0, cfo_est = _estimate_phase(context, cfo)
            with torch.no_grad():
                logits = context_logits(model, context, None, visible="data")
                ber = _ber(logits, frame)
            arm["tail"] = tail_from_logits(logits, arm["tail"], 1.0)
            rows.append({
                "seed": int(seed), "snr_db": int(snr), "frame_index": int(frame.frame_index),
                "arm": name, "ber": ber, "nmse": _nmse(arm["cir"], truth),
            })

            started = time.perf_counter()
            pilot_design, pilot_obs, pilot_noise = _pilot_observations(
                frame, max_delay, noise_variance, phase0, cfo_est,
            )
            if name == "ls":
                arm["cir"] = _ls_update(pilot_design, pilot_obs, ridge).to(torch.complex64)
            elif name == "lmmse":
                estimate, _ = posterior_update(
                    prior_mean, prior_cov, pilot_design, pilot_obs, pilot_noise,
                )
                arm["cir"] = estimate
            elif name == "bayes":
                arm["cir"], arm["cov"] = pilot_posterior(
                    frame, arm["cir"], arm["cov"], max_delay, noise_variance, phase0, cfo_est,
                )
                arm["cir"], arm["cov"] = data_posterior(
                    frame, logits, arm["cir"], arm["cov"], max_delay, noise_variance, phase0, cfo_est,
                )
                arm["cov"] = add_process_noise(arm["cov"], process_noise)
            else:
                data_design, data_obs, data_noise = _data_observations(
                    frame, logits, arm["cir"], max_delay, noise_variance, phase0, cfo_est,
                )
                design = torch.cat((pilot_design, data_design), dim=0)
                observation = torch.cat((pilot_obs, data_obs), dim=0)
                row_noise = torch.cat((pilot_noise, data_noise), dim=0)
                estimate, _, _ = _sbl_update(
                    design, observation, row_noise, arm["prior"], arm["gamma0"], sbl_iterations,
                )
                arm["cir"] = (estimate / estimate.norm().clamp_min(1e-12)).to(torch.complex64)
            elapsed_ms = (time.perf_counter() - started) * 1e3
            rows[-1]["update_ms"] = elapsed_ms
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[110001, 110002, 110003])
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--process-noise", type=float, default=1e-4)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--sbl-iterations", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if (config.get("main_level") != "B" or config.get("pilot_layout") != "prefix"
            or config.get("max_delay") != 116 or config.get("pilot_total") != 256
            or config.get("reward_pilot_total") != 32
            or config.get("main_snrs") != [0, 5, 10, 15]):
        raise ValueError("诊断只允许固定 Level B、116 符号长回波、前缀 Pilot=256 和四个指定 SNR。")
    output = Path(args.output)
    if output.exists():
        raise ValueError("输出文件已存在，请使用新路径。")

    device = torch.device(args.device)
    model, _ = _load_branch(Path(args.checkpoint), config, device)
    model.eval()
    result = {
        "schema_version": "sbl-vs-bayes-diagnostic-v1",
        "diagnostic_only": True,
        "online_updates_executed": False,
        "data_labels_used_for": "事后 BER 评分；不进入估计",
        "true_cir_used_for": "事后 NMSE 评分；不进入估计",
        "config": str(Path(args.config)),
        "checkpoint": str(Path(args.checkpoint)),
        "seeds": args.seeds,
        "frames": args.frames,
        "process_noise": args.process_noise,
        "ridge": args.ridge,
        "sbl_iterations": args.sbl_iterations,
        "per_snr": {},
        "rows": [],
    }
    for snr in config["main_snrs"]:
        rows = [
            row for seed in args.seeds
            for row in _run(
                model, config, seed, snr, args.frames,
                args.process_noise, args.ridge, args.sbl_iterations, device,
            )
        ]
        result["rows"].extend(rows)
        summary = {}
        for arm in ("ls", "lmmse", "bayes", "sbl"):
            selected = [row for row in rows if row["arm"] == arm]
            count = len(selected)
            summary[arm] = {
                "ber": sum(row["ber"] for row in selected) / count,
                "nmse": sum(row["nmse"] for row in selected) / count,
                "update_ms": sum(row["update_ms"] for row in selected) / count,
                "frames": count,
            }
        result["per_snr"][str(snr)] = summary
        print(
            f"snr={snr:2d} " + " ".join(
                f"{arm}: ber={summary[arm]['ber']:.6f} nmse={summary[arm]['nmse']:.4f} "
                f"ms={summary[arm]['update_ms']:.2f}"
                for arm in ("ls", "lmmse", "bayes", "sbl")
            ),
            flush=True,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"written {output}", flush=True)


if __name__ == "__main__":
    main()
