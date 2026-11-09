# -*- coding: utf-8 -*-
"""只读诊断：Pilot 半分割交叉验证的自适应过程噪声 Q。

每帧把有效 Adapt Pilot 行分成前一半（更新）与后一半（评分），
在候选 Q 网格中选择后半 Pilot 预测误差最小者；该 Q 只用于下一帧的
先验协方差。全程只用已知 Pilot 符号与接收波形，不读 Data 标签或真实 SNR。

三臂对照：fixed_1e-4（当前）、fixed_0（参考上界）、adaptive（交叉验证选 Q）。
Data 标签只事后 BER，真实 CIR 只事后 NMSE。
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
from scripts.diagnose_sbl_vs_bayes import (
    _ber,
    _data_observations,
    _estimate_phase,
    _gaussian_posterior,
    _nmse,
    _pilot_observations,
)
from training.bayesian_cir import acquisition_prior
from training.pilot_transfer import context_logits, make_context, make_episode, tail_from_logits


def _select_q(prior_mean, prior_cov, design, observation, row_noise, candidates):
    """前半 Pilot 更新、后半 Pilot 评分，返回预测误差最小的候选 Q。"""

    rows = design.shape[0]
    split = max(rows // 2, 1)
    train = slice(0, split)
    score = slice(split, rows)
    if split >= rows:
        return float(candidates[0])
    identity = torch.eye(
        design.shape[1], dtype=design.dtype, device=design.device,
    )
    best_q = float(candidates[0])
    best_error = float("inf")
    for q in candidates:
        covariance = prior_cov + float(q) * identity
        mean, _ = _gaussian_posterior(
            prior_mean, covariance, design[train], observation[train], row_noise[train],
        )
        prediction = design[score] @ mean
        error = float((observation[score] - prediction).abs().square().mean())
        if error < best_error:
            best_q, best_error = float(q), error
    return best_q


def _run(model, config, seed, snr, frames, ridge, candidates, device):
    _, _, cfo, tail, received, acquisition = make_episode(
        config, model, seed=seed, snr=snr, frames=frames,
        state_split="offline_train", device=device, return_acquisition=True,
    )
    max_delay = int(config["max_delay"])
    noise_variance = 10.0 ** (-float(snr) / 10.0)
    prior_mean, prior_cov = acquisition_prior(
        acquisition, cfo, max_delay, noise_variance, ridge=ridge,
    )
    identity = torch.eye(max_delay + 1, device=device, dtype=torch.complex64)
    arms = {
        name: {"cir": prior_mean.clone(), "cov": prior_cov.clone(), "tail": tail.clone()}
        for name in ("fixed_1e-4", "fixed_0", "adaptive")
    }
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
            nmse = _nmse(arm["cir"], truth)

            pilot_design, pilot_obs, pilot_noise = _pilot_observations(
                frame, max_delay, noise_variance, phase0, cfo_est,
            )
            if name == "adaptive":
                q = _select_q(
                    arm["cir"], arm["cov"], pilot_design, pilot_obs, pilot_noise, candidates,
                )
            elif name == "fixed_0":
                q = 0.0
            else:
                q = 1e-4
            cir, cov = _gaussian_posterior(
                arm["cir"], arm["cov"] + q * identity, pilot_design, pilot_obs, pilot_noise,
            )
            data_design, data_obs, data_noise = _data_observations(
                frame, logits, cir, max_delay, noise_variance, phase0, cfo_est,
            )
            cir, cov = _gaussian_posterior(cir, cov, data_design, data_obs, data_noise)
            norm = cir.norm().clamp_min(1e-12)
            arm["cir"] = (cir / norm).detach().to(torch.complex64)
            arm["cov"] = (cov / norm.square()).detach()
            rows.append({
                "seed": int(seed), "snr_db": int(snr), "frame_index": int(frame.frame_index),
                "arm": name, "ber": ber, "nmse": nmse, "q": float(q),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[110001, 110002, 110003])
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--q-candidates", nargs="+", type=float,
                        default=[0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
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
        "schema_version": "adaptive-q-online-diagnostic-v1",
        "diagnostic_only": True,
        "config": str(Path(args.config)),
        "checkpoint": str(Path(args.checkpoint)),
        "seeds": args.seeds,
        "frames": args.frames,
        "ridge": args.ridge,
        "q_candidates": args.q_candidates,
        "per_snr": {},
        "rows": [],
    }
    for snr in config["main_snrs"]:
        rows = [
            row for seed in args.seeds
            for row in _run(model, config, seed, snr, args.frames, args.ridge,
                            args.q_candidates, device)
        ]
        result["rows"].extend(rows)
        summary = {}
        for arm in ("fixed_1e-4", "fixed_0", "adaptive"):
            selected = [row for row in rows if row["arm"] == arm]
            count = len(selected)
            summary[arm] = {
                "ber": sum(row["ber"] for row in selected) / count,
                "nmse": sum(row["nmse"] for row in selected) / count,
                "q_mean": sum(row["q"] for row in selected) / count,
                "frames": count,
            }
        result["per_snr"][str(snr)] = summary
        print(
            f"snr={snr:2d} " + " ".join(
                f"{arm}: ber={summary[arm]['ber']:.6f} nmse={summary[arm]['nmse']:.4f} "
                f"q={summary[arm]['q_mean']:.2g}"
                for arm in ("fixed_1e-4", "fixed_0", "adaptive")
            ),
            flush=True,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"written {output}", flush=True)


if __name__ == "__main__":
    main()
