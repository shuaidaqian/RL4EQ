# -*- coding: utf-8 -*-
"""只读诊断：离线学习的 CIR 去偏映射能否把 data LS 拉向真实 CIR。

每帧特征（在线合法）：
- acquisition CIR（Pilot 估计）；
- 当前帧 Data soft 判决的加权 LS（不读 Data 标签）；
- Data 置信度统计、LS 残差、Pilot 相位特征。

训练目标：真实 CIR（仅离线诊断），按 acquisition 相位对齐。
评估：held-out seed 的 CIR NMSE；不是 BER 结论。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pilot_transfer import _load_branch
from training.data_aided_cir import data_aided_cir_update
from training.pilot_transfer import context_logits, make_context, make_episode, next_pilot_tail


def _align_phase(reference: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ref = reference.reshape(-1).to(torch.complex128)
    tgt = target.reshape(-1).to(torch.complex128)
    phase = torch.angle(torch.vdot(ref, tgt))
    return (tgt * torch.exp(-1j * phase)).to(torch.complex64)


def _nmse(estimate: torch.Tensor, truth: torch.Tensor) -> float:
    est = estimate.reshape(-1).to(torch.complex128)
    tru = truth.reshape(-1).to(torch.complex128)
    scale = torch.vdot(est, tru) / torch.vdot(est, est).real.clamp_min(1e-12)
    residual = tru - scale * est
    return float((residual.abs().square().sum() / tru.abs().square().sum().clamp_min(1e-12)).real)


def _complex_to_real(value: torch.Tensor) -> np.ndarray:
    value = value.reshape(-1).cpu().numpy()
    return np.concatenate([value.real, value.imag])


def _real_to_complex(value: np.ndarray) -> torch.Tensor:
    count = value.size // 2
    return torch.from_numpy(value[:count] + 1j * value[count:]).to(torch.complex128)


def _vector_nmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    est = _real_to_complex(estimate)
    tru = _real_to_complex(truth)
    scale = torch.vdot(est, tru) / torch.vdot(est, est).real.clamp_min(1e-12)
    residual = tru - scale * est
    return float((residual.abs().square().sum() / tru.abs().square().sum().clamp_min(1e-12)).real)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=list(range(94001, 94009)))
    parser.add_argument("--test-seeds", nargs="+", type=int, default=[94009, 94010])
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model, _ = _load_branch(Path(args.checkpoint), config, device)
    model.eval()
    max_delay = int(config["max_delay"])

    records = []
    for seed in args.train_seeds + args.test_seeds:
        for snr in config["main_snrs"]:
            env, cir, cfo, tail, received = make_episode(
                config, model, seed=seed, snr=snr, frames=args.frames,
                state_split="offline_train", device=device,
            )
            for frame in received:
                context = make_context(frame, cir, cfo, tail, snr)
                with torch.no_grad():
                    logits = context_logits(model, context, None, visible="data")
                latent = context.condition.latent_residual
                phase0 = float(latent[0, 0].detach().cpu()) if latent is not None else 0.0
                cfo_est = float(latent[0, 1].detach().cpu()) if latent is not None else 0.0
                soft = torch.tanh(logits / 2.0)
                data_mask = frame.data_mask.bool()
                ls = data_aided_cir_update(
                    frame, logits, max_delay, cir, alpha=1.0, confidence_threshold=0.5,
                    ridge=1e-3, phase0=phase0, cfo=cfo_est,
                )
                truth = _align_phase(cir, frame.true_cir.to(device=device, dtype=torch.complex64))
                confidence = soft[data_mask].abs()
                scalar = np.array([
                    float(confidence.mean()), float((confidence > 0.5).float().mean()),
                    float((1.0 - confidence.square()).mean()), float(phase0), float(cfo_est),
                ], dtype=np.float32)
                records.append({
                    "seed": int(seed), "snr_db": int(snr), "frame_index": int(frame.frame_index),
                    "acq": _complex_to_real(cir), "ls": _complex_to_real(ls),
                    "truth": _complex_to_real(truth), "scalar": scalar,
                    "nmse_acq": _nmse(cir, truth), "nmse_ls": _nmse(ls, truth),
                })
                tail = next_pilot_tail(model, context, alpha=1.0, mode="dd_soft")

    train = [r for r in records if r["seed"] in args.train_seeds]
    test = [r for r in records if r["seed"] in args.test_seeds]
    X_train = np.stack([np.concatenate([r["acq"], r["ls"], r["scalar"]]) for r in train])
    Y_train = np.stack([r["truth"] for r in train])
    X_test = np.stack([np.concatenate([r["acq"], r["ls"], r["scalar"]]) for r in test])
    Y_test = np.stack([r["truth"] for r in test])
    mean = X_train.mean(axis=0, keepdims=True)
    scale = X_train.std(axis=0, keepdims=True).clip(min=1e-6)
    Xn = (X_train - mean) / scale
    Xt = (X_test - mean) / scale
    gram = Xn.T @ Xn + args.ridge * np.eye(Xn.shape[1])
    weight = np.linalg.solve(gram, Xn.T @ Y_train)
    pred = Xt @ weight

    result = {"train_seeds": args.train_seeds, "test_seeds": args.test_seeds,
              "frames": args.frames, "ridge": args.ridge, "per_snr": {}}
    for snr in config["main_snrs"]:
        idx = [i for i, r in enumerate(test) if r["snr_db"] == snr]
        acq = np.mean([test[i]["nmse_acq"] for i in idx])
        ls = np.mean([test[i]["nmse_ls"] for i in idx])
        learned = np.mean([_vector_nmse(pred[i], test[i]["truth"]) for i in idx])
        # 简单 blend：在训练集上选择 per-SNR 权重
        train_idx = [i for i, r in enumerate(train) if r["snr_db"] == snr]
        best_w, best_nmse = 0.0, 1e9
        for w in np.linspace(0.0, 1.0, 21):
            vals = [_vector_nmse((1.0 - w) * train[i]["acq"] + w * train[i]["ls"], train[i]["truth"])
                    for i in train_idx]
            if np.mean(vals) < best_nmse:
                best_nmse, best_w = float(np.mean(vals)), float(w)
        blend_test = np.mean([_vector_nmse((1.0 - best_w) * test[i]["acq"] + best_w * test[i]["ls"], test[i]["truth"])
                              for i in idx])
        result["per_snr"][str(snr)] = {"nmse_acq": acq, "nmse_ls": ls, "nmse_learned": learned,
                                        "blend_weight": best_w, "nmse_blend": blend_test}
        print(f"snr={snr:2d} NMSE acq={acq:.4f} ls={ls:.4f} learned={learned:.4f} "
              f"blend_w={best_w:.2f} blend={blend_test:.4f}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
