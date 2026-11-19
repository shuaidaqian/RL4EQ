# -*- coding: utf-8 -*-
"""持续均衡四臂开发筛查；接收链路与事后评分分离。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from env.comm_env import CommunicationEnvironment
from env.experiment_config import build_comm_env_config
from evaluation.pilot_transfer import _load_branch
from evaluation.seed_registry import reserve_seeds
from training.continual_receiver import ARMS, ContinualReceiver, estimate_acquisition, receiver_frame
from training.pilot_transfer import frame_to_device


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summarize(rows):
    summary = {}
    for snr in sorted({r["snr_db"] for r in rows}):
        methods = {}
        for arm in sorted({r["arm"] for r in rows}):
            current = [r for r in rows if r["snr_db"] == snr and r["arm"] == arm]
            sessions = sorted({r["seed"] for r in current})
            if not current:
                continue
            methods[arm] = {
                "ber": sum(r["errors"] for r in current) / sum(r["bits"] for r in current),
                "mean_elapsed_ms": sum(r["elapsed_ms"] for r in current) / len(current),
                "session_ber": {str(seed): sum(r["errors"] for r in current if r["seed"] == seed)
                                / sum(r["bits"] for r in current if r["seed"] == seed)
                                for seed in sessions},
                "windows": [],
            }
            length = max(r["frame_index"] for r in current) + 1
            edges = sorted(set(x for x in [0, 1, 4, 8, 16, 32, 64, 128, length] if x <= length))
            for left, right in zip(edges, edges[1:]):
                window = [r for r in current if left <= r["frame_index"] < right]
                methods[arm]["windows"].append({
                    "start": left, "stop": right,
                    "ber": sum(r["errors"] for r in window) / sum(r["bits"] for r in window),
                })
        summary[str(snr)] = methods
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eme_long_memory_v2_online_dd_tail_v1.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--snrs", nargs="+", type=int, default=[0, 5, 10, 15])
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--state-split", default="offline_train")
    parser.add_argument("--process-noise", type=float, default=1e-4)
    parser.add_argument("--noise-source", choices=["estimated", "oracle_reference"], default="estimated")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.frames < 1 or len(set(args.arms)) != len(args.arms) or len(set(args.snrs)) != len(args.snrs):
        parser.error("帧数必须为正，接收臂及 SNR 不能重复。")
    output = Path(args.output)
    if output.exists():
        raise ValueError("输出已存在，禁止覆盖或隐式重跑。")
    torch.set_num_threads(1)
    device = torch.device(args.device)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model, _ = _load_branch(Path(args.checkpoint), config, device)
    protocol = vars(args).copy()
    protocol.update({"config_sha256": fingerprint(args.config), "checkpoint_sha256": fingerprint(args.checkpoint),
                     "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                     "receiver_sha256": fingerprint(ROOT / "training/continual_receiver.py"),
                     "torch_version": torch.__version__, "timing": "current_pilot_then_decode",
                     "status": "development", "statistical_unit": "session"})
    reserve_seeds(ROOT, experiment=args.experiment, seeds=args.seeds,
                  purpose="development", details=protocol)
    result = {"protocol": protocol, "acquisitions": [], "rows": [], "status": "running"}
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        result["summary"] = summarize(result["rows"])
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)

    save()
    for snr in args.snrs:
        for seed in args.seeds:
            env_config = build_comm_env_config(config, level="B", snr_db=snr, seed=seed,
                max_delay=int(config["max_delay"]), total_pilot=int(config["pilot_total"]),
                reward_pilot_total=int(config["reward_pilot_total"]), pilot_layout="prefix",
                state_split=args.state_split)
            env = CommunicationEnvironment(env_config)
            start = env.reset_episode()
            acquisition = estimate_acquisition(
                start.acquisition.tx_symbols, start.acquisition.rx_symbols, int(config["max_delay"]),
                cfo_limit=float(config.get("acquisition_cfo_limit", 0.004)),
                noise_reference=(10 ** (-snr / 10) if args.noise_source == "oracle_reference" else None),
            )
            # 环境真噪声仅用于保存事后诊断，绝不传入估计噪声接收臂。
            result["acquisitions"].append({"seed": seed, "snr_db": snr,
                "noise_estimate": acquisition.noise_variance, "noise_truth_diagnostic": 10 ** (-snr / 10),
                "cfo_estimate": acquisition.cfo, "cir_norm": float(acquisition.mean.norm()),
                "rank": acquisition.rank, "observations": acquisition.observations,
                "noise_source": acquisition.noise_source})
            from dataclasses import replace
            acquisition = replace(acquisition, mean=acquisition.mean.to(device),
                                  covariance=acquisition.covariance.to(device))
            receivers = {arm: ContinualReceiver(acquisition, start.initial_soft_tail.to(device),
                         arm=arm, process_noise=args.process_noise) for arm in args.arms}
            for _ in range(args.frames):
                original = frame_to_device(env.next_frame(), device)
                visible = receiver_frame(original)
                rx_hash = hashlib.sha256(original.rx_symbols.cpu().numpy().tobytes()).hexdigest()
                for arm, receiver in receivers.items():
                    synchronize(device)
                    began = time.perf_counter()
                    logits, diagnostics = receiver.step(model, visible)
                    synchronize(device)
                    elapsed = (time.perf_counter() - began) * 1000
                    # 以下标签/真实状态访问全部在接收计算结束后，只作评分。
                    mask = original.data_mask
                    errors = int(((logits[mask] >= 0) != original.bits[mask].bool()).sum())
                    estimate = diagnostics.pop("decode_cir")
                    truth = original.true_cir.flatten()
                    scale = torch.vdot(estimate, truth) / estimate.abs().square().sum().clamp_min(1e-12)
                    nmse = float((truth - scale * estimate).abs().square().sum() / truth.abs().square().sum())
                    result["rows"].append({"seed": seed, "snr_db": snr, "frame_index": original.frame_index,
                        "arm": arm, "errors": errors, "bits": int(mask.sum()), "ber": errors / int(mask.sum()),
                        "elapsed_ms": elapsed, "aligned_nmse_diagnostic": nmse,
                        "noise_variance_used": acquisition.noise_variance, "rx_sha256": rx_hash, **diagnostics})
                if original.frame_index % 8 == 0:
                    save()
            save()
            scores = result["summary"][str(snr)]
            print(f"SNR={snr} seed={seed} " + " ".join(
                f"{arm}={scores[arm]['session_ber'][str(seed)]:.6f}" for arm in args.arms), flush=True)
    result["status"] = "completed"
    save()


if __name__ == "__main__":
    main()
