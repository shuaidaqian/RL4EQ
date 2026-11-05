# -*- coding: utf-8 -*-
"""只读估计非线性Pilot上下文在固定半径内的Data可达上界。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.detection_subspace import DetectionSubspace
from evaluation.subspace_development import read_manifest, write_json
from training.subspace_adaptation import project


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze(manifest_path: Path, feature_cache: Path, checkpoint: Path, output_path: Path,
            device="cpu", steps=100, learning_rate=0.03):
    if output_path.exists():
        raise ValueError("输出文件已存在，禁止覆盖。")
    spec = read_manifest(manifest_path)
    sessions = torch.load(feature_cache, map_location=device, weights_only=False)
    allowed = set(spec["data_roles"]["evaluation"]["seeds"])
    if not sessions or any(int(session["seed"]) not in allowed for session in sessions):
        raise ValueError("只允许清单隔离的evaluation开发缓存。")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = DetectionSubspace(int(payload["dimension"]), architecture=payload["architecture"],
        learned_gradient_transport=payload.get("learned_gradient_transport", False),
        learned_context_update=payload.get("learned_context_update", False)).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    radius = float(spec["update"]["total_radius"])
    records = []
    for session in sessions:
        for frame in session["frames"]:
            observation = frame["observation"]
            mask = frame["offline"]["data_mask"].to(device)
            labels = frame["offline"]["bits"].to(device)[mask].float()
            origin = model.coefficients.detach().clone()
            with torch.no_grad():
                before = model.logits(observation, origin)[mask]
            coefficients = torch.nn.Parameter(origin.clone())
            optimizer = torch.optim.Adam([coefficients], lr=learning_rate)
            best_coefficients = origin.clone()
            best_loss = float(F.binary_cross_entropy_with_logits(before, labels))
            for _ in range(steps):
                optimizer.zero_grad(set_to_none=True)
                logits = model.logits(observation, coefficients)[mask]
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    projected = project(coefficients, origin, radius)
                    coefficients.copy_(projected)
                    trial = model.logits(observation, coefficients)[mask]
                    trial_loss = float(F.binary_cross_entropy_with_logits(trial, labels))
                    if trial_loss < best_loss:
                        best_loss = trial_loss
                        best_coefficients = coefficients.detach().clone()
            with torch.no_grad():
                oracle = model.logits(observation, best_coefficients)[mask]
                before_errors = int(((before >= 0) != labels.bool()).sum())
                oracle_errors = int(((oracle >= 0) != labels.bool()).sum())
            records.append(dict(seed=int(session["seed"]), snr_db=int(session["snr_db"]),
                frame=int(frame["identity"]["frame"]), symbols=int(labels.numel()),
                baseline_errors=before_errors, oracle_errors=oracle_errors,
                baseline_bce=float(F.binary_cross_entropy_with_logits(before, labels)),
                oracle_bce=best_loss, oracle_delta_norm=float((best_coefficients-origin).norm()),
                data_used_for_oracle_only=True, oracle_used_for_online_control=False))

    grouped = []
    for snr in sorted({row["snr_db"] for row in records}):
        active = [row for row in records if row["snr_db"] == snr]
        symbols = sum(row["symbols"] for row in active)
        baseline = sum(row["baseline_errors"] for row in active)
        oracle = sum(row["oracle_errors"] for row in active)
        grouped.append(dict(snr_db=snr, frames=len(active), symbols=symbols,
            baseline_ber=baseline/symbols, oracle_ber=oracle/symbols,
            oracle_ber_gain=(baseline-oracle)/symbols,
            frames_with_fewer_errors=sum(row["oracle_errors"] < row["baseline_errors"] for row in active),
            frames_with_more_errors=sum(row["oracle_errors"] > row["baseline_errors"] for row in active),
            mean_baseline_bce=sum(row["baseline_bce"] for row in active)/len(active),
            mean_oracle_bce=sum(row["oracle_bce"] for row in active)/len(active)))
    result = dict(role="development_read_only_nonlinear_data_oracle",
        core_confirmed=False, confirmation_ready=False, online_gain_claimed=False,
        online_control_used_data_labels=False,
        interpretation_limit="逐帧用Data BCE优化非线性上下文的乐观可达性上界；Data只作离线诊断，不是合法在线算法，也不证明Pilot可达。",
        dimension=int(payload["dimension"]), architecture=payload["architecture"], radius=radius,
        optimizer=dict(name="projected_adam", steps=steps, learning_rate=learning_rate),
        manifest_sha256=sha256(manifest_path),
        inputs={name: {"path": str(path), "sha256": sha256(path)} for name, path in {
            "manifest": manifest_path, "feature_cache": feature_cache, "checkpoint": checkpoint}.items()},
        records=len(records), by_snr=grouped, frames=records)
    write_json(output_path, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    args = parser.parse_args()
    result = analyze(Path(args.manifest), Path(args.feature_cache), Path(args.checkpoint),
        Path(args.output), args.device, args.steps, args.learning_rate)
    print(json.dumps(result["by_snr"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
