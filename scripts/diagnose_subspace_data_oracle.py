# -*- coding: utf-8 -*-
"""只读估计固定神经子空间在既定系数半径内的Data可达上界。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.detection_subspace import DetectionSubspace
from evaluation.subspace_development import read_manifest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def projected_data_oracle(basis, base_logits, labels, radius: float,
                          steps: int = 160, learning_rate: float = 0.03):
    """用Data BCE优化单帧系数；仅用于离线事后上界，不接入在线控制。"""
    origin = torch.zeros(basis.shape[1], device=basis.device, dtype=basis.dtype)
    coeff = torch.nn.Parameter(origin.clone())
    optimizer = torch.optim.Adam([coeff], lr=learning_rate)
    best = origin.clone()
    best_loss = float(F.binary_cross_entropy_with_logits(base_logits, labels))
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = base_logits + basis @ coeff
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            delta = coeff - origin
            norm = delta.norm()
            if norm > radius:
                coeff.copy_(origin + delta * (radius / norm))
            value = float(F.binary_cross_entropy_with_logits(base_logits + basis @ coeff, labels))
            if value < best_loss:
                best_loss = value
                best = coeff.detach().clone()
    return best, best_loss


def analyze(manifest_path: Path, feature_cache: Path, checkpoint: Path,
            output_path: Path, device: str = "cpu") -> dict:
    if output_path.exists():
        raise ValueError("输出文件已存在，禁止覆盖。")
    spec = read_manifest(manifest_path)
    sessions = torch.load(feature_cache, map_location=device, weights_only=False)
    allowed = set(spec["data_roles"]["evaluation"]["seeds"])
    if not sessions or any(int(item["seed"]) not in allowed for item in sessions):
        raise ValueError("oracle只允许清单隔离的evaluation开发会话。")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = DetectionSubspace(int(payload["dimension"]), architecture=payload["architecture"],
        learned_gradient_transport=payload.get("learned_gradient_transport", False),
        learned_context_update=payload.get("learned_context_update", False)).to(device)
    model.use_context_update = payload.get("use_context_update", payload.get("learned_context_update", False))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    radius = float(spec["update"]["total_radius"])
    rows = []
    with torch.enable_grad():
        for session in sessions:
            for frame in session["frames"]:
                observation = frame["observation"]
                data = frame["offline"]
                mask = data["data_mask"].to(device)
                labels = data["bits"].to(device)[mask].float()
                with torch.no_grad():
                    basis = model.basis(observation.to(device) if hasattr(observation, "to") else observation)
                    base = observation.base_logits.to(device)[mask]
                    basis = basis[mask].detach()
                oracle, oracle_bce = projected_data_oracle(basis, base.detach(), labels, radius)
                with torch.no_grad():
                    before_errors = int(((base > 0) != (labels > 0.5)).sum())
                    oracle_errors = int((((base + basis @ oracle) > 0) != (labels > 0.5)).sum())
                    rows.append({
                        "snr_db": int(session["snr_db"]), "seed": int(session["seed"]),
                        "frame": int(frame["identity"]["frame"]),
                        "symbols": int(labels.numel()), "baseline_errors": before_errors,
                        "oracle_errors": oracle_errors,
                        "baseline_bce": float(F.binary_cross_entropy_with_logits(base, labels)),
                        "oracle_bce": oracle_bce, "oracle_delta_norm": float(oracle.norm()),
                        "data_used_for_oracle_only": True,
                        "oracle_used_for_online_control": False,
                    })
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["snr_db"]].append(row)
    by_snr = []
    for snr, group in sorted(grouped.items()):
        symbols = sum(row["symbols"] for row in group)
        before = sum(row["baseline_errors"] for row in group)
        after = sum(row["oracle_errors"] for row in group)
        by_snr.append({"snr_db": snr, "seed_sessions": len({r["seed"] for r in group}),
            "frames": len(group), "symbols": symbols,
            "baseline_ber": before / symbols, "oracle_ber": after / symbols,
            "oracle_ber_gain": (before - after) / symbols,
            "baseline_bce": sum(r["baseline_bce"] for r in group) / len(group),
            "oracle_bce": sum(r["oracle_bce"] for r in group) / len(group),
            "oracle_flipped_errors": sum(max(0, r["baseline_errors"] - r["oracle_errors"]) for r in group),
            "oracle_added_errors": sum(max(0, r["oracle_errors"] - r["baseline_errors"]) for r in group)})
    result = {"schema_version": "subspace-data-oracle-v1",
        "role": "development_read_only_upper_bound_diagnostic", "core_confirmed": False,
        "confirmation_ready": False, "online_gain_claimed": False,
        "data_labels_used_for_online_control": False,
        "interpretation_limit": "逐帧用Data BCE优化同一低维基系数的乐观诊断上界；不是合法在线算法，也不证明Pilot可达",
        "radius": radius, "optimizer": {"name": "projected_adam", "steps": 160, "learning_rate": 0.03},
        "by_snr": by_snr, "frames": rows,
        "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in {
            "manifest": manifest_path, "feature_cache": feature_cache, "checkpoint": checkpoint}.items()}}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = analyze(Path(args.manifest), Path(args.feature_cache), Path(args.checkpoint),
                     Path(args.output), args.device)
    print(json.dumps(result["by_snr"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

