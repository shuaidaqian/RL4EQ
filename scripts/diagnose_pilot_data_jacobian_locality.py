# -*- coding: utf-8 -*-
"""只读测量Adapt Pilot与Data位置的神经子空间Jacobian影响关系。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

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


def analyze(manifest_path: Path, feature_cache: Path, checkpoint: Path,
            output_path: Path, device: str = "cpu") -> dict:
    if output_path.exists():
        raise ValueError("输出文件已存在，禁止覆盖。")
    spec = read_manifest(manifest_path)
    sessions = torch.load(feature_cache, map_location=device, weights_only=False)
    allowed = set(spec["data_roles"]["evaluation"]["seeds"])
    if not sessions or any(int(item["seed"]) not in allowed for item in sessions):
        raise ValueError("Pilot Jacobian诊断只允许清单隔离的evaluation开发会话。")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = DetectionSubspace(int(payload["dimension"]), architecture=payload["architecture"],
        learned_gradient_transport=payload.get("learned_gradient_transport", False),
        learned_context_update=payload.get("learned_context_update", False)).to(device)
    model.use_context_update = payload.get("use_context_update", payload.get("learned_context_update", False))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    bins = [(0, 1), (2, 4), (5, 8), (9, 16), (17, 24), (25, 32), (33, 48), (49, 64), (65, 10**9)]
    values = defaultdict(lambda: defaultdict(list))
    rows = []
    with torch.no_grad():
        for session in sessions:
            for frame in session["frames"]:
                obs = frame["observation"]
                basis = model.basis(obs.to(device) if hasattr(obs, "to") else obs)
                adapt_positions = torch.where(obs.adapt_mask.to(device))[0]
                data_positions = torch.where(frame["offline"]["data_mask"].to(device))[0]
                a = basis[adapt_positions]
                d = basis[data_positions]
                a_norm = a.norm(dim=1).clamp_min(1e-12)
                d_norm = d.norm(dim=1).clamp_min(1e-12)
                cosine = (d @ a.transpose(0, 1)) / (d_norm[:, None] * a_norm[None, :])
                distance = (data_positions[:, None] - adapt_positions[None, :]).abs()
                row = {"snr_db": int(session["snr_db"]), "seed": int(session["seed"]),
                    "frame": int(frame["identity"]["frame"]),
                    "adapt_symbols": int(adapt_positions.numel()),
                    "data_symbols": int(data_positions.numel()),
                    "adapt_jacobian_rank": int(torch.linalg.matrix_rank(a).item()),
                    "adapt_jacobian_condition": float(torch.linalg.cond(a).item()),
                    "bins": {}}
                for low, high in bins:
                    name = f"{low}-{high if high < 10**9 else 'plus'}"
                    mask = (distance >= low) & (distance <= high)
                    if mask.any():
                        selected = cosine.abs()[mask]
                        row["bins"][name] = {"pairs": int(mask.sum()),
                            "mean_abs_cosine": float(selected.mean()),
                            "p90_abs_cosine": float(torch.quantile(selected, .9))}
                        values[(int(session["snr_db"]), name)]["mean"].append(float(selected.mean()))
                        values[(int(session["snr_db"]), name)]["p90"].append(float(torch.quantile(selected, .9)))
                rows.append(row)
    by_snr = []
    for snr in sorted({int(s["snr_db"]) for s in sessions}):
        entry = {"snr_db": snr, "frames": sum(r["snr_db"] == snr for r in rows), "distance_bins": []}
        for low, high in bins:
            name = f"{low}-{high if high < 10**9 else 'plus'}"
            record = values[(snr, name)]
            if record["mean"]:
                entry["distance_bins"].append({"symbols_apart": name,
                    "mean_abs_cosine": sum(record["mean"]) / len(record["mean"]),
                    "mean_frame_p90_abs_cosine": sum(record["p90"]) / len(record["p90"]),
                    "frames": len(record["mean"])})
        by_snr.append(entry)
    result = {"schema_version": "pilot-data-jacobian-locality-v1",
        "role": "development_read_only_geometry_diagnostic", "core_confirmed": False,
        "confirmation_ready": False, "labels_read": False, "online_gain_claimed": False,
        "interpretation_limit": "Jacobian余弦刻画系数对Pilot/Data logit的方向相似性；不含梯度标签、噪声、优化器或验证门控，不能代表可部署增益",
        "distance_bins": [f"{a}-{b if b < 10**9 else 'plus'}" for a, b in bins],
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

