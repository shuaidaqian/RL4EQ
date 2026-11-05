# -*- coding: utf-8 -*-
"""只读比较Adapt损失梯度与离线Data梯度的方向，不驱动在线更新。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.detection_subspace import DetectionSubspace
from evaluation.subspace_development import read_manifest
from training.deterministic_adaptation import DeterministicSettings
from training.subspace_adaptation import update_subspace


METHODS = ("bce_t2", "bce_t1", "hard25", "hard50", "focal2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adapt_loss(logits: torch.Tensor, labels: torch.Tensor, method: str) -> torch.Tensor:
    signed = (labels.float() * 2. - 1.) * logits.float()
    per_symbol = F.softplus(-signed)
    if method == "bce_t2":
        return F.softplus(-signed / 2.).mean()
    if method == "bce_t1":
        return per_symbol.mean()
    if method in {"hard25", "hard50"}:
        fraction = .25 if method == "hard25" else .5
        count = max(1, int(len(per_symbol) * fraction + .999999))
        return torch.topk(per_symbol, count, largest=True).values.mean()
    if method == "focal2":
        return ((1. - torch.sigmoid(signed)).square() * per_symbol).mean()
    raise ValueError("Adapt损失名称未登记。")


def gradient(model: DetectionSubspace, observations: list, coefficients: torch.Tensor,
             method: str) -> torch.Tensor:
    per_frame = []
    for observation in observations:
        logits = model.logits(observation, coefficients)[observation.adapt_mask]
        labels = observation.adapt_labels[observation.adapt_mask]
        value = adapt_loss(logits, labels, method)
        grad, = torch.autograd.grad(value, coefficients)
        per_frame.append(grad)
    return torch.stack(per_frame).mean(0)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denominator = left.norm() * right.norm()
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 1e-20:
        return None
    value = torch.dot(left, right) / denominator
    return float(value) if bool(torch.isfinite(value)) else None


def analyze(manifest_path: Path, feature_cache: Path, checkpoint: Path,
            output_path: Path, device: str = "cpu") -> dict:
    if output_path.exists():
        raise ValueError("输出文件已存在，禁止覆盖。")
    spec = read_manifest(manifest_path)
    cache = torch.load(feature_cache, map_location=device, weights_only=False)
    sessions = cache["selection"]
    allowed = set(spec["data_roles"]["selection"]["seeds"])
    if not sessions or any(int(item["seed"]) not in allowed for item in sessions):
        raise ValueError("梯度诊断只允许清单隔离的selection会话。")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = DetectionSubspace(int(payload["dimension"]), architecture=payload["architecture"],
        learned_gradient_transport=payload.get("learned_gradient_transport", False),
        learned_context_update=payload.get("learned_context_update", False)).to(device)
    model.use_context_update = payload.get("use_context_update", payload.get("learned_context_update", False))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    settings = DeterministicSettings(**spec["update"])
    rows = []
    for session in sessions:
        origin = model.coefficients.detach().clone()
        current = origin.clone()
        history = []
        for index, frame in enumerate(session["frames"]):
            observation = frame["observation"]
            history = (history + [observation])[-int(session.get("inner_history_frames", 8)):]
            _, current, control = update_subspace(model, observation, current, origin,
                settings, adapt_history=history)
            current = current.detach().clone()
            coefficients = current.detach().clone().requires_grad_(True)
            data_logits = model.logits(observation, coefficients)
            data = frame["offline"]
            mask = data["data_mask"].to(data_logits.device)
            labels = data["bits"].to(data_logits.device).float()
            data_loss = F.binary_cross_entropy_with_logits(data_logits[mask], labels[mask])
            data_gradient, = torch.autograd.grad(data_loss, coefficients)
            for method in METHODS:
                adapt_gradient = gradient(model, history, coefficients, method)
                rows.append({
                    "snr_db": int(session["snr_db"]), "seed": int(session["seed"]),
                    "frame": int(frame["identity"]["frame"]), "method": method,
                    "adapt_gradient_norm": float(adapt_gradient.norm()),
                    "data_gradient_norm": float(data_gradient.norm()),
                    "adapt_data_cosine": cosine(adapt_gradient, data_gradient),
                    "online_control_used_data_gradient": False,
                    "online_candidate_accepted": bool(control["accepted"]),
                })

    by_session = defaultdict(list)
    for row in rows:
        by_session[(row["method"], row["snr_db"], row["seed"])].append(row)
    session_rows = []
    for (method, snr, seed), values in sorted(by_session.items()):
        cosines = [row["adapt_data_cosine"] for row in values if row["adapt_data_cosine"] is not None]
        session_rows.append({
            "method": method, "snr_db": snr, "seed": seed, "frames": len(values),
            "mean_cosine": sum(cosines) / len(cosines) if cosines else None,
            "mean_adapt_gradient_norm": sum(row["adapt_gradient_norm"] for row in values) / len(values),
        })
    by_snr = []
    for method in METHODS:
        for snr in spec["snrs_db"]:
            items = [row for row in session_rows if row["method"] == method and row["snr_db"] == snr]
            valid = [row["mean_cosine"] for row in items if row["mean_cosine"] is not None]
            norms = [row["mean_adapt_gradient_norm"] for row in items]
            by_snr.append({
                "method": method, "snr_db": snr, "seed_sessions": len(items),
                "usable_seed_sessions": len(valid),
                "mean_of_session_cosines": sum(valid) / len(valid) if valid else None,
                "mean_adapt_gradient_norm": sum(norms) / len(norms) if norms else None,
            })
    result = {
        "schema_version": "adapt-loss-alignment-diagnostic-v1",
        "role": "development_selection_read_only_diagnostic",
        "data_gradient_used_for_online_control": False,
        "independent_unit": "selection seed-session; 每SNR为2个既有开发会话",
        "description": "逐帧按原v11更新回放，只比较Adapt Pilot损失梯度与离线Data BCE梯度；不运行候选损失更新、不据此声称性能提升",
        "methods": list(METHODS), "by_snr": by_snr, "sessions": session_rows, "frames": rows,
        "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in {
            "manifest": manifest_path, "feature_cache": feature_cache, "checkpoint": checkpoint}.items()},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main() -> None:
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
