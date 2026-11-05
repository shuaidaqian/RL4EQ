# -*- coding: utf-8 -*-
"""只读比较GN Adapt方向、Data BCE方向和验证Pilot方向。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.detection_subspace import DetectionSubspace
from evaluation.subspace_development import read_manifest, write_json
from training.pilot_gauss_newton import damped_gauss_newton_candidate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cosine(left, right):
    denom = left.norm() * right.norm()
    if not torch.isfinite(denom) or float(denom) <= 1e-20:
        return None
    return float(torch.dot(left.flatten(), right.flatten()) / denom)


def analyze(manifest_path: Path, feature_cache: Path, checkpoint: Path, output_path: Path, device="cpu"):
    if output_path.exists():
        raise ValueError("输出文件已存在，禁止覆盖。")
    spec = read_manifest(manifest_path)
    sessions = torch.load(feature_cache, map_location=device, weights_only=False)
    allowed = set(spec["data_roles"]["evaluation"]["seeds"])
    if not sessions or any(int(session["seed"]) not in allowed for session in sessions):
        raise ValueError("方向诊断只允许清单指定的evaluation开发缓存。")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = DetectionSubspace(int(payload["dimension"]), architecture=payload["architecture"],
        learned_gradient_transport=payload.get("learned_gradient_transport", False),
        learned_context_update=payload.get("learned_context_update", False)).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    update = spec["update"]
    records = []
    for session in sessions:
        for frame in session["frames"]:
            obs = frame["observation"]
            origin = model.coefficients.detach().clone()
            z = origin.clone().requires_grad_(True)
            adapt_mask = obs.adapt_mask.bool()
            adapt_logits = model.logits(obs, z)[adapt_mask] / update["adapt_logit_temperature"]
            adapt_loss = F.binary_cross_entropy_with_logits(adapt_logits, obs.adapt_labels[adapt_mask].float())
            adapt_gradient, = torch.autograd.grad(adapt_loss, z)
            validation_mask = obs.validation_mask.bool()
            validation_logits = model.logits(obs, z)[validation_mask]
            validation_loss = F.binary_cross_entropy_with_logits(
                validation_logits, obs.validation_labels[validation_mask].float())
            validation_gradient, = torch.autograd.grad(validation_loss, z)
            data = frame["offline"]
            data_mask = data["data_mask"].bool()
            data_logits = model.logits(obs, z)[data_mask]
            data_loss = F.binary_cross_entropy_with_logits(data_logits, data["bits"][data_mask].float())
            data_gradient, = torch.autograd.grad(data_loss, z)
            candidate, geometry = damped_gauss_newton_candidate(model, obs, origin, origin,
                damping=update["gn_damping"], proximal_weight=update["proximal_weight"],
                temperature=update["adapt_logit_temperature"], step_radius=update["step_radius"],
                total_radius=update["total_radius"], max_backtracks=update["gn_max_backtracks"])
            delta = candidate - origin
            records.append(dict(seed=int(session["seed"]), snr_db=int(session["snr_db"]),
                frame=int(frame["identity"]["frame"]),
                adapt_data_cosine=cosine(adapt_gradient, data_gradient),
                gn_step_data_descent_cosine=cosine(delta, -data_gradient),
                validation_data_cosine=cosine(validation_gradient, data_gradient),
                adapt_gradient_norm=float(adapt_gradient.norm()),
                data_gradient_norm=float(data_gradient.norm()),
                validation_gradient_norm=float(validation_gradient.norm()),
                gn_step_norm=float(delta.norm()),
                gn_candidate_data_loss_change=float(data_loss.detach()) - float(F.binary_cross_entropy_with_logits(
                    model.logits(obs, candidate)[data_mask].detach(), data["bits"][data_mask].float())),
                gn_candidate_validation_loss_change=float(validation_loss.detach()) - float(F.binary_cross_entropy_with_logits(
                    model.logits(obs, candidate)[validation_mask].detach(), obs.validation_labels[validation_mask].float())),
                jacobian_rank=geometry["jacobian_rank"], fisher_condition=geometry["fisher_condition"],
                labels_used_for_update=False, data_labels_used_for_diagnostic=True))

    grouped = []
    for snr in sorted({row["snr_db"] for row in records}):
        active = [row for row in records if row["snr_db"] == snr]
        stats = {}
        for key in ("adapt_data_cosine", "gn_step_data_descent_cosine", "validation_data_cosine",
                    "adapt_gradient_norm", "data_gradient_norm", "validation_gradient_norm",
                    "gn_step_norm", "gn_candidate_data_loss_change",
                    "gn_candidate_validation_loss_change"):
            values = [row[key] for row in active if row[key] is not None]
            if values:
                ordered = sorted(values)
                stats[key] = dict(mean=sum(values)/len(values), median=ordered[len(ordered)//2],
                    positive_fraction=sum(value > 0 for value in values)/len(values), n=len(values))
        grouped.append(dict(snr_db=snr, frames=len(active), metrics=stats,
            full_rank_fraction=sum(row["jacobian_rank"] == int(payload["dimension"])
                                   for row in active)/len(active)))
    result = dict(role="development_read_only_gradient_alignment_diagnostic",
        core_confirmed=False, confirmation_ready=False,
        online_control_used_data_labels=False,
        interpretation_limit="Adapt、验证和Data梯度只作离线方向诊断；Data梯度与候选目标不进入在线动作。",
        dimension=int(payload["dimension"]), manifest_sha256=sha256(manifest_path),
        inputs={name: {"path": str(path), "sha256": sha256(path)} for name, path in {
            "manifest": manifest_path, "feature_cache": feature_cache, "checkpoint": checkpoint}.items()},
        records=len(records), by_snr=grouped, frames=records)
    write_json(output_path, result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--feature-cache",required=True)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--device",default="cpu")
    args=parser.parse_args()
    result=analyze(Path(args.manifest),Path(args.feature_cache),Path(args.checkpoint),Path(args.output),args.device)
    print(json.dumps(result["by_snr"],ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
