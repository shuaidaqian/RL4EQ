# -*- coding: utf-8 -*-
"""只读检查神经子空间能否在受限系数变化内触及Data判决边界。"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.detection_subspace import DetectionSubspace
from evaluation.subspace_development import read_manifest
from training.deterministic_adaptation import DeterministicSettings
from training.subspace_adaptation import update_subspace


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_sessions(frame_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped = defaultdict(list)
    for row in frame_rows:
        grouped[(row["snr_db"], row["seed"])].append(row)
    sessions = []
    for (snr, seed), rows in sorted(grouped.items()):
        total_errors = sum(row["before_errors"] for row in rows)
        total_candidates = sum(row["candidate_errors"] for row in rows)
        total_retained = sum(row["retained_errors"] for row in rows)
        error_rows = [row for row in rows if row["before_errors"]]
        denom = sum(row["before_errors"] for row in error_rows)
        sessions.append({
            "snr_db": snr, "seed": seed, "frames": len(rows),
            "before_errors": total_errors,
            "candidate_errors": total_candidates,
            "retained_errors": total_retained,
            "candidate_accepted_frames": sum(bool(row["accepted"]) for row in rows),
            "errors_reachable_within_step_radius": sum(
                row["before_errors_reachable"] for row in rows),
            "errors_reachable_within_frame_radius": sum(
                row["before_errors_reachable_within_frame_radius"] for row in rows),
            "error_symbols_with_positive_candidate_margin_change": sum(
                row["before_errors_improved_by_candidate"] for row in rows),
            "error_symbols_with_negative_candidate_margin_change": sum(
                row["before_errors_harmed_by_candidate"] for row in rows),
            "before_errors_crossed_by_candidate": sum(
                row["before_errors_crossed_by_candidate"] for row in rows),
            "reachable_fraction": (sum(row["before_errors_reachable"] for row in rows) / denom
                                   if denom else None),
            "positive_candidate_change_fraction": (
                sum(row["before_errors_improved_by_candidate"] for row in rows) / denom
                if denom else None),
        })
    by_snr = []
    for snr in sorted({row["snr_db"] for row in sessions}):
        rows = [row for row in sessions if row["snr_db"] == snr]
        errors = sum(row["before_errors"] for row in rows)
        reachable = sum(row["errors_reachable_within_step_radius"] for row in rows)
        frame_reachable = sum(row["errors_reachable_within_frame_radius"] for row in rows)
        improved = sum(row["error_symbols_with_positive_candidate_margin_change"] for row in rows)
        by_snr.append({
            "snr_db": snr, "seed_sessions": len(rows),
            "before_errors": errors,
            "candidate_errors": sum(row["candidate_errors"] for row in rows),
            "retained_errors": sum(row["retained_errors"] for row in rows),
            "accepted_frames": sum(row["candidate_accepted_frames"] for row in rows),
            "reachable_errors_within_step_radius": reachable,
            "reachable_errors_within_frame_radius": frame_reachable,
            "candidate_helped_errors": improved,
            "reachable_fraction": reachable / errors if errors else None,
            "frame_reachable_fraction": frame_reachable / errors if errors else None,
            "candidate_helped_fraction": improved / errors if errors else None,
            "candidate_harmed_errors": sum(
                row["error_symbols_with_negative_candidate_margin_change"] for row in rows),
            "candidate_crossed_errors": sum(
                row["before_errors_crossed_by_candidate"] for row in rows),
        })
    return sessions, by_snr


def analyze(manifest_path: Path, feature_cache: Path, checkpoint: Path,
            output_path: Path, device: str = "cpu") -> dict:
    if output_path.exists():
        raise ValueError("输出文件已存在，禁止覆盖。")
    spec = read_manifest(manifest_path)
    sessions = torch.load(feature_cache, map_location=device, weights_only=False)
    allowed = set(spec["data_roles"]["evaluation"]["seeds"])
    if not sessions or any(int(item["seed"]) not in allowed for item in sessions):
        raise ValueError("边界可达性诊断只允许清单隔离的evaluation开发会话。")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = DetectionSubspace(int(payload["dimension"]), architecture=payload["architecture"],
        learned_gradient_transport=payload.get("learned_gradient_transport", False),
        learned_context_update=payload.get("learned_context_update", False)).to(device)
    model.use_context_update = payload.get("use_context_update", payload.get("learned_context_update", False))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    settings = DeterministicSettings(**spec["update"])
    frame_rows = []
    for session in sessions:
        origin = model.coefficients.detach().clone()
        current = origin.clone()
        history = []
        for frame in session["frames"]:
            observation = frame["observation"]
            history = (history + [observation])[-int(session.get("inner_history_frames", 8)):]
            candidate, retained, control = update_subspace(model, observation, current,
                origin, settings, adapt_history=history)
            basis = model.basis(observation, current).detach()
            data = frame["offline"]
            mask = data["data_mask"].to(basis.device)
            labels = data["bits"].to(basis.device)[mask].float()
            signs = labels.mul(2.).sub(1.)
            frozen_logits = model.logits(observation, origin).detach().to(basis.device)[mask]
            before_logits = model.logits(observation, current).detach().to(basis.device)[mask]
            candidate_logits = model.logits(observation, candidate).detach().to(basis.device)[mask]
            retained_logits = model.logits(observation, retained).detach().to(basis.device)[mask]
            before_margin = signs * before_logits
            candidate_margin = signs * candidate_logits
            frozen_errors = int(((signs * frozen_logits) <= 0).sum())
            before_error_mask = before_margin <= 0
            candidate_error_mask = candidate_margin <= 0
            retained_error_mask = signs * retained_logits <= 0
            row_norm = basis[mask].norm(dim=1).clamp_min(1e-20)
            required_norm = (-before_margin).clamp_min(0) / row_norm
            current_errors = int(before_error_mask.sum())
            delta = candidate.detach() - current.detach()
            signed_change = signs * (candidate_logits - before_logits)
            local_radius = float(settings.step_radius)
            frame_rows.append({
                "snr_db": int(session["snr_db"]), "seed": int(session["seed"]),
                "frame": int(frame["identity"]["frame"]),
                "frozen_errors": frozen_errors, "before_errors": current_errors,
                "candidate_errors": int(candidate_error_mask.sum()),
                "retained_errors": int(retained_error_mask.sum()),
                "before_errors_reachable": int((before_error_mask & (required_norm <= local_radius)).sum()),
                "before_errors_reachable_within_frame_radius": int((before_error_mask &
                    (required_norm <= float(settings.step_radius) * int(settings.steps))).sum()),
                "before_errors_improved_by_candidate": int((before_error_mask & (signed_change > 0)).sum()),
                "before_errors_harmed_by_candidate": int((before_error_mask & (signed_change < 0)).sum()),
                "before_errors_crossed_by_candidate": int((before_error_mask & (candidate_margin > 0)).sum()),
                "mean_required_delta_norm_for_before_errors": (
                    float(required_norm[before_error_mask].mean()) if current_errors else None),
                "mean_signed_margin_change_on_before_errors": (
                    float(signed_change[before_error_mask].mean()) if current_errors else None),
                "candidate_delta_norm": float(delta.norm()),
                "accepted": bool(control["accepted"]),
                "data_labels_used_for_online_control": False,
            })
            current = retained.detach().clone()
    session_rows, by_snr = summarize_sessions(frame_rows)
    result = {
        "schema_version": "subspace-data-boundary-reachability-v1",
        "role": "development_evaluation_read_only_diagnostic",
        "core_confirmed": False,
        "confirmation_ready": False,
        "data_labels_used_for_online_control": False,
        "independent_unit": "evaluation seed-session; 每SNR为3个既有开发会话",
        "interpretation_limit": "单符号线性子空间局部可达性，不等于因果Pilot可识别性，也不等于部署性能",
        "step_radius": float(settings.step_radius),
        "maximum_frame_displacement_upper_bound": float(settings.step_radius) * int(settings.steps),
        "by_snr": by_snr,
        "sessions": session_rows,
        "frames": frame_rows,
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
