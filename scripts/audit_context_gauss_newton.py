# -*- coding: utf-8 -*-
"""在固定长回波v22开发缓存上审计Pilot上下文阻尼Gauss–Newton更新。"""

from __future__ import annotations

import argparse
import gzip
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
from evaluation.pilot_transfer import paired_interval
from evaluation.subspace_development import read_manifest, write_json
from training.pilot_gauss_newton import damped_gauss_newton_candidate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pilot_ber(logits, mask, labels):
    return float(((logits[mask] >= 0) != labels[mask].bool()).float().mean())


def run_audit(audit_manifest: Path, feature_cache: Path, checkpoint_a: Path,
              checkpoint_b: Path, output_dir: Path, device="cpu"):
    if output_dir.exists():
        raise ValueError("输出目录已存在，禁止覆盖。")
    audit = json.loads(audit_manifest.read_text(encoding="utf-8"))
    parent_path = ROOT / audit["parent_manifest"]
    parent = read_manifest(parent_path)
    sessions = torch.load(feature_cache, map_location=device, weights_only=False)
    expected_seeds = set(audit["parent_evaluation_seeds"])
    if (audit["channel_changes"] or audit["heldout_confirmation"]
            or audit["online_data_labels_used"]):
        raise ValueError("审计清单违反固定信道、非确认或Data标签边界。")
    if not sessions or {int(row["seed"]) for row in sessions} != expected_seeds:
        raise ValueError("缓存seed与v23审计清单不符。")
    if sorted({int(row["snr_db"]) for row in sessions}) != audit["snrs_db"]:
        raise ValueError("缓存SNR与v23审计清单不符。")
    update = audit["update"]

    models = {}
    for arm, checkpoint in (("A_supervised", checkpoint_a), ("B_post_adapt", checkpoint_b)):
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        if payload["architecture"] != audit["architecture"] or int(payload["dimension"]) != audit["dimension"]:
            raise ValueError("A/B checkpoint与审计清单中的网络结构或维度不符。")
        model = DetectionSubspace(int(payload["dimension"]), architecture=payload["architecture"],
            learned_gradient_transport=payload.get("learned_gradient_transport", False),
            learned_context_update=payload.get("learned_context_update", False)).to(device)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.use_context_update = payload.get("use_context_update", payload.get("learned_context_update", False))
        model.eval()
        models[arm] = model

    records = []
    for arm, model in models.items():
        for session in sessions:
            origin = model.coefficients.detach().clone()
            current = origin.clone()
            for row in session["frames"]:
                observation = row["observation"]
                with torch.no_grad():
                    frozen_logits = model.logits(observation, origin)
                    before_logits = model.logits(observation, current)
                    before_val_loss = float(F.binary_cross_entropy_with_logits(
                        before_logits[observation.validation_mask],
                        observation.validation_labels[observation.validation_mask].float()))
                    before_val_ber = pilot_ber(before_logits, observation.validation_mask,
                                               observation.validation_labels)
                candidate, geometry = damped_gauss_newton_candidate(model, observation, current, origin,
                    damping=update["damping"], proximal_weight=update["proximal_weight"],
                    temperature=update["adapt_logit_temperature"], step_radius=update["step_radius"],
                    total_radius=update["total_radius"], max_backtracks=update["max_backtracks"])
                with torch.no_grad():
                    candidate_logits = model.logits(observation, candidate)
                    candidate_val_loss = float(F.binary_cross_entropy_with_logits(
                        candidate_logits[observation.validation_mask],
                        observation.validation_labels[observation.validation_mask].float()))
                    candidate_val_ber = pilot_ber(candidate_logits, observation.validation_mask,
                                                  observation.validation_labels)
                candidate_accepted = bool(geometry["accepted_by_adapt_backtracking"]
                    and math.isfinite(candidate_val_loss)
                    and candidate_val_loss < before_val_loss - update["validation_min_improvement"]
                    and candidate_val_ber <= before_val_ber)
                retained = candidate if candidate_accepted else current
                if candidate_accepted and update["origin_retention_decay"] < 1:
                    retained = origin + update["origin_retention_decay"] * (retained - origin)
                with torch.no_grad():
                    retained_logits = model.logits(observation, retained)
                    retained_val_loss = float(F.binary_cross_entropy_with_logits(
                        retained_logits[observation.validation_mask],
                        observation.validation_labels[observation.validation_mask].float()))
                    retained_val_ber = pilot_ber(retained_logits, observation.validation_mask,
                                                 observation.validation_labels)
                    origin_logits = model.logits(observation, origin)
                    origin_val_loss = float(F.binary_cross_entropy_with_logits(
                        origin_logits[observation.validation_mask],
                        observation.validation_labels[observation.validation_mask].float()))
                    origin_val_ber = pilot_ber(origin_logits, observation.validation_mask,
                                               observation.validation_labels)
                accepted = bool(candidate_accepted and retained_val_loss < before_val_loss
                    - update["validation_min_improvement"] and retained_val_ber <= before_val_ber)
                guarded_loss = retained_val_loss if accepted else before_val_loss
                guarded_ber = retained_val_ber if accepted else before_val_ber
                guarded_coefficients = retained if accepted else current
                reset_to_origin = bool(update["reset_to_origin_on_validation"]
                    and origin_val_loss < guarded_loss - update["validation_min_improvement"]
                    and origin_val_ber <= guarded_ber
                    and float((guarded_coefficients - origin).norm()) > 0)
                if reset_to_origin:
                    retained, retained_logits = origin, origin_logits
                    accepted = False
                elif not accepted:
                    retained, retained_logits = current, before_logits
                else:
                    retained = retained.detach()

                # Data只在候选和门控完全确定后用于离线事后评分。
                offline = row["offline"]
                data_mask, bits = offline["data_mask"], offline["bits"]
                records.append(dict(**row["identity"], arm=arm,
                    ber_frozen=float(((frozen_logits[data_mask] >= 0) != bits[data_mask]).float().mean()),
                    ber_before=float(((before_logits[data_mask] >= 0) != bits[data_mask]).float().mean()),
                    ber_candidate=float(((candidate_logits[data_mask] >= 0) != bits[data_mask]).float().mean()),
                    ber_online=float(((retained_logits[data_mask] >= 0) != bits[data_mask]).float().mean()),
                    accepted=accepted, candidate_accepted=candidate_accepted,
                    reset_to_origin=reset_to_origin,
                    validation_loss_before=before_val_loss, validation_loss_candidate=candidate_val_loss,
                    validation_loss_retained=retained_val_loss,
                    validation_ber_before=before_val_ber, validation_ber_candidate=candidate_val_ber,
                    validation_ber_retained=retained_val_ber,
                    coefficient_values=retained.detach().cpu().tolist(), **geometry,
                    offline_labels_used_for_control=False))
                current = retained.detach()

    summary = []
    for snr in audit["snrs_db"]:
        arms = {}
        for arm in models:
            active = [row for row in records if row["snr_db"] == snr and row["arm"] == arm]
            seeds = sorted({row["seed"] for row in active})
            gains = [float(sum(row["ber_frozen"] - row["ber_online"] for row in active
                               if row["seed"] == seed) / sum(1 for row in active if row["seed"] == seed))
                     for seed in seeds]
            arms[arm] = dict(ber_frozen=float(sum(row["ber_frozen"] for row in active) / len(active)),
                ber_online=float(sum(row["ber_online"] for row in active) / len(active)),
                gain=paired_interval(gains), per_seed_gain=dict(zip(map(str, seeds), gains)),
                candidate_accepted=sum(row["candidate_accepted"] for row in active),
                accepted=sum(row["accepted"] for row in active),
                reset_to_origin=sum(row["reset_to_origin"] for row in active),
                finite_jacobian_fraction=sum(row["jacobian_condition"] is not None
                                             for row in active) / len(active),
                full_rank_fraction=sum(row["jacobian_rank"] == row["parameter_dimension"] for row in active) / len(active),
                mean_fisher_condition=sum(row["fisher_condition"] for row in active) / len(active))
        summary.append(dict(snr_db=snr, arms=arms,
            gain_interaction_B_minus_A=arms["B_post_adapt"]["gain"]["mean"]
                - arms["A_supervised"]["gain"]["mean"]))

    output_dir.mkdir(parents=True)
    metrics = output_dir / "frame_metrics.jsonl.gz"
    with gzip.open(metrics, "wt", encoding="utf-8", compresslevel=9) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    result = dict(role="development_numerical_audit", core_confirmed=False,
        confirmation_ready=False, matched_offline_training=False,
        interpretation_limit=audit["interpretation"], audit_manifest_sha256=sha256(audit_manifest),
        parent_manifest_sha256=sha256(parent_path), fixed_update=update,
        source_sha256={name: sha256(ROOT / name) for name in (
            "training/pilot_gauss_newton.py", "scripts/audit_context_gauss_newton.py",
            "agent/detection_subspace.py", "evaluation/subspace_development.py")},
        inputs={name: {"path": str(path), "sha256": sha256(path)} for name, path in {
            "audit_manifest": audit_manifest, "parent_manifest": parent_path,
            "feature_cache": feature_cache, "A_checkpoint": checkpoint_a,
            "B_checkpoint": checkpoint_b}.items()},
        snrs_db=audit["snrs_db"], evaluation_seeds=sorted(expected_seeds),
        records=len(records), per_snr=summary, frame_metrics_sha256=sha256(metrics))
    write_json(output_dir / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-manifest", default="configs/audits/detection_subspace_long_v23_gauss_newton_numerical_v1.json")
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_audit(Path(args.audit_manifest), Path(args.feature_cache), Path(args.checkpoint_a),
        Path(args.checkpoint_b), Path(args.output_dir), args.device)
    print(json.dumps(result["per_snr"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
