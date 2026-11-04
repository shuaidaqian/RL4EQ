# -*- coding: utf-8 -*-
"""训练并评估仅使用验证Pilot统计的离线动作风险门控。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.subspace_development import (  # noqa: E402
    data_loss, make_subspace, read_manifest, sha256, write_json,
)
from training.deterministic_adaptation import DeterministicSettings  # noqa: E402
from training.pilot_risk_gate import FEATURE_NAMES, PilotRiskRegressor  # noqa: E402
from training.subspace_adaptation import update_subspace  # noqa: E402


def _load_model(checkpoint_path, seed, device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = make_subspace(int(payload["dimension"]), seed, device, payload["architecture"],
        learned_gradient_transport=payload.get("learned_gradient_transport", False),
        learned_context_update=payload.get("learned_context_update", False))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.use_context_update = payload.get("use_context_update", model.learned_context_update)
    model.eval()
    return model, payload


def _collect_hard_gate_samples(model, sessions, settings, device):
    features, targets, rows = [], [], []
    for session in sessions:
        origin = model.coefficients.detach().clone()
        current = origin.clone()
        history = []
        for frame in session["frames"]:
            observation = frame["observation"]
            before_coefficients = current.detach().clone()
            history.append(observation)
            history = history[-int(session.get("inner_history_frames", 8)):]
            candidate, retained, control = update_subspace(model, observation, current, origin,
                settings, adapt_history=history,
                validation_risk_gate=lambda value: value.new_tensor(1.0),
                risk_gate_threshold=0.0)
            pilot_features = control["validation_risk_features"]
            if pilot_features is None:
                raise AssertionError("风险特征观察器未返回Pilot统计。")
            proposal_coefficients = origin + settings.origin_retention_decay * (candidate - origin)
            basis = model.basis(observation)
            before_logits = observation.base_logits + basis @ before_coefficients
            proposal_logits = observation.base_logits + basis @ proposal_coefficients
            target = float((data_loss(before_logits, frame) - data_loss(proposal_logits, frame)).detach())
            if not control["reset_to_origin"]:
                features.append(pilot_features)
                targets.append(target)
                rows.append(dict(seed=int(frame["identity"]["seed"]),
                    snr_db=int(frame["identity"]["snr_db"]), frame=int(frame["identity"]["frame"]),
                    target_data_bce_gain=target, hard_gate_accepted=bool(control["accepted"]),
                    reset_to_origin=False, data_labels_used_for_offline_risk_target=True))
            current = retained.detach()
    return torch.tensor(features, dtype=torch.float32, device=device), torch.tensor(targets,
        dtype=torch.float32, device=device), rows


def _score_predictions(predictions, targets):
    predictions = torch.as_tensor(predictions).flatten().cpu().float()
    targets = torch.as_tensor(targets).flatten().cpu().float()
    positive = predictions > 0
    actual = targets > 0
    tp = int((positive & actual).sum())
    fp = int((positive & ~actual).sum())
    fn = int((~positive & actual).sum())
    tn = int((~positive & ~actual).sum())
    if predictions.numel() > 1 and float(predictions.std()) > 0 and float(targets.std()) > 0:
        correlation = float(torch.corrcoef(torch.stack((predictions, targets)))[0, 1])
    else:
        correlation = None
    mse = float((predictions - targets).square().mean())
    return dict(samples=int(targets.numel()), predicted_beneficial=int(positive.sum()),
        actual_beneficial=int(actual.sum()), true_accept=tp, false_accept=fp,
        false_reject=fn, true_reject=tn, precision=tp/max(tp+fp, 1),
        recall=tp/max(tp+fn, 1), rmse=float(np.sqrt(mse)),
        pearson_prediction_target=correlation,
        mean_predicted_gain=float(predictions.mean()), mean_actual_gain=float(targets.mean()))


def _evaluate_branches(base_model, sessions, settings, gate, device):
    rows = []
    for session in sessions:
        for branch in ("hard_gate", "pilot_risk_gate"):
            origin = base_model.coefficients.detach().clone()
            current = origin.clone()
            history = []
            for frame in session["frames"]:
                observation = frame["observation"]
                history.append(observation)
                history = history[-int(session.get("inner_history_frames", 8)):]
                candidate, retained, control = update_subspace(base_model, observation, current,
                    origin, settings, adapt_history=history,
                    validation_risk_gate=gate if branch == "pilot_risk_gate" else None)
                basis = base_model.basis(observation)
                before_logits = observation.base_logits + basis @ current
                proposal = origin + settings.origin_retention_decay * (candidate - origin)
                candidate_logits = observation.base_logits + basis @ proposal
                retained_logits = observation.base_logits + basis @ retained
                data_mask = frame["offline"]["data_mask"]
                bits = frame["offline"]["bits"].bool()
                ber_before = float(((before_logits[data_mask] >= 0) != bits[data_mask]).float().mean())
                ber_candidate = float(((candidate_logits[data_mask] >= 0) != bits[data_mask]).float().mean())
                ber_retained = float(((retained_logits[data_mask] >= 0) != bits[data_mask]).float().mean())
                data_gain_candidate = float((data_loss(before_logits, frame) -
                                             data_loss(candidate_logits, frame)).detach())
                rows.append(dict(seed=int(frame["identity"]["seed"]),
                    snr_db=int(frame["identity"]["snr_db"]), frame=int(frame["identity"]["frame"]),
                    branch=branch, accepted=control["accepted"],
                    candidate_accepted=control["candidate_accepted"],
                    risk_rejected=control["validation_risk_gate_rejected"],
                    predicted_data_bce_gain=control["validation_risk_gate_prediction"],
                    candidate_data_bce_gain_posthoc=data_gain_candidate,
                    candidate_data_harm_posthoc=data_gain_candidate <= 0,
                    ber_gain_candidate_posthoc=ber_before-ber_candidate,
                    ber_gain_retained_posthoc=ber_before-ber_retained,
                    offline_labels_used_for_online_control=False))
                current = retained.detach()
    return rows


def summarize(rows):
    summary = []
    for snr in sorted({row["snr_db"] for row in rows}):
        arms = {}
        for branch in sorted({row["branch"] for row in rows}):
            active = [row for row in rows if row["snr_db"] == snr and row["branch"] == branch]
            seeds = sorted({row["seed"] for row in active})
            gains = [float(np.mean([row["ber_gain_retained_posthoc"] for row in active
                                    if row["seed"] == seed])) for seed in seeds]
            proposed = [row for row in active if row["candidate_accepted"]]
            accepted = [row for row in active if row["accepted"]]
            harms = [row for row in accepted if row["candidate_data_harm_posthoc"]]
            arms[branch] = dict(candidate_count=len(proposed), accepted=len(accepted),
                risk_rejected=sum(row["risk_rejected"] for row in active),
                false_accept_data_harm=sum(row["candidate_data_harm_posthoc"] for row in accepted),
                false_reject_data_benefit=sum(row["risk_rejected"]
                    and not row["candidate_data_harm_posthoc"] for row in active),
                harmful_accept_fraction_posthoc=len(harms)/max(len(accepted), 1),
                mean_candidate_data_bce_gain_posthoc=float(np.mean(
                    [row["candidate_data_bce_gain_posthoc"] for row in accepted])) if accepted else None,
                mean_retained_ber_gain_posthoc=float(np.mean(gains)),
                per_seed_ber_gain=dict(zip(map(str, seeds), gains)))
        summary.append(dict(snr_db=snr, branches=arms))
    return summary


def run(manifest_path, base_training_dir, output_dir, *, device="cuda"):
    risk_spec = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if risk_spec.get("schema_version") != "pilot-validation-risk-gate-v1" or risk_spec.get("role") != "development":
        raise ValueError("必须提供固定的Pilot风险门控开发清单。")
    if risk_spec.get("channel_changes") is not False or risk_spec.get("heldout_confirmation") is not False:
        raise ValueError("风险门控审计不能修改信道或冒充独立确认。")
    base_manifest_path = ROOT / risk_spec["base_manifest"]
    if sha256(base_manifest_path) != risk_spec["base_manifest_sha256"]:
        raise ValueError("v11基础清单指纹不符。")
    spec = read_manifest(base_manifest_path)
    policy = risk_spec["risk_gate"]
    if policy.get("ridge") != 0.1 or policy.get("threshold") != 0.0 \
            or policy.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("门控超参数或特征定义与冻结清单不符。")
    source = Path(base_training_dir)
    target = Path(output_dir)
    if target.exists():
        raise ValueError("风险门控输出目录已存在，禁止覆盖。")
    target.mkdir(parents=True)
    started = time.perf_counter()
    training_sessions = torch.load(source / "offline_features.pt", map_location=device,
                                  weights_only=False)["training"]
    selection_sessions = torch.load(source / "offline_features.pt", map_location=device,
                                    weights_only=False)["selection"]
    evaluation_sessions = torch.load(source / "evaluation_features.pt", map_location=device,
                                     weights_only=False)
    checkpoint_path = source / "B_post_adapt.pt"
    if (sha256(checkpoint_path) != risk_spec["base_checkpoint_sha256"]
            or sha256(source / "summary.json") != risk_spec["base_summary_sha256"]):
        raise ValueError("v11在线更新模型checkpoint指纹不符。")
    model, payload = _load_model(checkpoint_path, int(spec["initialization_seed"]), device)
    settings = DeterministicSettings(**spec["update"])

    train_x, train_y, train_rows = _collect_hard_gate_samples(model, training_sessions, settings, device)
    select_x, select_y, select_rows = _collect_hard_gate_samples(model, selection_sessions, settings, device)
    gate = PilotRiskRegressor.fit(train_x, train_y, ridge=policy["ridge"])
    selection_predictions = gate.predict(select_x)
    selection_score = _score_predictions(selection_predictions, select_y)

    rows = _evaluate_branches(model, evaluation_sessions, settings, gate, device)
    output = dict(schema_version="pilot-validation-risk-gate-result-v1", role="development",
        core_confirmed=False, confirmation_ready=False, selected_dimension=int(payload["dimension"]),
        feature_names=list(FEATURE_NAMES), training_samples=int(train_y.numel()),
        selection_samples=int(select_y.numel()), risk_model=gate.to_dict(),
        train_target_summary=_score_predictions(gate.predict(train_x), train_y),
        selection_metrics=selection_score, per_snr=summarize(rows), frame_rows=rows,
        source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        manifest_sha256=sha256(manifest_path), base_manifest_sha256=sha256(base_manifest_path),
        base_summary_sha256=sha256(source / "summary.json"),
        base_checkpoint_sha256=sha256(checkpoint_path),
        feature_cache_sha256={name: sha256(source / name) for name in
            ("offline_features.pt", "evaluation_features.pt")},
        source_sha256={name: sha256(ROOT / name) for name in
            ("scripts/train_pilot_risk_gate.py", "training/pilot_risk_gate.py",
             "training/subspace_adaptation.py")},
        training_samples_sha256=hashlib.sha256(json.dumps(train_rows, sort_keys=True).encode()).hexdigest(),
        selection_samples_sha256=hashlib.sha256(json.dumps(select_rows, sort_keys=True).encode()).hexdigest(),
        threshold=policy["threshold"], decision_rule="保留既有验证Pilot硬门控，仅当预估Data BCE改善大于0时额外接受",
        data_labels_used_for_online_control=False, wall_seconds=time.perf_counter()-started,
        device=str(device), gpu=torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else None)
    write_json(target / "risk_gate_result.json", output)
    write_json(target / "risk_gate.json", gate.to_dict())
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-training-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run(args.manifest, args.base_training_dir, args.output_dir,
                 device=args.device)
    print(json.dumps(result["per_snr"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
