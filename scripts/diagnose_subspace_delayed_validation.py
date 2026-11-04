# -*- coding: utf-8 -*-
"""只读检验把单帧候选暂存后跨帧累积合法验证Pilot证据的效果。"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.subspace_development import make_subspace, read_manifest, sha256
from training.deterministic_adaptation import DeterministicSettings
from training.subspace_adaptation import update_subspace


def _pilot_stats(model, candidate, incumbent, observations):
    """按验证符号累积候选相对原状态的配对BCE与硬错误。"""
    gains, errors_candidate, errors_incumbent = [], 0, 0
    count = 0
    for observation in observations:
        candidate_logits = observation.base_logits + model.basis(observation) @ candidate
        incumbent_logits = observation.base_logits + model.basis(observation) @ incumbent
        mask, labels = observation.validation_mask, observation.validation_labels.float()
        before = F.binary_cross_entropy_with_logits(incumbent_logits[mask].float(), labels[mask], reduction="none")
        after = F.binary_cross_entropy_with_logits(candidate_logits[mask].float(), labels[mask], reduction="none")
        gains.extend((before - after).detach().cpu().tolist())
        errors_candidate += int(((candidate_logits[mask] >= 0) != labels[mask].bool()).sum())
        errors_incumbent += int(((incumbent_logits[mask] >= 0) != labels[mask].bool()).sum())
        count += int(mask.sum())
    return {"symbols": count, "mean_bce_gain": sum(gains) / len(gains) if gains else None,
            "candidate_errors": errors_candidate, "incumbent_errors": errors_incumbent,
            "hard_error_nonincrease": errors_candidate <= errors_incumbent}


def _data_gain(model, candidate, incumbent, frame):
    """只在门控判定逻辑完成后调用的离线Data事后评分。"""
    observation, offline = frame["observation"], frame["offline"]
    mask, labels = offline["data_mask"], offline["bits"].float()
    basis = model.basis(observation)
    candidate_logits = observation.base_logits + basis @ candidate
    incumbent_logits = observation.base_logits + basis @ incumbent
    before = F.binary_cross_entropy_with_logits(incumbent_logits[mask].float(), labels[mask], reduction="none")
    after = F.binary_cross_entropy_with_logits(candidate_logits[mask].float(), labels[mask], reduction="none")
    ber_before = ((incumbent_logits[mask] >= 0) != labels[mask].bool()).float().mean()
    ber_after = ((candidate_logits[mask] >= 0) != labels[mask].bool()).float().mean()
    return {"data_bce_gain_posthoc": float((before.mean() - after.mean()).detach()),
            "data_ber_gain_posthoc": float((ber_before - ber_after).detach())}


def analyze(manifest_path: Path, training_dir: Path, trajectory_path: Path,
            *, device="cpu", horizons=(1, 2, 4, 8)):
    spec = read_manifest(manifest_path)
    sessions = torch.load(training_dir / "evaluation_features.pt", map_location=device, weights_only=False)
    payload = torch.load(training_dir / "B_post_adapt.pt", map_location=device, weights_only=False)
    model = make_subspace(int(payload["dimension"]), int(spec["initialization_seed"]),
                          device, payload["architecture"],
                          learned_gradient_transport=payload.get("learned_gradient_transport", False),
                          learned_context_update=payload.get("learned_context_update", False))
    model.use_context_update = payload.get("use_context_update", model.learned_context_update)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    with gzip.open(trajectory_path, "rt", encoding="utf-8") as handle:
        trajectory = [json.loads(line) for line in handle if line.strip()]
    by_key = {(int(row["seed"]), int(row["snr_db"]), int(row["frame"])): row
              for row in trajectory if row.get("training") == "B_post_adapt"}
    settings = DeterministicSettings(**spec["update"])
    all_rows = []
    for session in sessions:
        seed, snr = int(session["seed"]), int(session["snr_db"])
        origin = model.coefficients.detach().clone()
        current, history, steps = origin.clone(), [], []
        frames, previous = session["frames"], None
        for frame in frames:
            identity = frame["identity"]
            logged = by_key[(seed, snr, int(identity["frame"]))]
            if identity.get("rx_sha256") != logged.get("rx_sha256"):
                raise ValueError("缓存与已归档轨迹的当前接收波形不一致。")
            if previous is not None:
                current = torch.tensor(previous["coefficient_values"], device=device, dtype=origin.dtype)
            observation = frame["observation"]
            history.append(observation)
            history = history[-int(session.get("inner_history_frames", 8)):]
            before = current.detach().clone()
            candidate, _, control = update_subspace(model, observation, before, origin,
                settings, adapt_history=history)
            if abs(float(control["validation_loss_candidate"]) - float(logged["validation_loss_candidate"])) > 2e-6:
                raise ValueError("重建候选与日志验证损失不一致，拒绝延迟门控分析。")
            # 将在线规则中固定的锚点收缩纳入待验证候选，但不套用当前帧守门结果。
            delayed_candidate = origin + settings.origin_retention_decay * (candidate.detach() - origin)
            steps.append({"before": before, "candidate": delayed_candidate.detach(),
                          "control": control, "frame": frame})
            previous = logged
        for index, item in enumerate(steps):
            for horizon in horizons:
                end = index + horizon
                if end >= len(steps):
                    continue
                pilot_window = [steps[j]["frame"]["observation"] for j in range(index, end)]
                stats = _pilot_stats(model, item["candidate"], item["before"], pilot_window)
                decision_ok = (stats["mean_bce_gain"] is not None and
                               stats["mean_bce_gain"] > settings.min_validation_improvement and
                               stats["hard_error_nonincrease"])
                prospective_data = _data_gain(model, item["candidate"], item["before"], steps[end]["frame"])
                all_rows.append({"seed": seed, "snr_db": snr,
                    "candidate_frame": int(item["frame"]["identity"]["frame"]),
                    "decision_frame": int(steps[end]["frame"]["identity"]["frame"]),
                    "horizon_frames": horizon, "pilot_window": stats,
                    "delayed_gate_accept": bool(decision_ok),
                    "original_current_frame_accept": bool(item["control"]["candidate_accepted"]),
                    **prospective_data, "data_used_for_gate": False})
    summaries = []
    for snr in sorted({row["snr_db"] for row in all_rows}):
        for horizon in horizons:
            subset = [r for r in all_rows if r["snr_db"] == snr and r["horizon_frames"] == horizon]
            accepted = [r for r in subset if r["delayed_gate_accept"]]
            summaries.append({"snr_db": snr, "horizon_frames": horizon,
                "candidate_windows": len(subset),
                "original_current_frame_accepts": sum(r["original_current_frame_accept"] for r in subset),
                "delayed_accepts": len(accepted),
                "accepted_future_data_bce_gain_mean_posthoc": (sum(r["data_bce_gain_posthoc"] for r in accepted) / len(accepted)) if accepted else None,
                "accepted_future_data_ber_gain_mean_posthoc": (sum(r["data_ber_gain_posthoc"] for r in accepted) / len(accepted)) if accepted else None,
                "harmful_future_data_fraction_posthoc": (sum(r["data_bce_gain_posthoc"] < 0 for r in accepted) / len(accepted)) if accepted else None})
    return {"schema_version": "subspace-delayed-validation-diagnostic-v1",
        "role": "development_read_only_diagnostic", "heldout_confirmation": False,
        "data_used_for_gate": False,
        "method": "Adapt候选在生成时冻结；从当前帧起暂存，用连续1/2/4/8帧合法验证Pilot作动作后累积判定；通过后才部署候选。Data仅在判定后离线评分下一帧。",
        "limits": "候选窗口重叠且共享会话状态；评价为机制筛查，不是独立样本统计。跨帧P状态改变，故候选在后续帧的作用依同一冻结子空间与当前物理特征重新计算。",
        "inputs": {"manifest_sha256": sha256(manifest_path),
            "training_summary_sha256": sha256(training_dir / "summary.json"),
            "trajectory_sha256": hashlib.sha256(trajectory_path.read_bytes()).hexdigest()},
        "by_snr_horizon": summaries, "windows": all_rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("诊断输出已存在，拒绝覆盖。")
    result = analyze(Path(args.manifest), Path(args.training_dir), Path(args.trajectory), device=args.device)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["by_snr_horizon"], ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
