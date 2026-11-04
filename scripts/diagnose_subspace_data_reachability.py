# -*- coding: utf-8 -*-
"""离线隔离诊断：检查已学检测子空间的Data标签可达性上界。"""

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


def optimize_data_coefficients(model, observation, labels, mask, initial, *, steps=32, learning_rate=0.02):
    """用离线Data标签优化副本系数；只作表示可达性诊断，不是部署算法。"""
    if steps < 1 or learning_rate <= 0 or not bool(mask.any()):
        raise ValueError("Data可达性诊断参数非法。")
    basis = model.basis(observation).detach()
    base = observation.base_logits.detach()
    labels = labels.float()
    coefficient = initial.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([coefficient], lr=learning_rate)

    def score(value):
        logits = base + basis @ value
        loss = F.binary_cross_entropy_with_logits(logits[mask].float(), labels[mask])
        errors = ((logits[mask] >= 0) != labels[mask].bool()).sum()
        return float(loss.detach()), int(errors)

    initial_loss, initial_errors = score(coefficient)
    best = coefficient.detach().clone()
    best_loss, best_errors, best_step = initial_loss, initial_errors, 0
    trace = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = base + basis @ coefficient
        loss = F.binary_cross_entropy_with_logits(logits[mask].float(), labels[mask])
        loss.backward()
        if coefficient.grad is None or not bool(torch.isfinite(coefficient.grad).all()):
            raise FloatingPointError("Data可达性诊断梯度非有限。")
        optimizer.step()
        current_loss, current_errors = score(coefficient)
        trace.append(current_loss)
        if current_loss < best_loss:
            best, best_loss, best_errors, best_step = coefficient.detach().clone(), current_loss, current_errors, step
    return {"initial_bce": initial_loss, "best_bce": best_loss,
        "initial_errors": initial_errors, "best_errors": best_errors,
        "best_step": best_step, "coefficient_delta_norm": float((best - initial).norm().detach()),
        "bce_gain_oracle": initial_loss - best_loss,
        "ber_gain_oracle": (initial_errors - best_errors) / int(mask.sum()),
        "trace": trace}


def analyze(manifest_path, training_dir, trajectory_path, *, device="cpu", steps=32):
    spec = read_manifest(manifest_path)
    sessions = torch.load(training_dir / "evaluation_features.pt", map_location=device, weights_only=False)
    payload = torch.load(training_dir / "B_post_adapt.pt", map_location=device, weights_only=False)
    model = make_subspace(int(payload["dimension"]), int(spec["initialization_seed"]), device, payload["architecture"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    with gzip.open(trajectory_path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    by_key = {(int(r["seed"]), int(r["snr_db"]), int(r["frame"])): r
              for r in rows if r.get("training") == "B_post_adapt"}
    results = []
    for session in sessions:
        seed, snr = int(session["seed"]), int(session["snr_db"])
        for frame in session["frames"]:
            identity = frame["identity"]
            row = by_key[(seed, snr, int(identity["frame"]))]
            if identity.get("rx_sha256") != row.get("rx_sha256"):
                raise ValueError("Data可达性诊断的缓存与轨迹波形不匹配。")
            initial = torch.tensor(row["coefficient_values"], device=device, dtype=model.coefficients.dtype)
            offline = frame["offline"]
            result = optimize_data_coefficients(model, frame["observation"], offline["bits"],
                offline["data_mask"], initial, steps=steps)
            results.append({"seed": seed, "snr_db": snr, "frame": int(identity["frame"]), **result,
                "data_labels_used_only_for_isolated_offline_diagnostic": True})
    summaries = []
    for snr in sorted({r["snr_db"] for r in results}):
        subset = [r for r in results if r["snr_db"] == snr]
        summaries.append({"snr_db": snr, "frames": len(subset),
            "mean_oracle_bce_gain": sum(r["bce_gain_oracle"] for r in subset) / len(subset),
            "median_oracle_bce_gain": sorted(r["bce_gain_oracle"] for r in subset)[len(subset)//2],
            "fraction_bce_improved": sum(r["bce_gain_oracle"] > 1e-8 for r in subset) / len(subset),
            "fraction_ber_improved": sum(r["ber_gain_oracle"] > 0 for r in subset) / len(subset),
            "mean_ber_gain_oracle": sum(r["ber_gain_oracle"] for r in subset) / len(subset),
            "mean_best_step": sum(r["best_step"] for r in subset) / len(subset)})
    return {"schema_version": "subspace-data-reachability-diagnostic-v1",
        "role": "isolated_offline_development_diagnostic", "deployable": False,
        "online_control_used_data": False, "heldout_confirmation": False,
        "method": f"冻结在线来源模型与子空间基，仅对每帧副本的{model.dimension}维系数用该帧Data BCE做{steps}步Adam优化；Data不返回任何在线状态。",
        "interpretation_limit": "这是局部Data可达性诊断，不是接收机性能上界，也不是可部署方法。下降只能说明子空间局部存在能改善Data BCE的方向，不能证明合法Pilot能识别该方向。",
        "inputs": {"manifest_sha256": sha256(manifest_path),
            "training_summary_sha256": sha256(training_dir / "summary.json"),
            "trajectory_sha256": hashlib.sha256(trajectory_path.read_bytes()).hexdigest()},
        "by_snr": summaries, "frames": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("诊断输出已存在，拒绝覆盖。")
    result = analyze(Path(args.manifest), Path(args.training_dir), Path(args.trajectory),
                     device=args.device, steps=args.steps)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["by_snr"], ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
