# -*- coding: utf-8 -*-
"""用模型真实前向重算子空间开发评估，支持非线性Pilot上下文Adapter。"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.detection_subspace import DetectionSubspace
from evaluation.subspace_development import evaluate_subspaces, read_manifest, write_json
from training.deterministic_adaptation import DeterministicSettings


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recompute(manifest_path: Path, feature_cache: Path, checkpoint_a: Path,
              checkpoint_b: Path, output_dir: Path, device: str = "cpu") -> dict:
    if output_dir.exists():
        raise ValueError("重算输出目录已存在，禁止覆盖。")
    spec = read_manifest(manifest_path)
    sessions = torch.load(feature_cache, map_location=device, weights_only=False)
    allowed = set(spec["data_roles"]["evaluation"]["seeds"])
    if not sessions or any(int(session["seed"]) not in allowed for session in sessions):
        raise ValueError("评估缓存与清单的evaluation seed不匹配。")
    models = {}
    for name, checkpoint in (("A_supervised", checkpoint_a), ("B_post_adapt", checkpoint_b)):
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model = DetectionSubspace(int(payload["dimension"]), architecture=payload["architecture"],
            learned_gradient_transport=payload.get("learned_gradient_transport", False),
            learned_context_update=payload.get("learned_context_update", False)).to(device)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.use_context_update = payload.get("use_context_update", payload.get("learned_context_update", False))
        model.eval()
        models[name] = model
    rows, per_snr = evaluate_subspaces(models, sessions, DeterministicSettings(**spec["update"]), device)
    output_dir.mkdir(parents=True)
    raw_path = output_dir / "frame_metrics.jsonl"
    raw_path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        for row in rows), encoding="utf-8")
    compressed_path = output_dir / "frame_metrics.jsonl.gz"
    with raw_path.open("rb") as source, gzip.open(compressed_path, "wb", compresslevel=9) as target:
        target.write(source.read())
    raw_path.unlink()
    result = {"role": "corrected_development_actual_model_forward",
        "core_confirmed": False, "confirmation_ready": False,
        "scoring_forward": "DetectionSubspace.logits(observation, coefficients)",
        "linearized_basis_scoring_used": False,
        "interpretation_limit": "只修正开发评估的非线性模型前向评分；不是独立确认，也不改变训练、checkpoint或在线控制",
        "dimension": next(iter(models.values())).dimension,
        "manifest_sha256": sha256(manifest_path),
        "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in {
            "manifest": manifest_path, "feature_cache": feature_cache,
            "A_checkpoint": checkpoint_a, "B_checkpoint": checkpoint_b}.items()},
        "per_snr": per_snr, "records": len(rows),
        "frame_metrics_sha256": sha256(compressed_path)}
    write_json(output_dir / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = recompute(Path(args.manifest), Path(args.feature_cache), Path(args.checkpoint_a),
        Path(args.checkpoint_b), Path(args.output_dir), args.device)
    print(json.dumps(result["per_snr"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
