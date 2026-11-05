# -*- coding: utf-8 -*-
"""用冻结的子空间checkpoint做隔离开发会话复核。"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.subspace_development import (
    DeterministicSettings,
    capture_sessions,
    evaluate_subspaces,
    make_subspace,
    read_manifest,
    sha256,
    write_json,
)


def load_replay_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "subspace-frozen-checkpoint-replay-v1":
        raise ValueError("只接受固定版本的冻结checkpoint复核配置。")
    if config.get("snrs_db") != [0, 5, 10, 15] or config.get("channel_changes") is not False:
        raise ValueError("复核必须保留全部四个SNR且不得修改信道。")
    if config.get("heldout_confirmation") is not False:
        raise ValueError("该入口只能运行开发复核，不能冒充正式确认。")
    if config.get("evaluation_seeds") != [121800, 121801, 121802]:
        raise ValueError("v27 checkpoint复核seed必须预先固定为121800/121801/121802。")
    if config.get("frames_per_seed_snr") != 32:
        raise ValueError("复核帧预算必须固定为每seed/SNR 32帧。")
    manifest_path = ROOT / config["source_manifest"]
    if sha256(manifest_path) != config["source_manifest_sha256"]:
        raise ValueError("来源开发清单指纹不匹配。")
    manifest = read_manifest(manifest_path)
    if manifest.get("schema_version") != "fixed-channel-subspace-development-v27":
        raise ValueError("来源checkpoint必须来自v27清单。")
    occupied = {seed for role in manifest["data_roles"].values() for seed in role["seeds"]}
    if occupied.intersection(config["evaluation_seeds"]):
        raise ValueError("复核seed与v27审计、训练、选择或评估seed重叠。")
    for name in ("A_supervised", "B_post_adapt"):
        item = config["checkpoints"].get(name, {})
        checkpoint_path = ROOT / item.get("path", "")
        if not checkpoint_path.is_file() or sha256(checkpoint_path) != item.get("sha256"):
            raise ValueError(f"{name}冻结checkpoint缺失或指纹不匹配。")
    return config


def replay(config_path: Path, output_dir: Path, device: str = "cpu") -> dict:
    config = load_replay_config(config_path)
    target = output_dir if output_dir.is_absolute() else ROOT / output_dir
    if target.exists():
        raise ValueError("输出目录已存在，禁止覆盖。")
    target.mkdir(parents=True)
    source_path = ROOT / config["source_manifest"]
    spec = read_manifest(source_path)
    spec["data_roles"]["evaluation"] = {
        "seeds": config["evaluation_seeds"],
        "frames": config["frames_per_seed_snr"],
    }
    models = {}
    for name in ("A_supervised", "B_post_adapt"):
        path = ROOT / config["checkpoints"][name]["path"]
        payload = torch.load(path, map_location=device, weights_only=False)
        model = make_subspace(payload["dimension"], spec["initialization_seed"], device,
            payload["architecture"],
            learned_gradient_transport=payload.get("learned_gradient_transport", False),
            learned_context_update=payload.get("learned_context_update", False))
        model.use_context_update = payload.get("use_context_update", False)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        models[name] = model

    started = time.perf_counter()
    evaluation, _ = capture_sessions(spec, "evaluation", device)
    rows, per_snr = evaluate_subspaces(models, evaluation, DeterministicSettings(**spec["update"]), device)
    raw = target / "frame_metrics.jsonl"
    raw.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8")
    with raw.open("rb") as source, gzip.open(target / "frame_metrics.jsonl.gz", "wb", compresslevel=9) as compressed:
        while chunk := source.read(1 << 20):
            compressed.write(chunk)
    raw.unlink()
    result = {
        "role": "development_replication",
        "core_confirmed": False,
        "confirmation_ready": False,
        "checkpoint_source": "v27_frozen_A_B",
        "evaluation_seeds": config["evaluation_seeds"],
        "frames_per_seed_snr": config["frames_per_seed_snr"],
        "snrs_db": config["snrs_db"],
        "per_snr": per_snr,
        "records": len(rows),
        "wall_seconds": time.perf_counter() - started,
        "replay_config_sha256": sha256(config_path),
        "source_manifest_sha256": sha256(source_path),
        "checkpoint_sha256": {name: item["sha256"] for name, item in config["checkpoints"].items()},
        "metrics_sha256": sha256(target / "frame_metrics.jsonl.gz"),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    write_json(target / "summary.json", result)
    (target / "config.json").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结v27 A/B checkpoint的新开发会话复核")
    parser.add_argument("--config", default="configs/audits/subspace_v27_checkpoint_replay_v1.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    result = replay(ROOT / args.config, Path(args.output_dir), args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
