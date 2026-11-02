# -*- coding: utf-8 -*-
"""在四臂实验的相同波形上评分原始离线源模型，防止以 Frozen 退化制造适配差值。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.unfolded_equalizer import UnfoldedConfig
from evaluation.pilot_transfer import _load_branch, _metrics, paired_interval
from training.pilot_transfer import context_logits, make_context, make_episode, next_pilot_tail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("源模型参考输出已存在，禁止覆盖。")
    summary_path = args.comparison_dir / "summary.json"
    rows_path = args.comparison_dir / "frame_metrics.jsonl"
    comparison = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = comparison["manifest"]
    if not manifest["diagnostic_only"] or manifest["state_split"] != "heldout_edge":
        raise ValueError("仅支持已完成的 heldout_edge 四臂诊断。")
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    paired = {(r["snr_db"], r["seed"], r["frame"]): r for r in rows if r["branch"] == "baseline"}
    assert len(paired) == len(manifest["snrs"]) * len(manifest["seeds"]) * manifest["frames"]
    payload = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    source_config = UnfoldedConfig.from_dict(payload["model_config"]).to_dict()
    target_config = UnfoldedConfig.from_dict(manifest["config"]["model"]).to_dict()
    differences = {name for name in source_config if source_config[name] != target_config[name]}
    if differences not in (set(), {"enable_channel_state_branch"}):
        raise ValueError("本参考只允许相同模型配置，或源模型缺少新增神经信道状态分支。")
    if differences and (source_config["enable_channel_state_branch"] or not target_config["enable_channel_state_branch"]):
        raise ValueError("源模型和实验模型的状态支路关系错误。")
    config = {**manifest["config"], "model": source_config}
    model, payload = _load_branch(args.pretrained, config, args.device)
    training_seeds = {r["seed"] for r in payload.get("metrics", {}).get("history", [])}
    if training_seeds.intersection(manifest["seeds"]):
        raise ValueError("参考测试 seed 与源训练重叠。")
    output_rows = []
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.monotonic()
    with torch.no_grad():
        for snr in manifest["snrs"]:
            for seed in manifest["seeds"]:
                _, cir, cfo, tail, frames = make_episode(config, model, seed=seed, snr=snr,
                    frames=manifest.get("evaluation_history_frames", 0) + manifest["frames"],
                    state_split="heldout_edge", device=args.device)
                history_frames = manifest.get("evaluation_history_frames", 0)
                if history_frames and any(bool(frame.adapt_mask[-tail.numel():].any()) for frame in frames[:history_frames]):
                    raise ValueError("带历史的源评分目前要求前缀不覆盖帧尾。")
                frames = frames[history_frames:]
                for index, frame in enumerate(frames):
                    fingerprint = hashlib.sha256(frame.rx_symbols.cpu().numpy().tobytes()).hexdigest()
                    assert fingerprint == paired[(snr, seed, index)]["input_sha256"]
                    context = make_context(frame, cir, cfo, tail, snr)
                    ber, bce = _metrics(context_logits(model, context), frame, frame.data_mask)
                    output_rows.append({"snr_db": snr, "seed": seed, "frame": index,
                                        "ber_source": ber, "bce_source": bce, "input_sha256": fingerprint})
                    tail = next_pilot_tail(model, context, alpha=float(config.get("tail_update_alpha", .5)))
            print(json.dumps({"snr_db": snr, "complete": True}), flush=True)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    reference = {(r["snr_db"], r["seed"], r["frame"]): r for r in output_rows}
    statistics = []
    for snr in manifest["snrs"]:
        item = {"snr_db": snr, "ber_source": float(np.mean([r["ber_source"] for r in output_rows if r["snr_db"] == snr]))}
        for branch in ["baseline", "post_adapt"]:
            for state in ["frozen", "online"]:
                item[branch + "_" + state + "_gain_over_source"] = paired_interval([
                    np.mean([reference[(snr, seed, r["frame"])]["ber_source"] - r["ber_" + state]
                        for r in rows if (r["snr_db"], r["seed"], r["branch"]) == (snr, seed, branch)])
                    for seed in manifest["seeds"]])
        statistics.append(item)
    result = {"diagnostic_only": True, "main_aggregation_allowed": False, "waveforms_matched": True,
        "source_checkpoint": str(args.pretrained), "source_sha256": hashlib.sha256(args.pretrained.read_bytes()).hexdigest(),
        "comparison_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [summary_path, rows_path]},
        "elapsed_seconds": elapsed, "per_snr": statistics, "rows": output_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"elapsed_seconds": elapsed, "per_snr": statistics}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
