# -*- coding: utf-8 -*-
"""核查固定尺度控制与离线尺度学习的权重、样本及零 Frozen 契约。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def audit(source_path, control_path, post_path, config):
    paths = [Path(path) for path in [source_path, control_path, post_path]]
    source, control, post = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    source_sha = hashlib.sha256(paths[0].read_bytes()).hexdigest()
    base = source["state_dict"]
    settings = config["pilot_transfer"]
    signatures = []
    for payload, objective in [(control, "baseline"), (post, "post_adapt")]:
        metrics, state = payload["metrics"], payload["state_dict"]
        if (metrics["objective"] != objective or metrics["source_checkpoint"]["sha256"] != source_sha
                or metrics["settings"] != settings or metrics["outer_trainable_parameter_count"] != 34
                or metrics["initialized_modules"] != ["channel_state_memory"]):
            raise ValueError("来源、训练设置、目标或尺度参数预算不匹配。")
        additions = {key for key in state if key.startswith(("channel_state.memory.", "channel_state.memory_scale."))}
        if set(state) != set(base) | additions or len(additions) != 3:
            raise ValueError("只允许新增复数记忆及尺度权重。")
        for key, value in base.items():
            torch.testing.assert_close(value, state[key], atol=0, rtol=0)
        for key in additions:
            if (key.startswith("channel_state.memory.") or objective == "baseline") and bool(state[key].ne(0).any()):
                raise ValueError("离线记忆必须为零，固定尺度控制的尺度也必须为零。")
        signatures.append([(row["seed"], row["snr_db"], row["frame_index"], row["post_target_frame_indices"],
                            row["offline_sample_cache_key"], row["offline_sample_input_sha256"], row["channel_state"])
                           for row in metrics["history"]])
    if signatures[0] != signatures[1] or len(signatures[0]) != 64:
        raise ValueError("固定预算64个样本的A/B输入不匹配。")
    expected = [(settings["seed"] + index, config["main_snrs"][index % len(config["main_snrs"])],
                 settings["history_frame_choices"][index % len(settings["history_frame_choices"])]) for index in range(64)]
    if [tuple(row[:3]) for row in signatures[0]] != expected:
        raise ValueError("样本 seed、SNR、物理年龄不符合预定循环。")
    cache_hashes = {}
    for row in control["metrics"]["history"]:
        path = Path(settings["offline_sample_cache_dir"]) / (row["offline_sample_cache_key"] + ".pt")
        cached = torch.load(path, map_location="cpu", weights_only=False)
        digest = hashlib.sha256(json.dumps(cached["protocol"], sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        actual = hashlib.sha256(cached["frames"][0].rx_symbols.numpy().tobytes()).hexdigest()
        if (digest != row["offline_sample_cache_key"] or actual != row["offline_sample_input_sha256"]
                or cached["protocol"]["config"] != config or not cached["offline_supervision_only"]
                or [frame.frame_index for frame in cached["frames"]] != row["post_target_frame_indices"]):
            raise ValueError("缓存、协议或当前/未来帧指纹不匹配。")
        cache_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if len(cache_hashes) != 64:
        raise ValueError("训练样本不是64个唯一上下文。")
    if not any(bool(value.ne(0).any()) for key, value in post["state_dict"].items() if key.startswith("channel_state.memory_scale.")):
        raise ValueError("更新后目标没有改变尺度参数。")
    return {"diagnostic_only": True, "source_and_old_parameters_unchanged": True,
        "zero_memory_in_both_checkpoints": True, "control_scale_zero": True, "post_scale_changed": True,
        "frozen_equality_by_zero_residual_and_identical_base": True, "matched_unique_samples": 64,
        "checkpoints": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "cache_files": cache_hashes,
        "training": {name: {key: payload["metrics"][key] for key in ["elapsed_seconds", "offline_sample_cache_hits",
            "offline_sample_cache_misses", "outer_trainable_parameter_count"]}
            for name, payload in [("fixed_scale", control), ("post_adapt", post)]}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--post", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("禁止覆盖已有审计。")
    result = audit(args.source, args.control, args.post, json.loads(Path(args.config).read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "cache_files"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
