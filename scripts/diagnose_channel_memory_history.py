# -*- coding: utf-8 -*-
"""只在最后已接收帧评分的复记忆历史筛查，不作独立泛化结论。"""

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

from agent.neural_channel_state import known_pilot_equations
from evaluation.pilot_transfer import _load_branch, _metrics
from scripts.diagnose_channel_state_windows import window_candidates
from training.pilot_transfer import make_context, frame_to_device


def pilot_spectrum(contexts):
    """仅使用合法Adapt设计，报告代数可辨认性，不读取任何Data标签或波形。"""
    designs = []
    for context in contexts:
        frame = context.frame
        symbols = frame.receiver_view().adapt_symbols
        design, _ = known_pilot_equations(torch.zeros_like(symbols), symbols, frame.adapt_mask,
            frame.adapt_mask, context.condition.complex_cir.shape[-1] - 1)
        designs.append(design.detach().cpu().to(torch.complex128))
    matrix = torch.cat(designs)
    singular = torch.linalg.svdvals(matrix)
    rank = int((singular > singular[0] * 1e-10).sum())
    return {"rows": matrix.shape[0], "complex_columns": matrix.shape[1], "complex_rank": rank,
            "complex_nullity": matrix.shape[1] - rank,
            "condition_if_full_rank": float(singular[0] / singular[-1]) if rank == matrix.shape[1] else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.output.exists() or config.get("offline_state_split") != "offline_train":
        raise ValueError("需要新输出文件和训练分布。")
    model, payload = _load_branch(args.pretrained, config, args.device)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    selected = {}
    for row in payload["metrics"]["history"]:
        selected.setdefault((row["snr_db"], row["frame_index"]), row)
    if len(selected) != 20:
        raise ValueError("固定筛查需要四SNR、五年龄的原训练样本。")
    rows, spectra, sources = [], [], {}
    started = time.monotonic()
    for (snr, age), entry in sorted(selected.items()):
        path = Path(config["pilot_transfer"]["offline_sample_cache_dir"]) / (entry["offline_sample_cache_key"] + ".pt")
        cached = torch.load(path, map_location="cpu", weights_only=False)
        digest = hashlib.sha256(json.dumps(cached["protocol"], sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        if (digest != entry["offline_sample_cache_key"] or cached["protocol"]["config"] != config
                or not cached["offline_supervision_only"]
                or [frame.frame_index for frame in cached["frames"]] != [age, age + 8]
                or hashlib.sha256(cached["frames"][0].rx_symbols.numpy().tobytes()).hexdigest() != entry["offline_sample_input_sha256"]):
            raise ValueError("缓存协议、波形或物理年龄不一致。")
        contexts = [make_context(frame_to_device(frame, args.device), cached["cir"].to(args.device),
                    cached["cfo"], cached["tail"].to(args.device), snr) for frame in cached["frames"]]
        frame = contexts[-1].frame
        spectra.append({"snr_db": snr, "frame": frame.frame_index,
                        "last_only": pilot_spectrum(contexts[-1:]), "pooled": pilot_spectrum(contexts)})
        for objective in ["channel_state_signal", "channel_state_map"]:
            frozen, candidates, decisions = window_candidates(model, contexts, windows=(1, 2),
                groups=("channel_state_memory",), objective=objective, max_delta_norm=.8, require_consecutive=False)
            # 同一最后帧的Data仅用于本目标全部候选及guard完成后的评分。
            with torch.no_grad():
                ber0, bce0 = _metrics(frozen, frame, frame.data_mask)
                for decision in decisions:
                    ber1, bce1 = _metrics(candidates[decision["adapt_window"]], frame, frame.data_mask)
                    rows.append({"snr_db": snr, "seed": entry["seed"], "frame": frame.frame_index,
                        "received_frame_indices": [context.frame.frame_index for context in contexts],
                        "objective": objective, **decision, "ber_frozen": ber0, "bce_frozen": bce0,
                        "ber_candidate": ber1, "bce_candidate": bce1,
                        "ber_committed": ber1 if decision["accepted"] else ber0, "data_labels_used_for_control": False})
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(json.dumps({"snr_db": snr, "frame": frame.frame_index, "complete": True}), flush=True)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    summary = []
    for snr in config["main_snrs"]:
        for objective in ["channel_state_signal", "channel_state_map"]:
            for aw in [1, 2]:
                for rw in [1, 2]:
                    v = [r for r in rows if (r["snr_db"], r["objective"], r["adapt_window"], r["reward_window"]) == (snr, objective, aw, rw)]
                    summary.append({"snr_db": snr, "objective": objective, "adapt_window": aw, "reward_window": rw,
                        "contexts": len(v), "accepted": sum(r["accepted"] for r in v),
                        "harmful_acceptances": sum(r["accepted"] and r["ber_candidate"] > r["ber_frozen"] for r in v),
                        "candidate_gain": float(np.mean([r["ber_frozen"] - r["ber_candidate"] for r in v])),
                        "committed_gain": float(np.mean([r["ber_frozen"] - r["ber_committed"] for r in v]))})
    result = {"diagnostic_only": True, "main_aggregation_allowed": False, "state_split": "reused_offline_training_samples",
        "checkpoint": args.pretrained, "checkpoint_sha256": hashlib.sha256(Path(args.pretrained).read_bytes()).hexdigest(),
        "config": config, "source_files": sources, "elapsed_seconds": time.monotonic() - started,
        "all_model_parameters_unchanged": True, "data_labels_used_for_control": False,
        "spectra": spectra, "per_snr": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
