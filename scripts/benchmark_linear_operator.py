# -*- coding: utf-8 -*-
"""核验长记忆线性算子 FFT 后端的精度、硬判决与实际耗时。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.linear_operator import LinearChannelOperator
from evaluation.pilot_transfer import _load_branch
from training.pilot_transfer import context_logits, make_context, make_episode


def _time_call(function, device, repetitions):
    for _ in range(2):
        function()
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repetitions):
        result = function()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / repetitions, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    checkpoint = Path(args.pretrained)
    model, _ = _load_branch(checkpoint, config, args.device)
    _, cir, cfo, tail, frames = make_episode(
        config, model, seed=41997, snr=10, frames=1,
        state_split="heldout_edge", device=args.device,
    )
    context = make_context(frames[0], cir, cfo, tail, 10)
    x = frames[0].rx_symbols
    timings, outputs = {}, {}
    previous = os.environ.get("RL4EQ_LINEAR_OPERATOR_BACKEND")
    try:
        with torch.no_grad():
            for backend in ("direct", "fft"):
                os.environ["RL4EQ_LINEAR_OPERATOR_BACKEND"] = backend
                operator = LinearChannelOperator(model.config.frame_len, model.config.max_delay, backend=backend)
                operator_ms, _ = _time_call(lambda: operator.adjoint(operator.forward(x, cir, tail), cir), args.device, 20)
                model_ms, logits = _time_call(lambda: context_logits(model, context), args.device, 3)
                timings[backend] = {"operator_pair_ms": operator_ms, "neural_frame_ms": model_ms}
                outputs[backend] = logits.detach().cpu()
    finally:
        if previous is None:
            os.environ.pop("RL4EQ_LINEAR_OPERATOR_BACKEND", None)
        else:
            os.environ["RL4EQ_LINEAR_OPERATOR_BACKEND"] = previous
    delta = (outputs["direct"] - outputs["fft"]).abs()
    data_mask = frames[0].data_mask.detach().cpu()
    result = {"device": args.device,
              "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "seed": 41997, "snr_db": 10, "timings": timings,
              "neural_frame_speedup": timings["direct"]["neural_frame_ms"] / timings["fft"]["neural_frame_ms"],
              "logit_max_abs_difference": float(delta.max()),
              "logit_mean_abs_difference": float(delta.mean()),
              "data_hard_decision_disagreements": int(((outputs["direct"][data_mask] >= 0) != (outputs["fft"][data_mask] >= 0)).sum())}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
