# -*- coding: utf-8 -*-
"""复用已审计训练缓存筛查复记忆容量；不产生留出成功证据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pilot_transfer import _load_branch, _metrics
from scripts.diagnose_channel_state_capacity import candidates_and_guards, MEMORY_ACTIONS as ACTIONS
from training.pilot_transfer import PilotTransferTrainer, context_logits, differentiable_adapt, frame_to_device, make_context


def offline_scale_gradients(model, contexts):
    """只读离线当前/未来标签导数，不生成更新、不参与任何在线控制。"""
    parameters = model.peft.parameters({"channel_state_memory_scale"})
    if not parameters or not all(parameter.requires_grad for parameter in parameters) or len(contexts) != 2:
        raise ValueError("需要启用梯度的尺度参数及当前/未来两帧。")
    rows = []
    with sdpa_kernel(SDPBackend.MATH):
        base = torch.stack([F.binary_cross_entropy_with_logits(context_logits(model, context).float(),
                            context.frame.bits.float()) for context in contexts]).mean()
        base_grad = torch.cat([value.flatten() for value in torch.autograd.grad(base, parameters)])
        for objective in ("channel_state_signal", "channel_state_map"):
            fast, _ = differentiable_adapt(model, [contexts[0]], learning_rate=.02, steps=3,
                max_delta_norm=.8, groups={"channel_state_memory"}, objective=objective,
                receiver_input="full_rx", create_graph=True)
            losses = []
            for context in contexts:
                logits, mask = context_logits(model, context, fast), context.frame.data_mask
                losses.append(F.binary_cross_entropy_with_logits(logits[mask].float(), context.frame.bits[mask].float()))
            gradients = [torch.cat([value.flatten() for value in torch.autograd.grad(loss, parameters, retain_graph=index == 0)])
                         for index, loss in enumerate(losses)]
            if not all(bool(torch.isfinite(value).all()) for value in [base_grad, *gradients]):
                raise FloatingPointError("离线尺度导数非有限。")
            mean = torch.stack(gradients).mean(dim=0)
            rows.append({"objective": objective, "base_gradient_norm": float(base_grad.norm()),
                "post_gradient_norm": float(mean.norm()), "current_gradient_norm": float(gradients[0].norm()),
                "future_gradient_norm": float(gradients[1].norm()),
                "current_future_cosine": float(F.cosine_similarity(gradients[0], gradients[1], dim=0)),
                "post_gradient": mean.detach().cpu().tolist(), "data_gradients_used_for_control": False})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--meta-gradient", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("禁止覆盖已有诊断。")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    source_config = json.loads(Path(args.source_config).read_text(encoding="utf-8"))
    source, payload = _load_branch(args.source, source_config, args.device)
    trainer = PilotTransferTrainer(config, args.device)
    trainer.load_resume(args.source)
    model = trainer.model.eval()
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    selected = {}
    for entry in payload["metrics"]["history"]:
        selected.setdefault((entry["snr_db"], entry["frame_index"]), entry)
    if set(selected) != {(snr, age) for snr in [0, 5, 10, 15] for age in [0, 127, 1023, 2047, 4095]}:
        raise ValueError("要求四 SNR、五年龄的固定训练样本。")
    rows, sources, gradient_rows = [], {}, []
    started = time.monotonic()
    for (snr, age), entry in sorted(selected.items()):
        path = Path(source_config["pilot_transfer"]["offline_sample_cache_dir"]) / (entry["offline_sample_cache_key"] + ".pt")
        cached = torch.load(path, map_location="cpu", weights_only=False)
        digest = hashlib.sha256(json.dumps(cached["protocol"], sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        frame = cached["frames"][0]
        fingerprint = hashlib.sha256(frame.rx_symbols.numpy().tobytes()).hexdigest()
        if (digest != entry["offline_sample_cache_key"] or not cached["offline_supervision_only"]
                or fingerprint != entry["offline_sample_input_sha256"] or frame.frame_index != age):
            raise ValueError("训练缓存协议、波形或年龄审计失败。")
        context = make_context(frame_to_device(frame, args.device), cached["cir"].to(args.device),
            cached["cfo"], cached["tail"].to(args.device), snr)
        with torch.no_grad():
            torch.testing.assert_close(context_logits(source, context), context_logits(model, context), atol=0, rtol=0)
        frozen, candidates, decisions = candidates_and_guards(model, context,
            objectives=("channel_state_signal", "channel_state_map"), actions=ACTIONS, groups=("channel_state_memory",))
        # 所有候选和验收完成后才读取 Data 标签；不进行事后动作选择。
        with torch.no_grad():
            ber0, bce0 = _metrics(frozen, context.frame, context.frame.data_mask)
            for decision in decisions:
                ber1, bce1 = _metrics(candidates[decision["action"]], context.frame, context.frame.data_mask)
                rows.append({"snr_db": snr, "frame": age, "seed": entry["seed"], **decision,
                    "ber_frozen": ber0, "ber_candidate": ber1, "ber_committed": ber1 if decision["accepted"] else ber0,
                    "bce_frozen": bce0, "bce_candidate": bce1, "data_labels_used_for_control": False})
        if args.meta_gradient:
            if [frame.frame_index for frame in cached["frames"]] != [age, age + 8]:
                raise ValueError("尺度导数要求已审计的当前/未来 +8 缓存。")
            future = make_context(frame_to_device(cached["frames"][1], args.device), cached["cir"].to(args.device),
                                  cached["cfo"], cached["tail"].to(args.device), snr)
            gradient_rows.extend({"snr_db": snr, "frame": age, "seed": entry["seed"], **item}
                                 for item in offline_scale_gradients(model, [context, future]))
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(json.dumps({"snr_db": snr, "frame": age, "complete": True}), flush=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    result = {"diagnostic_only": True, "main_aggregation_allowed": False,
        "state_split": "reused_offline_training_samples", "source_checkpoint": args.source,
        "source_sha256": hashlib.sha256(Path(args.source).read_bytes()).hexdigest(),
        "config": config, "source_files": sources, "frozen_exactly_preserved": True,
        "data_labels_used_for_control": False, "all_base_parameters_unchanged": True,
        "elapsed_seconds": time.monotonic() - started, "rows": rows, "offline_gradient_rows": gradient_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
