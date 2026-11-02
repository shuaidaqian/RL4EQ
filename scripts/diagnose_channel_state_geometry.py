# -*- coding: utf-8 -*-
"""只读离线几何诊断；Data 导数仅解释，不生成或选择候选。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.func import functional_call
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pilot_transfer import _load_branch
from scripts.diagnose_channel_state_capacity import _StateOutput
from training.pilot_transfer import ChannelStateSignalObjective, context_logits, differentiable_adapt, frame_to_device, make_context


def real_vector(value):
    return torch.cat((value.real.reshape(-1), value.imag.reshape(-1)))


def state_jacobian(model, context):
    """对 head 参数求真实 CIR 映射的 Jacobian，不访问 Data 标签。"""
    named = list(model.peft.named_group_parameters({"channel_state_head"}))
    theta = torch.cat([value.detach().reshape(-1) for _, value in named])
    wrapper = _StateOutput(model)

    def state(parameters):
        fast, offset = {}, 0
        for name, value in named:
            fast["model." + name] = parameters[offset:offset + value.numel()].view_as(value)
            offset += value.numel()
        return functional_call(wrapper, fast, (context,))

    jacobian = torch.autograd.functional.jacobian(lambda x: real_vector(state(x)), theta, vectorize=True)
    return state(theta).detach(), jacobian.detach(), theta, state


def probe_loss(model, context, cir, role):
    """在状态分支输出点探测损失；不会回传到或改写模型参数。"""
    def substitute(module, args, original):
        torch.testing.assert_close(original.detach(), cir.detach(), atol=2e-6, rtol=2e-6)
        return cir

    hook = model.channel_state.register_forward_hook(substitute)
    try:
        if role == "adapt":
            return ChannelStateSignalObjective(model)(context)
        if role != "offline_data":
            raise ValueError("只支持 Adapt 和离线 Data 导数。")
        logits = context_logits(model, context)
        mask = context.frame.data_mask
        return F.binary_cross_entropy_with_logits(logits[mask].float(), context.frame.bits[mask].float())
    finally:
        hook.remove()


def _cosine(left, right):
    denominator = left.norm() * right.norm()
    return float(torch.dot(left, right) / denominator) if float(denominator) > 1e-18 else None


def analyze_geometry(model, context):
    cir, jacobian, _, _ = state_jacobian(model, context)
    probe = cir.clone().requires_grad_(True)
    adapt_loss = probe_loss(model, context, probe, "adapt")
    adapt_grad, = torch.autograd.grad(adapt_loss, probe)
    data_loss = probe_loss(model, context, probe, "offline_data")
    data_grad, = torch.autograd.grad(data_loss, probe)
    matrix = jacobian.double()
    u, singular, _ = torch.linalg.svd(matrix, full_matrices=False)
    rank = int((singular > singular[0] * 1e-5).sum()) if float(singular[0]) > 0 else 0
    basis = u[:, :rank]
    ga, gd = real_vector(adapt_grad).double(), real_vector(data_grad).double()
    pa, pd = basis.T @ ga, basis.T @ gd
    # 实际候选只读 Adapt；离线 Data 导数不传入任何候选函数。
    fast, update = differentiable_adapt(model, [context], learning_rate=.02, steps=3,
        max_delta_norm=.2, groups={"channel_state_head"}, objective="channel_state_signal", receiver_input="full_rx")
    fast = {name: value.detach() for name, value in fast.items()}
    with torch.no_grad():
        after = functional_call(_StateOutput(model), {"model." + name: value for name, value in fast.items()}, (context,))
        logits = context_logits(model, context, fast)
        mask = context.frame.data_mask
        after_loss = F.binary_cross_entropy_with_logits(logits[mask].float(), context.frame.bits[mask].float())
    step = real_vector(after - cir).double()
    return {"head_parameters": matrix.shape[1], "cir_real_dimensions": matrix.shape[0],
        "rank_relative_1e5": rank, "rank_relative_1e3": int((singular > singular[0] * 1e-3).sum()),
        "singular_values": singular.tolist(), "first_singular_energy_fraction": float(singular[0].square() / singular.square().sum().clamp_min(1e-30)),
        "adapt_gradient_norm": float(ga.norm()), "offline_data_gradient_norm": float(gd.norm()),
        "offline_data_reachable_energy_fraction": float(pd.square().sum() / gd.square().sum()) if float(gd.norm()) > 1e-12 else None,
        "adapt_reachable_energy_fraction": float(pa.square().sum() / ga.square().sum()) if float(ga.norm()) > 1e-12 else None,
        "full_gradient_cosine": _cosine(ga, gd), "reachable_gradient_cosine": _cosine(pa, pd),
        "actual_cir_step_norm": float(step.norm()), "actual_head_delta_norm": update["delta_norm"],
        "offline_data_step_directional_derivative": float(torch.dot(gd, step)),
        "offline_data_bce_before": float(data_loss.detach()), "offline_data_bce_after": float(after_loss),
        "data_gradient_used_for_action": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.output.exists() or config.get("offline_state_split") != "offline_train":
        raise ValueError("需要新的输出文件和 offline_train 配置。")
    model, payload = _load_branch(args.pretrained, config, "cpu")
    selected = {}
    for entry in payload["metrics"]["history"]:
        selected.setdefault((entry["snr_db"], entry["frame_index"]), entry)
    if set(selected) != {(snr, age) for snr in [0, 5, 10, 15] for age in [0, 127, 1023, 2047, 4095]}:
        raise ValueError("本诊断要求四 SNR 和五个预设训练年龄。")
    rows, source_hashes = [], {}
    started = time.monotonic()
    for (snr, age), entry in sorted(selected.items()):
        path = Path(config["pilot_transfer"]["offline_sample_cache_dir"]) / (entry["offline_sample_cache_key"] + ".pt")
        cached = torch.load(path, map_location="cpu", weights_only=False)
        digest = hashlib.sha256(json.dumps(cached["protocol"], sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        if digest != entry["offline_sample_cache_key"] or not cached["offline_supervision_only"]:
            raise ValueError("缓存协议指纹不匹配。")
        frame = frame_to_device(cached["frames"][0], "cpu")
        fingerprint = hashlib.sha256(frame.rx_symbols.numpy().tobytes()).hexdigest()
        if fingerprint != entry["offline_sample_input_sha256"] or frame.frame_index != age:
            raise ValueError("缓存波形或物理年龄与训练记录不一致。")
        context = make_context(frame, cached["cir"], cached["cfo"], cached["tail"], snr)
        rows.append({"snr_db": snr, "frame": age, "seed": entry["seed"], **analyze_geometry(model, context)})
        source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(json.dumps({"snr_db": snr, "frame": age, "complete": True}), flush=True)
    result = {"diagnostic_only": True, "main_aggregation_allowed": False, "state_split": "reused_offline_training_samples",
        "checkpoint": args.pretrained, "checkpoint_sha256": hashlib.sha256(Path(args.pretrained).read_bytes()).hexdigest(),
        "data_gradient_used_for_action": False, "config": config, "elapsed_seconds": time.monotonic() - started,
        "source_files": source_hashes, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
