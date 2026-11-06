# -*- coding: utf-8 -*-
"""开发 smoke：不确定性条件化 FiLM + ELBO 跨帧更新。

- 两臂共享 acquisition CIR + dd_soft tail；Frozen 的 FiLM 为零（严格等价原模型）。
- Online 在 head 之前对 64 维隐藏特征做 FiLM：
  h' = h * (1 + tanh(ctx @ Ws + bs)) + (ctx @ Wv + bv)
  FiLM 参数由当前帧 ELBO 梯度更新（Adam + 信赖域 + Reward BCE 门控），作用于下一帧。
- ctx 是 Data 置信度与 ELBO 残差统计（合法），在 base 模型上计算并 stop-gradient。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pilot_transfer import _load_branch
from training.data_aided import data_aided_elbo
from training.pilot_transfer import context_logits, make_context, make_episode, next_pilot_tail


def _register_film(model):
    state = {"ctx": None, "ws": None, "bs": None, "wv": None, "bv": None}

    def hook(module, inputs):
        hidden = inputs[0]
        if state["ctx"] is None:
            return None
        ctx = state["ctx"]
        scale = torch.tanh(ctx @ state["ws"].transpose(0, 1) + state["bs"])
        shift = ctx @ state["wv"].transpose(0, 1) + state["bv"]
        return (hidden * (1.0 + scale) + shift,)

    model.head.register_forward_pre_hook(hook)
    return state


def _zero_params(d_model, context_dim, device):
    return {
        "ws": torch.zeros(d_model, context_dim, device=device),
        "bs": torch.zeros(d_model, device=device),
        "wv": torch.zeros(d_model, context_dim, device=device),
        "bv": torch.zeros(d_model, device=device),
    }


def _set_state(state, params, ctx):
    state["ctx"] = ctx
    state["ws"] = params["ws"]
    state["bs"] = params["bs"]
    state["wv"] = params["wv"]
    state["bv"] = params["bv"]


def _context(model, context, frame, state, zero):
    _set_state(state, zero, None)
    state["ctx"] = None
    with torch.no_grad():
        logits = context_logits(model, context, None, visible="data")
        elbo = data_aided_elbo(model, context, None, variance_weight=1.0, anchor_weight=0.0)
        mask = frame.data_mask.bool()
        soft = torch.tanh(logits / 2.0)
        conf = soft[mask].abs()
        ctx = torch.stack([
            conf.mean(), (conf > 0.5).float().mean(), (1.0 - conf.square()).mean(),
            elbo["reconstruction"], elbo["variance"], elbo["elbo"],
        ]).detach()
    return ctx


def _forward(model, context, state, params, ctx, visible="data"):
    _set_state(state, params, ctx)
    with torch.no_grad():
        return context_logits(model, context, None, visible=visible)


def _ber(logits, frame):
    mask = frame.data_mask.bool()
    return float(((logits[mask] >= 0) != frame.bits[mask].bool()).float().mean())


def _reward_bce(logits, frame):
    mask = frame.reward_mask.bool()
    return float(F.binary_cross_entropy_with_logits(
        logits[mask].float(), frame.bits[mask].float(),
    ))


def _flat(params):
    return torch.cat([params[name].reshape(-1) for name in ("ws", "bs", "wv", "bv")])


def _project(params, radius):
    flat = _flat(params)
    norm = flat.norm().clamp_min(1e-12)
    if float(norm) <= radius:
        return params
    scale = radius / norm
    return {name: value * scale for name, value in params.items()}


def _clone_requires_grad(params):
    return {name: value.detach().clone().requires_grad_(True) for name, value in params.items()}


def _run(model, config, seed, snr, frames, state, zero, lr, radius, gate, steps, device):
    _, cir, cfo, tail, received = make_episode(
        config, model, seed=seed, snr=snr, frames=frames,
        state_split="offline_train", device=device,
    )
    d_model = model.head.in_features
    context_dim = 6
    params = _clone_requires_grad(_zero_params(d_model, context_dim, device))
    moment = {name: torch.zeros_like(value) for name, value in params.items()}
    variance = {name: torch.zeros_like(value) for name, value in params.items()}
    rows = []
    for frame in received:
        context = make_context(frame, cir, cfo, tail, snr)
        ctx = _context(model, context, frame, state, zero)
        frozen_logits = _forward(model, context, state, zero, ctx, visible="data")
        online_logits = _forward(model, context, state, params, ctx, visible="data")
        frozen_ber = _ber(frozen_logits, frame)
        online_ber = _ber(online_logits, frame)

        _set_state(state, zero, ctx)
        tail = next_pilot_tail(model, context, alpha=1.0, mode="dd_soft")

        accepted = 0
        for step in range(1, int(steps) + 1):
            _set_state(state, params, ctx)
            elbo = data_aided_elbo(model, context, None, variance_weight=1.0, anchor_weight=0.0)
            grads = torch.autograd.grad(
                elbo["elbo"], [params[name] for name in ("ws", "bs", "wv", "bv")],
            )
            candidate = {}
            for name, grad in zip(("ws", "bs", "wv", "bv"), grads):
                moment[name] = 0.9 * moment[name] + 0.1 * grad
                variance[name] = 0.999 * variance[name] + 0.001 * grad.square()
                m_hat = moment[name] / (1.0 - 0.9 ** step)
                v_hat = variance[name] / (1.0 - 0.999 ** step)
                candidate[name] = params[name].detach() - lr * m_hat / (v_hat.sqrt() + 1e-8)
            candidate = _project(candidate, radius)
            reward_before = _reward_bce(_forward(model, context, state, params, ctx, "data"), frame)
            reward_after = _reward_bce(_forward(model, context, state, candidate, ctx, "data"), frame)
            if reward_after <= reward_before + gate:
                params = _clone_requires_grad(candidate)
                accepted += 1

        rows.append({
            "seed": int(seed), "snr_db": int(snr), "frame_index": int(frame.frame_index),
            "frozen_ber": frozen_ber, "online_ber": online_ber,
            "gain": frozen_ber - online_ber, "accepted_steps": accepted,
            "param_norm": float(_flat(params).norm()),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[120101, 120102])
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--gate", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model, _ = _load_branch(Path(args.checkpoint), config, device)
    model.eval()
    state = _register_film(model)
    zero = {name: value.detach() for name, value in _zero_params(model.head.in_features, 6, device).items()}

    result = {"config": str(Path(args.config)), "checkpoint": str(Path(args.checkpoint)),
              "seeds": args.seeds, "frames": args.frames, "lr": args.lr,
              "radius": args.radius, "gate": args.gate, "steps": args.steps,
              "per_snr": {}, "rows": []}
    for snr in config["main_snrs"]:
        rows = [row for seed in args.seeds
                for row in _run(model, config, seed, snr, args.frames, state, zero,
                                args.lr, args.radius, args.gate, args.steps, device)]
        result["rows"].extend(rows)
        n = len(rows)
        frozen = sum(r["frozen_ber"] for r in rows) / n
        online = sum(r["online_ber"] for r in rows) / n
        result["per_snr"][str(snr)] = {
            "frozen_ber": frozen, "online_ber": online, "gain": frozen - online,
            "accepted_steps": sum(r["accepted_steps"] for r in rows),
            "param_norm_mean": sum(r["param_norm"] for r in rows) / n,
            "frames": n,
        }
        print(f"snr={snr:2d} frozen={frozen:.6f} online={online:.6f} gain={frozen-online:+.6f} "
              f"accepted={result['per_snr'][str(snr)]['accepted_steps']} "
              f"|p|={result['per_snr'][str(snr)]['param_norm_mean']:.4f}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
