# -*- coding: utf-8 -*-
"""开发筛查：用离线Pilot Fisher主方向约束强离线网络head微调。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pilot_transfer import _load_branch, paired_interval
from scripts.diagnose_head_fisher import head_fisher
from training.pilot_transfer import (
    context_logits, differentiable_adapt, make_context, make_episode, next_pilot_tail,
    rank1_gauss_newton_adapt,
)
from training.shared_pilot_state import adapt_only_history_context
from training.rank1_online import damped_rank1_gauss_newton_candidate
from training.windowed_discrete_ppo import should_accept_reward_guard


def _full_head_jacobian(model, context):
    theta = torch.cat((model.head.weight.detach().reshape(-1), model.head.bias.detach().reshape(-1)))
    def logits_from_head(value):
        parameters = {
            "head.weight": value[:model.head.weight.numel()].reshape_as(model.head.weight),
            "head.bias": value[model.head.weight.numel():].reshape_as(model.head.bias),
        }
        return context_logits(model, context, parameters, visible="adapt")[context.frame.adapt_mask]
    theta.requires_grad_(True)
    with torch.enable_grad():
        jacobian = torch.autograd.functional.jacobian(logits_from_head, theta, vectorize=True)
        logits = logits_from_head(theta)
    return jacobian.detach(), logits.detach()


def estimate_shared_basis(config, checkpoint, device, seed):
    """用独立训练角色seed的四个SNR Adapt Pilot Fisher平均初始化固定主方向。"""
    model, _ = _load_branch(Path(checkpoint), config, device)
    fisher_sum = None
    for snr in config["main_snrs"]:
        _, cir, cfo, tail, frames = make_episode(
            config, model, seed=seed, snr=snr, frames=1,
            state_split="offline_train", device=device)
        context = make_context(frames[0], cir, cfo, tail, snr)
        jacobian, logits = _full_head_jacobian(model, context)
        fisher = head_fisher(jacobian, logits, torch.ones_like(logits, dtype=torch.bool))
        fisher_sum = fisher if fisher_sum is None else fisher_sum + fisher
    eigenvalues, eigenvectors = torch.linalg.eigh(fisher_sum.double())
    basis = eigenvectors[:, -1].to(dtype=model.head.weight.dtype, device=device)
    basis = basis / basis.norm().clamp_min(1e-12)
    return model, basis.detach(), {
        "basis_seed": int(seed),
        "basis_snrs_db": list(config["main_snrs"]),
        "basis_source": "training_role_adapt_pilot_mean_fisher",
        "fisher_top_eigenvalue": float(eigenvalues[-1]),
        "fisher_energy_fraction": float(eigenvalues[-1] / eigenvalues.clamp_min(0).sum()),
    }


def _guard_metrics(logits, frame):
    symbols = frame.receiver_view().reward_symbols
    labels = (symbols.real > 0).float()
    mask = frame.reward_mask.bool()
    selected = logits[mask].float()
    target = labels[mask].to(selected.device)
    loss = F.binary_cross_entropy_with_logits(selected, target)
    ber = ((selected >= 0) != target.bool()).float().mean()
    return float(ber.detach()), float(loss.detach())


def _flatten_group(values, names):
    return torch.cat([values[name].reshape(-1) for name in names])


def _unflatten_group(template, names, vector):
    out, offset = {}, 0
    for name in names:
        count = template[name].numel()
        out[name] = vector[offset:offset + count].reshape_as(template[name])
        offset += count
    if offset != vector.numel():
        raise ValueError("head向量无法按来源参数形状还原。")
    return out


def _project_candidate(candidate, before, origin, names, basis, radius):
    before_vector = _flatten_group(before, names)
    candidate_vector = _flatten_group(candidate, names)
    origin_vector = _flatten_group(origin, names)
    local_delta = candidate_vector - before_vector
    projected_delta = basis * torch.dot(basis, local_delta)
    projected = before_vector + projected_delta
    total_delta = projected - origin_vector
    scale = (float(radius) / total_delta.norm().clamp_min(1e-12)).clamp(max=1.0)
    projected = origin_vector + total_delta * scale
    return _unflatten_group(before, names, projected), float(local_delta.norm()), float(projected_delta.norm())


def _rank1_adapt_candidate(model, context, parameters, names, basis, settings):
    """直接在固定神经Fisher方向的一维系数上执行Adapt BCE Adam。"""
    origin = _flatten_group(parameters, names).detach()
    coefficient = torch.zeros((), device=origin.device, dtype=origin.dtype, requires_grad=True)
    first = torch.zeros_like(coefficient)
    second = torch.zeros_like(coefficient)
    labels = (context.frame.receiver_view().adapt_symbols.real > 0).float()
    mask = context.frame.adapt_mask.bool()
    losses = []
    for step in range(1, int(settings["inner_steps"]) + 1):
        vector = origin + basis * coefficient
        head = _unflatten_group(parameters, names, vector)
        current_parameters = dict(parameters)
        current_parameters.update(head)
        logits = context_logits(model, context, current_parameters, visible="adapt")
        loss = F.binary_cross_entropy_with_logits(logits[mask].float(), labels[mask].to(logits.device).float())
        gradient, = torch.autograd.grad(loss, coefficient)
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient.square()
        mean = first / (1.0 - 0.9 ** step)
        variance = second / (1.0 - 0.999 ** step)
        coefficient = (coefficient - float(settings["inner_learning_rate"])
                       * mean / (variance.sqrt() + 1e-8)).detach().requires_grad_(True)
        losses.append(float(loss.detach()))
    vector = origin + basis * coefficient.detach()
    candidate = _unflatten_group(parameters, names, vector)
    return candidate, float((vector - origin).norm()), losses


def _rank1_gauss_newton_candidate(model, context, parameters, names, basis, settings):
    """用当前帧Adapt Pilot执行一次阻尼一维Gauss-Newton更新。"""
    origin = _flatten_group(parameters, names).detach()
    labels = (context.frame.receiver_view().adapt_symbols.real > 0).float()
    mask = context.frame.adapt_mask.bool()
    adapt_labels = labels[mask]

    def logits_for_group(vector):
        head = _unflatten_group(parameters, names, vector)
        current = dict(parameters)
        current.update(head)
        return context_logits(model, context, current, visible="adapt")[mask]

    candidate_vector, diagnostics = damped_rank1_gauss_newton_candidate(
        parameters=origin, basis=basis, adapt_logits_fn=logits_for_group,
        adapt_labels=adapt_labels,
        damping=float(settings.get("rank1_gn_damping", 0.01)),
        max_step_norm=float(settings["max_delta_norm"]),
    )
    return _unflatten_group(parameters, names, candidate_vector), diagnostics


def _run_branch(model, frames, cir, cfo, initial_tail, snr, settings, branch, basis, warmup_frames=0):
    names = [name for name, _ in model.peft.named_group_parameters({"head"})]
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    origin = {name: initial[name].clone() for name in names}
    active = {name: origin[name].clone() for name in names}
    active_rank1_coefficient = torch.zeros((), device=basis.device, dtype=basis.dtype)
    tail = initial_tail.detach().clone()
    adapt_history = []
    for frame in frames[:warmup_frames]:
        warmup_context = make_context(frame, cir, cfo, tail, snr)
        tail = next_pilot_tail(model, warmup_context, alpha=float(config_tail_alpha(settings)), parameters=initial)
        adapt_history.append(adapt_only_history_context(frame, warmup_context.condition, warmup_context.tail))
    rows = []
    history_count = 8 if branch == "fisher_rank1_trajectory_gn_h8" else 1
    for frame in frames[warmup_frames:]:
        context = make_context(frame, cir, cfo, tail, snr)
        params_before = dict(initial)
        params_before.update(active)
        frozen_logits = context_logits(model, context, initial)
        candidate = {name: value for name, value in params_before.items()}
        accepted, update_norm, projected_norm = False, 0.0, 0.0
        reward_before = reward_after = None
        reward_loss_before = reward_loss_after = None
        if branch != "frozen":
            candidate = dict(params_before)
            if branch in {"fisher_rank1", "fisher_rank1_gn", "fisher_rank1_trajectory_gn",
                          "fisher_rank1_trajectory_gn_h8", "meta_rank1_gn"}:
                if branch in {"fisher_rank1_trajectory_gn", "fisher_rank1_trajectory_gn_h8", "meta_rank1_gn"}:
                    update_history = (adapt_history[-(history_count - 1):] if history_count > 1 else []) + [context]
                    fast, candidate_alpha, gn_diagnostics = rank1_gauss_newton_adapt(
                        model, update_history, parameters=initial, names=names, basis=basis,
                        coefficient=active_rank1_coefficient, steps=int(settings["inner_steps"]),
                        learning_rate=float(settings.get("trajectory_rank1_step_scale", 1.0)),
                        damping=float(settings.get("trajectory_rank1_damping", 0.01)),
                        max_step_norm=float(settings["max_delta_norm"]),
                        max_total_norm=float(settings["max_delta_norm"]), create_graph=False,
                    )
                    gn_diagnostics = {
                        key: (float(value.detach().cpu()) if isinstance(value, torch.Tensor) else value)
                        for key, value in gn_diagnostics.items()
                    }
                    update_norm = float((candidate_alpha - active_rank1_coefficient).detach().abs())
                    adapt_losses = []
                    candidate.update({name: value.detach() for name, value in fast.items() if name in names})
                elif branch == "fisher_rank1_gn":
                    fast, gn_diagnostics = _rank1_gauss_newton_candidate(
                        model, context, params_before, names, basis, settings)
                    update_norm = gn_diagnostics["raw_step_norm"]
                    adapt_losses = []
                else:
                    fast, update_norm, adapt_losses = _rank1_adapt_candidate(
                        model, context, params_before, names, basis, settings)
                    gn_diagnostics = None
                candidate.update({name: value.detach() for name, value in fast.items()})
                candidate_group, _, projected_norm = _project_candidate(
                    candidate, params_before, origin, names, basis,
                    float(settings["max_delta_norm"]))
                candidate.update({name: value.detach() for name, value in candidate_group.items()})
            else:
                fast, update = differentiable_adapt(
                    model, [context], learning_rate=float(settings["inner_learning_rate"]),
                    steps=int(settings["inner_steps"]), max_delta_norm=float(settings["max_delta_norm"]),
                    initial_parameters=params_before, groups={"head"}, objective="bce",
                    receiver_input="adapt_only")
                candidate.update({name: fast[name].detach() for name in names})
                update_norm = float(update["delta_norm"])
                candidate_vector = _flatten_group(candidate, names)
                origin_vector = _flatten_group(origin, names)
                delta = candidate_vector - origin_vector
                radius = float(settings.get("max_total_delta_norm", settings["max_delta_norm"]))
                scale = (radius / delta.norm().clamp_min(1e-12)).clamp(max=1.0)
                projected_vector = origin_vector + delta * scale
                projected_total = _unflatten_group(origin, names, projected_vector)
                candidate.update({name: value.detach() for name, value in projected_total.items()})
                projected_norm = float(_flatten_group(candidate, names).sub(
                    _flatten_group(params_before, names)).norm())
            with torch.no_grad():
                before_reward_logits = context_logits(model, context, params_before, visible="pilot")
                after_reward_logits = context_logits(model, context, candidate, visible="pilot")
                reward_before, reward_loss_before = _guard_metrics(before_reward_logits, frame)
                reward_after, reward_loss_after = _guard_metrics(after_reward_logits, frame)
            accepted = should_accept_reward_guard(
                reward_loss_before=reward_loss_before, reward_loss_after=reward_loss_after,
                reward_ber_before=reward_before, reward_ber_after=reward_after,
                allow_loss_only=True, peft_delta_norm=float(_flatten_group(candidate, names).sub(
                    _flatten_group(params_before, names)).norm()))
            if accepted:
                active = {name: candidate[name].detach().clone() for name in names}
                if branch in {"fisher_rank1_trajectory_gn", "fisher_rank1_trajectory_gn_h8", "meta_rank1_gn"}:
                    active_rank1_coefficient = candidate_alpha.detach().clone()
            params_committed = dict(initial)
            params_committed.update(active)
        else:
            params_committed = initial
        # 候选及守门完成后才读取Data标签作事后评分。
        online_logits = context_logits(model, context, params_committed)
        mask = frame.data_mask
        frozen_ber = float(((frozen_logits[mask] >= 0) != frame.bits[mask]).float().mean())
        online_ber = float(((online_logits[mask] >= 0) != frame.bits[mask]).float().mean())
        rows.append({
            "frame_index": int(frame.frame_index), "accepted": accepted,
            "reward_ber_before": reward_before, "reward_ber_candidate": reward_after,
            "reward_bce_before": reward_loss_before, "reward_bce_candidate": reward_loss_after,
            "delta_norm": update_norm, "projected_delta_norm": projected_norm,
            "adapt_losses": adapt_losses if branch in {
                "fisher_rank1", "fisher_rank1_gn", "fisher_rank1_trajectory_gn",
                "fisher_rank1_trajectory_gn_h8", "meta_rank1_gn"
            } else update.get("adapt_losses", []) if branch not in {"frozen"} else [],
            "rank1_gn_diagnostics": gn_diagnostics if branch in {
                "fisher_rank1_gn", "fisher_rank1_trajectory_gn", "fisher_rank1_trajectory_gn_h8", "meta_rank1_gn"
            } else None,
            "rank1_coefficient": float(active_rank1_coefficient.detach()) if branch in {
                "fisher_rank1_trajectory_gn", "fisher_rank1_trajectory_gn_h8", "meta_rank1_gn"
            } else None,
            "ber_frozen": frozen_ber, "ber_online": online_ber,
            "data_labels_used_for_control": False,
        })
        tail = next_pilot_tail(model, context, alpha=float(config_tail_alpha(settings)), parameters=params_committed)
        adapt_history.append(adapt_only_history_context(frame, context.condition, context.tail))
        adapt_history = adapt_history[-7:]
    return rows


def config_tail_alpha(settings):
    return float(settings.get("tail_alpha", 0.5))


def run(config, checkpoint, *, basis_seed, evaluation_seeds, frames, device, output,
        basis_checkpoint=None, branches=None):
    target = Path(output)
    if target.exists():
        raise ValueError("输出文件已存在，请指定新路径。")
    if (config.get("main_level") != "B" or config.get("max_delay") != 116
            or config.get("pilot_layout") != "prefix" or config.get("pilot_total") != 256
            or config.get("reward_pilot_total") != 32 or config.get("main_snrs") != [0, 5, 10, 15]):
        raise ValueError("开发原型只允许固定Level B长回波、前缀Pilot=256和四个指定SNR。")
    if frames < 1 or not evaluation_seeds or len(set(evaluation_seeds)) != len(evaluation_seeds):
        raise ValueError("评估帧数或seed非法。")
    branch_names = list(branches) if branches is not None else [
        "frozen", "full_head_adam", "fisher_rank1", "fisher_rank1_gn",
    ] + (["fisher_rank1_trajectory_gn", "meta_rank1_gn"] if basis_checkpoint is not None else [])
    if "frozen" not in branch_names or len(set(branch_names)) != len(branch_names):
        raise ValueError("branch列表必须包含唯一的frozen基线。")
    allowed_branches = {"frozen", "full_head_adam", "fisher_rank1", "fisher_rank1_gn",
                        "fisher_rank1_trajectory_gn", "fisher_rank1_trajectory_gn_h8", "meta_rank1_gn"}
    if set(branch_names) - allowed_branches:
        raise ValueError("branch列表含有未知方法。")
    if basis_checkpoint is None and "meta_rank1_gn" in branch_names:
        raise ValueError("meta_rank1_gn分支需要--basis-checkpoint。")
    need_fisher = basis_checkpoint is None or "fisher_rank1_trajectory_gn" in branch_names
    if need_fisher:
        model, fisher_basis, basis_info = estimate_shared_basis(config, checkpoint, device, basis_seed)
    else:
        model, _ = _load_branch(Path(checkpoint), config, device)
        fisher_basis, basis_info = None, None
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    training_seeds = {row["seed"] for row in payload.get("metrics", {}).get("history", []) if "seed" in row}
    basis_payload = None
    basis_training_seeds = set()
    if basis_checkpoint is not None:
        basis_path = Path(basis_checkpoint)
        basis_payload = torch.load(basis_path, map_location="cpu", weights_only=False)
        basis_metrics = basis_payload.get("metrics", {})
        if basis_metrics.get("trajectory_update_rule") != "rank1_gn":
            raise ValueError("basis checkpoint必须来自trajectory_update_rule=rank1_gn训练。")
        basis_source = basis_metrics.get("trajectory_rank1_basis_source", {})
        if basis_source.get("base_checkpoint_sha256") != hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest():
            raise ValueError("learned basis与在线评估使用的强离线checkpoint来源不一致。")
        learned_basis = basis_metrics.get("trajectory_rank1_basis")
        if not isinstance(learned_basis, list) or not learned_basis:
            raise ValueError("训练checkpoint缺少trajectory_rank1_basis。")
        basis = torch.as_tensor(learned_basis, dtype=next(model.parameters()).dtype, device=device)
        norm = basis.norm()
        if not torch.isfinite(norm) or float(norm) <= 1e-12:
            raise ValueError("learned rank-1 basis必须为非零有限向量。")
        basis = (basis / norm).detach()
        learned_basis = basis
        basis_training_seeds = {row["seed"] for row in basis_metrics.get("history", []) if "seed" in row}
        basis_info = {
            "basis_source": "offline_trajectory_learned_rank1_gn",
            "basis_checkpoint": str(basis_path.resolve()),
            "basis_checkpoint_sha256": hashlib.sha256(basis_path.read_bytes()).hexdigest(),
            "source_basis_metadata": basis_source,
            "basis_norm": float(norm),
        }
    else:
        basis = fisher_basis
    if (training_seeds.intersection(evaluation_seeds) or basis_training_seeds.intersection(evaluation_seeds)
            or basis_seed in evaluation_seeds or basis_seed in training_seeds):
        raise ValueError("Pilot方向训练seed、离线checkpoint训练seed与评估seed必须隔离。")
    settings = config["pilot_transfer"]
    warmup_frames = 7 if "fisher_rank1_trajectory_gn_h8" in branch_names else 0
    all_rows = []
    for snr in config["main_snrs"]:
        for seed in evaluation_seeds:
            env, cir, cfo, tail, received = make_episode(
                config, model, seed=seed, snr=snr, frames=frames + warmup_frames,
                state_split="offline_train", device=device)
            branches = {}
            for branch in branch_names:
                branch_model, _ = _load_branch(Path(checkpoint), config, device)
                branch_basis = (fisher_basis if branch == "fisher_rank1_trajectory_gn" else basis)
                with sdpa_kernel(SDPBackend.MATH):
                    rows = _run_branch(branch_model, received, cir, cfo, tail, snr,
                                       settings, branch, branch_basis, warmup_frames=warmup_frames)
                branches[branch] = rows
            for index in range(frames):
                frozen_independent = branches["frozen"][index]["ber_online"]
                for branch in branches:
                    row = dict(branches[branch][index], seed=seed, snr_db=snr,
                               branch=branch, rx_sha256=hashlib.sha256(
                                   received[index].rx_symbols.detach().cpu().numpy().tobytes()).hexdigest())
                    row["ber_frozen_same_online_tail"] = row["ber_frozen"]
                    row["ber_frozen_independent_trajectory"] = frozen_independent
                    row["gain_same_online_tail"] = row["ber_frozen_same_online_tail"] - row["ber_online"]
                    row["gain"] = frozen_independent - row["ber_online"]
                    all_rows.append(row)
            print(json.dumps({"seed": seed, "snr_db": snr, "frames": frames}, ensure_ascii=False), flush=True)
    window = max(1, frames // 4)
    summary = []
    for snr in config["main_snrs"]:
        for branch in branch_names:
            if branch == "frozen":
                continue
            rows = [r for r in all_rows if r["snr_db"] == snr and r["branch"] == branch]
            per_seed = []
            for seed in evaluation_seeds:
                session = sorted((r for r in rows if r["seed"] == seed), key=lambda r: r["frame_index"])
                gains = [float(np.mean([r["gain"] for r in session[i:i + window]]))
                         for i in range(0, frames, window)]
                per_seed.append({"seed": seed, "overall_gain": float(np.mean([r["gain"] for r in session])),
                                 "window_gains": gains, "late_minus_early": gains[-1] - gains[0]})
            summary.append({"snr_db": snr, "branch": branch,
                "overall_gain": paired_interval([r["overall_gain"] for r in per_seed]),
                "late_minus_early": paired_interval([r["late_minus_early"] for r in per_seed]),
                "window_gains_by_session": per_seed,
                "accepted": sum(r["accepted"] for r in rows), "candidate_count": len(rows),
                "harmful_accepted": sum(r["accepted"] and r["ber_online"] > r["ber_frozen"] for r in rows)})
    result = {
        "schema_version": "head-fisher-rank1-online-development-v1",
        "diagnostic_only": True, "heldout_confirmation": False,
        "main_aggregation_allowed": False, "config": config,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "prototype_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rank1_update_module_sha256": hashlib.sha256(
            (ROOT / "training" / "rank1_online.py").read_bytes()).hexdigest(),
        "basis": basis.detach().cpu().tolist(), "basis_info": basis_info,
        "fisher_basis": (fisher_basis.detach().cpu().tolist() if fisher_basis is not None else None),
        "basis_checkpoint": None if basis_checkpoint is None else str(Path(basis_checkpoint).resolve()),
        "basis_checkpoint_sha256": (None if basis_checkpoint is None else
                                     hashlib.sha256(Path(basis_checkpoint).read_bytes()).hexdigest()),
        "basis_seed": basis_seed, "evaluation_seeds": evaluation_seeds,
        "frames": frames, "window_frames": window,
        "warmup_frames": warmup_frames,
        "snrs_db": list(config["main_snrs"]), "data_labels_used_for_control": False,
        "methods": branch_names,
        "per_snr": summary, "rows": all_rows,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--basis-seed", type=int, default=63000)
    parser.add_argument("--seed-start", type=int, default=63010)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--basis-checkpoint")
    parser.add_argument("--branches", help="逗号分隔分支，必须包含frozen")
    args = parser.parse_args()
    if args.num_seeds < 1:
        parser.error("num-seeds必须为正数。")
    run(json.loads(Path(args.config).read_text(encoding="utf-8")), args.pretrained,
        basis_seed=args.basis_seed, evaluation_seeds=list(range(args.seed_start, args.seed_start + args.num_seeds)),
        frames=args.frames, device=args.device, output=args.output,
        basis_checkpoint=args.basis_checkpoint,
        branches=None if args.branches is None else args.branches.split(","))


if __name__ == "__main__":
    main()
