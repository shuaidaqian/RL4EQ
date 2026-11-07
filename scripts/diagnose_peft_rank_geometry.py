# -*- coding: utf-8 -*-
"""神经 PEFT 秩几何只读诊断（不训练、不改在线）。

三项诊断分别回答：
1. 秩噪声基线（arXiv:2610.22438）：跨帧梯度矩阵的秩度量是否真的低于同形状随机基线，
   即“Fisher 有效秩≈1”是结构还是度量假象；
2. LoRA 内在维度（arXiv:2604.08749）：秩-r 子空间 oracle 拟合的 BER 在哪个 r 饱和；
3. 响应谱集中度（arXiv:2609.33765）：多帧响应主方向对留出帧响应的捕获比例，
   决定 P2（响应几何子空间 + RLS）是否值得做。

约束：两臂共享 acquisition CIR + dd_soft tail；不引入 Bayes CIR。
Data 标签只用于事后 oracle 拟合与 BER 评分，不进入任何在线更新或门控。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pilot_transfer import _load_branch
from scripts.diagnose_head_fisher import head_fisher, summarize_fisher
from training.data_aided import data_aided_elbo
from training.pilot_transfer import context_logits, make_context, make_episode, next_pilot_tail


# ---------------------------------------------------------------------------
# 纯函数：秩度量、子空间捕获、噪声基线
# ---------------------------------------------------------------------------

def rank_measures(matrix: torch.Tensor) -> dict:
    """arXiv:2610.22438 的五种秩度量，均归一化为最大秩的比例。"""

    if matrix.ndim != 2:
        raise ValueError("秩度量输入必须为二维矩阵。")
    if matrix.numel() == 0:
        raise ValueError("秩度量输入不能为空。")
    singular = torch.linalg.svdvals(matrix.double())
    if float(singular[0]) <= 0.0:
        raise ValueError("秩度量输入的谱必须非零。")
    maximum_rank = singular.numel()
    energy = singular.square()
    probabilities = energy / energy.sum()
    cumulative = torch.cumsum(probabilities, dim=0)
    threshold_index = int(torch.searchsorted(
        cumulative,
        torch.tensor(0.9, dtype=cumulative.dtype, device=cumulative.device),
    ))
    energy_rank = min(threshold_index + 1, maximum_rank)
    positive = probabilities[probabilities > 0]
    entropy = float(-(positive * positive.log()).sum())
    return {
        "threshold_count": float((singular >= 0.01 * singular[0]).sum()) / maximum_rank,
        "energy_rank_90": float(energy_rank) / maximum_rank,
        "entropy_rank": float(torch.exp(torch.tensor(entropy, dtype=torch.float64))) / maximum_rank,
        "participation_ratio": float(1.0 / (maximum_rank * probabilities.square().sum())),
        "stable_rank": float(energy.sum() / singular[0].square()) / maximum_rank,
        "top1_energy": float(probabilities[0]),
        "maximum_rank": int(maximum_rank),
    }


def subspace_capture(basis: torch.Tensor, target: torch.Tensor) -> float:
    """target 行向量能量被 basis 列空间解释的比例。"""

    if basis.ndim != 2 or target.ndim != 2:
        raise ValueError("basis 与 target 必须为二维。")
    if basis.shape[0] != target.shape[1]:
        raise ValueError("basis 行数必须等于 target 特征维数。")
    projected = target.double() @ basis.double()
    return float(projected.square().sum() / target.double().square().sum().clamp_min(1e-30))


def gaussian_baseline(shape: tuple[int, int], *, trials: int = 16, seed: int = 0) -> dict:
    """同形状高斯随机矩阵的秩度量均值，作为无结构基线。"""

    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("基线形状必须为二维正数。")
    keys = ("threshold_count", "energy_rank_90", "entropy_rank",
            "participation_ratio", "stable_rank")
    generator = torch.Generator().manual_seed(int(seed))
    totals = {key: 0.0 for key in keys}
    for _ in range(int(trials)):
        sample = torch.randn(*shape, generator=generator, dtype=torch.float64)
        measures = rank_measures(sample)
        for key in keys:
            totals[key] += measures[key]
    return {key: value / int(trials) for key, value in totals.items()}


def _rank_gap(measures: dict, baseline: dict) -> dict:
    """gap = 1 - update/noise；正值表示比噪声更集中。"""

    return {
        key: 1.0 - float(measures[key]) / float(baseline[key])
        for key in ("threshold_count", "energy_rank_90", "entropy_rank",
                    "participation_ratio", "stable_rank")
        if baseline.get(key, 0.0) > 0.0
    }


# ---------------------------------------------------------------------------
# 模型侧工具
# ---------------------------------------------------------------------------

def _flat_head(model) -> torch.Tensor:
    return torch.cat([
        model.head.weight.detach().reshape(-1),
        model.head.bias.detach().reshape(-1),
    ])


def _head_params(theta: torch.Tensor, model) -> dict:
    weight_size = model.head.weight.numel()
    return {
        "head.weight": theta[:weight_size].reshape_as(model.head.weight),
        "head.bias": theta[weight_size:].reshape_as(model.head.bias),
    }


def _head_grad(loss, model) -> torch.Tensor:
    grads = torch.autograd.grad(
        loss, [model.head.weight, model.head.bias], retain_graph=True, allow_unused=True,
    )
    return torch.cat([
        (g if g is not None else torch.zeros_like(p)).reshape(-1)
        for g, p in zip(grads, [model.head.weight, model.head.bias])
    ]).detach()


def _head_jacobian(model, context, adapt_count: int, theta: torch.Tensor) -> torch.Tensor:
    """Adapt Pilot 行对 65 维 head 参数的响应 Jacobian。"""

    def head_logits(value):
        return context_logits(model, context, _head_params(value, model), visible="adapt")

    source = theta.detach().clone().requires_grad_(True)
    jacobian = torch.autograd.functional.jacobian(head_logits, source, vectorize=True)
    return jacobian.detach()[:adapt_count]


def _ber(logits: torch.Tensor, frame) -> float:
    mask = frame.data_mask.bool()
    return float(((logits[mask] >= 0) != frame.bits[mask].bool()).float().mean())


def _collect_session(model, config, seed, snr, frames, device):
    """固定 Frozen head 与 dd_soft tail 轨迹，收集梯度、响应与 logits。"""

    _, cir, cfo, tail, received = make_episode(
        config, model, seed=seed, snr=snr, frames=frames,
        state_split="offline_train", device=device,
    )
    theta0 = _flat_head(model)
    adapt_grads, data_grads, elbo_grads = [], [], []
    jacobians, contexts, bers = [], [], []
    for frame in received:
        context = make_context(frame, cir, cfo, tail, snr)
        contexts.append(context)
        with torch.no_grad():
            frozen_logits = context_logits(model, context, None, visible="data")
            bers.append(_ber(frozen_logits, frame))
        adapt_mask = frame.adapt_mask.bool()
        data_mask = frame.data_mask.bool()
        logits = context_logits(model, context, None, visible="data")
        adapt_grads.append(_head_grad(F.binary_cross_entropy_with_logits(
            logits[adapt_mask].float(), frame.bits[adapt_mask].float(),
        ), model))
        data_grads.append(_head_grad(F.binary_cross_entropy_with_logits(
            logits[data_mask].float(), frame.bits[data_mask].float(),
        ), model))
        elbo = data_aided_elbo(model, context, None, variance_weight=1.0, anchor_weight=0.0)
        elbo_grads.append(_head_grad(elbo["elbo"], model))
        jacobians.append(_head_jacobian(model, context, int(adapt_mask.sum()), theta0))
        with torch.no_grad():
            fisher = head_fisher(
                jacobians[-1].detach(), logits[adapt_mask].detach(),
                torch.ones(int(adapt_mask.sum()), dtype=torch.bool),
            )
        jacobians[-1] = jacobians[-1].detach()
        tail = next_pilot_tail(model, context, alpha=1.0, mode="dd_soft")
    return {
        "theta0": theta0,
        "contexts": contexts,
        "frames": list(received),
        "adapt_grads": torch.stack(adapt_grads),
        "data_grads": torch.stack(data_grads),
        "elbo_grads": torch.stack(elbo_grads),
        "jacobians": torch.stack(jacobians),
        "fisher": fisher.detach(),
        "frozen_ber": sum(bers) / len(bers),
        "frozen_bers": bers,
    }


def _fit_shared_coefficient(model, contexts, frames, theta0, basis, steps, lr, device):
    """在偶数训练帧上拟合共享秩-r 系数，Data 标签仅用于 oracle 诊断。"""

    rank = basis.shape[1]
    coefficient = torch.zeros(rank, device=device)
    if rank == 0:
        return coefficient
    coefficient.requires_grad_(True)
    optimizer = torch.optim.Adam([coefficient], lr=lr)
    train_indices = [index for index in range(len(frames)) if index % 2 == 0]
    for _ in range(int(steps)):
        optimizer.zero_grad()
        total = None
        for index in train_indices:
            frame = frames[index]
            mask = frame.data_mask.bool()
            logits = context_logits(
                model, contexts[index],
                _head_params(theta0 + basis @ coefficient, model), visible="data",
            )
            loss = F.binary_cross_entropy_with_logits(
                logits[mask].float(), frame.bits[mask].float(),
            )
            total = loss if total is None else total + loss
        total.backward()
        optimizer.step()
    return coefficient.detach()


def _evaluate_head(model, contexts, frames, theta0, delta, device) -> dict:
    """用 theta0+delta 评估训练帧与留出帧 BER。"""

    train_bers, heldout_bers = [], []
    with torch.no_grad():
        for index, (context, frame) in enumerate(zip(contexts, frames)):
            logits = context_logits(
                model, context, _head_params(theta0 + delta, model), visible="data",
            )
            target = train_bers if index % 2 == 0 else heldout_bers
            target.append(_ber(logits, frame))
    return {
        "train_ber": sum(train_bers) / len(train_bers),
        "heldout_ber": sum(heldout_bers) / len(heldout_bers),
    }


# ---------------------------------------------------------------------------
# 三项诊断
# ---------------------------------------------------------------------------

def _diagnosis_rank_noise(session: dict, frames: int, seed: int) -> dict:
    baseline = gaussian_baseline((frames, session["theta0"].numel()), trials=32, seed=seed)
    result = {"gaussian_baseline": baseline}
    for name in ("adapt_grads", "data_grads", "elbo_grads"):
        measures = rank_measures(session[name])
        result[name] = measures
        result[f"{name}_gap"] = _rank_gap(measures, baseline)
    fisher = session["fisher"]
    fisher_summary = summarize_fisher(fisher)
    result["fisher"] = {
        "effective_rank": fisher_summary["effective_rank"],
        "numerical_rank": fisher_summary["numerical_rank"],
        "stable_rank": (
            float(torch.tensor(fisher_summary["eigenvalues"]).square().sum()
                  / max(fisher_summary["eigenvalue_max"], 1e-300))
            / max(fisher_summary["dimension"], 1)
        ),
        "top1_energy": (
            fisher_summary["eigenvalue_max"]
            / max(sum(fisher_summary["eigenvalues"]), 1e-300)
        ),
    }
    return result


def _diagnosis_intrinsic_dimension(model, session, ranks, steps, lr, device) -> dict:
    theta0 = session["theta0"]
    contexts = session["contexts"]
    frames = session["frames"]
    parameter_count = theta0.numel()
    rows = {str(rank): {} for rank in ranks}
    training = session["data_grads"][0::2]
    for rank in ranks:
        effective = min(int(rank), parameter_count, len(training))
        if effective < 1:
            continue
        # 基来自训练帧 Data 梯度（oracle 内在维度；不进入在线更新）。
        singular_basis = torch.linalg.svd(
            training.double(), full_matrices=False,
        ).vh[:effective].transpose(0, 1)
        basis = singular_basis.to(device=device, dtype=theta0.dtype)
        coefficient = _fit_shared_coefficient(
            model, contexts, frames, theta0, basis, steps, lr, device,
        )
        delta = basis @ coefficient
        metrics = _evaluate_head(model, contexts, frames, theta0, delta, device)
        rows[str(rank)] = {
            "basis_rank": effective,
            "train_ber": metrics["train_ber"],
            "heldout_ber": metrics["heldout_ber"],
            "train_gain": session["frozen_ber"] - metrics["train_ber"],
            "heldout_gain": session["frozen_ber"] - metrics["heldout_ber"],
            "coefficient_norm": float(coefficient.norm()),
        }
    return {"frozen_ber": session["frozen_ber"], "ranks": rows}


def _diagnosis_response_spectrum(session, ranks) -> dict:
    jacobians = session["jacobians"].double()
    frame_count, adapt_count, dimension = jacobians.shape
    # 全帧聚合响应协方差的主方向能量占比。
    flat = jacobians.reshape(-1, dimension)
    _, singular, vh = torch.linalg.svd(flat, full_matrices=False)
    energy = singular.square()
    share = energy / energy.sum().clamp_min(1e-300)
    cumulative = torch.cumsum(share, dim=0)
    energy_share = {
        str(rank): float(cumulative[min(rank, cumulative.numel()) - 1])
        for rank in ranks
    }
    # 前一半帧建基，后一半帧测 out-of-sample 捕获。
    split = max(frame_count // 2, 1)
    early = jacobians[:split].reshape(-1, dimension)
    late = jacobians[split:].reshape(-1, dimension) if split < frame_count else jacobians[:1].reshape(-1, dimension)
    _, _, vh_early = torch.linalg.svd(early, full_matrices=False)
    persistence = {}
    for rank in ranks:
        effective = min(int(rank), dimension, vh_early.shape[0])
        basis = vh_early[:effective].transpose(0, 1)
        persistence[str(rank)] = subspace_capture(basis, late)
    # 同维随机正交基对照。
    generator = torch.Generator().manual_seed(17)
    random_basis = torch.linalg.qr(
        torch.randn(dimension, max(ranks), generator=generator, dtype=torch.float64),
    )[0]
    random_control = {
        str(rank): subspace_capture(random_basis[:, :rank], late) for rank in ranks
    }
    return {
        "energy_share": energy_share,
        "persistence_early_to_late": persistence,
        "random_basis_control": random_control,
        "principal_energy_top1": float(share[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[110001, 110002, 110003])
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 65])
    parser.add_argument("--fit-steps", type=int, default=8)
    parser.add_argument("--fit-lr", type=float, default=1e-2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if (config.get("main_level") != "B" or config.get("pilot_layout") != "prefix"
            or config.get("max_delay") != 116 or config.get("pilot_total") != 256
            or config.get("reward_pilot_total") != 32
            or config.get("main_snrs") != [0, 5, 10, 15]):
        raise ValueError("诊断只允许固定 Level B、116 符号长回波、前缀 Pilot=256 和四个指定 SNR。")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("诊断 seed 必须非空且不重复。")
    output = Path(args.output)
    if output.exists():
        raise ValueError("输出文件已存在，请使用新路径。")

    device = torch.device(args.device)
    model, _ = _load_branch(Path(args.checkpoint), config, device)
    model.eval()
    for parameter in model.head.parameters():
        parameter.requires_grad_(True)

    result = {
        "schema_version": "peft-rank-geometry-diagnostic-v1",
        "diagnostic_only": True,
        "online_updates_executed": False,
        "data_labels_used": True,
        "data_labels_used_for": "oracle 秩-r 拟合与 BER 评分；不回流在线更新或门控",
        "state_split": "offline_train",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "config": str(Path(args.config)),
        "seeds": args.seeds,
        "frames": args.frames,
        "ranks": args.ranks,
        "fit_steps": args.fit_steps,
        "fit_lr": args.fit_lr,
        "per_snr": {},
    }
    for snr in config["main_snrs"]:
        sessions = [
            _collect_session(model, config, seed, snr, args.frames, device)
            for seed in args.seeds
        ]
        rank_noise = [
            _diagnosis_rank_noise(session, args.frames, seed)
            for session, seed in zip(sessions, args.seeds)
        ]
        intrinsic = [
            _diagnosis_intrinsic_dimension(
                model, session, args.ranks, args.fit_steps, args.fit_lr, device,
            )
            for session in sessions
        ]
        spectrum = [
            _diagnosis_response_spectrum(session, args.ranks) for session in sessions
        ]
        result["per_snr"][str(snr)] = {
            "rank_noise": rank_noise,
            "intrinsic_dimension": intrinsic,
            "response_spectrum": spectrum,
        }
        # 控制台摘要：取 seed 平均。
        def mean(key, rows):
            return sum(row[key] for row in rows) / len(rows)

        adapt_gap = mean("stable_rank", [row["adapt_grads_gap"] for row in rank_noise])
        noise_stable = mean("stable_rank", [row["gaussian_baseline"] for row in rank_noise])
        print(
            f"snr={snr:2d} adapt_stable_gap={adapt_gap:+.4f} "
            f"noise_stable={noise_stable:.4f} "
            f"spectrum_top4={mean('4', [row['energy_share'] for row in spectrum]):.4f} "
            f"persist_top4={mean('4', [row['persistence_early_to_late'] for row in spectrum]):.4f} "
            f"random_top4={mean('4', [row['random_basis_control'] for row in spectrum]):.4f}",
            flush=True,
        )
        for rank in args.ranks:
            gains = [
                row["ranks"].get(str(rank), {}).get("heldout_gain") for row in intrinsic
            ]
            gains = [value for value in gains if value is not None]
            if gains:
                print(f"  rank={rank:2d} heldout_gain={sum(gains) / len(gains):+.6f}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"written {output}", flush=True)


if __name__ == "__main__":
    main()
