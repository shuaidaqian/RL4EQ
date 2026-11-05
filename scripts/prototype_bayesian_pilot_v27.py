# -*- coding: utf-8 -*-
"""v27缓存上的低维Pilot贝叶斯/RLS机制原型；不进入正式训练或在线入口。"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.detection_subspace import DetectionSubspace
from evaluation.subspace_development import (
    DeterministicSettings,
    data_loss,
    make_subspace,
    paired_interval,
    read_manifest,
    sha256,
    synchronize,
    write_json,
)
from training.subspace_adaptation import project, update_subspace


ARTIFACT = "detection_subspace_v27_bayesian_pilot_rls_screen_v1_20261105"
MANIFEST = "configs/audits/detection_subspace_long_v27_gauss_newton_online_gain_proximal001_v1.json"
CACHE_DIR = "logs/detection_subspace_training_v27_gauss_newton_online_gain_proximal001_v1_20261105"
SOURCE_CHECKPOINT = "docs/results/detection_subspace_long_v27_gauss_newton_online_gain_proximal001_v1_20261105/B_post_adapt.pt"
GRID_FORGETTING = (0.98, 0.995, 1.0)
GRID_INITIAL_COVARIANCE = (0.1, 1.0, 10.0)
PRESCREEN_TRAINING_SEEDS = (121100, 121101)
SELECTION_STAGE_TOP_K = 3
WINDOW_FRAMES = 8


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_model(checkpoint: dict, seed: int, device: str) -> DetectionSubspace:
    model = make_subspace(int(checkpoint["dimension"]), seed, device,
        checkpoint["architecture"],
        learned_gradient_transport=checkpoint.get("learned_gradient_transport", False),
        learned_context_update=checkpoint.get("learned_context_update", False))
    model.use_context_update = checkpoint.get("use_context_update", False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def pilot_validation(model, observation, coefficients):
    logits = model.logits(observation, coefficients)
    mask = observation.validation_mask.bool()
    labels = observation.validation_labels[mask].bool()
    loss = float(F.binary_cross_entropy_with_logits(logits[mask], labels.to(logits.dtype)).detach())
    ber = float(((logits[mask] >= 0) != labels).float().mean())
    return loss, ber


def bayesian_candidate(model, observation, current, origin, covariance, settings, forgetting):
    """用当前Pilot的Laplace似然信息递推低维系数后验，再做与GN相同的局部投影。"""
    mask = observation.adapt_mask.bool()
    labels = observation.adapt_labels[mask].to(dtype=current.dtype)
    if labels.numel() == 0:
        raise ValueError("Adapt Pilot为空。")
    temperature = settings.adapt_logit_temperature
    start = current.detach().clone()
    variable = start.detach().requires_grad_(True)
    logits = model.logits(observation, variable)[mask] / temperature
    probabilities = torch.sigmoid(logits.detach())
    jacobian = torch.autograd.functional.jacobian(
        lambda coefficients: model.logits(observation, coefficients)[mask] / temperature,
        variable, vectorize=True).detach()
    weights = (probabilities * (1. - probabilities)).clamp_min(1e-7)
    count = max(int(labels.numel()), 1)
    # 使用帧内平均对数似然而非符号级似然和：该帧作为一批相关Pilot观测，
    # 因而后验精度是按帧归一化的经验Fisher递推，不宣称符号独立的精确Bayes后验。
    # 初始协方差与遗忘率也按此标度在training/selection Pilot验证集上校准。
    score = jacobian.transpose(0, 1) @ (probabilities - labels) / count
    fisher = jacobian.transpose(0, 1) @ (weights.unsqueeze(1) * jacobian) / count

    predicted_covariance = covariance / forgetting
    dimension = start.numel()
    identity = torch.eye(dimension, device=start.device, dtype=start.dtype)
    prior_precision = torch.linalg.inv(predicted_covariance)
    proximal_gradient = 2. * settings.proximal_weight * (start - origin)
    system = prior_precision + fisher + 2. * settings.proximal_weight * identity
    step = torch.linalg.solve(system, -(score + proximal_gradient))
    if not bool(torch.isfinite(step).all()):
        raise FloatingPointError("Pilot RLS产生非有限更新。")
    step = step * (settings.step_radius / step.norm().clamp_min(1e-12)).clamp(max=1.)

    def objective(coefficients):
        pilot_logits = model.logits(observation, coefficients)[mask] / temperature
        likelihood = F.binary_cross_entropy_with_logits(pilot_logits, labels)
        prox = settings.proximal_weight * (coefficients - origin).square().sum()
        temporal = .5 * (coefficients - start).dot(prior_precision @ (coefficients - start))
        return likelihood + prox + temporal

    before_objective = float(objective(start).detach())
    candidate = start
    accepted_by_adapt = False
    backtracks = 0
    for backtracks in range(settings.gn_max_backtracks + 1):
        trial = project(project(start + step, start, settings.step_radius), origin, settings.total_radius)
        value = float(objective(trial).detach())
        if np.isfinite(value) and value <= before_objective:
            candidate = trial.detach()
            accepted_by_adapt = True
            break
        step = step * .5
    posterior_covariance = torch.linalg.inv(prior_precision + fisher)
    return candidate, posterior_covariance.detach(), dict(
        accepted_by_adapt_backtracking=accepted_by_adapt,
        adapt_objective_before=before_objective,
        adapt_objective_candidate=float(objective(candidate).detach()),
        jacobian_rank=int(torch.linalg.matrix_rank(jacobian)),
        jacobian_rows=int(jacobian.shape[0]),
        parameter_dimension=int(jacobian.shape[1]),
        fisher_trace=float(torch.trace(fisher)),
        posterior_trace=float(torch.trace(posterior_covariance)),
        proposed_step_norm=float(step.detach().norm()),
        applied_step_norm=float((candidate - start).norm()),
        backtracks=backtracks,
        adapt_only=True,
        validation_used_for_candidate=False,
        data_used_for_candidate=False)


def apply_validation_gate(model, observation, current, origin, candidate, geometry, settings):
    """复刻v27的验证Pilot验收、锚点收缩及初值回退规则。"""
    before_loss, before_ber = pilot_validation(model, observation, current)
    candidate_loss, candidate_ber = pilot_validation(model, observation, candidate)
    origin_loss, origin_ber = pilot_validation(model, observation, origin)
    candidate_accepted = bool(geometry["accepted_by_adapt_backtracking"]
        and np.isfinite(candidate_loss)
        and candidate_loss < before_loss - settings.min_validation_improvement
        and candidate_ber <= before_ber)
    retained = candidate if candidate_accepted else current
    if candidate_accepted and settings.origin_retention_decay < 1:
        retained = origin + settings.origin_retention_decay * (retained - origin)
    retained_loss, retained_ber = pilot_validation(model, observation, retained)
    accepted = bool(candidate_accepted
        and retained_loss < before_loss - settings.min_validation_improvement
        and retained_ber <= before_ber)
    if not accepted:
        retained = current
        retained_loss, retained_ber = before_loss, before_ber
    guarded_loss = retained_loss if accepted else before_loss
    guarded_ber = retained_ber if accepted else before_ber
    reset_to_origin = bool(settings.reset_to_origin_on_validation
        and origin_loss < guarded_loss - settings.min_validation_improvement
        and origin_ber <= guarded_ber and float((retained - origin).norm()) > 0)
    if reset_to_origin:
        retained = origin
        accepted = False
        retained_loss, retained_ber = origin_loss, origin_ber
    return retained.detach(), dict(
        candidate_accepted=candidate_accepted, accepted=accepted,
        reset_to_origin=reset_to_origin,
        validation_loss_before=before_loss, validation_loss_candidate=candidate_loss,
        validation_loss_origin=origin_loss, validation_loss_retained=retained_loss,
        validation_ber_before=before_ber, validation_ber_candidate=candidate_ber,
        validation_ber_origin=origin_ber, validation_ber_retained=retained_ber)


def run_rls_session(model, session, settings, forgetting, initial_covariance, score_data):
    origin = model.coefficients.detach().clone()
    current = origin.clone()
    covariance = initial_covariance * torch.eye(origin.numel(), device=origin.device, dtype=origin.dtype)
    rows = []
    for item in session["frames"]:
        observation = item["observation"]
        before_coefficients = current.detach().clone()
        candidate, posterior_covariance, geometry = bayesian_candidate(
            model, observation, current, origin, covariance, settings, forgetting)
        retained, gate = apply_validation_gate(model, observation, current, origin,
            candidate, geometry, settings)
        reset = gate["reset_to_origin"]
        if gate["accepted"]:
            current = retained
            covariance = posterior_covariance
        elif reset:
            current = origin.clone()
            covariance = initial_covariance * torch.eye(origin.numel(), device=origin.device, dtype=origin.dtype)
        else:
            # 被拒绝的观测不进入后验；只按遗忘因子传播不确定度。
            covariance = covariance / forgetting
        row = dict(**item["identity"], method="bayesian_pilot_rls",
            **geometry, **gate, coefficient_values=current.detach().cpu().tolist(),
            covariance_trace=float(torch.trace(covariance)),
            data_labels_used_for_control=False, parameter_count=int(current.numel()))
        if score_data:
            # 只有候选、门控、回滚全部结束后才能读取Data标签。
            with torch.no_grad():
                frozen_logits = model.logits(observation, origin)
                before_logits = model.logits(observation, before_coefficients)
                online_logits = model.logits(observation, current)
                mask = item["offline"]["data_mask"]
                labels = item["offline"]["bits"][mask]
                row.update(ber_frozen=float(((frozen_logits[item["offline"]["data_mask"]] >= 0)
                    != item["offline"]["bits"][item["offline"]["data_mask"]]).float().mean()),
                    ber_before=float(((before_logits[mask] >= 0) != labels).float().mean()),
                    ber_online=float(((online_logits[item["offline"]["data_mask"]] >= 0)
                    != item["offline"]["bits"][item["offline"]["data_mask"]]).float().mean()),
                    frozen_data_loss=float(data_loss(frozen_logits, item)),
                    online_data_loss=float(data_loss(online_logits, item)))
        rows.append(row)
    return rows


def run_gn_session(model, session, settings):
    origin = model.coefficients.detach().clone()
    current = origin.clone()
    rows = []
    for item in session["frames"]:
        observation = item["observation"]
        before_coefficients = current.detach().clone()
        candidate, retained, control = update_subspace(model, observation, current, origin, settings)
        current = retained.detach()
        row = dict(**item["identity"], method="single_frame_gn", **control,
            coefficient_values=current.cpu().tolist())
        # 所有控制动作完成后才读取Data标签。
        with torch.no_grad():
            frozen_logits = model.logits(observation, origin)
            before_logits = model.logits(observation, before_coefficients)
            online_logits = model.logits(observation, current)
            mask = item["offline"]["data_mask"]
            row.update(ber_frozen=float(((frozen_logits[mask] >= 0) != item["offline"]["bits"][mask]).float().mean()),
                ber_before=float(((before_logits[mask] >= 0) != item["offline"]["bits"][mask]).float().mean()),
                ber_online=float(((online_logits[mask] >= 0) != item["offline"]["bits"][mask]).float().mean()),
                frozen_data_loss=float(data_loss(frozen_logits, item)),
                online_data_loss=float(data_loss(online_logits, item)))
        rows.append(row)
    return rows


def aggregate(rows, window_frames=WINDOW_FRAMES):
    by_session = {}
    for row in rows:
        by_session.setdefault((row["seed"], row["snr_db"], row["method"]), []).append(row)
    session_stats = []
    for (seed, snr, method), active in sorted(by_session.items()):
        active.sort(key=lambda row: row["frame"])
        windows = []
        for start in range(0, len(active), window_frames):
            block = active[start:start+window_frames]
            if not block:
                continue
            gain = float(np.mean([row["ber_frozen"] - row["ber_online"] for row in block]))
            windows.append(dict(start_frame=start, end_frame=start+len(block)-1,
                gain_frozen_minus_online=gain,
                ber_frozen=float(np.mean([row["ber_frozen"] for row in block])),
                ber_online=float(np.mean([row["ber_online"] for row in block])),
                accepted=sum(bool(row["accepted"]) for row in block),
                candidate_accepted=sum(bool(row["candidate_accepted"]) for row in block),
                reset_to_origin=sum(bool(row["reset_to_origin"]) for row in block),
                harmful_accepted=sum(bool(row["accepted"] and row["ber_online"] > row.get("ber_before", row["ber_online"]))
                    for row in block),
                mean_parameter_norm=float(np.mean([np.linalg.norm(np.asarray(row["coefficient_values"])) for row in block]))))
        centers = np.asarray([.5*(w["start_frame"]+w["end_frame"]) for w in windows], dtype=float)
        gains = np.asarray([w["gain_frozen_minus_online"] for w in windows], dtype=float)
        slope = float(np.polyfit(centers, gains, 1)[0]) if len(windows) > 1 else 0.
        cumulative = []
        for end in range(window_frames, len(active)+1, window_frames):
            prefix = active[:end]
            cumulative.append(dict(through_frame=end-1,
                gain_frozen_minus_online=float(np.mean([r["ber_frozen"]-r["ber_online"] for r in prefix]))))
        session_stats.append(dict(seed=seed, snr_db=snr, method=method,
            windows=windows, slope_gain_per_frame=slope,
            late_minus_early_gain=(windows[-1]["gain_frozen_minus_online"] - windows[0]["gain_frozen_minus_online"]
                if len(windows) > 1 else None), cumulative_gain=cumulative,
            records=len(active), accepted=sum(bool(r["accepted"]) for r in active),
            candidate_accepted=sum(bool(r["candidate_accepted"]) for r in active),
            reset_to_origin=sum(bool(r["reset_to_origin"]) for r in active)))

    summary = []
    for snr in sorted({row["snr_db"] for row in rows}):
        methods = {}
        for method in sorted({row["method"] for row in rows}):
            active = [item for item in session_stats if item["snr_db"] == snr and item["method"] == method]
            seed_gains = [float(np.mean([row["ber_frozen"]-row["ber_online"] for row in rows
                if row["seed"] == item["seed"] and row["snr_db"] == snr and row["method"] == method]))
                for item in active]
            slopes = [item["slope_gain_per_frame"] for item in active]
            late_early = [item["late_minus_early_gain"] for item in active if item["late_minus_early_gain"] is not None]
            methods[method] = dict(
                ber_frozen=float(np.mean([row["ber_frozen"] for row in rows
                    if row["snr_db"] == snr and row["method"] == method])),
                ber_online=float(np.mean([row["ber_online"] for row in rows
                    if row["snr_db"] == snr and row["method"] == method])),
                overall_gain=paired_interval(seed_gains),
                trend_slope_per_frame=paired_interval(slopes),
                late_minus_early_gain=paired_interval(late_early),
                sessions=len(active),
                accepted=sum(item["accepted"] for item in active),
                candidate_accepted=sum(item["candidate_accepted"] for item in active),
                reset_to_origin=sum(item["reset_to_origin"] for item in active))
        summary.append(dict(snr_db=snr, methods=methods))
    return session_stats, summary


def run(output_dir: Path, device="cpu"):
    if output_dir.exists():
        raise ValueError("输出目录已存在，禁止覆盖。")
    manifest_path = ROOT / MANIFEST
    spec = read_manifest(manifest_path)
    cache_root = ROOT / CACHE_DIR
    cache = torch.load(cache_root / "offline_features.pt", map_location=device, weights_only=False)
    evaluation = torch.load(cache_root / "evaluation_features.pt", map_location=device, weights_only=False)
    checkpoint_path = ROOT / SOURCE_CHECKPOINT
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = make_model(checkpoint, spec["initialization_seed"], device)
    settings = DeterministicSettings(**spec["update"])

    # 第一阶段只在两个预定训练seed上用Pilot验证BCE预筛9组参数；不读取任何Data标签。
    tuning = []
    training_sessions = [session for session in cache["training"]
        if int(session["seed"]) in PRESCREEN_TRAINING_SEEDS]
    if not training_sessions:
        raise ValueError("训练Pilot预筛seed不在v27缓存中。")
    for forgetting in GRID_FORGETTING:
        for initial_covariance in GRID_INITIAL_COVARIANCE:
            train_pilot_losses = []
            accepted = 0
            for session in training_sessions:
                rows = run_rls_session(model, session, settings, forgetting,
                    initial_covariance, score_data=False)
                train_pilot_losses.extend(row["validation_loss_retained"] for row in rows)
                accepted += sum(row["accepted"] for row in rows)
            tuning.append(dict(forgetting_factor=forgetting,
                initial_covariance=initial_covariance,
                train_validation_bce=float(np.mean(train_pilot_losses)),
                prescreen_accepted=accepted))
    prescreened = sorted(tuning, key=lambda item: (item["train_validation_bce"],
        -item["forgetting_factor"], item["initial_covariance"]))[:SELECTION_STAGE_TOP_K]
    # 第二阶段仅在预筛Top-K上用v27 selection角色的Pilot验证BCE定参。
    selection_scores = []
    for candidate in prescreened:
        selection_pilot_losses = []
        accepted = 0
        for session in cache["selection"]:
            rows = run_rls_session(model, session, settings,
                candidate["forgetting_factor"], candidate["initial_covariance"], score_data=False)
            selection_pilot_losses.extend(row["validation_loss_retained"] for row in rows)
            accepted += sum(row["accepted"] for row in rows)
        selection_scores.append(dict(**candidate,
            selection_validation_bce=float(np.mean(selection_pilot_losses)),
            selection_accepted=accepted))
    chosen = min(selection_scores, key=lambda item: (item["selection_validation_bce"],
        item["train_validation_bce"], -item["forgetting_factor"], item["initial_covariance"]))
    forgetting = chosen["forgetting_factor"]
    initial_covariance = chosen["initial_covariance"]

    all_rows = []
    for session in evaluation:
        all_rows.extend(run_rls_session(model, session, settings, forgetting, initial_covariance, score_data=True))
        all_rows.extend(run_gn_session(model, session, settings))
    # 生成cache顺序无关的排序，方便压缩diff与重算。
    all_rows.sort(key=lambda row: (row["snr_db"], row["seed"], row["method"], row["frame"]))
    session_stats, per_snr = aggregate(all_rows)
    output_dir.mkdir(parents=True)
    raw = output_dir / "frame_metrics.jsonl"
    raw.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n" for row in all_rows),
        encoding="utf-8")
    with raw.open("rb") as source, gzip.open(output_dir / "frame_metrics.jsonl.gz", "wb", compresslevel=9) as target:
        while chunk := source.read(1 << 20):
            target.write(chunk)
    raw.unlink()
    result = dict(role="development_mechanism_screen", core_confirmed=False, confirmation_ready=False,
        hypothesis="持久低维Pilot贝叶斯/RLS更新是否能在同一session内逐渐积累在线增益",
        source_manifest=MANIFEST, source_manifest_sha256=sha256(manifest_path),
        source_checkpoint=SOURCE_CHECKPOINT, source_checkpoint_sha256=file_sha(checkpoint_path),
        source_cache_sha256={name: sha256(cache_root / name)
            for name in ("offline_features.pt", "evaluation_features.pt")},
        tuning_roles=["training_prescreen", "selection_final_choice"], tuning_uses_data_labels=False,
        tuning_training_seeds=list(PRESCREEN_TRAINING_SEEDS), tuning_grid=selection_scores,
        tuning_training_grid=tuning, selected_hyperparameters=chosen,
        evaluation_seeds=sorted({int(session["seed"]) for session in evaluation}),
        snrs_db=spec["snrs_db"], frames_per_session=min(len(session["frames"]) for session in evaluation),
        methods=["single_frame_gn", "bayesian_pilot_rls"],
        update_parameters=int(checkpoint["dimension"]), per_snr=per_snr,
        session_trends=session_stats, records=len(all_rows),
        frame_metrics_sha256=file_sha(output_dir / "frame_metrics.jsonl.gz"),
        algorithm=dict(forgetting_factor=forgetting, initial_covariance=initial_covariance,
            posterior="Laplace logistic Fisher information in coefficient space",
            update="Ppred=P/forgetting; delta=-(Ppred^-1+F+2*prox I)^-1*(g+2*prox*(theta-origin)); validation gate from v27"),
        information_boundary="在线候选只使用Adapt Pilot标签；验证Pilot只门控；Data只在动作、接受/回滚完成后评分。",
        limitations=["v27 eval cache只有3个独立seed且每seed/SNR仅32帧，只能作机制筛查，不能确认持续增长。",
            "RLS超参数只按v27训练/选择角色的Pilot验证BCE选取，不用Data标签或v27 eval结果。",
            "原型未改正式入口、训练器或主配置。"])
    write_json(output_dir / "summary.json", result)
    write_json(output_dir / "tuning_pilot_only.json", dict(grid=tuning, selected=chosen,
        data_labels_used=False, roles=["training", "selection"]))
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="v27持久低维贝叶斯/RLS Pilot更新机制筛查")
    parser.add_argument("--output-dir", default=f"docs/results/{ARTIFACT}")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    target = Path(args.output_dir)
    if not target.is_absolute():
        target = ROOT / target
    print(json.dumps(run(target, args.device), ensure_ascii=False, indent=2))
