# -*- coding: utf-8 -*-
"""固定长回波上的子空间开发：分离在线观测、离线评分和数据角色。"""

from __future__ import annotations

import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from agent.detection_subspace import DetectionSubspace, capture_observation
from baseline.traditional_equalizers import estimate_acquisition_cir_with_cfo
from env.comm_env import CommunicationEnvironment
from env.experiment_config import build_comm_env_config
from evaluation.adaptability_audit import LegalStateTracker, deployable_frame, information_metrics, sha256, tensor_sha256
from evaluation.pilot_transfer import _load_branch, paired_interval
from training.deterministic_adaptation import DeterministicMemoryAdapter, DeterministicSettings
from training.pilot_transfer import frame_to_device
from training.subspace_adaptation import update_subspace


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def read_manifest(path):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec.get("schema_version") not in {"fixed-channel-subspace-development-v1", "fixed-channel-subspace-development-v2", "fixed-channel-subspace-development-v3", "fixed-channel-subspace-development-v4", "fixed-channel-subspace-development-v5", "fixed-channel-subspace-development-v6", "fixed-channel-subspace-development-v7", "fixed-channel-subspace-development-v8", "fixed-channel-subspace-development-v9", "fixed-channel-subspace-development-v10", "fixed-channel-subspace-development-v11", "fixed-channel-subspace-development-v12", "fixed-channel-subspace-development-v13", "fixed-channel-subspace-development-v14", "fixed-channel-subspace-development-v15", "fixed-channel-subspace-development-v16"} or spec.get("role") != "development":
        raise ValueError("必须提供固定信道子空间开发清单。")
    for field in ("config", "checkpoint"):
        if sha256(ROOT / spec[field]) != spec[field + "_sha256"]:
            raise ValueError("来源配置或checkpoint指纹不符。")
    allowed_dimensions = ([2, 4, 8], [2, 4, 8, 16], [2, 4, 8, 10, 12, 14, 16])
    if spec["snrs_db"] != [0, 5, 10, 15] or spec["dimensions"] not in allowed_dimensions:
        raise ValueError("开发清单必须完整列出四SNR及已登记的递增维数候选。")
    all_seeds = [seed for role in spec["data_roles"].values() for seed in role["seeds"]]
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("审计、训练、选择和评估seed必须隔离。")
    if spec.get("channel_changes") is not False or spec.get("heldout_confirmation") is not False:
        raise ValueError("本入口只允许固定信道开发，不能冒充正式确认。")
    required_roles = {"inspection", "training", "selection", "evaluation"}
    if set(spec["data_roles"]) != required_roles or any(type(seed) is not int or seed < 0 for seed in all_seeds):
        raise ValueError("数据角色或seed非法。")
    for role in spec["data_roles"].values():
        if not role["seeds"] or type(role["frames"]) is not int or role["frames"] < 1:
            raise ValueError("每种数据角色必须有固定正预算。")
    train = spec["training"]
    if train["outer_target_offsets"] != [0,1] or train["outer_steps"] < 1 or train["sequence_update_frames"] < 1:
        raise ValueError("本版固定当前/下一帧外层目标与正训练预算。")
    if any(spec["data_roles"][role]["frames"] <= train["sequence_update_frames"] for role in ("training","selection")):
        raise ValueError("训练与选择需要额外下一帧，仅供离线评分。")
    if not train["checkpoint_steps"] or any(step < 1 or step > train["outer_steps"] for step in train["checkpoint_steps"]):
        raise ValueError("checkpoint选择步数超出训练预算。")
    DeterministicSettings(**spec["update"])
    if spec.get("schema_version") == "fixed-channel-subspace-development-v10":
        policy = spec.get("context_update_policy", {})
        if (spec.get("learned_context_update") is not True
                or spec.get("learned_gradient_transport") is not False
                or policy.get("enabled") is not True
                or spec["inner_history_frames"] != 8):
            raise ValueError("v10必须只启用固定8帧历史的上下文更新器。")
    if spec.get("schema_version") == "fixed-channel-subspace-development-v11":
        objective = spec.get("training", {}).get("outer_objective")
        temperature = spec.get("training", {}).get("smooth_ber_temperature")
        if (spec.get("learned_context_update") is not True
                or objective != "smooth_ber"
                or not isinstance(temperature, (int, float))
                or not np.isfinite(temperature) or temperature <= 0):
            raise ValueError("v11只允许固定正温度的平滑BER外层目标。")
    if spec.get("schema_version") == "fixed-channel-subspace-development-v12":
        training = spec.get("training", {})
        if (spec.get("learned_context_update") is not True
                or spec.get("learned_gradient_transport") is not False
                or training.get("outer_objective") != "online_gain"
                or training.get("smooth_ber_temperature") != 1.0):
            raise ValueError("v12只允许固定T=1平滑BER下的配对在线增益外层目标。")
    if spec.get("schema_version") == "fixed-channel-subspace-development-v13":
        training = spec.get("training", {})
        if (spec.get("learned_context_update") is not True
                or spec.get("learned_gradient_transport") is not False
                or training.get("outer_objective") != "frozen_error_conditioned"
                or training.get("smooth_ber_temperature") != 1.0):
            raise ValueError("v13只允许T=1的Frozen错误条件化Online外层目标。")
    if spec.get("schema_version") == "fixed-channel-subspace-development-v16":
        if spec.get("training", {}).get("outer_objective") != "frozen_error_conditioned":
            raise ValueError("v16只允许沿用Frozen错误条件化外层目标比较新增可辨识维数。")
    return spec


def source_fingerprints():
    # 保留全部已跟踪Python源文件的指纹，避免隐含依赖漏记。
    names = subprocess.check_output(["git", "ls-files", "*.py"], cwd=ROOT, text=True).splitlines()
    names += ["agent/detection_subspace.py", "training/subspace_adaptation.py", "evaluation/subspace_development.py"]
    return {name: sha256(ROOT / name) for name in sorted(set(names))}


def ber(logits, offline):
    mask = offline["data_mask"]
    return float(((logits[mask] >= 0) != offline["bits"][mask]).float().mean())


def capture_sessions(spec, role, device, *, dense_diagnostic=False):
    """状态和特征仅消费合法观测；评分标签放入独立离线封套。"""
    config = json.loads((ROOT / spec["config"]).read_text(encoding="utf-8"))
    model, payload = _load_branch(ROOT / spec["checkpoint"], config, device)
    if payload["metrics"].get("receiver_lookahead_len") != config["receiver_lookahead_len"]:
        raise ValueError("来源checkpoint与现有接收窗口协议不一致。")
    training_seeds = {row["seed"] for row in payload["metrics"].get("history", []) if "seed" in row}
    if training_seeds.intersection(spec["data_roles"][role]["seeds"]):
        raise ValueError("开发seed与来源checkpoint训练seed重叠。")
    sessions, dense_rows = [], []
    count = spec["data_roles"][role]["frames"]
    base_weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
    for seed in spec["data_roles"][role]["seeds"]:
        for snr in spec["snrs_db"]:
            env = CommunicationEnvironment(build_comm_env_config(config, level="B", snr_db=snr,
                seed=seed, state_split="offline_train"))
            start = env.reset_episode()
            cir, cfo = estimate_acquisition_cir_with_cfo(start.acquisition, model.config.max_delay,
                cfo_limit=config["acquisition_cfo_limit"])
            cir, tail = cir.to(device), start.initial_soft_tail.to(device)
            tracker = LegalStateTracker(cir, 10**(-snr/10), config["frame_len"]-config["max_delay"], config["rho_frame"])
            dense_settings = replace(DeterministicSettings(**spec["update"]),
                reset_to_origin_on_validation=False, origin_retention_decay=1.0,
                adapt_logit_temperature=1.0) if dense_diagnostic else None
            adapter = DeterministicMemoryAdapter(copy.deepcopy(model), dense_settings) if dense_diagnostic else None
            frames = []
            for index in range(count):
                original = frame_to_device(env.next_frame(config["receiver_lookahead_len"]), device)
                frame = deployable_frame(original)
                synchronize(device)
                tick = time.perf_counter()
                context, _ = tracker.context(model, frame, cfo, tail, snr)
                observation = capture_observation(model, context)
                synchronize(device)
                physical_ms = 1000*(time.perf_counter()-tick)
                identity = dict(role=role, seed=seed, snr_db=snr, frame=index,
                    rx_sha256=tensor_sha256(frame.rx_symbols), lookahead_rx_sha256=tensor_sha256(frame.lookahead_rx_symbols))
                # 更新器只收到无真值context；全部Data指标在动作完成之后评分。
                if adapter is not None:
                    control, before, candidate, retained, _ = adapter.step(context)
                    dense_rows.append(dict(**identity, **control,
                        ber_P=ber(observation.base_logits, dict(data_mask=original.data_mask, bits=original.bits)),
                        ber_before=ber(before, dict(data_mask=original.data_mask, bits=original.bits)),
                        ber_candidate=ber(candidate, dict(data_mask=original.data_mask, bits=original.bits)),
                        ber_retained=ber(retained, dict(data_mask=original.data_mask, bits=original.bits)),
                        ber_reset_to_source=ber(observation.base_logits, dict(data_mask=original.data_mask, bits=original.bits)),
                        reset_is_read_only_counterfactual=True,
                        dense_reset_policy_applied=False,
                        reset_equivalence="仅memory.weight与来源不同；重置后与P为同一接收函数，不改变持续分支",
                        data_scoring_used_for_control=False))
                frames.append(dict(observation=observation, offline=dict(data_mask=original.data_mask.clone(),
                    bits=original.bits.clone()), identity=identity, physical_ms=physical_ms))
                values = torch.tanh(observation.base_logits[-model.config.max_delay:]/2)
                tail = torch.complex(values, torch.zeros_like(values)).detach()
            sessions.append(dict(seed=seed, snr_db=snr, frames=frames,
                                 inner_history_frames=int(spec.get("inner_history_frames", 1))))
            print(json.dumps(dict(role=role, seed=seed, snr_db=snr, captured=count), ensure_ascii=False), flush=True)
    if any(not torch.equal(base_weights[name], value) for name, value in model.state_dict().items()):
        raise AssertionError("来源Frozen网络被修改。")
    return sessions, dense_rows


def make_subspace(dimension, seed, device, architecture="pointwise_mlp_v1", *,
                  learned_gradient_transport=False, learned_context_update=False):
    # 初始化隔离于环境的专用RNG；候选共享预先固定种子，不能按收益重抽。
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return DetectionSubspace(dimension, architecture=architecture,
            learned_gradient_transport=learned_gradient_transport,
            learned_context_update=learned_context_update).to(device)


def inspect_candidates(spec, sessions, device):
    result = []
    for dimension in spec["dimensions"]:
        model = make_subspace(dimension, spec["initialization_seed"], device,
            spec.get("architecture", "pointwise_mlp_v1"),
            learned_gradient_transport=spec.get("learned_gradient_transport", False),
            learned_context_update=spec.get("learned_context_update", False))
        for session in sessions:
            for row in session["frames"]:
                if row["identity"]["frame"] not in spec["geometry_frames"]:
                    continue
                observation = row["observation"]
                basis = model.basis(observation)
                mask = observation.adapt_mask
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    model.logits(observation)[mask], observation.adapt_labels[mask])
                gradient, = torch.autograd.grad(loss, model.coefficients)
                geometry = information_metrics(basis[mask].detach())
                data_mask = row["offline"]["data_mask"]
                # 每个符号单独考虑的球约束上界；并不意味着同一更新可同时翻转所有符号。
                pointwise_bound = spec["update"]["total_radius"] * basis.detach().norm(dim=1)
                wrong = ((observation.base_logits >= 0) != row["offline"]["bits"]) & data_mask
                result.append(dict(**row["identity"], dimension=dimension, geometry=geometry,
                    gradient_norm=float(gradient.norm()), finite_gradient=bool(torch.isfinite(gradient).all()),
                    data_jacobian_norm=float(basis[data_mask].detach().norm()),
                    data_errors=int(wrong.sum()), pointwise_potentially_flippable_errors=int(
                        (wrong & (pointwise_bound >= observation.base_logits.abs())).sum()),
                    selection_uses_data_ber=False))
    eligible = [dimension for dimension in spec["dimensions"] if all(
        row["geometry"]["rank"] == dimension and row["finite_gradient"]
        and row["data_jacobian_norm"] > 0
        and row["geometry"]["full_column_condition_number"] <= spec["maximum_condition_number"]
        for row in result if row["dimension"] == dimension)]
    return dict(candidate_rows=result, eligible_dimensions=eligible,
        selected_dimension=max(eligible) if eligible else None,
        selection_rule="全部指定开发帧满列秩、有限梯度、有Data作用且条件数不超过预定上限，取最大合格维数；不按Data增益选择",
        core_confirmed=False)


def reachability_smoke(spec, sessions, dimension, device):
    """独立副本上的普通监督作用检查，只证明表示可训练，不证明在线泛化。"""
    model = make_subspace(dimension, spec["initialization_seed"], device,
        spec.get("architecture", "pointwise_mlp_v1"),
        learned_gradient_transport=spec.get("learned_gradient_transport", False),
        learned_context_update=spec.get("learned_context_update", False))
    rows = [session["frames"][0] for session in sessions if session["seed"] == spec["data_roles"]["inspection"]["seeds"][0]]
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["training"]["outer_learning_rate"], weight_decay=0.)
    initial_state = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        before = [model.logits(row["observation"]).clone() for row in rows]
        initial_loss = float(torch.stack([data_loss(logits,row) for logits,row in zip(before,rows)]).mean())
    for _ in range(spec["reachability_smoke_steps"]):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.stack([data_loss(model.logits(row["observation"]),row) for row in rows]).mean()
        loss.backward()
        missing_or_nonfinite = [name for name, parameter in model.named_parameters()
            if name != "gradient_transport_raw" and not name.startswith("context_update_network.") and
            (parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()))]
        if not bool(torch.isfinite(loss)) or missing_or_nonfinite:
            raise FloatingPointError("普通监督作用检查的损失或梯度非有限。")
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
    with torch.no_grad():
        after = [model.logits(row["observation"]) for row in rows]
        final_loss = float(torch.stack([data_loss(logits,row) for logits,row in zip(after,rows)]).mean())
        basis_change = sum(float((value-initial_state[name]).square().sum()) for name,value in model.state_dict().items() if name != "coefficients")**.5
        coefficient_change = float((model.coefficients-initial_state["coefficients"]).norm())
        output_change = max(float((left[row["offline"]["data_mask"]]-right[row["offline"]["data_mask"]]).abs().max())
                            for left,right,row in zip(before,after,rows))
    return dict(steps=spec["reachability_smoke_steps"], snrs_db=spec["snrs_db"],
        initial_data_loss=initial_loss, final_data_loss=final_loss,
        basis_parameter_delta_norm=basis_change, coefficient_delta_norm=coefficient_change,
        maximum_data_logit_change=output_change,
        reachable=basis_change>0 and coefficient_change>0 and output_change>0,
        scope="开发副本的离线监督作用检查，不计入四臂checkpoint，不是在线收益证据")


def run_inspection(manifest_path, output_dir, device="cpu"):
    spec = read_manifest(manifest_path)
    target = Path(output_dir)
    if target.exists():
        raise ValueError("输出目录已存在，禁止覆盖。")
    target.mkdir(parents=True)
    started = time.perf_counter()
    fingerprints = source_fingerprints()
    write_json(target / "manifest.json", spec)
    sessions, dense_rows = capture_sessions(spec, "inspection", device, dense_diagnostic=True)
    result = inspect_candidates(spec, sessions, device)
    if result["selected_dimension"] is not None:
        result["reachability_smoke"] = reachability_smoke(spec, sessions, result["selected_dimension"], device)
    result.update(role="development", source_revision=subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT,text=True).strip(),
        source_sha256=fingerprints, manifest_sha256=sha256(manifest_path),
        config_sha256=spec["config_sha256"], checkpoint_sha256=spec["checkpoint_sha256"],
        dense_memory_rows=dense_rows, records=sum(len(s["frames"]) for s in sessions),
        wall_seconds=time.perf_counter()-started, device=str(device),
        gpu=torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else None,
        shared_tail_source="P_full_rx_soft_decisions", autonomous_closed_loop_validated=False)
    if fingerprints != source_fingerprints():
        raise RuntimeError("运行期间源码改变，保留中间结果但不签发完整结果。")
    torch.save(sessions, target / "inspection_features.pt")
    result["feature_cache_sha256"] = sha256(target / "inspection_features.pt")
    write_json(target / "inspection.json", result)
    return result


def data_loss(logits, row):
    """只供离线外层、checkpoint选择或事后评估调用。"""
    offline = row["offline"]
    mask = offline["data_mask"]
    return torch.nn.functional.binary_cross_entropy_with_logits(logits[mask], offline["bits"][mask].float())


def outer_data_loss(logits, row, objective="bce", smooth_ber_temperature=1.0):
    """离线训练目标；线上更新器不接收Data标签。"""
    if objective == "bce":
        return data_loss(logits, row)
    if objective != "smooth_ber":
        raise ValueError("离线外层目标未登记。")
    if not np.isfinite(smooth_ber_temperature) or smooth_ber_temperature <= 0:
        raise ValueError("平滑BER温度必须为正有限数。")
    offline = row["offline"]
    mask = offline["data_mask"]
    labels = offline["bits"][mask].float().mul(2).sub(1)
    return torch.sigmoid(-labels * logits[mask] / smooth_ber_temperature).mean()


def paired_online_gain_loss(frozen_loss, online_loss):
    """离线配对目标：同时降低Frozen和Online误差，并额外惩罚Online落后。"""
    gap = online_loss - frozen_loss
    return frozen_loss + online_loss + torch.nn.functional.softplus(gap)


def frozen_error_conditioned_loss(frozen_logits, online_logits, row, smooth_ber_temperature=1.0):
    """离线按Frozen错判分桶，平衡纠错与维持正确位边距。"""
    if not np.isfinite(smooth_ber_temperature) or smooth_ber_temperature <= 0:
        raise ValueError("平滑BER温度必须为正有限数。")
    offline = row["offline"]
    mask = offline["data_mask"]
    labels = offline["bits"][mask].float().mul(2).sub(1)
    frozen_margin = labels * frozen_logits[mask]
    online_margin = labels * online_logits[mask]
    frozen_error = frozen_margin.detach() <= 0
    online_per_symbol = torch.sigmoid(-online_margin / smooth_ber_temperature)
    if bool(frozen_error.any()) and bool((~frozen_error).any()):
        error_loss = online_per_symbol[frozen_error].mean()
        correct_loss = online_per_symbol[~frozen_error].mean()
        online_loss = .5 * (error_loss + correct_loss)
    else:
        online_loss = online_per_symbol.mean()
    frozen_loss = torch.sigmoid(-frozen_margin / smooth_ber_temperature).mean()
    return frozen_loss + online_loss


def sequence_loss(model, session, settings, update_frames, *, adapt, create_graph,
                  outer_objective="bce", smooth_ber_temperature=1.0):
    """联合最近合法Adapt历史更新；当前帧守门后评分当前和下一帧Data。"""
    origin = model.coefficients
    current = origin
    losses = []
    history = []
    history_limit = int(session.get("inner_history_frames", update_frames))
    for index, row in enumerate(session["frames"][:update_frames]):
        if adapt:
            history.append(row["observation"])
            history = history[-history_limit:]
            _, current, _ = update_subspace(model, row["observation"], current, origin,
                settings, adapt_history=history, create_graph=create_graph)
        for offset in (0, 1):
            target = session["frames"][index + offset]
            online_loss = outer_data_loss(model.logits(target["observation"], current), target,
                "smooth_ber" if outer_objective in {"online_gain", "frozen_error_conditioned"}
                else outer_objective,
                smooth_ber_temperature)
            if outer_objective == "online_gain":
                frozen_loss = outer_data_loss(model.logits(target["observation"], origin), target,
                    "smooth_ber", smooth_ber_temperature)
                losses.append(paired_online_gain_loss(frozen_loss, online_loss))
            elif outer_objective == "frozen_error_conditioned":
                frozen_logits = model.logits(target["observation"], origin)
                online_logits = model.logits(target["observation"], current)
                losses.append(frozen_error_conditioned_loss(frozen_logits, online_logits,
                    target, smooth_ber_temperature))
            else:
                losses.append(online_loss)
    outer = torch.stack(losses).mean()
    if model.learned_gradient_transport and settings.gradient_transport_regularization > 0:
        transport = model.gradient_transport_matrix()
        identity = torch.eye(model.dimension, device=transport.device, dtype=transport.dtype)
        outer = outer + settings.gradient_transport_regularization * (transport - identity).square().mean()
    return outer


def selection_loss(model, sessions, settings, update_frames, *, outer_objective="bce",
                   smooth_ber_temperature=1.0):
    losses = []
    for session in sessions:
        # 选择标准两种训练方式完全相同，兼顾Frozen与Online，不只选最好的终点。
        for adapt in (False, True):
            losses.append(float(sequence_loss(model, session, settings, update_frames,
                adapt=adapt, create_graph=False, outer_objective=outer_objective,
                smooth_ber_temperature=smooth_ber_temperature).detach()))
    return float(np.mean(losses))


def evaluate_subspaces(models, sessions, settings, device):
    rows = []
    for name, model in models.items():
        for session in sessions:
            origin = model.coefficients.detach().clone()
            current = origin.clone()
            history = []
            for row in session["frames"]:
                observation = row["observation"]
                history.append(observation)
                history = history[-int(session.get("inner_history_frames", 8)):]
                synchronize(device)
                tick = time.perf_counter()
                with torch.no_grad():
                    basis = model.basis(observation)
                    frozen = observation.base_logits + basis @ origin
                    before = observation.base_logits + basis @ current
                synchronize(device)
                frozen_ms = 1000 * (time.perf_counter() - tick)
                tick = time.perf_counter()
                candidate, retained, control = update_subspace(model, observation, current, origin, settings,
                                                               adapt_history=history)
                synchronize(device)
                update_ms = 1000 * (time.perf_counter() - tick)
                with torch.no_grad():
                    candidate_logits = observation.base_logits + basis @ candidate
                    retained_logits = observation.base_logits + basis @ retained
                # 所有分支控制完成后才作Data评分。
                rows.append(dict(**row["identity"], training=name, **control,
                    ber_source=ber(observation.base_logits, row["offline"]),
                    ber_frozen=ber(frozen, row["offline"]), ber_before=ber(before, row["offline"]),
                    ber_candidate=ber(candidate_logits, row["offline"]), ber_online=ber(retained_logits, row["offline"]),
                    frozen_data_loss=float(data_loss(frozen, row)), online_data_loss=float(data_loss(retained_logits, row)),
                    coefficient_values=retained.cpu().tolist(), source_physical_ms=row["physical_ms"],
                    subspace_frozen_ms=frozen_ms, subspace_update_ms=update_ms,
                    offline_labels_used_for_online_control=False))
                current = retained
    summary = []
    for snr in sorted({session["snr_db"] for session in sessions}):
        arms = {}
        for name in models:
            active = [row for row in rows if row["snr_db"] == snr and row["training"] == name]
            seeds = sorted({row["seed"] for row in active})
            gains = [float(np.mean([row["ber_frozen"]-row["ber_online"] for row in active if row["seed"] == seed]))
                     for seed in seeds]
            arms[name] = dict(ber_frozen=float(np.mean([row["ber_frozen"] for row in active])),
                ber_online=float(np.mean([row["ber_online"] for row in active])), gain=paired_interval(gains),
                per_seed_gain=dict(zip(map(str,seeds),gains)), accepted=sum(row["accepted"] for row in active),
                candidate_accepted=sum(row["candidate_accepted"] for row in active),
                reset_to_origin=sum(row["reset_to_origin"] for row in active),
                harmful_accepted=sum(row["accepted"] and row["ber_online"]>row["ber_before"] for row in active),
                harmful_retained=sum(row["ber_online"]>row["ber_frozen"] for row in active),
                mean_incremental_update_ms=float(np.mean([row["subspace_update_ms"] for row in active])))
        summary.append(dict(snr_db=snr, arms=arms,
            gain_interaction_B_minus_A=arms["B_post_adapt"]["gain"]["mean"]-arms["A_supervised"]["gain"]["mean"]))
    return rows, summary


def run_training(manifest_path, inspection_dir, output_dir, device="cpu"):
    """先通过固定几何门槛，再按清单预算进行四臂开发。"""
    spec = read_manifest(manifest_path)
    inspection_dir, target = Path(inspection_dir), Path(output_dir)
    inspection = json.loads((inspection_dir / "inspection.json").read_text(encoding="utf-8"))
    if inspection["manifest_sha256"] != sha256(manifest_path):
        raise ValueError("检查与训练清单不一致。")
    dimension = inspection["selected_dimension"]
    if dimension is None:
        raise ValueError("没有通过几何检查的结构；先修表示，不启动训练。")
    if not inspection.get("reachability_smoke",{}).get("reachable"):
        raise ValueError("普通监督作用检查未通过，不启动四臂训练。")
    if sha256(inspection_dir / "inspection_features.pt") != inspection["feature_cache_sha256"]:
        raise ValueError("检查缓存指纹损坏。")
    if target.exists():
        raise ValueError("输出目录已存在，禁止覆盖。")
    target.mkdir(parents=True)
    started, fingerprints = time.perf_counter(), source_fingerprints()
    write_json(target / "manifest.json", spec)
    settings = DeterministicSettings(**spec["update"])
    training, _ = capture_sessions(spec, "training", device)
    selection, _ = capture_sessions(spec, "selection", device)
    # 离线缓存只供受控开发，部署更新器没有OfflineFrame参数。
    torch.save(dict(training=training, selection=selection), target / "offline_features.pt")
    models, history, checkpoints = {}, [], {}
    training_settings = spec["training"]
    for name in ("A_supervised", "B_post_adapt"):
        model = make_subspace(dimension, spec["initialization_seed"], device,
            spec.get("architecture", "pointwise_mlp_v1"),
            learned_gradient_transport=spec.get("learned_gradient_transport", False),
            learned_context_update=spec.get("learned_context_update", False))
        model.use_context_update = name == "B_post_adapt"
        optimizer = torch.optim.AdamW(model.parameters(), lr=training_settings["outer_learning_rate"], weight_decay=0.)
        best, best_state, best_step = float("inf"), None, None
        tick = time.perf_counter()
        for step in range(1, training_settings["outer_steps"] + 1):
            session = training[(step-1) % len(training)]
            optimizer.zero_grad(set_to_none=True)
            loss = sequence_loss(model, session, settings, training_settings["sequence_update_frames"],
                adapt=name == "B_post_adapt", create_graph=name == "B_post_adapt",
                outer_objective=training_settings.get("outer_objective", "bce"),
                smooth_ber_temperature=training_settings.get("smooth_ber_temperature", 1.0))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("离线外层损失非有限。")
            loss.backward()
            if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
                raise FloatingPointError("离线外层梯度非有限。")
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
            optimizer.step()
            record = dict(training=name, step=step, seed=session["seed"], snr_db=session["snr_db"],
                outer_loss=float(loss.detach()), gradient_norm=gradient_norm)
            if step in training_settings["checkpoint_steps"]:
                value = selection_loss(model, selection, settings, training_settings["sequence_update_frames"],
                    outer_objective=training_settings.get("outer_objective", "bce"),
                    smooth_ber_temperature=training_settings.get("smooth_ber_temperature", 1.0))
                record["selection_loss"] = value
                if value < best:
                    best, best_step = value, step
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                print(json.dumps(record, ensure_ascii=False), flush=True)
            history.append(record)
        synchronize(device)
        model.load_state_dict(best_state, strict=True)
        models[name] = model
        path = target / (name + ".pt")
        torch.save(dict(state_dict=best_state, dimension=dimension, architecture=model.architecture,
            learned_gradient_transport=model.learned_gradient_transport,
            learned_context_update=model.learned_context_update,
            use_context_update=model.use_context_update,
            training=name, selected_step=best_step,
            selection_loss=best, manifest_sha256=sha256(manifest_path),
            source_checkpoint_sha256=spec["checkpoint_sha256"]), path)
        checkpoints[name] = dict(path=path.name, sha256=sha256(path), selected_step=best_step,
            selection_loss=best, train_and_selection_seconds=time.perf_counter()-tick)
    # 模型与checkpoint固定后才生成配对评估数据。
    evaluation, _ = capture_sessions(spec, "evaluation", device)
    torch.save(evaluation, target / "evaluation_features.pt")
    rows, summary = evaluate_subspaces(models, evaluation, settings, device)
    raw = target / "frame_metrics.jsonl"
    raw.write_text("".join(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n" for row in rows),encoding="utf-8")
    if fingerprints != source_fingerprints():
        raise RuntimeError("运行期间源码改变，保留中间结果但不签发完整结果。")
    result = dict(role="development", core_confirmed=False, confirmation_ready=False,
        dimension=dimension, manifest_sha256=sha256(manifest_path), source_sha256=fingerprints,
        source_revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        checkpoints=checkpoints, training_history=history, per_snr=summary,
        records=len(rows), raw_sha256=sha256(raw), wall_seconds=time.perf_counter()-started,
        config_sha256=spec["config_sha256"], checkpoint_sha256=spec["checkpoint_sha256"],
        shared_tail_source="P_full_rx_soft_decisions", autonomous_closed_loop_validated=False,
        timing_scope="来源P跟踪与冻结特征提取单独计时；子空间代价为缓存后真实执行，不含acquisition/环境/接收等待，非端到端部署延迟",
        cache_sha256={name:sha256(target/name) for name in ("offline_features.pt","evaluation_features.pt")})
    write_json(target / "summary.json", result)
    return result
