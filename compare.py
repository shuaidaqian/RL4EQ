# -*- coding: utf-8 -*-
"""正式比较入口与标签隔离方法封装。"""

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from agent.continual_policy import ContinualPolicy, ObservationEncoder
from agent.cir_estimator import (
    CIRCondition,
    PilotPhysicalState,
    condition_from_cir,
    decision_directed_cir_update,
    pilot_sparse_cir_update,
    track_pilot_physical_state,
)
from agent.pilot_state import PilotStateEmbedding, build_pilot_state_summary
from agent.pilot_risk import compute_pilot_risk
from agent.state_tracking import (
    PhysicalStatePosterior,
    commit_if_safe,
    inflate_tail_uncertainty,
    is_safe_to_commit,
    initial_state,
    update_from_adapt_pilot,
    update_tail_from_adapt_pilot,
)
from agent.discrete_safe_policy import DiscreteSafePolicy, initialize_safe_discrete_policy_prior, safe_modulation_actions
from agent.modulation import ModulationConfig, ModulationState
from agent.safe_contextual_bandit import SafeContextualBandit, SafeUpdateAction
from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from baseline.block_equalizers import bit_error_rate, perfect_csi_bpsk_refine_detect
from baseline.synchronization_compensation import apply_phase_correction
from baseline.traditional_equalizers import (
    TRADITIONAL_BASELINES,
    TraditionalPhaseState,
    estimate_acquisition_cir_with_cfo,
    estimate_phase_residual_features,
    estimate_phase_residual_vector,
    run_traditional_equalizer,
)
from env.comm_env import CommunicationEnvironment, ReceiverState
from env.linear_operator import LinearChannelOperator
from env.experiment_config import (
    build_comm_env_config,
    effective_channel_metadata,
    validate_model_dimensions,
)
from evaluation.bootstrap import paired_block_bootstrap
from evaluation.metrics import FrameMetric, summarize_main_matrix
from training.continual_ppo import (
    _confidence,
    _decision_directed_cir_update,
    _detector_settings,
    _initialize_safe_policy_prior,
    _masked_bce,
    _policy_view,
    _ppo_update,
)
from training.meta_training import _estimate_cir_from_known_frame
from training.windowed_discrete_ppo import (
    WindowedDiscreteOnlineState,
    _load_discrete_policy_if_available,
    run_windowed_discrete_frame,
)
from training.online_adaptation import PilotDrivenOnlineAdapter, PilotResidualRLSAdapter


FORMAL_METHODS = (
    "Perfect-CSI Block",
    "Sparse CIR + Kalman/RLS",
    "Block LMMSE/CG",
    "DFE-RLS",
    "Analytic Iterative BPSK",
    "No Adapt",
    "Best Fixed",
    "Drift-Aware Pilot Rule",
    "Contextual Bandit",
    "Continual PPO",
)

TRADITIONAL_METHODS = (
    "LMMSE-FIR",
    "LMS",
    "NLMS",
    "RLS Linear",
    "DFE-RLS",
    "SC-FDE-MMSE",
    "CFO-Corrected LMMSE-FIR",
    "CFO-Corrected DFE-RLS",
    "CFO+DD-Phase LMMSE-FIR",
    "CFO+DD-Phase DFE-RLS",
)

NEURAL_IMPLEMENTATION_METHODS = (
    "F0 Acquisition Frozen NN",
    "F1 Pilot-Conditioned Frozen NN",
    "F2 Pilot State Tracking Frozen NN",
    "Frozen Offline NN",
    "Pilot CIR only",
    "Pilot-Driven Online Adaptation",
    "NN + Fixed Modulation",
    "NN + Rule Modulation",
    "NN + Discrete PEFT Scheduler",
    "RL-Modulated Neural Block Equalizer",
)

# 当前论文主矩阵只比较冻结离线模型和在线微调模型；其余实现保留为
# 历史兼容或专项消融，不能通过主方法组混入主平均。
FROZEN_NEURAL_BASELINES = (
    "F0 Acquisition Frozen NN",
    "F1 Pilot-Conditioned Frozen NN",
)
PRIMARY_PROPOSED_METHODS = (
    "F2 Pilot State Tracking Frozen NN",
)
PROPOSED_METHODS = PRIMARY_PROPOSED_METHODS

DIAGNOSTIC_METHODS = (
    "Perfect-CSI Block",
    "Perfect-CSI + Pilot Phase",
    "Fixed CG-BPSK-DD Block Detector",
)

PROFILE_PRIORS = {
    "eme_slow_drift_v1": {
        "cfo_abs_cycles_per_symbol": 0.0008,
        "phase_noise_std": 0.003,
        "residual_cfo_limit": 0.0012,
        "acquisition_cfo_limit": 0.004,
    },
}


def method_group(name: str) -> tuple[str, ...]:
    """返回正式比较方法组。

    traditional 只包含非神经、非 RL 传统均衡器；强模型驱动块检测器只作为
    diagnostic reference，不进入主 baseline 成功门槛。
    """

    groups = {
        "traditional": TRADITIONAL_METHODS,
        "proposed": PRIMARY_PROPOSED_METHODS,
        "diagnostic": DIAGNOSTIC_METHODS,
        "main": TRADITIONAL_METHODS + FROZEN_NEURAL_BASELINES + PRIMARY_PROPOSED_METHODS,
        "all": TRADITIONAL_METHODS + FROZEN_NEURAL_BASELINES + PRIMARY_PROPOSED_METHODS + DIAGNOSTIC_METHODS,
    }
    if name not in groups:
        raise ValueError(f"未知方法组：{name}")
    return groups[name]


def _profile_prior_from_config(config: dict, impairment_profile: str) -> dict:
    """合并命名 profile 与配置文件显式给出的 profile 级先验。"""

    prior = dict(PROFILE_PRIORS.get(str(impairment_profile), {}))
    explicit = config.get("profile_prior", {})
    if isinstance(explicit, dict):
        prior.update(explicit)
    for field in ("residual_cfo_limit", "acquisition_cfo_limit"):
        if field in config:
            prior[field] = float(config[field])
    return prior


def _apply_online_cli_overrides(config: dict, overrides: dict[str, object]) -> dict:
    """把在线适配实验参数写入本次运行的有效配置。"""

    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def _online_condition_source_from_config(
    config: dict,
    override: str | None,
) -> str:
    """解析在线方法的条件来源，支持严格的参数微调消融。"""

    source = str(
        override
        if override is not None
        else config.get("online_condition_source", "pilot_cir_phase")
    )
    allowed = {"acquisition", "pilot_phase", "pilot_cir_phase"}
    if source not in allowed:
        raise ValueError(
            "online_condition_source 必须是 acquisition、pilot_phase 或 pilot_cir_phase。"
        )
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--config", default="configs/level_b_state_tracking.json")
    parser.add_argument("--pretrained", default="pretrained/model_best.pt")
    parser.add_argument("--policy", default="logs/online/policy.pt")
    parser.add_argument("--method-group", choices=["traditional", "proposed", "diagnostic", "main", "all"], default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--delays", nargs="*", type=int, default=None)
    parser.add_argument("--snrs", nargs="*", type=float, default=[0, 5, 10, 15])
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--pilot-total", type=int, default=None)
    parser.add_argument("--reward-pilot-total", type=int, default=None)
    parser.add_argument(
        "--acquisition-to-data-gap-seconds",
        type=float,
        default=None,
        help="仅覆盖本次运行的 acquisition 到首个数据帧间隔。",
    )
    parser.add_argument("--pilot-layout", default=None)
    parser.add_argument("--state-split", choices=["offline_train", "heldout_edge", "drift"], default=None)
    parser.add_argument(
        "--state-tracking-mode",
        choices=["joint", "phase_only", "cir_only"],
        default="joint",
        help="F2 物理状态消融；单变量模式冻结 soft-tail belief。",
    )
    parser.add_argument(
        "--impairment-profile",
        default=None,
        choices=[
            "clean",
            "cfo_tiny",
            "phase_tiny",
            "cfo_phase_tiny",
            "cfo_light",
            "phase_light",
            "cfo_phase_light",
            "cfo_phase_mid",
            "eme_slow_drift_v1",
        ],
    )
    parser.add_argument("--update-interval", type=int, default=32)
    parser.add_argument(
        "--scheduler",
        choices=["fixed", "bandit"],
        default=None,
        help="在线调度器；未指定时沿用配置文件中的 online_scheduler，默认 fixed。",
    )
    parser.add_argument("--cir-update", choices=["fixed", "pilot_sparse", "decision_directed"], default="fixed")
    parser.add_argument("--cir-alpha", type=float, default=0.2)
    parser.add_argument("--online-groups", nargs="*", default=None)
    parser.add_argument(
        "--online-algorithm",
        choices=["sgd", "rls"],
        default=None,
        help="覆盖配置中的在线参数更新算法。",
    )
    parser.add_argument("--online-learning-rate", type=float, default=None)
    parser.add_argument("--online-steps", type=int, default=None)
    parser.add_argument(
        "--online-optimizer",
        choices=["sgd", "adamw"],
        default=None,
        help="选择在线 PEFT 梯度优化器；默认沿用配置中的 sgd。",
    )
    parser.add_argument("--online-max-delta-norm", type=float, default=None)
    parser.add_argument("--online-rls-max-total-delta-norm", type=float, default=None)
    parser.add_argument("--online-proximal-weight", type=float, default=None)
    parser.add_argument(
        "--online-objective",
        choices=["bce", "pilot_reconstruction", "pilot_signal_reconstruction"],
        default=None,
    )
    parser.add_argument("--online-min-reward-improvement", type=float, default=None)
    parser.add_argument("--online-relative-min-reward-improvement", type=float, default=None)
    parser.add_argument("--online-reward-windows", type=int, default=None)
    parser.add_argument(
        "--online-reward-ranking",
        choices=[
            "aggregate_bce",
            "robust_lcb",
            "worst_window",
            "pilot_consensus",
            "pilot_balanced",
            "pilot_stable_consensus",
            "pilot_margin_consensus",
        ],
        default=None,
        help="只用 Reward Pilot 选择候选的排序方式；pilot_margin_consensus 额外使用已知 Pilot 的软边际改善。",
    )
    parser.add_argument("--online-reward-ranking-dispersion", type=float, default=None)
    parser.add_argument(
        "--online-require-reward-signal-reconstruction",
        action="store_true",
        help="要求 Reward Pilot 的复数信号重构误差不退化后才保留 PEFT 更新。",
    )
    parser.add_argument("--online-reward-signal-tolerance", type=float, default=None)
    parser.add_argument(
        "--online-reward-logit-change-tolerance",
        type=float,
        default=None,
        help="限制候选在 Reward Pilot 上的平均 logits 改变量；用于拦截当前帧过大的 PEFT 更新。",
    )
    parser.add_argument(
        "--online-defer-peft-commit",
        action="store_true",
        help="候选先留在当前帧之后，用下一帧 Reward Pilot 复核通过后再提交 Adapter 更新。",
    )
    parser.add_argument("--online-cross-frame-tolerance", type=float, default=None)
    parser.add_argument("--online-phase-smoothing", type=float, default=None)
    parser.add_argument("--online-phase-min-confidence", type=float, default=None)
    parser.add_argument("--online-freeze-below-snr-db", type=float, default=None)
    parser.add_argument(
        "--online-drift-gate-threshold",
        type=float,
        default=None,
        help="仅当 Adapt Pilot 估计的 CIR 相对残差达到该阈值时才尝试 PEFT 更新。",
    )
    parser.add_argument(
        "--online-condition-source",
        choices=["acquisition", "pilot_phase", "pilot_cir_phase"],
        default=None,
    )
    parser.add_argument("--output-dir", default="logs/compare")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.seed_start < 0:
        parser.error("--seed-start 必须是非负整数。")
    if args.acquisition_to_data_gap_seconds is not None and (
        not math.isfinite(args.acquisition_to_data_gap_seconds)
        or args.acquisition_to_data_gap_seconds < 0.0
    ):
        parser.error("--acquisition-to-data-gap-seconds 必须是有限非负数。")
    if args.version:
        print("RL4EQ state-tracking schema-v1")
        return
    selected_methods = method_group(args.method_group) if args.method_group else _select_methods(args.methods)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config["state_tracking_mode"] = str(args.state_tracking_mode)
    if (
        args.method_group is None
        and args.methods is None
        and str(config.get("channel_profile", "legacy_sparse_v1")) == "legacy_sparse_v1"
    ):
        # 旧 legacy 配置没有可用的主路线 checkpoint，保留传统 smoke 兼容；
        # canonical Level B 配置仍默认选择 F0/F1/F2。
        selected_methods = TRADITIONAL_METHODS
    scheduler = str(
        args.scheduler
        if args.scheduler is not None
        else config.get("online_scheduler", "fixed")
    )
    if scheduler not in {"fixed", "bandit"}:
        parser.error(
            "配置项 online_scheduler 必须是 fixed 或 bandit，"
            f"当前为 {scheduler!r}。"
        )
    if args.acquisition_to_data_gap_seconds is not None:
        config["acquisition_to_data_gap_seconds"] = args.acquisition_to_data_gap_seconds
        config["aggregation_role"] = "diagnostic"
        config["diagnostic_only"] = True
        config["main_aggregation_allowed"] = False
        profile_prior = config.get("profile_prior")
        if isinstance(profile_prior, dict):
            profile_prior = dict(profile_prior)
            profile_prior["candidate_id"] = (
                "B-acquisition-gap-"
                f"{args.acquisition_to_data_gap_seconds:g}s-diagnostic"
            )
            config["profile_prior"] = profile_prior
    _apply_online_cli_overrides(
        config,
        {
            "online_adaptation_learning_rate": args.online_learning_rate,
            "online_adaptation_algorithm": args.online_algorithm,
            "online_adaptation_steps": args.online_steps,
            "online_adaptation_optimizer": args.online_optimizer,
            "online_adaptation_max_delta_norm": args.online_max_delta_norm,
            "online_rls_max_total_delta_norm": args.online_rls_max_total_delta_norm,
            "online_adaptation_proximal_weight": args.online_proximal_weight,
            "online_adaptation_objective": args.online_objective,
            "online_adaptation_min_reward_improvement": args.online_min_reward_improvement,
            "online_adaptation_relative_min_reward_improvement": args.online_relative_min_reward_improvement,
            "online_reward_windows": args.online_reward_windows,
            "online_reward_ranking": args.online_reward_ranking,
            "online_reward_ranking_dispersion": args.online_reward_ranking_dispersion,
            "online_require_reward_signal_reconstruction": (
                True if args.online_require_reward_signal_reconstruction else None
            ),
            "online_reward_signal_tolerance": args.online_reward_signal_tolerance,
            "online_reward_logit_change_tolerance": args.online_reward_logit_change_tolerance,
            "online_defer_peft_commit": True if args.online_defer_peft_commit else None,
            "online_cross_frame_rollback_tolerance": args.online_cross_frame_tolerance,
            "online_phase_tracking_smoothing": args.online_phase_smoothing,
            "online_phase_tracking_min_confidence": args.online_phase_min_confidence,
            "online_adaptation_freeze_below_snr_db": args.online_freeze_below_snr_db,
            "online_drift_gate_threshold": args.online_drift_gate_threshold,
            "online_condition_source": args.online_condition_source,
            "online_scheduler": scheduler,
        },
    )
    policy_explicit = any(argument == "--policy" or argument.startswith("--policy=") for argument in sys.argv[1:])
    policy_required = policy_explicit and "RL-Modulated Neural Block Equalizer" in selected_methods
    policy_path = Path(args.policy) if policy_explicit and args.policy else None
    if policy_path is not None and not policy_path.exists():
        if policy_required:
            raise FileNotFoundError(f"显式指定的 policy checkpoint 不存在：{policy_path}")
        policy_path = None
    pilot_total = int(args.pilot_total if args.pilot_total is not None else config.get("pilot_total", 128))
    reward_pilot_total = int(
        args.reward_pilot_total
        if args.reward_pilot_total is not None
        else config.get("reward_pilot_total", pilot_total // 4)
    )
    pilot_layout = str(args.pilot_layout if args.pilot_layout is not None else config.get("pilot_layout", "prefix"))
    impairment_profile = str(
        args.impairment_profile if args.impairment_profile is not None else config.get("impairment_profile", "clean")
    )
    selected_delays = args.delays or [int(value) for value in config.get("main_delays", [20, 30, 40])]
    env_configs = {
        (int(delay), float(snr_db), int(seed)): build_comm_env_config(
            config,
            level="B",
            snr_db=float(snr_db),
            seed=50_000 + int(seed),
            max_delay=int(delay),
            total_pilot=pilot_total,
            reward_pilot_total=reward_pilot_total,
            pilot_layout=pilot_layout,
            impairment_profile=impairment_profile,
            state_split=args.state_split,
        )
        for delay in selected_delays
        for snr_db in args.snrs
        for seed in range(args.seed_start, args.seed_start + args.num_seeds)
    }
    if not env_configs:
        raise ValueError("对比实验至少需要一个 delay/SNR/seed 配置。")
    effective_channel = effective_channel_metadata(next(iter(env_configs.values())))
    profile_prior = _profile_prior_from_config(config, impairment_profile)
    residual_cfo_limit = float(profile_prior.get("residual_cfo_limit", 0.001))
    acquisition_cfo_limit = float(profile_prior.get("acquisition_cfo_limit", max(0.004, residual_cfo_limit)))
    pretrained_path = Path(args.pretrained) if args.pretrained else None
    neural_methods = {
        "F0 Acquisition Frozen NN",
        "F1 Pilot-Conditioned Frozen NN",
        "F2 Pilot State Tracking Frozen NN",
        "Frozen Offline NN",
        "Pilot CIR only",
        "Pilot-Driven Online Adaptation",
        "RL-Modulated Neural Block Equalizer",
    }
    if any(method in neural_methods for method in selected_methods):
        if pretrained_path is None or not pretrained_path.is_file():
            raise FileNotFoundError(
                "正式神经方法必须提供存在的 --pretrained checkpoint；禁止静默回退随机初始化。"
            )
    elif pretrained_path is not None and not pretrained_path.is_file():
        pretrained_path = None
    if config.get("channel_profile") in {"eme_measurement_v1", "eme_long_memory_v2"} and any(
        method in neural_methods for method in selected_methods
    ):
        validate_model_dimensions(
            _load_model_config(config, pretrained_path),
            next(iter(env_configs.values())),
        )
    target = Path(args.output_dir)
    target.mkdir(parents=True, exist_ok=True)
    jsonl = target / "frame_metrics.jsonl"
    existing_rows = _load_existing_rows(jsonl) if args.resume else []
    existing_profiles = {str(row.get("profile_name", "<missing>")) for row in existing_rows}
    expected_profile = str(effective_channel["profile_name"])
    if existing_profiles and existing_profiles != {expected_profile}:
        raise ValueError(
            "--resume 检测到不兼容的 profile_name："
            f"已有 {sorted(existing_profiles)}，当前 {expected_profile}。"
        )
    existing_gaps = {
        float(
            row.get(
                "acquisition_to_data_gap_seconds",
                row.get("state_instance", {}).get("acquisition_to_data_gap_seconds", 0.0),
            )
        )
        for row in existing_rows
    }
    expected_gap = float(effective_channel["acquisition_to_data_gap_seconds"])
    if existing_gaps and existing_gaps != {expected_gap}:
        raise ValueError(
            "--resume 检测到不兼容的 acquisition_to_data_gap_seconds："
            f"已有 {sorted(existing_gaps)}，当前 {expected_gap:g}。"
        )
    existing_keys = {_row_key(row) for row in existing_rows}
    rows = list(existing_rows)
    mode = "a" if args.resume else "w"
    with jsonl.open(mode, encoding="utf-8") as handle:
        for delay in selected_delays:
            for snr_db in args.snrs:
                for seed in range(args.seed_start, args.seed_start + args.num_seeds):
                    env_config = env_configs[(int(delay), float(snr_db), int(seed))]
                    env = CommunicationEnvironment(env_config)
                    start = env.reset_episode()
                    if impairment_profile == "clean":
                        acquisition_cir = _estimate_cir_from_known_frame(start.acquisition, int(env_config.max_delay))
                        acquisition_cfo = 0.0
                        acquisition_phase_features = torch.zeros(16, dtype=torch.float32)
                    else:
                        acquisition_cir, acquisition_cfo = estimate_acquisition_cir_with_cfo(
                            start.acquisition,
                            int(env_config.max_delay),
                            cfo_limit=acquisition_cfo_limit,
                        )
                        acquisition_phase_features = estimate_phase_residual_vector(
                            start.acquisition.receiver_view(),
                            acquisition_cir,
                            start.initial_soft_tail,
                            blocks=4,
                            cfo_hint=float(acquisition_cfo),
                        )
                    method_states = _build_method_states(
                        selected_methods,
                        start.initial_soft_tail,
                        acquisition_cir,
                        config,
                        int(env_config.max_delay),
                        float(snr_db),
                        int(seed),
                        pretrained_path=pretrained_path,
                        policy_path=policy_path,
                        policy_required=policy_required,
                        device=str(args.device),
                        cir_update_mode=str(args.cir_update),
                        cir_alpha=float(args.cir_alpha),
                        residual_cfo_limit=residual_cfo_limit,
                        acquisition_cfo=float(acquisition_cfo),
                        acquisition_phase_features=acquisition_phase_features,
                        online_groups_override=args.online_groups,
                        online_condition_source_override=args.online_condition_source,
                        scheduler=scheduler,
                    )
                    for frame_index in range(1, args.frames + 1):
                        frame = env.next_frame()
                        frame_results = _run_methods_for_frame(
                            selected_methods,
                            frame,
                            float(snr_db),
                            method_states,
                            delay=int(env_config.max_delay),
                            frame_index=int(frame_index),
                            update_interval=int(args.update_interval),
                        )
                        for method in selected_methods:
                            result = frame_results[method]
                            metric = FrameMetric(
                                method=method,
                                level=str(env_config.level),
                                delay=int(env_config.max_delay),
                                snr_db=snr_db,
                                rho=float(env_config.rho),
                                pilot_total=int(env_config.total_pilot),
                                pilot_layout=str(env_config.layout),
                                seed=seed,
                                frame=frame_index,
                                ber_data=result.ber_data,
                                latency_ms=result.latency_ms,
                                ber_reward_pilot=result.ber_reward_pilot,
                                ber_adapt_pilot=result.ber_adapt_pilot,
                                detector_iterations=result.detector_iterations,
                                adapt_params=int(result.extra.get("adapt_params", 0)),
                                adapt_steps=int(result.extra.get("adapt_steps", 0)),
                                parameter_delta_norm=float(result.extra.get("parameter_delta_norm", 0.0)),
                            )
                            payload = metric.to_json()
                            payload["profile_name"] = str(env_config.profile_name)
                            payload["impairment_profile"] = impairment_profile
                            payload["profile_residual_cfo_limit"] = residual_cfo_limit
                            payload["profile_acquisition_cfo_limit"] = acquisition_cfo_limit
                            payload["acquisition_to_data_gap_seconds"] = float(
                                env_config.acquisition_to_data_gap_seconds
                            )
                            payload["reward_pilot_total"] = int(frame.reward_mask.sum().item())
                            payload["adapt_pilot_total"] = int(frame.adapt_mask.sum().item())
                            payload["state_split"] = args.state_split
                            payload["state_instance"] = env.state_metadata()
                            payload["input_hash"] = result.input_hash
                            payload.update(result.extra)
                            key = _row_key(payload)
                            if key in existing_keys:
                                continue
                            rows.append(payload)
                            existing_keys.add(key)
                            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                            handle.flush()
    primary_method = _primary_summary_method(selected_methods)
    primary_rows = [row for row in rows if row["method"] == primary_method]
    summary = summarize_main_matrix(
        primary_rows,
        main_delays=config.get("main_delays"),
        main_snrs=config.get("main_snrs"),
    )
    interval = paired_block_bootstrap(primary_rows, seed=0, repetitions=200, block_length=min(10, max(1, args.frames)))
    per_method = {}
    for method in selected_methods:
        method_rows = [row for row in rows if row["method"] == method]
        method_summary = summarize_main_matrix(
            method_rows,
            main_delays=config.get("main_delays"),
            main_snrs=config.get("main_snrs"),
        )
        per_method[method] = {
            "per_config": [item.__dict__ for item in method_summary.per_config],
            "generalization": method_summary.generalization,
        }
    summary_payload = {
        "schema_version": "state-tracking-compare-v1",
        "methods": list(selected_methods),
        "primary_summary_method": primary_method,
        "main_level": "B",
        "level_c_separate": True,
        "per_config": [item.__dict__ for item in summary.per_config],
        "generalization": summary.generalization,
        "per_method": per_method,
        "bootstrap": interval.__dict__,
        "effective_channel": effective_channel,
        "impairment_profile": impairment_profile,
        "state_split": args.state_split,
        "aggregation_role": str(config.get("aggregation_role", "main")),
        "diagnostic_only": bool(config.get("diagnostic_only", False)),
        "main_aggregation_allowed": bool(config.get("main_aggregation_allowed", True)),
        "online_condition_source": config.get("online_condition_source", "pilot_cir_phase"),
        "online_scheduler": scheduler,
        "profile_prior": profile_prior,
        "forbidden": {"data_label_upper_bound_method_present": any("数据标签上界" in method for method in selected_methods)},
    }
    (target / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {args.output_dir}")


def _primary_summary_method(methods: tuple[str, ...]) -> str:
    """选择默认主结果方法，优先使用当前在线 proposed 主线。"""

    priority = (
        "F2 Pilot State Tracking Frozen NN",
        "F1 Pilot-Conditioned Frozen NN",
        "F0 Acquisition Frozen NN",
        "RL-Modulated Neural Block Equalizer",
        "Frozen Offline NN",
        "Pilot-Driven Online Adaptation",
    )
    for method in priority:
        if method in methods:
            return method
    if not methods:
        raise ValueError("至少需要一个比较方法。")
    return str(methods[0])


@dataclass(frozen=True)
class RealMethodResult:
    method: str
    input_hash: str
    ber_data: float
    ber_reward_pilot: float
    ber_adapt_pilot: float
    latency_ms: float
    detector_iterations: int
    extra: dict


@dataclass
class BaselineMethodState:
    cir: torch.Tensor
    receiver_state: ReceiverState
    phase_state: TraditionalPhaseState | None = None
    residual_cfo_limit: float = 0.001
    cir_update_mode: str = "fixed"
    cir_update_alpha: float = 0.2
    acquisition_cfo: float = 0.0
    pilot_physical_state: PilotPhysicalState | None = None


@dataclass
class PPOMethodState:
    cir: torch.Tensor
    receiver_state: ReceiverState
    policy: ContinualPolicy
    optimizer: torch.optim.Optimizer
    encoder: ObservationEncoder
    hidden: torch.Tensor
    previous_reward: float
    last_parameter_delta_norm: float
    rollout: list[dict]
    last_good_cir: torch.Tensor
    last_good_tail: torch.Tensor


@dataclass
class NeuralMethodState:
    cir: torch.Tensor
    receiver_state: ReceiverState
    model: UnfoldedEqualizer
    modulation: ModulationState
    pretrained_loaded: bool
    acquisition_cfo: float = 0.0
    cir_update_mode: str = "fixed"
    cir_update_alpha: float = 0.2
    condition_update_mode: str = "fixed"
    condition_source: str = "acquisition"
    acquisition_phase_features: torch.Tensor | None = None
    physical_state: PilotPhysicalState | None = None
    phase_tracking_smoothing: float = 0.5
    phase_tracking_min_confidence: float = 0.15
    phase_tracking_cfo_limit: float = 0.0012
    tail_update_alpha: float = 1.0
    posterior: PhysicalStatePosterior | None = None
    state_tracking_mode: str = "joint"


@dataclass
class RLModulatedMethodState:
    online_state: WindowedDiscreteOnlineState
    pretrained_loaded: bool
    policy_loaded: bool
    acquisition_cfo: float = 0.0


@dataclass
class PilotOnlineMethodState:
    model: UnfoldedEqualizer
    adapter: PilotDrivenOnlineAdapter
    cir: torch.Tensor
    receiver_state: ReceiverState
    pretrained_loaded: bool
    reference_cir: torch.Tensor | None = None
    acquisition_cfo: float = 0.0
    acquisition_phase_features: torch.Tensor | None = None
    cir_update_mode: str = "fixed"
    cir_update_alpha: float = 0.2
    tail_update_alpha: float = 0.5
    candidate_specs: tuple[dict, ...] = ()
    freeze_online_below_snr_db: float | None = None
    physical_state: PilotPhysicalState | None = None
    phase_tracking_smoothing: float = 0.5
    phase_tracking_min_confidence: float = 0.15
    phase_tracking_cfo_limit: float = 0.0012
    min_reward_improvement: float = 0.001
    relative_min_reward_improvement: float = 0.0
    reward_pilot_windows: int = 2
    # 候选排序只使用 Reward Pilot 各子窗口的改善。robust_lcb 用中位数减
    # 离散度惩罚，避免某个偶然的高收益窗口把不稳定候选排在前面。
    reward_ranking_mode: str = "aggregate_bce"
    reward_ranking_dispersion: float = 0.5
    cross_frame_rollback_tolerance: float = 0.00001
    # 在线参数微调可固定 acquisition 条件，以单独度量 PEFT 的增量价值。
    condition_source: str = "pilot_cir_phase"
    last_pre_update_snapshot: object | None = None
    last_pre_update_groups: frozenset[str] = frozenset()
    scheduler: str = "fixed"
    bandit: SafeContextualBandit | None = None
    active_action: SafeUpdateAction | None = None
    hold_remaining: int = 0
    previous_bandit_reward: float = 0.0
    rollback_count: int = 0
    consecutive_rejections: int = 0
    previous_parameter_delta_norm: float = 0.0
    bandit_action_cost: float = 0.000001
    drift_gate_threshold: float | None = None
    require_reward_signal_reconstruction: bool = False
    reward_signal_tolerance: float = 0.0
    # Reward Pilot 输出变化守门：限制一次 PEFT 候选对留出 Pilot logits 的平均改变量。
    # 设为 None 时关闭，避免把 Data 标签引入在线决策。
    reward_logit_change_tolerance: float | None = None
    # 延迟提交：当前帧只生成候选，下一帧用新的 Reward Pilot 验收后才写入模型。
    defer_peft_commit: bool = False
    # 状态 embedding 条件化默认关闭；打开时只改变在线 Adapter 的调制路径。
    state_embedding_conditioned: bool = False
    pending_peft_snapshot: object | None = None
    pending_peft_base_snapshot: object | None = None
    pending_peft_groups: frozenset[str] = frozenset()
    pending_peft_name: str = "skip"
    pending_peft_logit_delta: float = 0.0


@dataclass
class PilotRLSMethodState:
    """Pilot 驱动 RLS 残差 Adapter 的逐帧状态。"""

    model: UnfoldedEqualizer
    rls_adapter: PilotResidualRLSAdapter
    cir: torch.Tensor
    receiver_state: ReceiverState
    pretrained_loaded: bool
    acquisition_cfo: float = 0.0
    acquisition_phase_features: torch.Tensor | None = None
    condition_source: str = "acquisition"
    tail_update_alpha: float = 0.5
    min_reward_improvement: float = 0.001
    relative_min_reward_improvement: float = 0.0
    reward_pilot_windows: int = 2


def _select_methods(methods: list[str] | None) -> tuple[str, ...]:
    if not methods:
        return method_group("main")
    allowed = set(FORMAL_METHODS) | set(method_group("all")) | set(NEURAL_IMPLEMENTATION_METHODS)
    unknown = [method for method in methods if method not in allowed]
    if unknown:
        raise ValueError(f"未知比较方法：{unknown}")
    deduplicated = []
    for method in methods:
        if method not in deduplicated:
            deduplicated.append(method)
    return tuple(deduplicated)


def _build_method_states(
    methods: tuple[str, ...],
    initial_soft_tail: torch.Tensor,
    acquisition_cir: torch.Tensor,
    config: dict,
    delay: int,
    snr_db: float,
    seed: int,
    pretrained_path: Path | None = None,
    policy_path: Path | None = None,
    policy_required: bool = False,
    device: str = "cpu",
    cir_update_mode: str = "fixed",
    cir_alpha: float = 0.2,
    residual_cfo_limit: float = 0.001,
    acquisition_cfo: float = 0.0,
    acquisition_phase_features: torch.Tensor | None = None,
    online_groups_override: list[str] | None = None,
    online_condition_source_override: str | None = None,
    scheduler: str = "fixed",
) -> dict[str, BaselineMethodState | PPOMethodState | NeuralMethodState | RLModulatedMethodState]:
    if scheduler not in {"fixed", "bandit"}:
        raise ValueError(f"未知在线调度器：{scheduler}")
    states: dict[str, BaselineMethodState | PPOMethodState | NeuralMethodState | RLModulatedMethodState] = {}
    model_config = _load_model_config(config, pretrained_path)
    for method in methods:
        if method == "Pilot-Driven Online Adaptation":
            model = _build_equalizer(model_config, pretrained_path, device)
            if str(config.get("online_adaptation_algorithm", "sgd")) == "rls":
                rls_adapter = PilotResidualRLSAdapter(
                    model,
                    forgetting_factor=float(config.get("online_rls_forgetting_factor", 0.99)),
                    ridge=float(config.get("online_rls_ridge", 1.0)),
                    warmup_symbols=int(
                        config.get("online_rls_warmup_symbols", model_config.max_delay)
                    ),
                    target_logit=float(config.get("online_rls_target_logit", 2.0)),
                    max_delta_norm=float(
                        config.get("online_adaptation_max_delta_norm", 1.0)
                    ),
                    max_total_delta_norm=float(
                        config.get(
                            "online_rls_max_total_delta_norm",
                            config.get("online_adaptation_max_delta_norm", 1.0),
                        )
                    ),
                )
                states[method] = PilotRLSMethodState(
                    model=model,
                    rls_adapter=rls_adapter,
                    cir=acquisition_cir.clone().to(device),
                    receiver_state=ReceiverState(initial_soft_tail.clone().to(device)),
                    pretrained_loaded=pretrained_path is not None,
                    acquisition_cfo=float(acquisition_cfo),
                    acquisition_phase_features=(
                        acquisition_phase_features.clone().to(device)
                        if acquisition_phase_features is not None
                        else None
                    ),
                    condition_source=(
                        online_condition_source_override
                        if online_condition_source_override is not None
                        else str(
                            config.get(
                                "online_rls_condition_source",
                                config.get("online_condition_source", "acquisition"),
                            )
                        )
                    ),
                    tail_update_alpha=float(config.get("tail_update_alpha", 0.5)),
                    min_reward_improvement=float(
                        config.get("online_adaptation_min_reward_improvement", 0.001)
                    ),
                    relative_min_reward_improvement=float(
                        config.get("online_adaptation_relative_min_reward_improvement", 0.0)
                    ),
                    reward_pilot_windows=max(1, int(config.get("online_reward_windows", 2))),
                )
                continue
            online_groups, candidate_config = _online_groups_from_config(config, online_groups_override)
            if "physics_blend" in online_groups:
                model.attach_online_physics_blend_adapter()
            if "physics_residual" in online_groups:
                model.attach_online_physics_residual_adapter()
            if "phase_trend" in online_groups:
                model.attach_online_phase_trend_adapter()
            if "channel_residual" in online_groups:
                model.attach_online_channel_residual_adapter()
            if "logit_affine" in online_groups:
                model.attach_online_logit_affine_adapter()
            if "input_affine" in online_groups:
                model.attach_online_input_affine_adapter()
            if "input_trend" in online_groups:
                model.attach_online_input_trend_adapter()
            if "input_fir" in online_groups:
                model.attach_online_input_fir_adapter()
            if "logit_fir" in online_groups:
                model.attach_online_logit_fir_adapter()
            adapter = PilotDrivenOnlineAdapter(
                model,
                groups=online_groups,
                learning_rate=float(config.get("online_adaptation_learning_rate", 1e-4)),
                steps=int(config.get("online_adaptation_steps", 1)),
                max_delta_norm=float(config.get("online_adaptation_max_delta_norm", 0.5)),
                proximal_weight=float(config.get("online_adaptation_proximal_weight", 0.0)),
                hard_example_weighting=bool(config.get("online_hard_example_weighting", False)),
                hard_example_temperature=float(
                    config.get("online_hard_example_temperature", 0.5)
                ),
                objective=str(config.get("online_adaptation_objective", "bce")),
                optimizer=str(config.get("online_adaptation_optimizer", "sgd")),
            )
            candidate_specs = _online_candidate_specs(candidate_config, online_groups)
            states[method] = PilotOnlineMethodState(
                model=model,
                adapter=adapter,
                cir=acquisition_cir.clone().to(device),
                reference_cir=acquisition_cir.clone().to(device),
                receiver_state=ReceiverState(initial_soft_tail.clone().to(device)),
                pretrained_loaded=pretrained_path is not None,
                acquisition_cfo=float(acquisition_cfo),
                acquisition_phase_features=(
                    acquisition_phase_features.clone().to(device)
                    if acquisition_phase_features is not None
                    else None
                ),
                cir_update_mode=str(cir_update_mode),
                cir_update_alpha=float(cir_alpha),
                tail_update_alpha=float(config.get("tail_update_alpha", 0.5)),
                candidate_specs=candidate_specs,
                freeze_online_below_snr_db=(
                    float(config["online_adaptation_freeze_below_snr_db"])
                    if config.get("online_adaptation_freeze_below_snr_db") is not None
                    else None
                ),
                physical_state=PilotPhysicalState(cfo_cycles_per_symbol=float(acquisition_cfo)),
                phase_tracking_smoothing=float(config.get("online_phase_tracking_smoothing", 0.5)),
                phase_tracking_min_confidence=float(config.get("online_phase_tracking_min_confidence", 0.15)),
                phase_tracking_cfo_limit=float(config.get("online_phase_tracking_cfo_limit", residual_cfo_limit)),
                min_reward_improvement=float(config.get("online_adaptation_min_reward_improvement", 0.001)),
                relative_min_reward_improvement=float(
                    config.get("online_adaptation_relative_min_reward_improvement", 0.0)
                ),
                reward_pilot_windows=max(1, int(config.get("online_reward_windows", 2))),
                reward_ranking_mode=str(config.get("online_reward_ranking", "aggregate_bce")),
                reward_ranking_dispersion=float(
                    config.get("online_reward_ranking_dispersion", 0.5)
                ),
                cross_frame_rollback_tolerance=float(
                    config.get("online_cross_frame_rollback_tolerance", 0.00001)
                ),
                condition_source=_online_condition_source_from_config(
                    config,
                    online_condition_source_override,
                ),
                scheduler=str(scheduler),
                bandit=(
                    SafeContextualBandit(seed=90_000 + int(seed))
                    if scheduler == "bandit"
                    else None
                ),
                bandit_action_cost=float(config.get("online_bandit_action_cost", 0.000001)),
                drift_gate_threshold=(
                    float(config["online_drift_gate_threshold"])
                    if config.get("online_drift_gate_threshold") is not None
                    else None
                ),
                require_reward_signal_reconstruction=bool(
                    config.get("online_require_reward_signal_reconstruction", False)
                ),
                reward_signal_tolerance=float(
                    config.get("online_reward_signal_tolerance", 0.0)
                ),
                reward_logit_change_tolerance=(
                    float(config["online_reward_logit_change_tolerance"])
                    if config.get("online_reward_logit_change_tolerance") is not None
                    else None
                ),
                defer_peft_commit=bool(config.get("online_defer_peft_commit", False)),
                state_embedding_conditioned=bool(
                    config.get("online_state_embedding_conditioned", False)
                ),
            )
        elif method == "RL-Modulated Neural Block Equalizer":
            torch.manual_seed(_method_seed("rl_modulated", seed, delay, snr_db))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(_method_seed("rl_modulated", seed, delay, snr_db))
            model = _build_equalizer(model_config, pretrained_path, device)
            actions = safe_modulation_actions(len(model.blocks), device=device)
            from agent.rl_modulator import ModulationObservationEncoder

            encoder = ModulationObservationEncoder()
            policy = DiscreteSafePolicy(len(encoder.FIELDS), len(actions)).to(device)
            initialize_safe_discrete_policy_prior(policy, actions)
            policy_loaded = _load_discrete_policy_if_available(
                policy,
                policy_path,
                device,
                required=policy_required,
                expected_action_names=[action.name for action in actions],
            )
            states[method] = RLModulatedMethodState(
                online_state=WindowedDiscreteOnlineState(
                    cir=acquisition_cir.clone().to(device),
                    receiver_state=ReceiverState(initial_soft_tail.clone().to(device)),
                    model=model,
                    policy=policy,
                    optimizer=torch.optim.AdamW(policy.parameters(), lr=3e-5),
                    encoder=encoder,
                    actions=actions,
                    hidden=policy.initial_hidden(batch_size=1, device=torch.device(device)),
                    window_size=int(config.get("ppo_window_size", 8)),
                    previous_window_reward=0.0,
                    last_action_delta_norm=0.0,
                    cir_update_mode=str(cir_update_mode),
                    cir_update_alpha=float(cir_alpha),
                    tail_refinement_passes=int(config.get("tail_refinement_passes", 0)),
                    tail_update_alpha=float(config.get("tail_update_alpha", 1.0)),
                    rollout=[],
                ),
                pretrained_loaded=pretrained_path is not None,
                policy_loaded=policy_loaded,
                acquisition_cfo=float(acquisition_cfo),
            )
        elif method in NEURAL_IMPLEMENTATION_METHODS:
            model = _build_equalizer(model_config, pretrained_path, device)
            modulation_config = ModulationConfig(num_adapter_gates=len(model.blocks), num_lora_scales=len(model.blocks))
            if method == "F0 Acquisition Frozen NN":
                condition_update_mode = "fixed"
                condition_source = "acquisition"
            elif method == "F1 Pilot-Conditioned Frozen NN":
                condition_update_mode = "fixed"
                condition_source = "pilot_cir_phase"
            elif method == "Frozen Offline NN":
                # Frozen 只冻结离线网络参数；条件来源由配置显式指定，
                # 以保证 Frozen 与 Online 的纯 PEFT 比较使用同一状态条件。
                condition_update_mode = "fixed"
                condition_source = _online_condition_source_from_config(
                    config,
                    online_condition_source_override,
                )
            elif method == "Pilot CIR only":
                condition_update_mode = "pilot_sparse"
                condition_source = "pilot_cir_phase"
            elif method == "F2 Pilot State Tracking Frozen NN":
                # 正式主路线只更新 Pilot 后验状态，不更新任何神经参数。
                condition_update_mode = "pilot_sparse"
                condition_source = "pilot_cir_phase"
            else:
                condition_update_mode = str(cir_update_mode)
                condition_source = _online_condition_source_from_config(
                    config,
                    online_condition_source_override,
                )
            posterior = None
            if method == "F2 Pilot State Tracking Frozen NN":
                posterior = initial_state(
                    acquisition_cir.to(device), initial_soft_tail.to(device)
                )
                posterior = replace(
                    posterior,
                    phase=PilotPhysicalState(
                        cfo_cycles_per_symbol=float(acquisition_cfo),
                        confidence=0.5,
                    ),
                )
            states[method] = NeuralMethodState(
                cir=acquisition_cir.clone().to(device),
                receiver_state=ReceiverState(initial_soft_tail.clone().to(device)),
                model=model,
                modulation=_fixed_modulation_for(method, modulation_config, device),
                pretrained_loaded=pretrained_path is not None,
                acquisition_cfo=float(acquisition_cfo),
                cir_update_mode=str(cir_update_mode),
                cir_update_alpha=float(cir_alpha),
                condition_update_mode=condition_update_mode,
                condition_source=condition_source,
                acquisition_phase_features=(
                    acquisition_phase_features.clone().to(device)
                    if acquisition_phase_features is not None
                    else None
                ),
                physical_state=PilotPhysicalState(cfo_cycles_per_symbol=float(acquisition_cfo)),
                phase_tracking_smoothing=float(config.get("online_phase_tracking_smoothing", 0.5)),
                phase_tracking_min_confidence=float(config.get("online_phase_tracking_min_confidence", 0.15)),
                phase_tracking_cfo_limit=float(config.get("online_phase_tracking_cfo_limit", residual_cfo_limit)),
                tail_update_alpha=float(config.get("tail_update_alpha", 1.0)),
                posterior=posterior,
                state_tracking_mode=str(config.get("state_tracking_mode", "joint")),
            )
        elif method == "Continual PPO":
            torch.manual_seed(90_000 + int(seed) + int(delay) * 17 + int(float(snr_db)) * 31)
            policy = ContinualPolicy()
            _initialize_safe_policy_prior(policy)
            states[method] = PPOMethodState(
                cir=acquisition_cir.clone(),
                receiver_state=ReceiverState(initial_soft_tail.clone()),
                policy=policy,
                optimizer=torch.optim.AdamW(policy.parameters(), lr=3e-5),
                encoder=ObservationEncoder(),
                hidden=torch.zeros(1, 1, policy.hidden_size),
                previous_reward=0.0,
                last_parameter_delta_norm=0.0,
                rollout=[],
                last_good_cir=acquisition_cir.clone(),
                last_good_tail=initial_soft_tail.clone(),
            )
        else:
            states[method] = BaselineMethodState(
                cir=acquisition_cir.clone(),
                receiver_state=ReceiverState(initial_soft_tail.clone()),
                phase_state=TraditionalPhaseState(),
                residual_cfo_limit=float(residual_cfo_limit),
                cir_update_mode=str(cir_update_mode),
                cir_update_alpha=float(cir_alpha),
                acquisition_cfo=float(acquisition_cfo),
                pilot_physical_state=PilotPhysicalState(cfo_cycles_per_symbol=float(acquisition_cfo)),
            )
    return states


def _method_seed(method_key: str, seed: int, delay: int, snr_db: float) -> int:
    """为随机策略生成稳定种子，保证正式配对矩阵可复现。"""

    digest = hashlib.sha256(
        json.dumps(
            {"method": method_key, "seed": int(seed), "delay": int(delay), "snr_db": float(snr_db)},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return int(digest[:8], 16)


def _run_methods_for_frame(
    methods: tuple[str, ...],
    frame,
    snr_db: float,
    states: dict[str, BaselineMethodState | PPOMethodState | NeuralMethodState | RLModulatedMethodState],
    delay: int,
    frame_index: int,
    update_interval: int,
) -> dict[str, RealMethodResult]:
    results: dict[str, RealMethodResult] = {}
    legacy_batch = [
        method
        for method in methods
        if method in FORMAL_METHODS and method != "Continual PPO" and method not in NEURAL_IMPLEMENTATION_METHODS and method not in TRADITIONAL_BASELINES
    ]
    if legacy_batch:
        results.update(_run_baseline_method_batch(legacy_batch, frame, snr_db, states, delay=delay))
    for method in methods:
        if method in results:
            continue
        results[method] = _run_new_or_single_method(method, frame, snr_db, states[method], delay, frame_index, update_interval)
    return results


def _run_new_or_single_method(
    method: str,
    frame,
    snr_db: float,
    state: BaselineMethodState | PPOMethodState | NeuralMethodState | RLModulatedMethodState,
    delay: int,
    frame_index: int,
    update_interval: int,
) -> RealMethodResult:
    if method == "Continual PPO":
        return _run_real_method(method, frame, snr_db, state, delay, frame_index, update_interval)
    if method in TRADITIONAL_BASELINES:
        if not isinstance(state, BaselineMethodState):
            raise TypeError("传统 baseline 需要 BaselineMethodState。")
        if state.cir_update_mode == "pilot_sparse":
            state.pilot_physical_state = track_pilot_physical_state(
                frame,
                state.cir,
                state.receiver_state.soft_tail,
                previous=state.pilot_physical_state,
                cfo_limit=float(state.residual_cfo_limit),
            )
            state.cir = pilot_sparse_cir_update(
                frame,
                state.cir,
                state.receiver_state.soft_tail,
                max_paths=24,
                alpha=float(state.cir_update_alpha),
                cfo_hint=float(
                    state.pilot_physical_state.cfo_cycles_per_symbol
                    if state.pilot_physical_state is not None
                    else state.acquisition_cfo
                ),
            ).to(state.cir.device) if _pilot_state_reliable(state.pilot_physical_state, 0.15) else state.cir
        result = run_traditional_equalizer(
            method,
            frame.receiver_view(),
            state.cir,
            state.receiver_state.soft_tail,
            snr_db,
            phase_state=state.phase_state,
            residual_cfo_limit=state.residual_cfo_limit,
        )
        state.receiver_state.update_tail(result.soft_tail)
        return _result_from_logits(method, result.logits, frame, result.iterations, result.extra)
    if method == "RL-Modulated Neural Block Equalizer":
        if not isinstance(state, RLModulatedMethodState):
            raise TypeError("RL-Modulated 方法需要 RLModulatedMethodState。")
        frame_device = _frame_to_device(frame, next(state.online_state.model.parameters()).device)
        if state.online_state.cir_update_mode == "pilot_sparse":
            state.online_state.cir = pilot_sparse_cir_update(
                frame_device,
                state.online_state.cir,
                state.online_state.receiver_state.soft_tail,
                max_paths=24,
                alpha=float(state.online_state.cir_update_alpha),
                cfo_hint=float(state.acquisition_cfo),
            ).to(state.online_state.cir.device)
        phase_features = estimate_phase_residual_vector(
            frame.receiver_view(),
            state.online_state.cir,
            state.online_state.receiver_state.soft_tail,
            blocks=4,
            cfo_hint=state.acquisition_cfo,
        )
        condition = condition_from_cir(state.online_state.cir, snr_db, phase_features=phase_features)
        row = run_windowed_discrete_frame(state.online_state, frame_device, condition, snr_db, frame_index, update_interval)
        return RealMethodResult(
            method=method,
            input_hash=_hash_json({"frame": int(frame.frame_index), "method": method, "visible": "rx+adapt+reward"}),
            ber_data=float(row["ber_data"]),
            ber_reward_pilot=float(row["ber_reward_pilot"]),
            ber_adapt_pilot=float(row["ber_adapt_pilot"]),
            latency_ms=0.0,
            detector_iterations=int(row.get("detector_iterations", 0)),
            extra={**row, "pretrained_loaded": state.pretrained_loaded, "policy_loaded": state.policy_loaded},
        )
    if method == "Pilot-Driven Online Adaptation":
        if isinstance(state, PilotRLSMethodState):
            return _run_pilot_rls_method(
                method,
                frame,
                snr_db,
                state,
                delay,
                frame_index,
                update_interval,
            )
        if not isinstance(state, PilotOnlineMethodState):
            raise TypeError("Pilot-Driven Online Adaptation 需要 PilotOnlineMethodState。")
        return _run_pilot_online_method(
            method,
            frame,
            snr_db,
            state,
            delay,
            frame_index,
            update_interval,
        )
    if method in NEURAL_IMPLEMENTATION_METHODS:
        if not isinstance(state, NeuralMethodState):
            raise TypeError("神经 proposed 消融需要 NeuralMethodState。")
        device = next(state.model.parameters()).device
        frame_device = _frame_to_device(frame, device)
        if method == "F2 Pilot State Tracking Frozen NN":
            if state.posterior is None:
                raise RuntimeError("Pilot State Tracking 缺少初始物理状态后验。")
            update = update_from_adapt_pilot(
                frame_device,
                state.posterior,
                snr_db=float(snr_db),
                cfo_limit=float(state.phase_tracking_cfo_limit),
                phase_smoothing=float(state.phase_tracking_smoothing),
                min_confidence=float(state.phase_tracking_min_confidence),
                cir_alpha=float(state.cir_update_alpha),
                max_paths=24,
                frame_index=int(frame_index),
                track_cir=state.state_tracking_mode != "phase_only",
                track_phase=state.state_tracking_mode != "cir_only",
            )
            state_update_accepted = is_safe_to_commit(update, minimum_confidence=0.15)
            state.posterior = commit_if_safe(update, minimum_confidence=0.15)
            if state.state_tracking_mode == "joint" and state_update_accepted:
                previous_tail = state.posterior.tail
                pilot_tail = update_tail_from_adapt_pilot(
                    frame_device,
                    state.posterior,
                    snr_db=float(snr_db),
                )
                if (
                    torch.equal(pilot_tail.mean, previous_tail.mean)
                    and torch.equal(pilot_tail.variance, previous_tail.variance)
                ):
                    pilot_tail = inflate_tail_uncertainty(previous_tail)
                state.posterior = replace(state.posterior, tail=pilot_tail)
            state.cir = state.posterior.cir
            state.physical_state = state.posterior.phase
            phase_features = estimate_phase_residual_vector(
                frame_device.receiver_view(),
                state.posterior.cir,
                state.posterior.tail.mean,
                blocks=int(state.model.config.phase_correction_segments),
                cfo_hint=float(state.posterior.phase.cfo_cycles_per_symbol),
            ).to(device)
            condition = condition_from_cir(
                state.posterior.cir,
                float(snr_db),
                phase_features=phase_features,
            )
            full_rx_iq = torch.stack(
                (frame_device.rx_symbols.real, frame_device.rx_symbols.imag), dim=-1
            ).unsqueeze(0).float()
            adapt_symbols = frame_device.receiver_view().adapt_symbols.unsqueeze(0).to(torch.complex64)
            adapt_mask = frame_device.adapt_mask.unsqueeze(0).bool()
            region_ids = frame_device.model_region_ids.unsqueeze(0).long()
            risk = compute_pilot_risk(
                frame_device,
                state.posterior,
                noise_variance=10.0 ** (-float(snr_db) / 10.0),
            )
            with torch.no_grad():
                logits, _ = state.model(
                    full_rx_iq,
                    condition,
                    region_ids,
                    state.posterior.tail.mean.unsqueeze(0).to(device=device, dtype=torch.complex64),
                    modulation=state.modulation,
                    adapt_symbols=adapt_symbols,
                    adapt_mask=adapt_mask,
                )
            logits = logits.squeeze(0)
            state.receiver_state.update_tail(state.posterior.tail.mean)
            return _result_from_logits(
                method,
                logits,
                frame_device,
                0,
                {
                    "uses_neural_network": True,
                    "uses_rl": False,
                    "pretrained_loaded": state.pretrained_loaded,
                    "online_parameter_update": False,
                    "physical_state_update": bool(state_update_accepted),
                    "physical_state_update_reason": update.reason,
                    "state_tracking_mode": str(state.state_tracking_mode),
                    "adapt_pilot_count": int(update.adapt_pilot_count),
                    "physical_state_confidence": float(state.posterior.confidence),
                    "residual_cfo": float(state.posterior.phase.cfo_cycles_per_symbol),
                    "tail_variance_mean": float(state.posterior.tail.variance.mean().detach().cpu()),
                    "tail_confidence": float(state.posterior.tail.confidence),
                    "tail_update_source": (
                        "adapt_pilot_map"
                        if state.state_tracking_mode == "joint" and state_update_accepted
                        else "frozen_prior"
                    ),
                    "pilot_risk_score": float(risk.score),
                    "pilot_risk_abstain": bool(risk.abstain),
                    "pilot_risk_validated_for_data_ber": False,
                    "data_labels_used_online": False,
                    "online_waveform_visibility": "adapt_pilot_for_state; full_rx_for_frozen_detection; no_data_labels_for_update",
                },
            )
        if state.condition_update_mode == "pilot_sparse":
            state.physical_state = track_pilot_physical_state(
                frame_device,
                state.cir,
                state.receiver_state.soft_tail,
                previous=state.physical_state,
                cfo_limit=float(state.phase_tracking_cfo_limit),
                smoothing=float(state.phase_tracking_smoothing),
                min_confidence=float(state.phase_tracking_min_confidence),
            )
            state.cir = pilot_sparse_cir_update(
                frame_device,
                state.cir,
                state.receiver_state.soft_tail,
                max_paths=24,
                alpha=float(state.cir_update_alpha),
                cfo_hint=float(
                    state.physical_state.cfo_cycles_per_symbol
                    if state.physical_state is not None
                    else state.acquisition_cfo
                ),
            ).to(device) if _pilot_state_reliable(
                state.physical_state,
                state.phase_tracking_min_confidence,
            ) else state.cir
        full_rx_iq = torch.stack(
            (frame_device.rx_symbols.real, frame_device.rx_symbols.imag), dim=-1
        ).unsqueeze(0).float()
        pilot_visible_mask = (frame_device.adapt_mask | frame_device.reward_mask).to(
            device=device, dtype=torch.bool
        )
        online_rx_symbols = torch.where(
            pilot_visible_mask,
            frame_device.rx_symbols,
            torch.zeros_like(frame_device.rx_symbols),
        )
        rx_iq = torch.stack((online_rx_symbols.real, online_rx_symbols.imag), dim=-1).unsqueeze(0).float()
        region_ids = frame_device.model_region_ids.unsqueeze(0).long()
        phase_features = _neural_phase_features(
            condition_source=state.condition_source,
            receiver_view=frame.receiver_view(),
            cir=state.cir,
            soft_tail=state.receiver_state.soft_tail,
            acquisition_cfo=float(state.acquisition_cfo),
            tracked_cfo=_tracked_cfo_or_none(
                state.physical_state,
                state.phase_tracking_min_confidence,
            ),
            acquisition_phase_features=state.acquisition_phase_features,
        )
        condition = condition_from_cir(
            state.cir,
            snr_db,
            cfo_residual=float(state.acquisition_cfo),
            phase_features=phase_features,
        )
        logits, _ = state.model(
            rx_iq,
            condition,
            region_ids,
            state.receiver_state.soft_tail.unsqueeze(0).to(torch.complex64),
            modulation=state.modulation,
            adapt_symbols=frame_device.receiver_view().adapt_symbols.unsqueeze(0).to(torch.complex64),
            adapt_mask=frame_device.adapt_mask.unsqueeze(0).bool(),
        )
        logits = logits.squeeze(0)
        tail_len = state.receiver_state.soft_tail.numel()
        detected_tail = torch.complex(
            torch.tanh(logits[-tail_len:] / 2.0),
            torch.zeros_like(logits[-tail_len:]),
        )
        tail_visible_mask = pilot_visible_mask[-tail_len:]
        detected_tail = torch.where(
            tail_visible_mask,
            detected_tail,
            state.receiver_state.soft_tail,
        )
        state.receiver_state.update_tail(
            _update_receiver_tail(
                state.receiver_state.soft_tail,
                detected_tail,
                state.tail_update_alpha,
            )
        )
        if state.cir_update_mode == "decision_directed":
            state.cir = decision_directed_cir_update(
                frame_device,
                logits.detach(),
                int(delay),
                state.cir,
                alpha=float(state.cir_update_alpha),
            ).to(device)
        # 在线状态更新已经完成；完整波形只用于事后输出评估。
        with torch.no_grad():
            full_logits, _ = state.model(
                full_rx_iq,
                condition,
                region_ids,
                state.receiver_state.soft_tail.unsqueeze(0).to(torch.complex64),
                modulation=state.modulation,
                adapt_symbols=frame_device.receiver_view().adapt_symbols.unsqueeze(0).to(torch.complex64),
                adapt_mask=frame_device.adapt_mask.unsqueeze(0).bool(),
            )
        return _result_from_logits(
            method,
            full_logits.squeeze(0),
            frame_device,
            0,
            {
                "uses_neural_network": True,
                "uses_rl": False,
                "pretrained_loaded": state.pretrained_loaded,
                "cir_update_mode": str(state.cir_update_mode),
                "condition_update_mode": str(state.condition_update_mode),
                "condition_source": str(state.condition_source),
                "pilot_phase_used": bool(state.condition_source != "acquisition"),
                "cir_update_applied": bool(state.condition_update_mode == "pilot_sparse"),
                "peft_update_applied": False,
                "cir_update_alpha": float(state.cir_update_alpha),
                "cir_update_uses_data_labels": False,
                "data_labels_used_online": False,
            },
        )
    if method in DIAGNOSTIC_METHODS:
        if not isinstance(state, BaselineMethodState):
            raise TypeError("诊断方法需要 BaselineMethodState。")
        cir = frame.true_cir if method in {"Perfect-CSI Block", "Perfect-CSI + Pilot Phase"} else state.cir
        rx = frame.rx_symbols
        diagnostic_extra = {"diagnostic": True, "phase_compensation": False}
        if method == "Perfect-CSI + Pilot Phase":
            phase0, cfo = estimate_phase_residual_features(
                frame.receiver_view(),
                frame.true_cir,
                state.receiver_state.soft_tail,
            )
            rx = apply_phase_correction(frame.rx_symbols, phase0, cfo)
            diagnostic_extra.update(
                {
                    "phase_compensation": True,
                    "pilot_phase_used": True,
                    "diagnostic_phase0": float(phase0),
                    "diagnostic_cfo_cycles_per_symbol": float(cfo),
                }
            )
        result = perfect_csi_bpsk_refine_detect(
            rx,
            cir,
            state.receiver_state.soft_tail,
            torch.tensor(10.0 ** (-float(snr_db) / 10.0)),
            cg_iterations=32,
            refine_iterations=2,
        )
        state.receiver_state.update_tail(result.soft_tail)
        return _result_from_logits(method, result.logits, frame, result.iterations, diagnostic_extra)
    return _run_real_method(method, frame, snr_db, state, delay, frame_index, update_interval)


def _run_pilot_rls_method(
    method: str,
    frame,
    snr_db: float,
    state: PilotRLSMethodState,
    delay: int,
    frame_index: int,
    update_interval: int,
) -> RealMethodResult:
    """运行一帧只由有效 Pilot 驱动的 RLS 残差 Adapter。"""

    del delay, frame_index, update_interval
    device = next(state.model.parameters()).device
    frame_device = _frame_to_device(frame, device)
    # RLS 的在线决策只能看到两段已知 Pilot。完整波形保留到所有参数
    # 更新、Reward 守门和状态维护完成之后，仅用于最终 Data 评估。
    full_rx_iq = torch.stack(
        (frame_device.rx_symbols.real, frame_device.rx_symbols.imag),
        dim=-1,
    ).unsqueeze(0).float()
    pilot_visible_mask = (frame_device.adapt_mask | frame_device.reward_mask).to(
        device=device, dtype=torch.bool
    )
    online_rx_symbols = torch.where(
        pilot_visible_mask,
        frame_device.rx_symbols,
        torch.zeros_like(frame_device.rx_symbols),
    )
    rx_iq = torch.stack(
        (online_rx_symbols.real, online_rx_symbols.imag),
        dim=-1,
    ).unsqueeze(0).float()
    region_ids = frame_device.model_region_ids.unsqueeze(0).long()
    tail = state.receiver_state.soft_tail.unsqueeze(0).to(torch.complex64)
    phase_features = _neural_phase_features(
        condition_source=state.condition_source,
        receiver_view=frame_device.receiver_view(),
        cir=state.cir,
        soft_tail=state.receiver_state.soft_tail,
        acquisition_cfo=float(state.acquisition_cfo),
        acquisition_phase_features=state.acquisition_phase_features,
    )
    condition = condition_from_cir(
        state.cir,
        snr_db,
        cfo_residual=float(state.acquisition_cfo),
        phase_features=phase_features,
    )
    pilot_state_summary = build_pilot_state_summary(
        cir=state.cir,
        reference_cir=state.cir,
        phase0=0.0,
        cfo_cycles_per_symbol=float(state.acquisition_cfo),
        noise_variance=condition.noise_variance,
        confidence=condition.confidence,
        reconstruction_error=0.0,
    )
    pilot_state_embedding = PilotStateEmbedding()(pilot_state_summary)
    adapt_symbols = frame_device.receiver_view().adapt_symbols.unsqueeze(0).to(torch.complex64)
    adapt_mask = frame_device.adapt_mask.unsqueeze(0).bool()
    old_weight = state.rls_adapter.module.linear.weight.detach().clone()
    old_bias = state.rls_adapter.module.linear.bias.detach().clone()
    old_covariance = state.rls_adapter.covariance.detach().clone()
    with torch.no_grad():
        logits_before, _ = state.model(
            rx_iq,
            condition,
            region_ids,
            tail,
            adapt_symbols=adapt_symbols,
            adapt_mask=adapt_mask,
        )
    adaptation = state.rls_adapter.adapt(frame_device, condition, tail)
    with torch.no_grad():
        logits_after, _ = state.model(
            rx_iq,
            condition,
            region_ids,
            tail,
            adapt_symbols=adapt_symbols,
            adapt_mask=adapt_mask,
        )
    logits_before = logits_before.squeeze(0)
    candidate_logits = logits_after.squeeze(0)
    reward_view = frame_device.receiver_view()
    reward_labels = (reward_view.reward_symbols.real > 0.0).to(frame_device.bits.dtype)
    reward_before = _masked_bce(
        logits_before,
        reward_labels,
        frame_device.reward_mask,
    )
    reward_after = _masked_bce(
        candidate_logits,
        reward_labels,
        frame_device.reward_mask,
    )
    reward_window_accepted, reward_window_gains = _accept_windowed_reward_update(
        logits_before,
        candidate_logits,
        reward_labels,
        frame_device.reward_mask,
        min_improvement=float(state.min_reward_improvement),
        windows=int(state.reward_pilot_windows),
        relative_min_improvement=float(state.relative_min_reward_improvement),
    )
    reward_hard_ber_accepted = _accept_reward_pilot_hard_ber(
        logits_before,
        candidate_logits,
        reward_mask=frame_device.reward_mask,
        reward_symbols=reward_view.reward_symbols,
    )
    pilot_state_summary = build_pilot_state_summary(
        cir=state.cir,
        reference_cir=state.cir,
        phase0=0.0,
        cfo_cycles_per_symbol=float(state.acquisition_cfo),
        noise_variance=condition.noise_variance,
        confidence=condition.confidence,
        reconstruction_error=0.0,
    )
    pilot_state_embedding = PilotStateEmbedding()(pilot_state_summary)
    accepted = bool(
        adaptation.accepted
        and reward_window_accepted
        and reward_hard_ber_accepted
    )
    if not accepted:
        # Reward Pilot 只负责留出验收；失败时恢复 RLS 参数和协方差，
        # 确保被拒绝的候选不会污染下一帧的在线状态。
        with torch.no_grad():
            state.rls_adapter.module.linear.weight.copy_(old_weight)
            state.rls_adapter.module.linear.bias.copy_(old_bias)
        state.rls_adapter.covariance = old_covariance
        state.rls_adapter.parameter_delta_norm = 0.0
        current_parameters = torch.cat(
            [old_weight.reshape(-1).float(), old_bias.reshape(-1).float()]
        )
        anchor_parameters = torch.cat(
            [
                state.rls_adapter.anchor_weight.reshape(-1).float(),
                state.rls_adapter.anchor_bias.reshape(-1).float(),
            ]
        )
        state.rls_adapter.cumulative_delta_norm = float(
            torch.linalg.vector_norm(current_parameters - anchor_parameters).cpu()
        )
        logits = logits_before
    else:
        logits = candidate_logits
    reward_hard_errors_before = int(
        (
            (logits_before[frame_device.reward_mask] >= 0.0)
            != (reward_labels[frame_device.reward_mask] >= 0.5)
        ).sum().item()
    )
    reward_hard_errors_after = int(
        (
            (candidate_logits[frame_device.reward_mask] >= 0.0)
            != (reward_labels[frame_device.reward_mask] >= 0.5)
        ).sum().item()
    )
    tail_len = state.receiver_state.soft_tail.numel()
    tail_visible_mask = pilot_visible_mask[-tail_len:]
    detected_tail = torch.complex(
        torch.tanh(logits[-tail_len:] / 2.0),
        torch.zeros_like(logits[-tail_len:]),
    )
    detected_tail = torch.where(
        tail_visible_mask,
        detected_tail,
        state.receiver_state.soft_tail,
    )
    state.receiver_state.update_tail(
        _update_receiver_tail(
            state.receiver_state.soft_tail,
            detected_tail,
            state.tail_update_alpha,
        )
    )
    # 在线决策已经结束。此处完整波形只生成事后评估输出，不参与 RLS
    # 参数、Reward 守门或跨帧状态更新。
    with torch.no_grad():
        full_logits, _ = state.model(
            full_rx_iq,
            condition,
            region_ids,
            tail,
            adapt_symbols=adapt_symbols,
            adapt_mask=adapt_mask,
        )
    final_logits = full_logits.squeeze(0)
    return _result_from_logits(
        method,
        final_logits,
        frame_device,
        0,
        {
            "uses_neural_network": True,
            "uses_rl": False,
            "pretrained_loaded": state.pretrained_loaded,
            "online_algorithm": "pilot_supervised_residual_rls",
            "online_update_source": "adapt_pilot_only",
            "condition_source": str(state.condition_source),
            "pilot_phase_used": bool(state.condition_source != "acquisition"),
            "adapt_pilot_count": int(adaptation.adapt_pilot_count),
            "adapt_loss_before": float(adaptation.adapt_loss_before),
            "adapt_loss_after": float(adaptation.adapt_loss_after),
            "pilot_mse_before": float(adaptation.pilot_mse_before),
            "pilot_mse_after": float(adaptation.pilot_mse_after),
            "adaptation_accepted": bool(adaptation.accepted),
            "peft_update_applied": bool(accepted),
            "parameter_delta_norm": float(
                adaptation.parameter_delta_norm if accepted else 0.0
            ),
            "cumulative_parameter_delta_norm": float(
                state.rls_adapter.cumulative_delta_norm
            ),
            "max_cumulative_parameter_delta_norm": float(
                state.rls_adapter.max_total_delta_norm
            ),
            "reward_pilot_guard": True,
            "reward_pilot_guard_passed": bool(accepted),
            "reward_pilot_window_gains": reward_window_gains,
            "reward_pilot_hard_ber_guard": bool(reward_hard_ber_accepted),
            "reward_pilot_hard_errors_before": reward_hard_errors_before,
            "reward_pilot_hard_errors_after": reward_hard_errors_after,
            "reward_pilot_loss_before": float(reward_before.detach().cpu()),
            "reward_pilot_loss_after": float(reward_after.detach().cpu()),
            "pilot_state_embedding": [float(value) for value in pilot_state_embedding.tolist()],
            "pilot_state_cir_residual_norm": float(pilot_state_summary.cir_residual_norm),
            "pilot_state_noise_variance": float(pilot_state_summary.noise_variance),
            "pilot_state_reconstruction_error": float(pilot_state_summary.reconstruction_error),
            "pilot_state_drift_gate_applied": False,
            "reward_pilot_min_improvement": float(state.min_reward_improvement),
            "reward_pilot_relative_min_improvement": float(
                state.relative_min_reward_improvement
            ),
            "pilot_state_embedding": [float(value) for value in pilot_state_embedding.tolist()],
            "pilot_state_cir_residual_norm": float(pilot_state_summary.cir_residual_norm),
            "pilot_state_noise_variance": float(pilot_state_summary.noise_variance),
            "pilot_state_reconstruction_error": float(pilot_state_summary.reconstruction_error),
            "pilot_state_drift_gate_applied": False,
            "data_labels_used_online": False,
            "data_waveform_used_online": False,
            "online_waveform_visibility": "adapt_pilot_for_update_reward_pilot_for_guard",
        },
    )


def _run_real_method(method: str, frame, snr_db: float, state: BaselineMethodState | PPOMethodState, delay: int, frame_index: int, update_interval: int) -> RealMethodResult:
    if method == "Continual PPO":
        if not isinstance(state, PPOMethodState):
            raise TypeError("Continual PPO 需要 PPOMethodState。")
        return _run_ppo_method(method, frame, snr_db, state, delay, frame_index, update_interval)
    if not isinstance(state, BaselineMethodState):
        raise TypeError("baseline 方法需要 BaselineMethodState。")
    sigma = torch.tensor(10.0 ** (-float(snr_db) / 10.0))
    settings = {
        "Perfect-CSI Block": (frame.true_cir, 32, 2),
        "Sparse CIR + Kalman/RLS": (state.cir, 32, 1),
        "Block LMMSE/CG": (state.cir, 32, 0),
        "DFE-RLS": (state.cir, 16, 1),
        "Analytic Iterative BPSK": (state.cir, 16, 2),
        "No Adapt": (state.cir, 8, 0),
        "Best Fixed": (state.cir, 32, 2),
        "Drift-Aware Pilot Rule": (state.cir, 32, 1),
        "Contextual Bandit": (state.cir, 32, 1),
    }
    if method not in settings:
        raise ValueError(f"未知比较方法：{method}")
    cir, cg_iterations, refine_iterations = settings[method]
    # 先用 Pilot-only 前向更新在线 CIR/tail。完整接收帧只在状态更新完成后
    # 生成最终评估输出，不能通过非因果块检测器进入在线状态。
    pilot_visible_mask = (frame.adapt_mask | frame.reward_mask).to(dtype=torch.bool)
    pilot_rx = torch.where(frame.adapt_mask | frame.reward_mask, frame.rx_symbols, torch.zeros_like(frame.rx_symbols))
    pilot_result = perfect_csi_bpsk_refine_detect(
        pilot_rx,
        cir,
        state.receiver_state.soft_tail,
        sigma,
        cg_iterations=cg_iterations,
        refine_iterations=refine_iterations,
    )
    tail_len = state.receiver_state.soft_tail.numel()
    if bool(pilot_visible_mask[-tail_len:].any()):
        visible_tail = pilot_visible_mask[-tail_len:]
        state.receiver_state.update_tail(
            torch.where(visible_tail, pilot_result.soft_tail, state.receiver_state.soft_tail)
        )
    _update_baseline_cir(method, frame, pilot_result.logits, delay, state)
    cir_for_output = frame.true_cir if method == "Perfect-CSI Block" else state.cir
    result = perfect_csi_bpsk_refine_detect(
        frame.rx_symbols,
        cir_for_output,
        state.receiver_state.soft_tail,
        sigma,
        cg_iterations=cg_iterations,
        refine_iterations=refine_iterations,
    )
    return RealMethodResult(
        method=method,
        input_hash=_hash_json({"seed": int(frame.frame_index), "method": method, "visible": "rx+adapt"}),
        ber_data=bit_error_rate(result.logits[frame.data_mask], frame.bits[frame.data_mask]),
        ber_reward_pilot=bit_error_rate(result.logits[frame.reward_mask], frame.bits[frame.reward_mask]),
        ber_adapt_pilot=bit_error_rate(result.logits[frame.adapt_mask], frame.bits[frame.adapt_mask]),
        latency_ms=0.0,
        detector_iterations=int(cg_iterations + refine_iterations),
        extra={
            **_baseline_extra(method),
            "data_waveform_used_online": False,
            "online_waveform_visibility": "adapt_pilot_for_update_reward_pilot_for_guard",
        },
    )


def _run_baseline_method_batch(methods: list[str], frame, snr_db: float, states: dict[str, BaselineMethodState | PPOMethodState], delay: int) -> dict[str, RealMethodResult]:
    # 该兼容批处理入口优先复用严格隔离的单方法实现。批量化只影响速度，
    # 不能让整帧块检测结果重新成为在线 CIR/tail 更新来源。
    return {
        method: _run_real_method(
            method,
            frame,
            snr_db,
            states[method],
            delay,
            frame_index=int(frame.frame_index),
            update_interval=1,
        )
        for method in methods
    }


def _baseline_settings(method: str, frame, state: BaselineMethodState) -> tuple[torch.Tensor, int, int]:
    settings = {
        "Perfect-CSI Block": (frame.true_cir, 32, 2),
        "Sparse CIR + Kalman/RLS": (state.cir, 32, 1),
        "Block LMMSE/CG": (state.cir, 32, 0),
        "DFE-RLS": (state.cir, 16, 1),
        "Analytic Iterative BPSK": (state.cir, 16, 2),
        "No Adapt": (state.cir, 8, 0),
        "Best Fixed": (state.cir, 32, 2),
        "Drift-Aware Pilot Rule": (state.cir, 32, 1),
        "Contextual Bandit": (state.cir, 32, 1),
    }
    if method not in settings:
        raise ValueError(f"未知 baseline 方法：{method}")
    return settings[method]


def _baseline_cir_update_alpha(method: str) -> float | None:
    """返回固定 baseline 的在线 CIR 跟踪强度；None 表示静态基线。"""

    if method in {"Sparse CIR + Kalman/RLS", "DFE-RLS", "Best Fixed", "Drift-Aware Pilot Rule", "Contextual Bandit"}:
        return 0.2
    return None


def _update_baseline_cir(method: str, frame, logits: torch.Tensor, delay: int, state: BaselineMethodState) -> None:
    alpha = _baseline_cir_update_alpha(method)
    if alpha is None:
        return
    state.cir = _decision_directed_cir_update(frame, logits, int(delay), state.cir, alpha=alpha)


def _baseline_extra(method: str) -> dict:
    alpha = _baseline_cir_update_alpha(method)
    if alpha is None:
        return {}
    return {"cir_update": "decision_directed", "cir_update_alpha": float(alpha)}


def _result_from_logits(method: str, logits: torch.Tensor, frame, iterations: int, extra: dict) -> RealMethodResult:
    return RealMethodResult(
        method=method,
        input_hash=_hash_json({"seed": int(frame.frame_index), "method": method, "visible": "rx+adapt"}),
        ber_data=bit_error_rate(logits[frame.data_mask], frame.bits[frame.data_mask]),
        ber_reward_pilot=bit_error_rate(logits[frame.reward_mask], frame.bits[frame.reward_mask]),
        ber_adapt_pilot=bit_error_rate(logits[frame.adapt_mask], frame.bits[frame.adapt_mask]),
        latency_ms=0.0,
        detector_iterations=int(iterations),
        extra=extra,
    )


def _data_eval_metrics(
    before_logits: torch.Tensor,
    after_logits: torch.Tensor,
    frame,
) -> dict[str, float]:
    """计算 Data 区域的事后诊断指标。

    该函数只应在在线候选排序、验收、回滚和状态更新全部结束后调用。返回值
    仅用于离线分析，不得回流到在线控制路径。
    """

    device = after_logits.device
    data_mask = frame.data_mask.to(device=device, dtype=torch.bool)
    bits = frame.bits.to(device=device)
    if not bool(data_mask.any()):
        return {
            "bce_before": 0.0,
            "bce_after": 0.0,
            "bce_improvement": 0.0,
            "ber_before": 0.0,
            "ber_after": 0.0,
            "ber_improvement": 0.0,
            "logits_abs_delta_mean": 0.0,
            "logits_direction_change_rate": 0.0,
            "soft_output_delta_mean": 0.0,
        }
    bce_before = _masked_bce(before_logits, bits, data_mask)
    bce_after = _masked_bce(after_logits, bits, data_mask)
    ber_before = bit_error_rate(before_logits[data_mask], bits[data_mask])
    ber_after = bit_error_rate(after_logits[data_mask], bits[data_mask])
    before_data = before_logits[data_mask]
    after_data = after_logits[data_mask]
    return {
        "bce_before": float(bce_before.detach().cpu()),
        "bce_after": float(bce_after.detach().cpu()),
        "bce_improvement": float((bce_before - bce_after).detach().cpu()),
        "ber_before": float(ber_before),
        "ber_after": float(ber_after),
        "ber_improvement": float(ber_before - ber_after),
        "logits_abs_delta_mean": float(
            torch.mean(torch.abs(after_data - before_data)).detach().cpu()
        ),
        "logits_direction_change_rate": float(
            ((before_data >= 0.0) != (after_data >= 0.0))
            .to(torch.float32)
            .mean()
            .detach()
            .cpu()
        ),
        "soft_output_delta_mean": float(
            torch.mean(
                torch.abs(torch.sigmoid(after_data) - torch.sigmoid(before_data))
            )
            .detach()
            .cpu()
        ),
    }


def _run_pilot_online_method(
    method: str,
    frame,
    snr_db: float,
    state: PilotOnlineMethodState,
    delay: int,
    frame_index: int,
    update_interval: int,
) -> RealMethodResult:
    """运行一帧不依赖 PPO 的 Pilot 驱动在线适配。"""

    device = next(state.model.parameters()).device
    frame_device = _frame_to_device(frame, device)
    online_parameter_groups = tuple(sorted(state.adapter.groups))
    online_parameter_items = state.model.peft.named_group_parameters(
        online_parameter_groups
    )
    online_parameter_count = int(
        sum(parameter.numel() for _, parameter in online_parameter_items)
    )
    cir_before_frame = state.cir.detach().clone()
    pilot_state_summary = build_pilot_state_summary(
        cir=state.cir,
        reference_cir=state.cir,
        phase0=float(state.physical_state.phase0 if state.physical_state is not None else 0.0),
        cfo_cycles_per_symbol=float(
            state.physical_state.cfo_cycles_per_symbol
            if state.physical_state is not None
            else state.acquisition_cfo
        ),
        noise_variance=0.0,
        confidence=float(state.physical_state.confidence if state.physical_state is not None else 0.0),
        reconstruction_error=0.0,
    )
    pilot_state_embedding = PilotStateEmbedding()(pilot_state_summary)
    updates_frozen = _online_updates_are_frozen(snr_db, state.freeze_online_below_snr_db)
    update_scheduled = _online_update_is_scheduled(frame_index, update_interval)
    pilot_drift_distance = 0.0
    pilot_drift_gate_applied = False
    pilot_drift_gate_allows_update = True
    # 在线更新和 Reward 验收严格使用 Pilot 可见波形。Adapt 更新只看
    # Adapt Pilot；Reward 验收可以看 Adapt + Reward Pilot；Data 波形只在
    # 下面的最终离线评估前向中使用。
    full_rx_iq = torch.stack(
        (frame_device.rx_symbols.real, frame_device.rx_symbols.imag), dim=-1
    ).unsqueeze(0).float()
    pilot_visible_mask = (frame_device.adapt_mask | frame_device.reward_mask).to(
        device=device, dtype=torch.bool
    )
    online_rx_symbols = torch.where(
        pilot_visible_mask,
        frame_device.rx_symbols,
        torch.zeros_like(frame_device.rx_symbols),
    )
    rx_iq = torch.stack((online_rx_symbols.real, online_rx_symbols.imag), dim=-1).unsqueeze(0).float()
    region_ids = frame_device.model_region_ids.unsqueeze(0).long()
    adapt_symbols = frame_device.receiver_view().adapt_symbols.unsqueeze(0).to(torch.complex64)
    adapt_mask = frame_device.adapt_mask.unsqueeze(0).bool()
    cir_update_accepted = False
    cir_reward_loss_before = None
    cir_reward_loss_after = None
    tracked_state = None
    # CIR 候选守门发生在状态 embedding 计算之前；该阶段不使用 embedding，
    # 先显式设为空，避免条件分支引用未初始化变量。
    online_state_embedding = None
    # 即使当前实验不允许把 Pilot CIR 写回条件，也先用 Adapt Pilot 观测
    # 当前物理状态，供状态 embedding 调制 PEFT；这不会改变冻结主干条件。
    if not updates_frozen:
        observed_state = track_pilot_physical_state(
            frame_device,
            state.cir,
            state.receiver_state.soft_tail,
            previous=state.physical_state,
            cfo_limit=float(state.phase_tracking_cfo_limit),
            smoothing=float(state.phase_tracking_smoothing),
            min_confidence=float(state.phase_tracking_min_confidence),
        )
        tracked_state = observed_state
        if state.cir_update_mode == "pilot_sparse":
            state.physical_state = observed_state
    if state.cir_update_mode == "pilot_sparse" and not updates_frozen:
        previous_cir = state.cir.clone()
        candidate_cir = state.cir
        if update_scheduled and _pilot_state_reliable(
            tracked_state,
            state.phase_tracking_min_confidence,
        ):
            candidate_cir = pilot_sparse_cir_update(
                frame_device,
                state.cir,
                state.receiver_state.soft_tail,
                max_paths=24,
                alpha=float(state.cir_update_alpha),
                cfo_hint=float(tracked_state.cfo_cycles_per_symbol),
            ).to(device)
        if update_scheduled and _pilot_state_reliable(tracked_state, state.phase_tracking_min_confidence):
            tail = state.receiver_state.soft_tail.unsqueeze(0).to(torch.complex64)

            def _condition_for(cir_value: torch.Tensor) -> CIRCondition:
                phase_value = _neural_phase_features(
                    condition_source=state.condition_source,
                    receiver_view=frame.receiver_view(),
                    cir=cir_value,
                    soft_tail=state.receiver_state.soft_tail,
                    acquisition_cfo=float(state.acquisition_cfo),
                    tracked_cfo=float(tracked_state.cfo_cycles_per_symbol),
                    acquisition_phase_features=state.acquisition_phase_features,
                )
                return condition_from_cir(
                    cir_value,
                    snr_db,
                    cfo_residual=float(state.acquisition_cfo),
                    phase_features=phase_value,
                )

            with torch.no_grad():
                previous_logits, _ = state.model(
                    rx_iq,
                    _condition_for(previous_cir),
                    region_ids,
                    tail,
                    adapt_symbols=adapt_symbols,
                    adapt_mask=adapt_mask,
                    online_state_embedding=online_state_embedding,
                )
                candidate_logits, _ = state.model(
                    rx_iq,
                    _condition_for(candidate_cir),
                    region_ids,
                    tail,
                    adapt_symbols=adapt_symbols,
                    adapt_mask=adapt_mask,
                    online_state_embedding=online_state_embedding,
                )
            reward_view = frame_device.receiver_view()
            reward_labels = (reward_view.reward_symbols.real > 0.0).to(frame_device.bits.dtype)
            cir_reward_loss_before = float(
                _masked_bce(previous_logits.squeeze(0), reward_labels, frame_device.reward_mask).detach().cpu()
            )
            cir_reward_loss_after = float(
                _masked_bce(candidate_logits.squeeze(0), reward_labels, frame_device.reward_mask).detach().cpu()
            )
            cir_update_accepted = _accept_pilot_cir_update(cir_reward_loss_before, cir_reward_loss_after)
            state.cir = candidate_cir if cir_update_accepted else previous_cir
    tail = state.receiver_state.soft_tail.unsqueeze(0).to(torch.complex64)
    phase_features = _neural_phase_features(
        condition_source=state.condition_source,
        receiver_view=frame.receiver_view(),
        cir=state.cir,
        soft_tail=state.receiver_state.soft_tail,
        acquisition_cfo=float(state.acquisition_cfo),
        tracked_cfo=(
            _tracked_cfo_or_none(
                state.physical_state,
                state.phase_tracking_min_confidence,
            )
            if not updates_frozen
            else None
        ),
        acquisition_phase_features=state.acquisition_phase_features,
    )
    condition = condition_from_cir(
        state.cir,
        snr_db,
        cfo_residual=float(state.acquisition_cfo),
        phase_features=phase_features,
    )
    # 这里的状态只来自当前 Adapt Pilot 可见量；Data 标签不会进入 embedding。
    embedding_cir = state.cir
    embedding_state = tracked_state if tracked_state is not None else state.physical_state
    if (
        state.state_embedding_conditioned
        and not updates_frozen
        and _pilot_state_reliable(embedding_state, state.phase_tracking_min_confidence)
    ):
        embedding_cir = pilot_sparse_cir_update(
            frame_device,
            state.cir,
            state.receiver_state.soft_tail,
            max_paths=24,
            alpha=1.0,
            cfo_hint=float(embedding_state.cfo_cycles_per_symbol),
        ).to(device)
    pilot_state_summary = build_pilot_state_summary(
        cir=embedding_cir,
        reference_cir=(
            state.reference_cir
            if state.reference_cir is not None
            else state.cir
        ),
        phase0=float(embedding_state.phase0 if embedding_state is not None else 0.0),
        cfo_cycles_per_symbol=float(
            embedding_state.cfo_cycles_per_symbol
            if embedding_state is not None
            else state.acquisition_cfo
        ),
        noise_variance=condition.noise_variance,
        confidence=(
            embedding_state.confidence
            if embedding_state is not None
            else condition.confidence
        ),
        reconstruction_error=0.0,
    )
    pilot_state_embedding = PilotStateEmbedding()(pilot_state_summary)
    online_state_embedding = (
        pilot_state_embedding.unsqueeze(0).to(device)
        if state.state_embedding_conditioned
        else None
    )
    if (
        state.drift_gate_threshold is not None
        and update_scheduled
        and not updates_frozen
    ):
        # 只用 Adapt Pilot 估计当前 CIR 与 acquisition 条件的差异，作为
        # “是否值得尝试参数微调”的门控；候选 CIR 不会写回 state.cir。
        pilot_estimated_cir = pilot_sparse_cir_update(
            frame_device,
            state.cir,
            state.receiver_state.soft_tail,
            max_paths=24,
            alpha=1.0,
            cfo_hint=float(state.acquisition_cfo),
        ).to(device)
        pilot_drift_distance = float(
            (
                torch.linalg.vector_norm(pilot_estimated_cir - state.cir)
                / torch.linalg.vector_norm(state.cir).clamp_min(1e-8)
            )
            .detach()
            .cpu()
        )
        pilot_drift_gate_applied = True
        pilot_drift_gate_allows_update = pilot_drift_distance >= float(
            state.drift_gate_threshold
        )
    with torch.no_grad():
        before_logits, _ = state.model(
            rx_iq,
            condition,
            region_ids,
            tail,
            adapt_symbols=adapt_symbols,
            adapt_mask=adapt_mask,
            online_state_embedding=online_state_embedding,
        )
    before = before_logits.squeeze(0)
    receiver_view = frame_device.receiver_view()
    reward_labels = (receiver_view.reward_symbols.real > 0.0).to(frame_device.bits.dtype)
    adapt_labels = (receiver_view.adapt_symbols.real > 0.0).to(frame_device.bits.dtype)
    reward_before = _masked_bce(before, reward_labels, frame_device.reward_mask)
    adapt_loss_before = _masked_bce(before, adapt_labels, frame_device.adapt_mask)
    ranking_requires_signal = state.reward_ranking_mode in {
        "pilot_consensus",
        "pilot_balanced",
        "pilot_stable_consensus",
        "pilot_margin_consensus",
    }
    reward_signal_before = None
    if state.require_reward_signal_reconstruction or ranking_requires_signal:
        reward_signal_before = float(
            _reward_signal_reconstruction_loss(
                state.model,
                frame_device,
                condition,
                tail,
                state_embedding=online_state_embedding,
            )
            .detach()
            .cpu()
        )
    phase_slope = float(
        torch.diff(phase_features.reshape(-1)).abs().mean().detach().cpu()
        if phase_features is not None and phase_features.numel() > 1
        else torch.zeros((), device=device)
    )
    cir_drift = float(
        torch.linalg.vector_norm(state.cir - cir_before_frame).detach().cpu()
        / torch.linalg.vector_norm(cir_before_frame).clamp_min(1e-8).detach().cpu()
    )
    bandit_context = {
        "adapt_loss": float(adapt_loss_before.detach().cpu()),
        "pilot_confidence": float(condition.confidence.mean().detach().cpu()),
        "residual_cfo": float(
            state.physical_state.cfo_cycles_per_symbol
            if state.physical_state is not None
            else state.acquisition_cfo
        ),
        "phase_slope": phase_slope,
        "cir_drift": cir_drift,
        "snr_db": float(snr_db),
        "reward_trend": float(state.previous_bandit_reward),
        "rollback_rate": float(state.rollback_count / max(1, frame_index - 1)),
        "consecutive_rejections": float(state.consecutive_rejections),
        "parameter_delta_norm": float(state.previous_parameter_delta_norm),
    }
    cross_frame_rollback = False
    previous_good_reward_loss = None
    pending_update_present = state.pending_peft_snapshot is not None
    pending_update_committed = False
    pending_update_rejected = False
    pending_update_reward_loss_before = None
    pending_update_reward_loss_after = None
    pending_update_logit_delta = 0.0
    pending_update_name = state.pending_peft_name if pending_update_present else "skip"
    if state.last_pre_update_snapshot is not None:
        current_snapshot = state.model.peft.snapshot(state.last_pre_update_groups)
        state.model.peft.restore(state.last_pre_update_snapshot)
        with torch.no_grad():
            previous_good_logits, _ = state.model(
                rx_iq,
                condition,
                region_ids,
                tail,
                adapt_symbols=adapt_symbols,
                adapt_mask=adapt_mask,
                online_state_embedding=online_state_embedding,
            )
        previous_good = previous_good_logits.squeeze(0)
        previous_good_reward_loss = float(
            _masked_bce(
                previous_good,
                reward_labels,
                frame_device.reward_mask,
            ).detach().cpu()
        )
        state.model.peft.restore(current_snapshot)
        if _previous_update_is_harmful(
            float(reward_before.detach().cpu()),
            previous_good_reward_loss,
            state.cross_frame_rollback_tolerance,
        ):
            state.model.peft.restore(state.last_pre_update_snapshot)
            before = previous_good
            reward_before = _masked_bce(before, reward_labels, frame_device.reward_mask)
            adapt_loss_before = _masked_bce(before, adapt_labels, frame_device.adapt_mask)
            cross_frame_rollback = True
            state.last_pre_update_snapshot = None
            state.last_pre_update_groups = frozenset()
    # 延迟提交模式把候选留到下一帧，用新的 Reward Pilot 再验收一次。
    # 如果上一帧已经触发跨帧回滚，待提交候选依赖的参数基线已经不存在，必须丢弃。
    if cross_frame_rollback and pending_update_present:
        state.pending_peft_snapshot = None
        state.pending_peft_base_snapshot = None
        state.pending_peft_groups = frozenset()
        state.pending_peft_name = "skip"
        state.pending_peft_logit_delta = 0.0
        pending_update_rejected = True
        pending_update_present = False
    if pending_update_present:
        pending_groups = state.pending_peft_groups
        current_pending_base = state.model.peft.snapshot(pending_groups)
        pending_snapshot = state.pending_peft_snapshot
        state.model.peft.restore(pending_snapshot)
        with torch.no_grad():
            pending_logits_value, _ = state.model(
                rx_iq,
                condition,
                region_ids,
                tail,
                adapt_symbols=adapt_symbols,
                adapt_mask=adapt_mask,
                online_state_embedding=online_state_embedding,
            )
        pending_logits = pending_logits_value.squeeze(0)
        pending_update_reward_loss_before = float(reward_before.detach().cpu())
        pending_update_reward_loss_after = float(
            _masked_bce(
                pending_logits,
                reward_labels,
                frame_device.reward_mask,
            ).detach().cpu()
        )
        pending_update_logit_delta = float(
            torch.mean(
                torch.abs(pending_logits[frame_device.reward_mask] - before[frame_device.reward_mask])
            )
            .detach()
            .cpu()
        )
        pending_window_accept, _ = _accept_windowed_reward_update(
            before,
            pending_logits,
            reward_labels,
            frame_device.reward_mask,
            state.min_reward_improvement,
            # 候选生成帧已经做过多窗口筛选；跨帧确认使用整个 Reward Pilot
            # 的聚合损失，避免把短窗口噪声误当成状态失配。
            windows=1,
            relative_min_improvement=state.relative_min_reward_improvement,
        )
        pending_hard_ber_accept = _accept_reward_pilot_hard_ber(
            before,
            pending_logits,
            reward_mask=frame_device.reward_mask,
            reward_symbols=receiver_view.reward_symbols,
        )
        pending_logit_accept = bool(
            state.reward_logit_change_tolerance is None
            or pending_update_logit_delta <= float(state.reward_logit_change_tolerance)
        )
        pending_signal_accept = True
        pending_signal_after = None
        if state.require_reward_signal_reconstruction or ranking_requires_signal:
            pending_signal_after = float(
                _reward_signal_reconstruction_loss(
                    state.model,
                    frame_device,
                    condition,
                    tail,
                    state_embedding=online_state_embedding,
                )
                .detach()
                .cpu()
            )
            pending_signal_accept = bool(
                reward_signal_before is not None
                and pending_signal_after
                <= reward_signal_before + float(state.reward_signal_tolerance)
            )
        if ranking_requires_signal and pending_signal_after is not None:
            pending_signal_accept = bool(
                pending_signal_accept
                and pending_signal_after
                <= reward_signal_before + float(state.reward_signal_tolerance)
            )
        pending_accept = bool(
            pending_window_accept
            and pending_hard_ber_accept
            and pending_logit_accept
            and pending_signal_accept
            and state.pending_peft_base_snapshot is not None
        )
        state.model.peft.restore(current_pending_base)
        if pending_accept:
            state.model.peft.restore(pending_snapshot)
            before = pending_logits
            reward_before = _masked_bce(before, reward_labels, frame_device.reward_mask)
            adapt_loss_before = _masked_bce(before, adapt_labels, frame_device.adapt_mask)
            if pending_signal_after is not None:
                reward_signal_before = pending_signal_after
            state.last_pre_update_snapshot = state.pending_peft_base_snapshot
            state.last_pre_update_groups = pending_groups
            pending_update_committed = True
        else:
            pending_update_rejected = True
        state.pending_peft_snapshot = None
        state.pending_peft_base_snapshot = None
        state.pending_peft_groups = frozenset()
        state.pending_peft_name = "skip"
        state.pending_peft_logit_delta = 0.0
    candidates = state.candidate_specs or (
        {
            "name": "default",
            "groups": tuple(sorted(state.adapter.groups)),
            "learning_rate_scale": 1.0,
            "steps": state.adapter.steps,
            "max_delta_scale": 1.0,
        },
    )
    freeze_below_db = state.freeze_online_below_snr_db
    candidates = (
        _online_candidates_for_snr(candidates, snr_db, freeze_below_db)
        if update_scheduled
        else ()
    )
    if pilot_drift_gate_applied and not pilot_drift_gate_allows_update:
        candidates = ()
    selected_action = SafeUpdateAction(
        "fixed",
        frozenset(state.adapter.groups),
        1.0,
        1.0,
        1,
        0.0,
    )
    if state.scheduler == "bandit" and state.bandit is not None and update_scheduled:
        allowed_names = {"skip"}
        for action in state.bandit.actions:
            if action.groups.issubset(state.adapter.groups) and _bandit_candidate_for_action(
                action,
                candidates,
            ):
                allowed_names.add(action.name)
        if state.hold_remaining <= 0 or state.active_action is None:
            selected_action = state.bandit.select(bandit_context, allowed_names=allowed_names)
            state.active_action = selected_action
            state.hold_remaining = int(selected_action.hold_frames)
        else:
            selected_action = state.active_action
        state.hold_remaining = max(0, state.hold_remaining - 1)
        candidates = _bandit_candidate_for_action(selected_action, candidates)
    elif state.scheduler == "bandit":
        selected_action = SafeUpdateAction("skip", frozenset(), 0.0, 0.0, 1, 0.0)
        candidates = ()
    candidate_groups = set().union(*(set(item["groups"]) for item in candidates)) if candidates else set()
    base_snapshot = state.model.peft.snapshot(candidate_groups)
    best_snapshot = None
    best_result = None
    best_reward = float(reward_before.detach().cpu())
    # 每个候选都和同一个冻结基线比较；不能让候选顺序改变验收结果。
    baseline_reward_logits = before
    best_reward_window_gains: list[float] = []
    best_name = "skip"
    best_after = before
    best_reward_signal_after = reward_signal_before
    best_reward_logit_delta = 0.0
    best_reward_ranking_score = float("-inf")
    best_reward_ranking_stats = {
        "median_gain": float("nan"),
        "mad_gain": float("nan"),
        "min_gain": float("nan"),
        "bce_lcb": float("nan"),
        "window_stability": float("nan"),
        "signal_gain_relative": float("nan"),
    }
    best_reward_signal_gain_relative = float("nan")
    reward_signal_candidates_evaluated = 0
    reward_signal_candidates_passed = 0
    reward_signal_candidates_rejected = 0
    reward_logit_candidates_evaluated = 0
    reward_logit_candidates_passed = 0
    reward_logit_candidates_rejected = 0
    candidate_ranking_records: list[dict[str, object]] = []
    candidate_snapshot_records: list[tuple[dict[str, object], object]] = []
    for candidate in candidates:
        state.model.peft.restore(base_snapshot)
        adaptation = state.adapter.adapt(
            frame_device,
            condition,
            tail,
            state_embedding=online_state_embedding,
            groups=set(candidate["groups"]),
            learning_rate=state.adapter.learning_rate * float(candidate["learning_rate_scale"]),
            steps=int(candidate["steps"]),
            max_delta_norm=state.adapter.max_delta_norm * float(candidate["max_delta_scale"]),
        )
        with torch.no_grad():
            after_logits, _ = state.model(
                rx_iq,
                condition,
                region_ids,
                tail,
                adapt_symbols=adapt_symbols,
                adapt_mask=adapt_mask,
                online_state_embedding=online_state_embedding,
            )
        after = after_logits.squeeze(0)
        reward_after = _masked_bce(after, reward_labels, frame_device.reward_mask)
        reward_value = float(reward_after.detach().cpu())
        reward_logit_delta = float(
            torch.mean(torch.abs(after[frame_device.reward_mask] - before[frame_device.reward_mask]))
            .detach()
            .cpu()
        )
        reward_logit_accept = True
        if state.reward_logit_change_tolerance is not None:
            reward_logit_candidates_evaluated += 1
            reward_logit_accept = bool(
                reward_logit_delta <= float(state.reward_logit_change_tolerance)
            )
            if reward_logit_accept:
                reward_logit_candidates_passed += 1
            else:
                reward_logit_candidates_rejected += 1
        reward_signal_after = None
        reward_signal_accept = True
        if state.require_reward_signal_reconstruction or ranking_requires_signal:
            reward_signal_candidates_evaluated += 1
            reward_signal_after = float(
                _reward_signal_reconstruction_loss(
                    state.model,
                    frame_device,
                    condition,
                    tail,
                    state_embedding=online_state_embedding,
                )
                .detach()
                .cpu()
            )
            reward_signal_accept = bool(
                reward_signal_before is not None
                and reward_signal_after
                <= reward_signal_before + float(state.reward_signal_tolerance)
            )
            if reward_signal_accept:
                reward_signal_candidates_passed += 1
            else:
                reward_signal_candidates_rejected += 1
        window_accept, window_gains = _accept_windowed_reward_update(
            baseline_reward_logits,
            after,
            reward_labels,
            frame_device.reward_mask,
            state.min_reward_improvement,
            windows=state.reward_pilot_windows,
            relative_min_improvement=state.relative_min_reward_improvement,
        )
        hard_ber_accept = _accept_reward_pilot_hard_ber(
            baseline_reward_logits,
            after,
            reward_mask=frame_device.reward_mask,
            reward_symbols=receiver_view.reward_symbols,
        )
        margin_window_gains = _reward_pilot_margin_gains(
            baseline_reward_logits,
            after,
            reward_labels,
            frame_device.reward_mask,
            windows=state.reward_pilot_windows,
        )
        baseline_margin = _reward_pilot_signed_margin(
            baseline_reward_logits,
            reward_labels,
            frame_device.reward_mask,
        )
        reward_margin_gain = _reward_pilot_signed_margin(
            after,
            reward_labels,
            frame_device.reward_mask,
        ) - baseline_margin
        reward_ranking_score, reward_ranking_stats = _pilot_reward_ranking_score(
            window_gains,
            float(reward_before.detach().cpu()) - reward_value,
            mode=state.reward_ranking_mode,
            dispersion=state.reward_ranking_dispersion,
            baseline_loss=float(reward_before.detach().cpu()),
            signal_gain_relative=(
                (
                    float(reward_signal_before) - float(reward_signal_after)
                )
                / max(abs(float(reward_signal_before)), 1e-8)
                if ranking_requires_signal
                and reward_signal_before is not None
                and reward_signal_after is not None
                else None
            ),
            margin_window_gains=margin_window_gains,
            margin_gain_relative=reward_margin_gain / max(abs(baseline_margin), 1.0),
            margin_baseline=baseline_margin,
        )
        consensus_signal_accept = True
        if ranking_requires_signal:
            consensus_signal_accept = bool(
                reward_signal_after is not None
                and reward_signal_before is not None
                and reward_signal_after
                <= reward_signal_before + float(state.reward_signal_tolerance)
            )
        candidate_ranking_records.append(
            {
                "name": str(candidate["name"]),
                "score": float(reward_ranking_score),
                "bce_gain": float(reward_before.detach().cpu() - reward_value),
                "window_gains": [float(value) for value in window_gains],
                "bce_lcb": float(reward_ranking_stats["bce_lcb"]),
                "window_stability": float(reward_ranking_stats["window_stability"]),
                "signal_gain_relative": float(
                    reward_ranking_stats["signal_gain_relative"]
                ),
                "margin_gain_relative": float(
                    reward_ranking_stats["margin_gain_relative"]
                ),
                "window_guard_passed": bool(window_accept),
                "hard_ber_guard_passed": bool(hard_ber_accept),
                "signal_guard_passed": bool(reward_signal_accept and consensus_signal_accept),
                "logit_guard_passed": bool(reward_logit_accept),
                "adaptation_accepted": bool(adaptation.accepted),
            }
        )
        # 快照只用于在线决策完成后的事后 Data 诊断，不参与当前帧的排序或守门。
        candidate_snapshot_records.append(
            (candidate_ranking_records[-1], state.model.peft.snapshot(candidate_groups))
        )
        if (
            adaptation.accepted
            and window_accept
            and hard_ber_accept
            and reward_logit_accept
            and reward_signal_accept
            and consensus_signal_accept
            and reward_ranking_score > best_reward_ranking_score
        ):
            best_snapshot = state.model.peft.snapshot(candidate_groups)
            best_result = adaptation
            best_reward = reward_value
            best_reward_window_gains = window_gains
            best_name = str(candidate["name"])
            best_after = after
            best_reward_signal_after = reward_signal_after
            best_reward_logit_delta = reward_logit_delta
            best_reward_ranking_score = float(reward_ranking_score)
            best_reward_ranking_stats = reward_ranking_stats
            best_reward_signal_gain_relative = float(
                reward_ranking_stats["signal_gain_relative"]
            )
    state.model.peft.restore(base_snapshot)
    update_applied = bool(candidates) and selected_action.name != "skip"
    pending_update_scheduled = False
    if best_snapshot is None:
        adaptation = None
        accepted = not update_applied
        final = before
        reward_after = reward_before
    elif state.defer_peft_commit:
        # 当前帧只使用提交前的模型输出，候选参数交给下一帧的 Reward Pilot 复核。
        state.model.peft.restore(base_snapshot)
        state.pending_peft_snapshot = best_snapshot
        state.pending_peft_base_snapshot = base_snapshot
        state.pending_peft_groups = frozenset(candidate_groups)
        state.pending_peft_name = best_name
        state.pending_peft_logit_delta = float(best_reward_logit_delta)
        pending_update_scheduled = True
        adaptation = best_result
        accepted = False
        final = before
        reward_after = reward_before
    else:
        state.model.peft.restore(best_snapshot)
        adaptation = best_result
        accepted = True
        final = best_after
        reward_after = torch.as_tensor(best_reward, device=reward_before.device)
    # 保存当前帧最终接受（或冻结）的 PEFT 参数。后面的事后 Data 诊断会临时
    # 恢复每个候选快照，完成后必须恢复这个快照，不能改变在线状态。
    final_peft_snapshot = state.model.peft.snapshot(candidate_groups)
    if accepted:
        state.last_pre_update_groups = frozenset(candidate_groups)
        state.last_pre_update_snapshot = base_snapshot
    raw_reward_gain = float(
        reward_before.detach().cpu() - reward_after.detach().cpu()
    )
    bandit_update_cost = 0.0
    bandit_action_cost = 0.0
    bandit_rollback_penalty = 0.0
    bandit_reward = None
    if state.scheduler == "bandit" and state.bandit is not None:
        bandit_update_cost = float(
            0.001
            * (adaptation.parameter_delta_norm if adaptation is not None and accepted else 0.0)
        )
        bandit_action_cost = state.bandit_action_cost if update_applied else 0.0
        bandit_rollback_penalty = 0.01 if update_applied and not accepted else 0.0
        bandit_reward = raw_reward_gain - bandit_update_cost - bandit_action_cost - bandit_rollback_penalty
        state.bandit.update(selected_action.name, bandit_context, bandit_reward, accepted)
        state.previous_bandit_reward = bandit_reward if accepted else -abs(bandit_reward)
        state.consecutive_rejections = (
            state.consecutive_rejections + 1 if not accepted else 0
        )
        state.rollback_count += int(update_applied and not accepted)
        state.previous_parameter_delta_norm = float(
            adaptation.parameter_delta_norm if adaptation is not None and accepted else 0.0
        )
    # 跨帧 soft tail 也只能由 Pilot 输出更新。prefix Pilot 通常不覆盖帧尾，
    # 此时保持旧 tail，避免把 Data 的硬/软判决带入下一帧在线状态。
    online_final = final
    tail_len = state.receiver_state.soft_tail.numel()
    tail_visible_mask = pilot_visible_mask[-tail_len:]
    if bool(tail_visible_mask.any()):
        old_tail = state.receiver_state.soft_tail
        detected_tail = torch.complex(
            torch.tanh(online_final[-tail_len:] / 2.0),
            torch.zeros_like(online_final[-tail_len:]),
        )
        detected_tail = torch.where(tail_visible_mask, detected_tail, old_tail)
        state.receiver_state.update_tail(
            _update_receiver_tail(old_tail, detected_tail, state.tail_update_alpha)
        )
    if state.cir_update_mode == "decision_directed":
        # 该历史兼容模式只允许在 Pilot 输出上更新；主路线不会启用它。
        state.cir = decision_directed_cir_update(
            frame_device,
            online_final.detach(),
            int(delay),
            state.cir,
            alpha=float(state.cir_update_alpha),
        ).to(device)

    # 至此本帧在线决策、接受/拒绝、回滚以及 Pilot 状态更新全部结束。
    # 从这一行开始才允许读取完整接收帧，所有结果都只用于事后诊断。
    state.model.peft.restore(base_snapshot)
    with torch.no_grad():
        full_before_logits, _ = state.model(
            full_rx_iq,
            condition,
            region_ids,
            tail,
            adapt_symbols=adapt_symbols,
            adapt_mask=adapt_mask,
            online_state_embedding=online_state_embedding,
        )
    full_before = full_before_logits.squeeze(0)
    for ranking_record, candidate_snapshot in candidate_snapshot_records:
        state.model.peft.restore(candidate_snapshot)
        with torch.no_grad():
            candidate_full_logits, _ = state.model(
                full_rx_iq,
                condition,
                region_ids,
                tail,
                adapt_symbols=adapt_symbols,
                adapt_mask=adapt_mask,
                online_state_embedding=online_state_embedding,
            )
        candidate_data = _data_eval_metrics(
            full_before,
            candidate_full_logits.squeeze(0),
            frame_device,
        )
        ranking_record.update(
            {
                "data_bce_before_eval_only": candidate_data["bce_before"],
                "data_bce_after_eval_only": candidate_data["bce_after"],
                "data_bce_improvement_eval_only": candidate_data["bce_improvement"],
                "data_ber_before_eval_only": candidate_data["ber_before"],
                "data_ber_after_eval_only": candidate_data["ber_after"],
                "data_ber_improvement_eval_only": candidate_data["ber_improvement"],
                "data_logits_abs_delta_mean_eval_only": candidate_data[
                    "logits_abs_delta_mean"
                ],
                "data_logits_direction_change_rate_eval_only": candidate_data[
                    "logits_direction_change_rate"
                ],
                "data_soft_output_delta_mean_eval_only": candidate_data[
                    "soft_output_delta_mean"
                ],
            }
        )
    state.model.peft.restore(final_peft_snapshot)
    with torch.no_grad():
        full_final_logits, _ = state.model(
            full_rx_iq,
            condition,
            region_ids,
            tail,
            adapt_symbols=adapt_symbols,
            adapt_mask=adapt_mask,
            online_state_embedding=online_state_embedding,
        )
    final = full_final_logits.squeeze(0)
    # 这些字段只用于离线评估更新对 Data 区域输出的影响；Data 标签不参与
    # 候选优化、Reward 守门或参数更新。
    final_data = _data_eval_metrics(full_before, final, frame_device)
    return _result_from_logits(
        method,
        final,
        frame_device,
        0,
        {
            "uses_neural_network": True,
            "uses_rl": bool(state.scheduler == "bandit"),
            "scheduler": str(state.scheduler),
            "online_algorithm": "pilot_driven_constrained_peft",
            "online_adaptation_objective": str(state.adapter.objective),
            "online_update_source": "adapt_pilot_only",
            "reward_pilot_guard": True,
            "cross_frame_reward_guard": True,
            "online_state_embedding_conditioned": bool(state.state_embedding_conditioned),
            "previous_good_reward_loss": previous_good_reward_loss,
            "rollback": bool(cross_frame_rollback),
            "defer_peft_commit": bool(state.defer_peft_commit),
            "pending_update_present": bool(pending_update_present),
            "pending_update_name": str(pending_update_name),
            "pending_update_scheduled": bool(pending_update_scheduled),
            "pending_update_committed": bool(pending_update_committed),
            "pending_update_rejected": bool(pending_update_rejected),
            "pending_update_reward_loss_before": pending_update_reward_loss_before,
            "pending_update_reward_loss_after": pending_update_reward_loss_after,
            "pending_update_logit_delta": float(pending_update_logit_delta),
            "adapt_pilot_count": int(
                adaptation.adapt_pilot_count
                if adaptation is not None
                else frame_device.adapt_mask.sum().item()
            ),
            "adapt_loss_before": float(
                adaptation.adapt_loss_before
                if adaptation is not None
                else adapt_loss_before.detach().cpu()
            ),
            "adapt_loss_after": float(
                adaptation.adapt_loss_after
                if adaptation is not None
                else adapt_loss_before.detach().cpu()
            ),
            "weighted_adapt_loss_before": float(
                adaptation.weighted_adapt_loss_before
                if adaptation is not None
                else adapt_loss_before.detach().cpu()
            ),
            "weighted_adapt_loss_after": float(
                adaptation.weighted_adapt_loss_after
                if adaptation is not None
                else adapt_loss_before.detach().cpu()
            ),
            "hard_example_weighting": bool(state.adapter.hard_example_weighting),
            "hard_example_temperature": float(state.adapter.hard_example_temperature),
            "adaptation_accepted": bool(accepted),
            "update_applied": bool(update_applied),
            "action": str(selected_action.name),
            "action_hold_frames": int(selected_action.hold_frames),
            "online_update_candidate": best_name,
            "online_update_candidate_count": len(candidates),
            "online_snr_layer": "fully_frozen" if updates_frozen else ("peft_frozen" if not candidates else "peft_enabled"),
            "online_update_scheduled": bool(update_scheduled),
            "online_update_interval": int(update_interval),
            "online_update_skipped": bool(not update_scheduled),
            "online_condition_update_applied": bool(
                state.condition_source != "acquisition" and not updates_frozen
            ),
            "online_adaptation_freeze_below_snr_db": freeze_below_db,
            "reward_pilot_loss_before": float(reward_before.detach().cpu()),
            "reward_pilot_loss_after": float(reward_after.detach().cpu()),
            "reward_signal_reconstruction_before": reward_signal_before,
            "reward_signal_reconstruction_after": best_reward_signal_after,
            "reward_signal_reconstruction_guard": bool(
                state.require_reward_signal_reconstruction
            ),
            "reward_signal_reconstruction_guard_passed": bool(
                not state.require_reward_signal_reconstruction
                or (
                    reward_signal_before is not None
                    and best_reward_signal_after is not None
                    and best_reward_signal_after
                    <= reward_signal_before + float(state.reward_signal_tolerance)
                )
            ),
            "online_reward_signal_tolerance": float(state.reward_signal_tolerance),
            "reward_signal_reconstruction_candidates_evaluated": int(
                reward_signal_candidates_evaluated
            ),
            "reward_signal_reconstruction_candidates_passed": int(
                reward_signal_candidates_passed
            ),
            "reward_signal_reconstruction_candidates_rejected": int(
                reward_signal_candidates_rejected
            ),
            "reward_signal_reconstruction_guard_rejected": bool(
                reward_signal_candidates_rejected > 0
            ),
            "reward_logit_change_before": 0.0,
            "reward_logit_change_after": float(best_reward_logit_delta),
            "reward_logit_change_guard": bool(
                state.reward_logit_change_tolerance is not None
            ),
            "reward_logit_change_guard_passed": bool(
                state.reward_logit_change_tolerance is None
                or best_reward_logit_delta <= float(state.reward_logit_change_tolerance)
            ),
            "online_reward_logit_change_tolerance": state.reward_logit_change_tolerance,
            "reward_logit_change_candidates_evaluated": int(
                reward_logit_candidates_evaluated
            ),
            "reward_logit_change_candidates_passed": int(
                reward_logit_candidates_passed
            ),
            "reward_logit_change_candidates_rejected": int(
                reward_logit_candidates_rejected
            ),
            "data_bce_before_eval_only": final_data["bce_before"],
            "data_bce_after_eval_only": final_data["bce_after"],
            "data_bce_improvement_eval_only": final_data["bce_improvement"],
            "data_ber_before_eval_only": final_data["ber_before"],
            "data_ber_after_eval_only": final_data["ber_after"],
            "data_ber_improvement_eval_only": final_data["ber_improvement"],
            "data_logit_delta_mean_abs_eval_only": final_data["logits_abs_delta_mean"],
            "data_logits_direction_change_rate_eval_only": final_data[
                "logits_direction_change_rate"
            ],
            "data_soft_output_delta_mean_eval_only": final_data[
                "soft_output_delta_mean"
            ],
            "reward_pilot_windows": int(state.reward_pilot_windows),
            "reward_pilot_window_gains": best_reward_window_gains,
            "reward_pilot_ranking_mode": str(state.reward_ranking_mode),
            "reward_pilot_ranking_score": float(best_reward_ranking_score),
            "reward_pilot_window_median_gain": float(
                best_reward_ranking_stats["median_gain"]
            ),
            "reward_pilot_window_mad_gain": float(
                best_reward_ranking_stats["mad_gain"]
            ),
            "reward_pilot_window_min_gain": float(
                best_reward_ranking_stats["min_gain"]
            ),
            "reward_pilot_window_stability": float(
                best_reward_ranking_stats["window_stability"]
            ),
            "reward_pilot_bce_lcb": float(best_reward_ranking_stats["bce_lcb"]),
            "reward_pilot_signal_gain_relative": float(best_reward_signal_gain_relative),
            "reward_pilot_candidate_ranking": candidate_ranking_records,
            "reward_pilot_hard_ber_guard": True,
            "reward_pilot_hard_ber_before": int(
                ((before[frame_device.reward_mask] >= 0.0) != (reward_labels[frame_device.reward_mask] >= 0.5)).sum().item()
            ),
            "reward_pilot_hard_ber_after": int(
                ((final[frame_device.reward_mask] >= 0.0) != (reward_labels[frame_device.reward_mask] >= 0.5)).sum().item()
            ),
            "online_min_reward_improvement": float(state.min_reward_improvement),
            "online_relative_min_reward_improvement": float(
                state.relative_min_reward_improvement
            ),
            "online_proximal_weight": float(state.adapter.proximal_weight),
            "peft_update_guarded": True,
            "online_parameter_groups": list(online_parameter_groups),
            "online_parameter_count": online_parameter_count,
            "online_parameter_role": "equalizer_internal_peft",
            "adapt_pilot_only": True,
            "reward_pilot_role": "acceptance_and_rollback",
            "reward_pilot_guard_only": bool(
                state.scheduler == "fixed" and len(candidates) <= 1
            ),
            "parameter_delta_norm": float(
                adaptation.parameter_delta_norm
                if adaptation is not None and accepted
                else 0.0
            ),
            "tail_update_alpha": float(state.tail_update_alpha),
            "cir_update_mode": str(state.cir_update_mode),
            "cir_update_alpha": float(state.cir_update_alpha),
            "condition_source": str(state.condition_source),
            "pilot_phase_used": bool(state.condition_source != "acquisition"),
            "cir_update_applied": bool(cir_update_accepted),
            "cir_update_guarded": bool(
                state.cir_update_mode == "pilot_sparse" and not updates_frozen and update_scheduled
            ),
            "cir_reward_pilot_loss_before": cir_reward_loss_before,
            "cir_reward_pilot_loss_after": cir_reward_loss_after,
            "peft_update_applied": bool(update_applied and accepted),
            "bandit_reward": bandit_reward,
            "bandit_update_cost": float(bandit_update_cost),
            "bandit_action_cost": float(bandit_action_cost),
            "bandit_rollback_penalty": float(bandit_rollback_penalty),
            "bandit_context": dict(bandit_context),
            "data_labels_used_online": False,
            "data_waveform_used_online": False,
            "online_waveform_visibility": "adapt_pilot_for_update_reward_pilot_for_guard",
            "pretrained_loaded": bool(state.pretrained_loaded),
            "frame_index": int(frame_index),
            "pilot_phase_tracking": bool(state.cir_update_mode == "pilot_sparse" and not updates_frozen),
            "tracked_phase0": float(state.physical_state.phase0 if state.physical_state is not None else 0.0),
            "tracked_cfo_cycles_per_symbol": float(
                state.physical_state.cfo_cycles_per_symbol
                if state.physical_state is not None
                else state.acquisition_cfo
            ),
            "pilot_phase_confidence": float(state.physical_state.confidence if state.physical_state is not None else 0.0),
            # 这些字段只用于后续 Pilot-only replay 和漂移门控校准；当前不改变动作。
            "pilot_state_embedding": [float(value) for value in pilot_state_embedding.tolist()],
            "pilot_state_cir_residual_norm": float(pilot_state_summary.cir_residual_norm),
            "pilot_state_noise_variance": float(pilot_state_summary.noise_variance),
            "pilot_state_reconstruction_error": float(pilot_state_summary.reconstruction_error),
            "pilot_state_drift_distance_to_frame_start": float(cir_drift),
            "pilot_state_drift_gate_applied": bool(pilot_drift_gate_applied),
            "pilot_state_drift_gate_threshold": (
                float(state.drift_gate_threshold)
                if state.drift_gate_threshold is not None
                else None
            ),
            "pilot_state_drift_gate_distance": float(pilot_drift_distance),
            "pilot_state_drift_gate_allows_update": bool(pilot_drift_gate_allows_update),
        },
    )


def _online_candidate_specs(config: dict, default_groups: set[str]) -> tuple[dict, ...]:
    """读取仅由 Adapt Pilot 更新、Reward Pilot 选择的离散候选。"""

    payload = config.get("online_adaptation_candidates")
    if not payload:
        return (
            {
                "name": "default",
                "groups": tuple(sorted(default_groups)),
                "learning_rate_scale": 1.0,
                "steps": int(config.get("online_adaptation_steps", 1)),
                "max_delta_scale": 1.0,
            },
        )
    candidates = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError("online_adaptation_candidates 的每项必须为对象。")
        groups = item.get("groups", sorted(default_groups))
        if isinstance(groups, str):
            groups = [groups]
        selected = tuple(sorted(str(group) for group in groups))
        if not selected:
            raise ValueError("online_adaptation_candidates 的 groups 不能为空。")
        candidates.append(
            {
                "name": str(item.get("name", f"candidate_{index}")),
                "groups": selected,
                "learning_rate_scale": float(item.get("learning_rate_scale", 1.0)),
                "steps": max(1, int(item.get("steps", config.get("online_adaptation_steps", 1)))),
                "max_delta_scale": float(item.get("max_delta_scale", 1.0)),
            }
        )
    return tuple(candidates)


def _bandit_candidate_for_action(
    action: SafeUpdateAction,
    candidates: tuple[dict, ...],
) -> tuple[dict, ...]:
    """把安全动作映射为一个候选更新；找不到匹配候选时安全退化为 skip。"""

    if action.name == "skip" or not action.groups:
        return ()
    for candidate in candidates:
        candidate_groups = frozenset(str(group) for group in candidate["groups"])
        if candidate_groups != action.groups:
            continue
        selected = dict(candidate)
        selected["learning_rate_scale"] = float(
            selected.get("learning_rate_scale", 1.0)
        ) * float(action.learning_rate_scale)
        selected["max_delta_scale"] = float(
            selected.get("max_delta_scale", 1.0)
        ) * float(action.max_delta_scale)
        selected["name"] = str(action.name)
        return (selected,)
    return ()


def _online_groups_from_config(
    config: dict,
    override: list[str] | None,
) -> tuple[set[str], dict]:
    """解析在线更新对象；命令行覆盖时同时清除配置文件候选，避免混用。"""

    if override is not None:
        groups = {str(group) for group in override if str(group)}
        if not groups:
            raise ValueError("online_groups 覆盖不能为空。")
        candidate_config = dict(config)
        candidate_config["online_adaptation_candidates"] = None
        return groups, candidate_config
    groups = {
        str(group)
        for group in config.get(
            "online_adaptation_groups",
            ["head", "conditioner_film"],
        )
    }
    if not groups:
        raise ValueError("online_adaptation_groups 不能为空。")
    return groups, config


def _online_candidates_for_snr(
    candidates: tuple[dict, ...],
    snr_db: float,
    freeze_below_db: float | None,
) -> tuple[dict, ...]:
    """按 SNR 可靠性分层决定是否允许 Pilot 驱动的 PEFT 更新。"""

    if freeze_below_db is not None and float(snr_db) < float(freeze_below_db):
        return ()
    return candidates


def _online_updates_are_frozen(snr_db: float, freeze_below_db: float | None) -> bool:
    """判断当前 SNR 是否低到不应相信 Pilot 驱动的状态更新。"""

    return freeze_below_db is not None and float(snr_db) < float(freeze_below_db)


def _online_update_is_scheduled(frame_index: int, update_interval: int) -> bool:
    """按帧调度在线状态更新，首帧建立初始状态，之后每隔指定帧更新一次。"""

    if int(frame_index) < 1:
        raise ValueError("frame_index 必须为正数。")
    if int(update_interval) <= 0:
        raise ValueError("update_interval 必须为正数。")
    return int(frame_index) == 1 or (int(frame_index) - 1) % int(update_interval) == 0


def _update_receiver_tail(
    previous: torch.Tensor,
    detected: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """统一所有神经方法的跨帧 soft-tail 平滑规则。"""

    selected_alpha = float(alpha)
    if not 0.0 <= selected_alpha <= 1.0:
        raise ValueError("tail_update_alpha 必须位于 [0, 1]。")
    return (1.0 - selected_alpha) * previous + selected_alpha * detected


def _pilot_state_reliable(state: PilotPhysicalState | None, threshold: float) -> bool:
    """只有当前 Adapt Pilot 物理状态足够可靠时才更新稀疏 CIR。"""

    return state is not None and float(state.confidence) >= float(threshold)


def _tracked_cfo_or_none(state: PilotPhysicalState | None, threshold: float) -> float | None:
    """仅在 Pilot 物理状态可靠时才把跟踪 CFO 注入神经条件。"""

    if not _pilot_state_reliable(state, threshold):
        return None
    return float(state.cfo_cycles_per_symbol)


def _accept_pilot_cir_update(reward_loss_before: float, reward_loss_after: float, tolerance: float = 1e-6) -> bool:
    """只接受不恶化留出 Reward Pilot 损失的 Pilot CIR 更新。"""

    return float(reward_loss_after) <= float(reward_loss_before) + float(tolerance)


def _accept_online_peft_update(
    reward_loss_before: float,
    reward_loss_after: float,
    min_improvement: float,
) -> bool:
    """要求 PEFT 在 Reward Pilot 上取得超过噪声阈值的真实改善。"""

    return float(reward_loss_before) - float(reward_loss_after) >= float(min_improvement)


def _accept_windowed_reward_update(
    logits_before: torch.Tensor,
    logits_after: torch.Tensor,
    labels: torch.Tensor,
    reward_mask: torch.Tensor,
    min_improvement: float,
    windows: int = 2,
    relative_min_improvement: float = 0.0,
) -> tuple[bool, list[float]]:
    """要求 Reward Pilot 的每个时间子窗口都支持同一个在线更新。"""

    before = logits_before.reshape(-1)
    after = logits_after.reshape(-1)
    target = labels.reshape(-1).to(device=before.device, dtype=before.dtype)
    positions = torch.nonzero(reward_mask.reshape(-1).to(device=before.device), as_tuple=False).flatten()
    if positions.numel() == 0:
        return False, []
    window_count = max(1, min(int(windows), int(positions.numel())))
    gains: list[float] = []
    for chunk in torch.tensor_split(positions, window_count):
        if chunk.numel() == 0:
            continue
        loss_before = torch.nn.functional.binary_cross_entropy_with_logits(
            before[chunk], target[chunk]
        )
        loss_after = torch.nn.functional.binary_cross_entropy_with_logits(
            after[chunk], target[chunk]
        )
        gains.append(float((loss_before - loss_after).detach().cpu()))
    relative_gains: list[float] = []
    for chunk in torch.tensor_split(positions, window_count):
        if chunk.numel() == 0:
            continue
        loss_before = torch.nn.functional.binary_cross_entropy_with_logits(
            before[chunk], target[chunk]
        )
        loss_after = torch.nn.functional.binary_cross_entropy_with_logits(
            after[chunk], target[chunk]
        )
        denominator = loss_before.detach().abs().clamp_min(1e-8)
        relative_gains.append(float(((loss_before - loss_after) / denominator).detach().cpu()))
    accepted = bool(gains) and all(
        gain >= float(min_improvement)
        or relative_gain >= float(relative_min_improvement)
        for gain, relative_gain in zip(gains, relative_gains)
    )
    return accepted, gains


def _pilot_reward_ranking_score(
    window_gains: list[float],
    aggregate_gain: float,
    *,
    mode: str = "aggregate_bce",
    dispersion: float = 0.5,
    baseline_loss: float = 1.0,
    signal_gain_relative: float | None = None,
    margin_window_gains: list[float] | None = None,
    margin_gain_relative: float | None = None,
    margin_baseline: float = 1.0,
) -> tuple[float, dict[str, float]]:
    """根据 Reward Pilot 子窗口给候选排序，只使用已知 Pilot 符号。

    ``aggregate_bce`` 延续旧策略，按整个 Reward Pilot 的 BCE 改善排序。
    ``robust_lcb`` 取窗口改善的中位数并减去 MAD 惩罚，能降低单个偶然
    窗口对候选排序的影响。``worst_window`` 直接按最差窗口排序，适合
    需要极保守更新的压力测试。``pilot_consensus`` 把 BCE 下界按基线损失
    归一化，并与 Reward Pilot 的复数信号重构相对改善取较小值；
    ``pilot_balanced`` 对两个归一化改善取几何平均；
    ``pilot_stable_consensus`` 在此基础上再乘以跨窗口稳定性，降低只在
    少数 Reward Pilot 子窗口偶然有效的候选排名。``pilot_margin_consensus``
    进一步要求已知 Pilot 的符号软边际也改善，适合 Reward Pilot 很短、硬
    BER 变化过于离散的场景。所有模式都只接收 Pilot 指标，不接收 Data。
    """

    selected_mode = str(mode).strip().lower()
    if selected_mode not in {
        "aggregate_bce",
        "robust_lcb",
        "worst_window",
        "pilot_consensus",
        "pilot_balanced",
        "pilot_stable_consensus",
        "pilot_margin_consensus",
    }:
        raise ValueError(
            "online_reward_ranking 必须是 aggregate_bce、robust_lcb、worst_window、pilot_consensus、pilot_balanced、pilot_stable_consensus 或 pilot_margin_consensus。"
        )
    values = [float(value) for value in window_gains if math.isfinite(float(value))]
    if not values or not math.isfinite(float(aggregate_gain)):
        return float("-inf"), {
            "median_gain": float("nan"),
            "mad_gain": float("nan"),
            "min_gain": float("nan"),
            "bce_lcb": float("nan"),
            "window_stability": float("nan"),
            "signal_gain_relative": float("nan"),
            "margin_gain_relative": float("nan"),
        }
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        median_gain = ordered[middle]
    else:
        median_gain = 0.5 * (ordered[middle - 1] + ordered[middle])
    deviations = sorted(abs(value - median_gain) for value in ordered)
    mad_middle = len(deviations) // 2
    if len(deviations) % 2:
        mad_gain = deviations[mad_middle]
    else:
        mad_gain = 0.5 * (deviations[mad_middle - 1] + deviations[mad_middle])
    minimum_gain = min(ordered)
    bce_lcb = float(median_gain - float(dispersion) * 1.4826 * mad_gain)
    normalized_bce_lcb = bce_lcb / max(abs(float(baseline_loss)), 1e-8)
    # 稳定性同时考虑稳健离散度和正收益窗口占比。仅看 MAD 会漏掉
    # “大多数窗口相同、少数窗口明显退化”的情况，因此加入符号一致性。
    positive_fraction = sum(value > 0.0 for value in values) / max(len(values), 1)
    relative_dispersion = (1.4826 * mad_gain) / max(abs(median_gain), 1e-8)
    dispersion_stability = max(0.0, min(1.0, 1.0 - relative_dispersion))
    window_stability = float(max(0.0, min(1.0, positive_fraction * dispersion_stability)))
    if selected_mode == "aggregate_bce":
        score = float(aggregate_gain)
    elif selected_mode == "worst_window":
        score = float(minimum_gain)
    elif selected_mode in {"pilot_consensus", "pilot_balanced", "pilot_stable_consensus", "pilot_margin_consensus"}:
        signal_score = (
            float(signal_gain_relative)
            if signal_gain_relative is not None and math.isfinite(float(signal_gain_relative))
            else float("-inf")
        )
        if selected_mode == "pilot_consensus":
            score = float(min(normalized_bce_lcb, signal_score))
        elif selected_mode == "pilot_balanced" and normalized_bce_lcb > 0.0 and signal_score > 0.0:
            # 几何平均不引入手工权重，能同时奖励两个 Pilot 视图，
            # 并惩罚其中一个视图接近零的候选。
            score = float(math.sqrt(normalized_bce_lcb * signal_score))
        elif selected_mode == "pilot_stable_consensus":
            consensus = min(normalized_bce_lcb, signal_score)
            score = float(consensus * window_stability)
        elif selected_mode == "pilot_margin_consensus":
            margin_values = [float(value) for value in (margin_window_gains or []) if math.isfinite(float(value))]
            if margin_values:
                margin_ordered = sorted(margin_values)
                margin_middle = len(margin_ordered) // 2
                margin_median = (
                    margin_ordered[margin_middle]
                    if len(margin_ordered) % 2
                    else 0.5 * (margin_ordered[margin_middle - 1] + margin_ordered[margin_middle])
                )
                margin_deviations = sorted(abs(value - margin_median) for value in margin_ordered)
                margin_mad_middle = len(margin_deviations) // 2
                margin_mad = (
                    margin_deviations[margin_mad_middle]
                    if len(margin_deviations) % 2
                    else 0.5 * (margin_deviations[margin_mad_middle - 1] + margin_deviations[margin_mad_middle])
                )
                margin_lcb = float(margin_median - float(dispersion) * 1.4826 * margin_mad)
                margin_score = float(margin_lcb / max(abs(float(margin_baseline)), 1.0))
            else:
                margin_score = float("-inf")
            # 三个 Pilot 证据都必须支持同一个候选；稳定性沿用 BCE 子窗口。
            score = float(min(normalized_bce_lcb, signal_score, margin_score) * window_stability)
        else:
            score = float(min(normalized_bce_lcb, signal_score))
    else:
        # 1.4826 把 MAD 换算为正态分布下的稳健标准差，dispersion
        # 控制保守程度；默认 0.5 保留收益同时惩罚跨窗口不一致。
        score = bce_lcb
    return score, {
        "median_gain": float(median_gain),
        "mad_gain": float(mad_gain),
        "min_gain": float(minimum_gain),
        "bce_lcb": float(bce_lcb),
        "window_stability": float(window_stability),
        "signal_gain_relative": (
            float(signal_gain_relative)
            if signal_gain_relative is not None and math.isfinite(float(signal_gain_relative))
            else float("nan")
        ),
        "margin_gain_relative": (
            float(margin_gain_relative)
            if margin_gain_relative is not None and math.isfinite(float(margin_gain_relative))
            else float("nan")
        ),
    }


def _reward_pilot_signed_margin(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reward_mask: torch.Tensor,
) -> float:
    """计算 Reward Pilot 已知符号对应的平均软边际。"""

    mask = reward_mask.reshape(-1).to(device=logits.device, dtype=torch.bool)
    if not bool(mask.any()):
        return 0.0
    target_sign = labels.reshape(-1).to(device=logits.device, dtype=logits.dtype) * 2.0 - 1.0
    return float((target_sign[mask] * logits.reshape(-1)[mask]).mean().detach().cpu())


def _reward_pilot_margin_gains(
    logits_before: torch.Tensor,
    logits_after: torch.Tensor,
    labels: torch.Tensor,
    reward_mask: torch.Tensor,
    *,
    windows: int = 2,
) -> list[float]:
    """按 Reward Pilot 子窗口计算软边际改善，只读取已知 Pilot 标签。"""

    mask = reward_mask.reshape(-1).to(device=logits_before.device, dtype=torch.bool)
    positions = torch.nonzero(mask, as_tuple=False).flatten()
    if positions.numel() == 0:
        return []
    target_sign = labels.reshape(-1).to(device=logits_before.device, dtype=logits_before.dtype) * 2.0 - 1.0
    gains: list[float] = []
    for chunk in torch.tensor_split(positions, max(1, min(int(windows), int(positions.numel())))):
        if chunk.numel() == 0:
            continue
        before = (target_sign[chunk] * logits_before.reshape(-1)[chunk]).mean()
        after = (target_sign[chunk] * logits_after.reshape(-1)[chunk]).mean()
        gains.append(float((after - before).detach().cpu()))
    return gains


def _reward_signal_reconstruction_loss(
    model: UnfoldedEqualizer,
    frame,
    condition: CIRCondition,
    tail: torch.Tensor,
    state_embedding: torch.Tensor | None = None,
) -> torch.Tensor:
    """计算 Reward Pilot 上的复数信号重构误差。

    该守门只使用当前 Reward Pilot 的已知符号和接收信号。Adapt 与 Reward
    符号可用于构造 Reward 位置之前的已知卷积输入，Data 区域始终置零；
    因此它只用于动作后的验收，不参与参数梯度更新。
    """

    # 该指标只用于动作后的验收，必须与 Adapt Pilot 梯度路径隔离。
    with torch.no_grad():
        view = frame.receiver_view()
        device = next(model.parameters()).device
        pilot_visible = (view.adapt_mask | view.reward_symbols.ne(0)).to(
            device=device, dtype=torch.bool
        )
        rx = view.rx_symbols.to(device).to(torch.complex64).reshape(-1)
        # Reward 信号守门只允许看到 Pilot 波形；Data 波形即使存在于整帧
        # 缓冲区，也不能通过非因果前向影响候选排序。
        rx = torch.where(pilot_visible.reshape(-1), rx, torch.zeros_like(rx))
        known_tx = (
            view.adapt_symbols.to(device).to(torch.complex64)
            + view.reward_symbols.to(device).to(torch.complex64)
        ).reshape(-1)
        reward_mask = view.reward_symbols.to(device).abs().reshape(-1) > 0.0
        if not bool(reward_mask.any()):
            return torch.zeros((), device=device, dtype=torch.float32)
        rx_iq = torch.stack((rx.real, rx.imag), dim=-1).unsqueeze(0)
        # 验收与实际前向都在既有相位补偿之后衡量 Adapter 的残差作用。
        corrected_iq = model._apply_phase_correction(rx_iq, condition)
        phase_adapter = model.online_phase_trend_adapter
        if phase_adapter is not None:
            corrected_iq = phase_adapter(
                corrected_iq,
                state_embedding=state_embedding,
            )
        corrected_iq = corrected_iq.squeeze(0)
        corrected = torch.complex(corrected_iq[:, 0], corrected_iq[:, 1]).to(torch.complex64)
        cir = condition.complex_cir.to(device).to(torch.complex64)
        channel_adapter = model.online_channel_residual_adapter
        if channel_adapter is not None:
            cir = channel_adapter(cir)
        tail = tail.to(device).to(torch.complex64).reshape(-1)
        operator = LinearChannelOperator(frame_len=known_tx.numel(), max_delay=cir.numel() - 1)
        predicted = operator.forward(known_tx, cir.reshape(-1), tail).reshape(-1)
        residual = corrected[reward_mask] - predicted[reward_mask]
        return torch.mean(torch.abs(residual) ** 2).real


def _accept_reward_pilot_hard_ber(
    logits_before: torch.Tensor,
    logits_after: torch.Tensor,
    reward_mask: torch.Tensor,
    reward_symbols: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
) -> bool:
    """要求候选更新在留出 Reward Pilot 上不增加硬判决错误数。"""

    mask = reward_mask.reshape(-1).to(device=logits_before.device, dtype=torch.bool)
    if not bool(mask.any()):
        return False
    if reward_symbols is None:
        if labels is None:
            raise ValueError("Reward Pilot 守门需要已知 Pilot 符号。")
        expected_positive = labels.reshape(-1).to(device=logits_before.device)[mask] >= 0.5
    else:
        expected_positive = reward_symbols.reshape(-1).real.to(device=logits_before.device)[mask] >= 0.0
    before_errors = ((logits_before.reshape(-1)[mask] >= 0.0) != expected_positive).sum()
    after_errors = ((logits_after.reshape(-1)[mask] >= 0.0) != expected_positive).sum()
    return bool(after_errors <= before_errors)


def _previous_update_is_harmful(
    current_reward_loss: float,
    previous_good_reward_loss: float,
    tolerance: float,
) -> bool:
    """判断上一轮已接受更新在当前 Reward Pilot 上是否已经恶化。"""

    return float(current_reward_loss) > float(previous_good_reward_loss) + float(tolerance)


def _neural_method_contract(method: str) -> dict[str, object]:
    """返回神经方法的 Pilot、CIR 和 PEFT 信息边界。"""

    contracts = {
        "Frozen Offline NN": {
            "condition_source": "pilot_cir_phase",
            "pilot_phase_used": True,
            "cir_update_applied": False,
            "peft_update_applied": False,
        },
        "Pilot CIR only": {
            "condition_source": "pilot_cir_phase",
            "pilot_phase_used": True,
            "cir_update_applied": True,
            "peft_update_applied": False,
        },
        "Pilot-Driven Online Adaptation": {
            "condition_source": "pilot_cir_phase",
            "pilot_phase_used": True,
            "cir_update_applied": True,
            "peft_update_applied": True,
        },
    }
    if method in contracts:
        return dict(contracts[method])
    if method in NEURAL_IMPLEMENTATION_METHODS:
        return {
            "condition_source": "pilot_cir_phase",
            "pilot_phase_used": True,
            "cir_update_applied": False,
            "peft_update_applied": False,
        }
    raise ValueError(f"未知神经方法：{method}")


def _neural_phase_features(
    *,
    condition_source: str,
    receiver_view,
    cir: torch.Tensor,
    soft_tail: torch.Tensor,
    acquisition_cfo: float,
    tracked_cfo: float | None = None,
    acquisition_phase_features: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """按显式条件来源生成相位特征；acquisition 模式绝不读取当前帧 Pilot。"""

    if condition_source == "acquisition":
        return acquisition_phase_features
    if condition_source not in {"pilot_phase", "pilot_cir_phase"}:
        raise ValueError(f"未知神经 condition_source：{condition_source}")
    return estimate_phase_residual_vector(
        receiver_view,
        cir,
        soft_tail,
        blocks=4,
        cfo_hint=float(acquisition_cfo if tracked_cfo is None else tracked_cfo),
    )


def _load_model_config(config: dict, pretrained_path: Path | None) -> UnfoldedConfig:
    if pretrained_path is None:
        return UnfoldedConfig.from_dict(config.get("model", {}))
    config_path = pretrained_path.parent / "model_config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    model_payload = payload.get("model", payload)
    # 维度和可学习结构必须服从 checkpoint；物理求解精度等非结构性推理参数
    # 允许由当前冻结实验配置覆盖，使旧权重可以复用已验证的长记忆求解器设置。
    runtime_fields = (
        "physics_warm_start_iterations",
        "physics_warm_start_scale",
        "analytic_logit_skip_scale",
        "neural_residual_scale",
        "phase_correction_initial_scale",
    )
    configured_model = config.get("model", {})
    if isinstance(configured_model, dict):
        model_payload = dict(model_payload)
        for field in runtime_fields:
            if field in configured_model:
                model_payload[field] = configured_model[field]
    return UnfoldedConfig.from_dict(model_payload)


def _build_equalizer(model_config: UnfoldedConfig, pretrained_path: Path | None, device: str) -> UnfoldedEqualizer:
    model = UnfoldedEqualizer(model_config).to(device)
    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
        if state_dict is None:
            raise KeyError("checkpoint 必须包含 model_state_dict 或 state_dict。")
        model.load_state_dict(state_dict, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _fixed_modulation_for(method: str, config: ModulationConfig, device: str) -> ModulationState:
    if method == "NN + Fixed Modulation":
        return ModulationState.identity(config, device=torch.device(device))
    if method == "NN + Rule Modulation":
        vector = torch.ones(config.action_dim, device=device)
        vector[-2] = 0.1
        return ModulationState.from_vector(vector, config)
    return ModulationState.identity(config, device=torch.device(device))


def _frame_to_device(frame, device: torch.device | str):
    from dataclasses import replace

    return replace(
        frame,
        bits=frame.bits.to(device),
        tx_symbols=frame.tx_symbols.to(device),
        rx_symbols=frame.rx_symbols.to(device),
        adapt_mask=frame.adapt_mask.to(device),
        reward_mask=frame.reward_mask.to(device),
        data_mask=frame.data_mask.to(device),
        model_region_ids=frame.model_region_ids.to(device),
        tail_symbols=frame.tail_symbols.to(device) if frame.tail_symbols is not None else None,
        true_cir=frame.true_cir.to(device) if frame.true_cir is not None else None,
    )


def _run_ppo_method(method: str, frame, snr_db: float, state: PPOMethodState, delay: int, frame_index: int, update_interval: int) -> RealMethodResult:
    sigma = torch.tensor(10.0 ** (-float(snr_db) / 10.0))
    pilot_mask = (frame.adapt_mask | frame.reward_mask).to(dtype=torch.bool)
    pilot_rx = torch.where(pilot_mask, frame.rx_symbols, torch.zeros_like(frame.rx_symbols))
    before = perfect_csi_bpsk_refine_detect(
        pilot_rx,
        state.cir,
        state.receiver_state.soft_tail,
        sigma,
        cg_iterations=8,
        refine_iterations=0,
    )
    view = _policy_view(
        frame=frame,
        cir=state.cir,
        snr_db=float(snr_db),
        previous_reward=state.previous_reward,
        last_parameter_delta_norm=state.last_parameter_delta_norm,
        confidence=_confidence(before.logits),
    )
    observation = state.encoder(view).tensor.unsqueeze(0)
    hidden_in = state.hidden.detach()
    action, log_prob, value, state.hidden = state.policy.sample(observation, hidden_in)
    if action.mode == "rollback":
        state.cir = state.last_good_cir.clone()
        state.receiver_state.update_tail(state.last_good_tail)
    cg_iterations, refine_iterations = _detector_settings(action)
    result = perfect_csi_bpsk_refine_detect(
        pilot_rx,
        state.cir,
        state.receiver_state.soft_tail,
        sigma,
        cg_iterations=cg_iterations,
        refine_iterations=refine_iterations,
    )
    pilot_labels = (frame.tx_symbols.real > 0.0).to(frame.bits.dtype)
    reward_loss_before = _masked_bce(before.logits, pilot_labels, frame.reward_mask)
    reward_loss_after = _masked_bce(result.logits, pilot_labels, frame.reward_mask)
    reward = float((reward_loss_before - reward_loss_after).detach().cpu())
    reward -= 0.0005 * max(0, cg_iterations - 8)
    reward -= 0.001 * refine_iterations
    if reward > 0:
        state.last_good_cir = state.cir.clone()
        state.last_good_tail = result.soft_tail.clone()
    parameter_delta_norm = float(torch.norm(state.cir - state.last_good_cir).detach().cpu())
    state.rollout.append(
        {
            "observation": observation.detach(),
            "hidden": hidden_in.detach(),
            "action": action,
            "old_log_prob": log_prob.detach(),
            "value": value.detach(),
            "reward": reward,
        }
    )
    should_update = frame_index % update_interval == 0
    policy_loss = None
    if should_update:
        policy_loss = _ppo_update(state.policy, state.optimizer, state.rollout)
        state.rollout.clear()
    tail_len = state.receiver_state.soft_tail.numel()
    if bool(pilot_mask[-tail_len:].any()):
        visible_tail = pilot_mask[-tail_len:]
        state.receiver_state.update_tail(
            torch.where(visible_tail, result.soft_tail, state.receiver_state.soft_tail)
        )
    cir_alpha = 0.2 if action.mode not in {"skip", "rollback"} else 0.05
    state.cir = _decision_directed_cir_update(frame, result.logits, int(delay), state.cir, alpha=cir_alpha * float(action.cir_trust))
    # Pilot 驱动的 CIR/tail 更新结束后，完整波形只用于最终事后评估。
    final = perfect_csi_bpsk_refine_detect(
        frame.rx_symbols,
        state.cir,
        state.receiver_state.soft_tail,
        sigma,
        cg_iterations=cg_iterations,
        refine_iterations=refine_iterations,
    )
    state.previous_reward = reward
    state.last_parameter_delta_norm = parameter_delta_norm
    return RealMethodResult(
        method=method,
        input_hash=_hash_json({"seed": int(frame.frame_index), "method": method, "visible": "rx+adapt+policy"}),
        ber_data=bit_error_rate(final.logits[frame.data_mask], frame.bits[frame.data_mask]),
        ber_reward_pilot=bit_error_rate(result.logits[frame.reward_mask], pilot_labels[frame.reward_mask]),
        ber_adapt_pilot=bit_error_rate(result.logits[frame.adapt_mask], pilot_labels[frame.adapt_mask]),
        latency_ms=0.0,
        detector_iterations=int(cg_iterations + refine_iterations),
        extra={
            "policy_learning": "clipped_ppo_reward_pilot",
            "policy_updated": should_update,
            "policy_loss": policy_loss,
            "policy_action_mode": action.mode,
            "policy_action_group": action.parameter_group,
            "reward": reward,
            "reward_pilot_loss_before": float(reward_loss_before.detach().cpu()),
            "reward_pilot_loss_after": float(reward_loss_after.detach().cpu()),
            "adapt_steps": int(action.steps if action.mode in {"update-channel", "update-equalizer", "joint-update"} else 0),
            "adapt_params": int(state.policy.parameter_count() if should_update else 0),
            "parameter_delta_norm": parameter_delta_norm,
            "cir_update": "decision_directed",
            "data_waveform_used_online": False,
            "online_waveform_visibility": "adapt_pilot_for_update_reward_pilot_for_guard",
        },
    )


def _load_existing_rows(jsonl: Path) -> list[dict]:
    if not jsonl.exists():
        return []
    return [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]


def _row_key(row: dict) -> tuple[str, int, float, int, int, int, int, str, str, str, float]:
    return (
        str(row["method"]),
        int(row["delay"]),
        float(row["snr_db"]),
        int(row["seed"]),
        int(row["frame"]),
        int(row.get("pilot_total", 0)),
        int(row.get("reward_pilot_total", int(row.get("pilot_total", 0)) // 4)),
        str(row.get("pilot_layout", "")),
        str(row.get("impairment_profile", "clean")),
        str(row.get("condition_source", "")),
        float(
            row.get(
                "acquisition_to_data_gap_seconds",
                row.get("state_instance", {}).get("acquisition_to_data_gap_seconds", 0.0),
            )
        ),
    )


def _hash_json(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
