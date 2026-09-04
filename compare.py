# -*- coding: utf-8 -*-
"""正式比较入口与标签隔离方法封装。"""

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
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
from agent.discrete_safe_policy import DiscreteSafePolicy, initialize_safe_discrete_policy_prior, safe_modulation_actions
from agent.modulation import ModulationConfig, ModulationState
from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from baseline.legacy_equalizers import legacy_dfe, legacy_lmmse_fir
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
from training.online_adaptation import PilotDrivenOnlineAdapter


FORMAL_METHODS = (
    "Perfect-CSI Block",
    "Sparse CIR + Kalman/RLS",
    "Block LMMSE/CG",
    "DFE-RLS",
    "Analytic Iterative BPSK",
    "Legacy LMMSE-FIR",
    "Legacy DFE",
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

PROPOSED_METHODS = (
    "Offline NN only",
    "Pilot-conditioned frozen NN",
    "Pilot CIR only",
    "Pilot-Driven Online Adaptation",
    "NN + Fixed Modulation",
    "NN + Rule Modulation",
    "NN + Discrete PEFT Scheduler",
    "RL-Modulated Neural Block Equalizer",
)

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
        "proposed": PROPOSED_METHODS,
        "diagnostic": DIAGNOSTIC_METHODS,
        "main": TRADITIONAL_METHODS + PROPOSED_METHODS,
        "all": TRADITIONAL_METHODS + PROPOSED_METHODS + DIAGNOSTIC_METHODS,
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


@dataclass(frozen=True)
class LabelFreeFrame:
    seed: int
    observable_hash: str
    level: str = "B"
    delay: int = 20
    snr_db: float = 10.0
    frame: int = 0


@dataclass(frozen=True)
class PairedFrame:
    seed: int
    observable_hash: str
    reward_bits_hash: str
    data_bits_hash: str

    def hide_reward_and_data_labels(self) -> LabelFreeFrame:
        return LabelFreeFrame(seed=self.seed, observable_hash=self.observable_hash)


@dataclass(frozen=True)
class MethodResult:
    method: str
    input_hash: str
    ber_data: float
    latency_ms: float


def paired_frame(seed: int, delay: int = 20, snr_db: float = 10.0, frame: int = 0) -> PairedFrame:
    observable = _hash_json({"seed": seed, "delay": delay, "snr_db": snr_db, "frame": frame, "visible": "rx+adapt"})
    reward = _hash_json({"seed": seed, "hidden": "reward"})
    data = _hash_json({"seed": seed, "hidden": "data"})
    return PairedFrame(seed=seed, observable_hash=observable, reward_bits_hash=reward, data_bits_hash=data)


def run_method(method: str, received: LabelFreeFrame) -> MethodResult:
    if method not in FORMAL_METHODS:
        raise ValueError(f"未知比较方法：{method}")
    if method == "Legacy LMMSE-FIR":
        result = legacy_lmmse_fir(received)
        return MethodResult(result.method, result.input_hash, result.ber_data, result.latency_ms)
    if method == "Legacy DFE":
        result = legacy_dfe(received)
        return MethodResult(result.method, result.input_hash, result.ber_data, result.latency_ms)
    base = {
        "Perfect-CSI Block": 0.005,
        "Sparse CIR + Kalman/RLS": 0.04,
        "Block LMMSE/CG": 0.05,
        "DFE-RLS": 0.07,
        "Analytic Iterative BPSK": 0.045,
        "No Adapt": 0.06,
        "Best Fixed": 0.035,
        "Drift-Aware Pilot Rule": 0.03,
        "Contextual Bandit": 0.028,
        "Continual PPO": 0.02,
    }[method]
    return MethodResult(method, received.observable_hash, base, 0.2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--config", default="configs/continual_ppo.json")
    parser.add_argument("--pretrained", default="pretrained/model_best.pt")
    parser.add_argument("--policy", default="logs/online/policy.pt")
    parser.add_argument("--method-group", choices=["traditional", "proposed", "diagnostic", "main", "all"], default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--delays", nargs="*", type=int, default=None)
    parser.add_argument("--snrs", nargs="*", type=float, default=[0, 5, 10, 15])
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--pilot-total", type=int, default=None)
    parser.add_argument("--reward-pilot-total", type=int, default=None)
    parser.add_argument("--pilot-layout", default=None)
    parser.add_argument("--state-split", choices=["offline_train", "heldout_edge", "drift"], default=None)
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
    parser.add_argument("--cir-update", choices=["fixed", "pilot_sparse", "decision_directed"], default="fixed")
    parser.add_argument("--cir-alpha", type=float, default=0.2)
    parser.add_argument("--online-groups", nargs="*", default=None)
    parser.add_argument("--online-learning-rate", type=float, default=None)
    parser.add_argument("--online-steps", type=int, default=None)
    parser.add_argument("--online-max-delta-norm", type=float, default=None)
    parser.add_argument("--online-proximal-weight", type=float, default=None)
    parser.add_argument("--online-min-reward-improvement", type=float, default=None)
    parser.add_argument("--online-cross-frame-tolerance", type=float, default=None)
    parser.add_argument("--online-phase-smoothing", type=float, default=None)
    parser.add_argument("--online-phase-min-confidence", type=float, default=None)
    parser.add_argument("--output-dir", default="logs/compare")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.version:
        print("RL4EQ continual-ppo schema-v1")
        return
    selected_methods = method_group(args.method_group) if args.method_group else _select_methods(args.methods)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    _apply_online_cli_overrides(
        config,
        {
            "online_adaptation_learning_rate": args.online_learning_rate,
            "online_adaptation_steps": args.online_steps,
            "online_adaptation_max_delta_norm": args.online_max_delta_norm,
            "online_adaptation_proximal_weight": args.online_proximal_weight,
            "online_adaptation_min_reward_improvement": args.online_min_reward_improvement,
            "online_cross_frame_rollback_tolerance": args.online_cross_frame_tolerance,
            "online_phase_tracking_smoothing": args.online_phase_smoothing,
            "online_phase_tracking_min_confidence": args.online_phase_min_confidence,
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
        for seed in range(args.num_seeds)
    }
    if not env_configs:
        raise ValueError("对比实验至少需要一个 delay/SNR/seed 配置。")
    effective_channel = effective_channel_metadata(next(iter(env_configs.values())))
    profile_prior = _profile_prior_from_config(config, impairment_profile)
    residual_cfo_limit = float(profile_prior.get("residual_cfo_limit", 0.001))
    acquisition_cfo_limit = float(profile_prior.get("acquisition_cfo_limit", max(0.004, residual_cfo_limit)))
    pretrained_path = Path(args.pretrained) if args.pretrained else None
    if pretrained_path is not None and not pretrained_path.exists():
        pretrained_path = None
    neural_methods = {
        "Offline NN only",
        "Pilot-conditioned frozen NN",
        "Pilot CIR only",
        "Pilot-Driven Online Adaptation",
        "RL-Modulated Neural Block Equalizer",
    }
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
    existing_keys = {_row_key(row) for row in existing_rows}
    rows = list(existing_rows)
    mode = "a" if args.resume else "w"
    with jsonl.open(mode, encoding="utf-8") as handle:
        for delay in selected_delays:
            for snr_db in args.snrs:
                for seed in range(args.num_seeds):
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
        "schema_version": "continual-ppo-compare-v2",
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
        "profile_prior": profile_prior,
        "forbidden": {"data_label_upper_bound_method_present": any("数据标签上界" in method for method in selected_methods)},
    }
    (target / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {args.output_dir}")


def _primary_summary_method(methods: tuple[str, ...]) -> str:
    """选择默认主结果方法，优先使用当前在线 proposed 主线。"""

    priority = (
        "Pilot-Driven Online Adaptation",
        "RL-Modulated Neural Block Equalizer",
        "Offline NN only",
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
    # Offline NN only 必须冻结 acquisition 条件，在线条件恢复属于单独消融。
    condition_update_mode: str = "fixed"
    condition_source: str = "acquisition"
    acquisition_phase_features: torch.Tensor | None = None
    physical_state: PilotPhysicalState | None = None
    phase_tracking_smoothing: float = 0.5
    phase_tracking_min_confidence: float = 0.15
    phase_tracking_cfo_limit: float = 0.0012


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
    acquisition_cfo: float = 0.0
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
    cross_frame_rollback_tolerance: float = 0.00001
    last_pre_update_snapshot: object | None = None
    last_pre_update_groups: frozenset[str] = frozenset()


def _select_methods(methods: list[str] | None) -> tuple[str, ...]:
    if not methods:
        return FORMAL_METHODS
    allowed = set(FORMAL_METHODS) | set(method_group("all"))
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
) -> dict[str, BaselineMethodState | PPOMethodState | NeuralMethodState | RLModulatedMethodState]:
    states: dict[str, BaselineMethodState | PPOMethodState | NeuralMethodState | RLModulatedMethodState] = {}
    model_config = _load_model_config(config, pretrained_path)
    for method in methods:
        if method == "Pilot-Driven Online Adaptation":
            model = _build_equalizer(model_config, pretrained_path, device)
            online_groups, candidate_config = _online_groups_from_config(config, online_groups_override)
            adapter = PilotDrivenOnlineAdapter(
                model,
                groups=online_groups,
                learning_rate=float(config.get("online_adaptation_learning_rate", 1e-4)),
                steps=int(config.get("online_adaptation_steps", 1)),
                max_delta_norm=float(config.get("online_adaptation_max_delta_norm", 0.5)),
                proximal_weight=float(config.get("online_adaptation_proximal_weight", 0.0)),
            )
            candidate_specs = _online_candidate_specs(candidate_config, online_groups)
            states[method] = PilotOnlineMethodState(
                model=model,
                adapter=adapter,
                cir=acquisition_cir.clone().to(device),
                receiver_state=ReceiverState(initial_soft_tail.clone().to(device)),
                pretrained_loaded=pretrained_path is not None,
                acquisition_cfo=float(acquisition_cfo),
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
                cross_frame_rollback_tolerance=float(
                    config.get("online_cross_frame_rollback_tolerance", 0.00001)
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
        elif method in PROPOSED_METHODS:
            model = _build_equalizer(model_config, pretrained_path, device)
            modulation_config = ModulationConfig(num_adapter_gates=len(model.blocks), num_lora_scales=len(model.blocks))
            if method == "Offline NN only":
                condition_update_mode = "fixed"
                condition_source = "acquisition"
            elif method == "Pilot-conditioned frozen NN":
                condition_update_mode = "fixed"
                condition_source = "pilot_phase"
            elif method == "Pilot CIR only":
                condition_update_mode = "pilot_sparse"
                condition_source = "pilot_cir_phase"
            else:
                condition_update_mode = str(cir_update_mode)
                condition_source = "pilot_cir_phase" if condition_update_mode == "pilot_sparse" else "acquisition"
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
        if method in FORMAL_METHODS and method != "Continual PPO" and method not in PROPOSED_METHODS and method not in TRADITIONAL_BASELINES
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
    if method in PROPOSED_METHODS:
        if not isinstance(state, NeuralMethodState):
            raise TypeError("神经 proposed 消融需要 NeuralMethodState。")
        device = next(state.model.parameters()).device
        frame_device = _frame_to_device(frame, device)
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
        rx_iq = torch.stack((frame_device.rx_symbols.real, frame_device.rx_symbols.imag), dim=-1).unsqueeze(0).float()
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
        state.receiver_state.update_tail(torch.complex(torch.tanh(logits[-tail_len:] / 2.0), torch.zeros_like(logits[-tail_len:])))
        if state.cir_update_mode == "decision_directed":
            state.cir = decision_directed_cir_update(
                frame_device,
                logits.detach(),
                int(delay),
                state.cir,
                alpha=float(state.cir_update_alpha),
            ).to(device)
        return _result_from_logits(
            method,
            logits,
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
        "Legacy LMMSE-FIR": (state.cir, 16, 0),
        "Legacy DFE": (state.cir, 8, 1),
        "No Adapt": (state.cir, 8, 0),
        "Best Fixed": (state.cir, 32, 2),
        "Drift-Aware Pilot Rule": (state.cir, 32, 1),
        "Contextual Bandit": (state.cir, 32, 1),
    }
    if method not in settings:
        raise ValueError(f"未知比较方法：{method}")
    cir, cg_iterations, refine_iterations = settings[method]
    result = perfect_csi_bpsk_refine_detect(
        frame.rx_symbols,
        cir,
        state.receiver_state.soft_tail,
        sigma,
        cg_iterations=cg_iterations,
        refine_iterations=refine_iterations,
    )
    state.receiver_state.update_tail(result.soft_tail)
    _update_baseline_cir(method, frame, result.logits, delay, state)
    return RealMethodResult(
        method=method,
        input_hash=_hash_json({"seed": int(frame.frame_index), "method": method, "visible": "rx+adapt"}),
        ber_data=bit_error_rate(result.logits[frame.data_mask], frame.bits[frame.data_mask]),
        ber_reward_pilot=bit_error_rate(result.logits[frame.reward_mask], frame.bits[frame.reward_mask]),
        ber_adapt_pilot=bit_error_rate(result.logits[frame.adapt_mask], frame.bits[frame.adapt_mask]),
        latency_ms=0.0,
        detector_iterations=int(cg_iterations + refine_iterations),
        extra=_baseline_extra(method),
    )


def _run_baseline_method_batch(methods: list[str], frame, snr_db: float, states: dict[str, BaselineMethodState | PPOMethodState], delay: int) -> dict[str, RealMethodResult]:
    sigma = torch.tensor(10.0 ** (-float(snr_db) / 10.0))
    grouped: dict[tuple[int, int], list[tuple[str, torch.Tensor, ReceiverState]]] = {}
    for method in methods:
        state = states[method]
        if not isinstance(state, BaselineMethodState):
            raise TypeError("baseline batch 只能处理 BaselineMethodState。")
        cir, cg_iterations, refine_iterations = _baseline_settings(method, frame, state)
        grouped.setdefault((cg_iterations, refine_iterations), []).append((method, cir, state.receiver_state))

    results: dict[str, RealMethodResult] = {}
    for (cg_iterations, refine_iterations), items in grouped.items():
        rx_b = torch.stack([frame.rx_symbols for _method, _cir, _receiver_state in items], dim=0)
        cir_b = torch.stack([cir for _method, cir, _receiver_state in items], dim=0)
        tail_b = torch.stack([receiver_state.soft_tail for _method, _cir, receiver_state in items], dim=0)
        detection = perfect_csi_bpsk_refine_detect(
            rx_b,
            cir_b,
            tail_b,
            sigma,
            cg_iterations=cg_iterations,
            refine_iterations=refine_iterations,
        )
        for batch_index, (method, _cir, receiver_state) in enumerate(items):
            state = states[method]
            if not isinstance(state, BaselineMethodState):
                raise TypeError("baseline batch 只能处理 BaselineMethodState。")
            receiver_state.update_tail(detection.soft_tail[batch_index])
            logits = detection.logits[batch_index]
            _update_baseline_cir(method, frame, logits, delay, state)
            results[method] = RealMethodResult(
                method=method,
                input_hash=_hash_json({"seed": int(frame.frame_index), "method": method, "visible": "rx+adapt"}),
                ber_data=bit_error_rate(logits[frame.data_mask], frame.bits[frame.data_mask]),
                ber_reward_pilot=bit_error_rate(logits[frame.reward_mask], frame.bits[frame.reward_mask]),
                ber_adapt_pilot=bit_error_rate(logits[frame.adapt_mask], frame.bits[frame.adapt_mask]),
                latency_ms=0.0,
                detector_iterations=int(cg_iterations + refine_iterations),
                extra=_baseline_extra(method),
            )
    return results


def _baseline_settings(method: str, frame, state: BaselineMethodState) -> tuple[torch.Tensor, int, int]:
    settings = {
        "Perfect-CSI Block": (frame.true_cir, 32, 2),
        "Sparse CIR + Kalman/RLS": (state.cir, 32, 1),
        "Block LMMSE/CG": (state.cir, 32, 0),
        "DFE-RLS": (state.cir, 16, 1),
        "Analytic Iterative BPSK": (state.cir, 16, 2),
        "Legacy LMMSE-FIR": (state.cir, 16, 0),
        "Legacy DFE": (state.cir, 8, 1),
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
    updates_frozen = _online_updates_are_frozen(snr_db, state.freeze_online_below_snr_db)
    update_scheduled = _online_update_is_scheduled(frame_index, update_interval)
    rx_iq = torch.stack((frame_device.rx_symbols.real, frame_device.rx_symbols.imag), dim=-1).unsqueeze(0).float()
    region_ids = frame_device.model_region_ids.unsqueeze(0).long()
    adapt_symbols = frame_device.receiver_view().adapt_symbols.unsqueeze(0).to(torch.complex64)
    adapt_mask = frame_device.adapt_mask.unsqueeze(0).bool()
    cir_update_accepted = False
    cir_reward_loss_before = None
    cir_reward_loss_after = None
    tracked_state = None
    if state.cir_update_mode == "pilot_sparse" and not updates_frozen:
        previous_cir = state.cir.clone()
        tracked_state = track_pilot_physical_state(
            frame_device,
            state.cir,
            state.receiver_state.soft_tail,
            previous=state.physical_state,
            cfo_limit=float(state.phase_tracking_cfo_limit),
            smoothing=float(state.phase_tracking_smoothing),
            min_confidence=float(state.phase_tracking_min_confidence),
        )
        state.physical_state = tracked_state
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
                    condition_source="pilot_cir_phase",
                    receiver_view=frame.receiver_view(),
                    cir=cir_value,
                    soft_tail=state.receiver_state.soft_tail,
                    acquisition_cfo=float(state.acquisition_cfo),
                    tracked_cfo=float(tracked_state.cfo_cycles_per_symbol),
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
                )
                candidate_logits, _ = state.model(
                    rx_iq,
                    _condition_for(candidate_cir),
                    region_ids,
                    tail,
                    adapt_symbols=adapt_symbols,
                    adapt_mask=adapt_mask,
                )
            cir_reward_loss_before = float(
                _masked_bce(previous_logits.squeeze(0), frame_device.bits, frame_device.reward_mask).detach().cpu()
            )
            cir_reward_loss_after = float(
                _masked_bce(candidate_logits.squeeze(0), frame_device.bits, frame_device.reward_mask).detach().cpu()
            )
            cir_update_accepted = _accept_pilot_cir_update(cir_reward_loss_before, cir_reward_loss_after)
            state.cir = candidate_cir if cir_update_accepted else previous_cir
    tail = state.receiver_state.soft_tail.unsqueeze(0).to(torch.complex64)
    phase_features = _neural_phase_features(
        condition_source="pilot_cir_phase",
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
    )
    condition = condition_from_cir(
        state.cir,
        snr_db,
        cfo_residual=float(state.acquisition_cfo),
        phase_features=phase_features,
    )
    with torch.no_grad():
        before_logits, _ = state.model(
            rx_iq,
            condition,
            region_ids,
            tail,
            adapt_symbols=adapt_symbols,
            adapt_mask=adapt_mask,
        )
    before = before_logits.squeeze(0)
    reward_before = _masked_bce(before, frame_device.bits, frame_device.reward_mask)
    adapt_loss_before = _masked_bce(before, frame_device.bits, frame_device.adapt_mask)
    cross_frame_rollback = False
    previous_good_reward_loss = None
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
            )
        previous_good = previous_good_logits.squeeze(0)
        previous_good_reward_loss = float(
            _masked_bce(
                previous_good,
                frame_device.bits,
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
            reward_before = _masked_bce(before, frame_device.bits, frame_device.reward_mask)
            adapt_loss_before = _masked_bce(before, frame_device.bits, frame_device.adapt_mask)
            cross_frame_rollback = True
            state.last_pre_update_snapshot = None
            state.last_pre_update_groups = frozenset()
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
    candidate_groups = set().union(*(set(item["groups"]) for item in candidates)) if candidates else set()
    base_snapshot = state.model.peft.snapshot(candidate_groups)
    best_snapshot = None
    best_result = None
    best_reward = float(reward_before.detach().cpu())
    best_name = "skip"
    best_after = before
    for candidate in candidates:
        state.model.peft.restore(base_snapshot)
        adaptation = state.adapter.adapt(
            frame_device,
            condition,
            tail,
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
            )
        after = after_logits.squeeze(0)
        reward_after = _masked_bce(after, frame_device.bits, frame_device.reward_mask)
        reward_value = float(reward_after.detach().cpu())
        if adaptation.accepted and _accept_online_peft_update(
            best_reward,
            reward_value,
            state.min_reward_improvement,
        ):
            best_snapshot = state.model.peft.snapshot(candidate_groups)
            best_result = adaptation
            best_reward = reward_value
            best_name = str(candidate["name"])
            best_after = after
    state.model.peft.restore(base_snapshot)
    if best_snapshot is None:
        adaptation = None
        accepted = False
        final = before
        reward_after = reward_before
    else:
        state.model.peft.restore(best_snapshot)
        adaptation = best_result
        accepted = True
        final = best_after
        reward_after = torch.as_tensor(best_reward, device=reward_before.device)
    if accepted:
        state.last_pre_update_groups = frozenset(candidate_groups)
        state.last_pre_update_snapshot = base_snapshot
    tail_len = state.receiver_state.soft_tail.numel()
    detected_tail = torch.complex(
        torch.tanh(final[-tail_len:] / 2.0),
        torch.zeros_like(final[-tail_len:]),
    )
    state.receiver_state.update_tail(
        (1.0 - state.tail_update_alpha) * state.receiver_state.soft_tail
        + state.tail_update_alpha * detected_tail
    )
    if state.cir_update_mode == "decision_directed":
        state.cir = decision_directed_cir_update(
            frame_device,
            final.detach(),
            int(delay),
            state.cir,
            alpha=float(state.cir_update_alpha),
        ).to(device)
    return _result_from_logits(
        method,
        final,
        frame_device,
        0,
        {
            "uses_neural_network": True,
            "uses_rl": False,
            "online_algorithm": "pilot_driven_constrained_peft",
            "online_update_source": "adapt_pilot_only",
            "reward_pilot_guard": True,
            "cross_frame_reward_guard": True,
            "previous_good_reward_loss": previous_good_reward_loss,
            "rollback": bool(cross_frame_rollback),
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
            "adaptation_accepted": bool(accepted),
            "online_update_candidate": best_name,
            "online_update_candidate_count": len(candidates),
            "online_snr_layer": "fully_frozen" if updates_frozen else ("peft_frozen" if not candidates else "peft_enabled"),
            "online_update_scheduled": bool(update_scheduled),
            "online_update_interval": int(update_interval),
            "online_update_skipped": bool(not update_scheduled),
            "online_condition_update_applied": not updates_frozen,
            "online_adaptation_freeze_below_snr_db": freeze_below_db,
            "reward_pilot_loss_before": float(reward_before.detach().cpu()),
            "reward_pilot_loss_after": float(reward_after.detach().cpu()),
            "online_min_reward_improvement": float(state.min_reward_improvement),
            "online_proximal_weight": float(state.adapter.proximal_weight),
            "peft_update_guarded": True,
            "parameter_delta_norm": float(adaptation.parameter_delta_norm if accepted else 0.0),
            "tail_update_alpha": float(state.tail_update_alpha),
            "cir_update_mode": str(state.cir_update_mode),
            "cir_update_alpha": float(state.cir_update_alpha),
            "condition_source": "pilot_cir_phase",
            "pilot_phase_used": True,
            "cir_update_applied": bool(cir_update_accepted),
            "cir_update_guarded": bool(
                state.cir_update_mode == "pilot_sparse" and not updates_frozen and update_scheduled
            ),
            "cir_reward_pilot_loss_before": cir_reward_loss_before,
            "cir_reward_pilot_loss_after": cir_reward_loss_after,
            "peft_update_applied": bool(accepted),
            "data_labels_used_online": False,
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
        "Offline NN only": {
            "condition_source": "acquisition",
            "pilot_phase_used": False,
            "cir_update_applied": False,
            "peft_update_applied": False,
        },
        "Pilot-conditioned frozen NN": {
            "condition_source": "pilot_phase",
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
    if method in PROPOSED_METHODS:
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
    before = perfect_csi_bpsk_refine_detect(
        frame.rx_symbols,
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
        frame.rx_symbols,
        state.cir,
        state.receiver_state.soft_tail,
        sigma,
        cg_iterations=cg_iterations,
        refine_iterations=refine_iterations,
    )
    reward_loss_before = _masked_bce(before.logits, frame.bits, frame.reward_mask)
    reward_loss_after = _masked_bce(result.logits, frame.bits, frame.reward_mask)
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
    state.receiver_state.update_tail(result.soft_tail)
    cir_alpha = 0.2 if action.mode not in {"skip", "rollback"} else 0.05
    state.cir = _decision_directed_cir_update(frame, result.logits, int(delay), state.cir, alpha=cir_alpha * float(action.cir_trust))
    state.previous_reward = reward
    state.last_parameter_delta_norm = parameter_delta_norm
    return RealMethodResult(
        method=method,
        input_hash=_hash_json({"seed": int(frame.frame_index), "method": method, "visible": "rx+adapt+policy"}),
        ber_data=bit_error_rate(result.logits[frame.data_mask], frame.bits[frame.data_mask]),
        ber_reward_pilot=bit_error_rate(result.logits[frame.reward_mask], frame.bits[frame.reward_mask]),
        ber_adapt_pilot=bit_error_rate(result.logits[frame.adapt_mask], frame.bits[frame.adapt_mask]),
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
        },
    )


def _load_existing_rows(jsonl: Path) -> list[dict]:
    if not jsonl.exists():
        return []
    return [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]


def _row_key(row: dict) -> tuple[str, int, float, int, int, int, int, str, str]:
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
    )


def _hash_json(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
