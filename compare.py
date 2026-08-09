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
from agent.cir_estimator import CIRCondition, condition_from_cir, decision_directed_cir_update
from agent.discrete_safe_policy import DiscreteSafePolicy, initialize_safe_discrete_policy_prior, safe_modulation_actions
from agent.modulation import ModulationConfig, ModulationState
from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from baseline.legacy_equalizers import legacy_dfe, legacy_lmmse_fir
from baseline.block_equalizers import bit_error_rate, perfect_csi_bpsk_refine_detect
from baseline.traditional_equalizers import TRADITIONAL_BASELINES, run_traditional_equalizer
from env.comm_env import CommEnvConfig, CommunicationEnvironment, ReceiverState
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
)

PROPOSED_METHODS = (
    "Offline NN only",
    "NN + Fixed Modulation",
    "NN + Rule Modulation",
    "NN + Discrete PEFT Scheduler",
    "RL-Modulated Neural Block Equalizer",
)

DIAGNOSTIC_METHODS = (
    "Perfect-CSI Block",
    "Fixed CG-BPSK-DD Block Detector",
)


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
    parser.add_argument("--delays", nargs="*", type=int, default=[20, 30, 40])
    parser.add_argument("--snrs", nargs="*", type=float, default=[0, 5, 10, 15])
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--pilot-total", type=int, default=None)
    parser.add_argument("--pilot-layout", default=None)
    parser.add_argument("--update-interval", type=int, default=32)
    parser.add_argument("--cir-update", choices=["fixed", "decision_directed"], default="fixed")
    parser.add_argument("--output-dir", default="logs/compare")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.version:
        print("RL4EQ continual-ppo schema-v1")
        return
    selected_methods = method_group(args.method_group) if args.method_group else _select_methods(args.methods)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    policy_explicit = any(argument == "--policy" or argument.startswith("--policy=") for argument in sys.argv[1:])
    policy_required = policy_explicit and "RL-Modulated Neural Block Equalizer" in selected_methods
    policy_path = Path(args.policy) if policy_explicit and args.policy else None
    if policy_path is not None and not policy_path.exists():
        if policy_required:
            raise FileNotFoundError(f"显式指定的 policy checkpoint 不存在：{policy_path}")
        policy_path = None
    pilot_total = int(args.pilot_total if args.pilot_total is not None else config.get("pilot_total", 128))
    pilot_layout = str(args.pilot_layout if args.pilot_layout is not None else config.get("pilot_layout", "prefix"))
    pretrained_path = Path(args.pretrained) if args.pretrained else None
    if pretrained_path is not None and not pretrained_path.exists():
        pretrained_path = None
    target = Path(args.output_dir)
    target.mkdir(parents=True, exist_ok=True)
    jsonl = target / "frame_metrics.jsonl"
    existing_rows = _load_existing_rows(jsonl) if args.resume else []
    existing_keys = {_row_key(row) for row in existing_rows}
    rows = list(existing_rows)
    mode = "a" if args.resume else "w"
    with jsonl.open(mode, encoding="utf-8") as handle:
        for delay in args.delays:
            for snr_db in args.snrs:
                for seed in range(args.num_seeds):
                    env = CommunicationEnvironment(
                        CommEnvConfig(
                            level="B",
                            max_delay=int(delay),
                            snr_db=float(snr_db),
                            rho=float(config.get("rho", 0.99)),
                            total_pilot=pilot_total,
                            layout=pilot_layout,
                            seed=50_000 + int(seed),
                        )
                    )
                    start = env.reset_episode()
                    acquisition_cir = _estimate_cir_from_known_frame(start.acquisition, int(delay))
                    method_states = _build_method_states(
                        selected_methods,
                        start.initial_soft_tail,
                        acquisition_cir,
                        config,
                        int(delay),
                        float(snr_db),
                        int(seed),
                        pretrained_path=pretrained_path,
                        policy_path=policy_path,
                        policy_required=policy_required,
                        device=str(args.device),
                        cir_update_mode=str(args.cir_update),
                    )
                    for frame_index in range(1, args.frames + 1):
                        frame = env.next_frame()
                        frame_results = _run_methods_for_frame(
                            selected_methods,
                            frame,
                            float(snr_db),
                            method_states,
                            delay=int(delay),
                            frame_index=int(frame_index),
                            update_interval=int(args.update_interval),
                        )
                        for method in selected_methods:
                            result = frame_results[method]
                            metric = FrameMetric(
                                method=method,
                                level="B",
                                delay=delay,
                                snr_db=snr_db,
                                rho=float(config.get("rho", 0.99)),
                                pilot_total=pilot_total,
                                pilot_layout=pilot_layout,
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
                            payload["input_hash"] = result.input_hash
                            payload.update(result.extra)
                            key = _row_key(payload)
                            if key in existing_keys:
                                continue
                            rows.append(payload)
                            existing_keys.add(key)
                            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                            handle.flush()
    ppo_rows = [row for row in rows if row["method"] == "Continual PPO"]
    summary = summarize_main_matrix(ppo_rows)
    interval = paired_block_bootstrap(ppo_rows, seed=0, repetitions=200, block_length=min(10, max(1, args.frames)))
    summary_payload = {
        "schema_version": "continual-ppo-compare-v2",
        "methods": list(selected_methods),
        "main_level": "B",
        "level_c_separate": True,
        "per_config": [item.__dict__ for item in summary.per_config],
        "generalization": summary.generalization,
        "bootstrap": interval.__dict__,
        "forbidden": {"data_label_upper_bound_method_present": any("数据标签上界" in method for method in selected_methods)},
    }
    (target / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {args.output_dir}")


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
    cir_update_mode: str = "fixed"


@dataclass
class RLModulatedMethodState:
    online_state: WindowedDiscreteOnlineState
    pretrained_loaded: bool
    policy_loaded: bool


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
) -> dict[str, BaselineMethodState | PPOMethodState | NeuralMethodState | RLModulatedMethodState]:
    states: dict[str, BaselineMethodState | PPOMethodState | NeuralMethodState | RLModulatedMethodState] = {}
    model_config = _load_model_config(config, pretrained_path)
    for method in methods:
        if method == "RL-Modulated Neural Block Equalizer":
            torch.manual_seed(_method_seed("rl_modulated", seed, delay, snr_db))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(_method_seed("rl_modulated", seed, delay, snr_db))
            model = _build_equalizer(model_config, pretrained_path, device)
            actions = safe_modulation_actions(len(model.blocks), device=device)
            from agent.rl_modulator import ModulationObservationEncoder

            encoder = ModulationObservationEncoder()
            policy = DiscreteSafePolicy(len(encoder.FIELDS), len(actions)).to(device)
            initialize_safe_discrete_policy_prior(policy)
            policy_loaded = _load_discrete_policy_if_available(policy, policy_path, device, required=policy_required)
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
                    rollout=[],
                ),
                pretrained_loaded=pretrained_path is not None,
                policy_loaded=policy_loaded,
            )
        elif method in PROPOSED_METHODS:
            model = _build_equalizer(model_config, pretrained_path, device)
            modulation_config = ModulationConfig(num_adapter_gates=len(model.blocks), num_lora_scales=len(model.blocks))
            states[method] = NeuralMethodState(
                cir=acquisition_cir.clone().to(device),
                receiver_state=ReceiverState(initial_soft_tail.clone().to(device)),
                model=model,
                modulation=_fixed_modulation_for(method, modulation_config, device),
                pretrained_loaded=pretrained_path is not None,
                cir_update_mode=str(cir_update_mode),
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
            states[method] = BaselineMethodState(cir=acquisition_cir.clone(), receiver_state=ReceiverState(initial_soft_tail.clone()))
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
        if method in FORMAL_METHODS and method != "Continual PPO" and method not in PROPOSED_METHODS
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
        result = run_traditional_equalizer(method, frame.receiver_view(), state.cir, state.receiver_state.soft_tail, snr_db)
        state.receiver_state.update_tail(result.soft_tail)
        return _result_from_logits(method, result.logits, frame, result.iterations, result.extra)
    if method == "RL-Modulated Neural Block Equalizer":
        if not isinstance(state, RLModulatedMethodState):
            raise TypeError("RL-Modulated 方法需要 RLModulatedMethodState。")
        frame_device = _frame_to_device(frame, next(state.online_state.model.parameters()).device)
        condition = condition_from_cir(state.online_state.cir, snr_db)
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
    if method in PROPOSED_METHODS:
        if not isinstance(state, NeuralMethodState):
            raise TypeError("神经 proposed 消融需要 NeuralMethodState。")
        device = next(state.model.parameters()).device
        frame_device = _frame_to_device(frame, device)
        rx_iq = torch.stack((frame_device.rx_symbols.real, frame_device.rx_symbols.imag), dim=-1).unsqueeze(0).float()
        region_ids = frame_device.model_region_ids.unsqueeze(0).long()
        condition = condition_from_cir(state.cir, snr_db)
        logits, _ = state.model(
            rx_iq,
            condition,
            region_ids,
            state.receiver_state.soft_tail.unsqueeze(0).to(torch.complex64),
            modulation=state.modulation,
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
                alpha=0.2,
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
                "cir_update_uses_data_labels": False,
            },
        )
    if method in DIAGNOSTIC_METHODS:
        if not isinstance(state, BaselineMethodState):
            raise TypeError("诊断方法需要 BaselineMethodState。")
        cir = frame.true_cir if method == "Perfect-CSI Block" else state.cir
        result = perfect_csi_bpsk_refine_detect(
            frame.rx_symbols,
            cir,
            state.receiver_state.soft_tail,
            torch.tensor(10.0 ** (-float(snr_db) / 10.0)),
            cg_iterations=32,
            refine_iterations=2,
        )
        state.receiver_state.update_tail(result.soft_tail)
        return _result_from_logits(method, result.logits, frame, result.iterations, {"diagnostic": True})
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


def _row_key(row: dict) -> tuple[str, int, float, int, int, int, str]:
    return (
        str(row["method"]),
        int(row["delay"]),
        float(row["snr_db"]),
        int(row["seed"]),
        int(row["frame"]),
        int(row.get("pilot_total", 0)),
        str(row.get("pilot_layout", "")),
    )


def _hash_json(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
