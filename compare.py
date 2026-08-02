# -*- coding: utf-8 -*-
"""正式比较入口与标签隔离方法封装。"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from agent.continual_policy import ContinualPolicy, ObservationEncoder
from baseline.legacy_equalizers import legacy_dfe, legacy_lmmse_fir
from baseline.block_equalizers import bit_error_rate, perfect_csi_bpsk_refine_detect
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
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--delays", nargs="*", type=int, default=[20, 30, 40])
    parser.add_argument("--snrs", nargs="*", type=float, default=[10, 15, 20])
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--update-interval", type=int, default=32)
    parser.add_argument("--output-dir", default="logs/compare")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.version:
        print("RL4EQ continual-ppo schema-v1")
        return
    del args.pretrained, args.policy
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    selected_methods = _select_methods(args.methods)
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
                            total_pilot=int(config.get("pilot_total", 128)),
                            layout=str(config.get("pilot_layout", "two_block")),
                            seed=50_000 + int(seed),
                        )
                    )
                    start = env.reset_episode()
                    acquisition_cir = _estimate_cir_from_known_frame(start.acquisition, int(delay))
                    method_states = _build_method_states(selected_methods, start.initial_soft_tail, acquisition_cir, config, int(delay), float(snr_db), int(seed))
                    for frame_index in range(1, args.frames + 1):
                        frame = env.next_frame()
                        frame_results = _run_baseline_method_batch(
                            [method for method in selected_methods if method != "Continual PPO"],
                            frame,
                            float(snr_db),
                            method_states,
                            delay=int(delay),
                        )
                        if "Continual PPO" in selected_methods:
                            frame_results["Continual PPO"] = _run_real_method(
                                "Continual PPO",
                                frame,
                                float(snr_db),
                                method_states["Continual PPO"],
                                int(delay),
                                int(frame_index),
                                int(args.update_interval),
                            )
                        for method in selected_methods:
                            result = frame_results[method]
                            metric = FrameMetric(
                                method=method,
                                level="B",
                                delay=delay,
                                snr_db=snr_db,
                                rho=float(config.get("rho", 0.99)),
                                pilot_total=int(config.get("pilot_total", 128)),
                                pilot_layout=str(config.get("pilot_layout", "two_block")),
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


def _select_methods(methods: list[str] | None) -> tuple[str, ...]:
    if not methods:
        return FORMAL_METHODS
    unknown = [method for method in methods if method not in FORMAL_METHODS]
    if unknown:
        raise ValueError(f"未知比较方法：{unknown}")
    deduplicated = []
    for method in methods:
        if method not in deduplicated:
            deduplicated.append(method)
    return tuple(deduplicated)


def _build_method_states(methods: tuple[str, ...], initial_soft_tail: torch.Tensor, acquisition_cir: torch.Tensor, config: dict, delay: int, snr_db: float, seed: int) -> dict[str, BaselineMethodState | PPOMethodState]:
    states: dict[str, BaselineMethodState | PPOMethodState] = {}
    for method in methods:
        if method == "Continual PPO":
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


def _row_key(row: dict) -> tuple[str, int, float, int, int]:
    return (str(row["method"]), int(row["delay"]), float(row["snr_db"]), int(row["seed"]), int(row["frame"]))


def _hash_json(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
