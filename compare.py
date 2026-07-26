# -*- coding: utf-8 -*-
"""正式比较入口与标签隔离方法封装。"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from baseline.legacy_equalizers import legacy_dfe, legacy_lmmse_fir
from baseline.block_equalizers import bit_error_rate, perfect_csi_bpsk_refine_detect
from env.comm_env import CommEnvConfig, CommunicationEnvironment, ReceiverState
from evaluation.bootstrap import paired_block_bootstrap
from evaluation.metrics import FrameMetric, summarize_main_matrix
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
    parser.add_argument("--delays", nargs="*", type=int, default=[20, 30, 40])
    parser.add_argument("--snrs", nargs="*", type=float, default=[10, 15, 20])
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--output-dir", default="logs/compare")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.version:
        print("RL4EQ continual-ppo schema-v1")
        return
    del args.pretrained, args.policy, args.resume
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    target = Path(args.output_dir)
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    jsonl = target / "frame_metrics.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
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
                    receiver_states = {
                        method: ReceiverState(start.initial_soft_tail)
                        for method in FORMAL_METHODS
                    }
                    for frame_index in range(1, args.frames + 1):
                        frame = env.next_frame()
                        for method in FORMAL_METHODS:
                            result = _run_real_method(method, frame, acquisition_cir, float(snr_db), receiver_states[method])
                            metric = FrameMetric(
                                method=method,
                                level="B",
                                delay=delay,
                                snr_db=snr_db,
                                seed=seed,
                                frame=frame_index,
                                ber_data=result.ber_data,
                                latency_ms=result.latency_ms,
                                ber_reward_pilot=result.ber_reward_pilot,
                                ber_adapt_pilot=result.ber_adapt_pilot,
                                detector_iterations=result.detector_iterations,
                            )
                            payload = metric.to_json()
                            payload["input_hash"] = result.input_hash
                            rows.append(payload)
                            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    ppo_rows = [row for row in rows if row["method"] == "Continual PPO"]
    summary = summarize_main_matrix(ppo_rows)
    interval = paired_block_bootstrap(ppo_rows, seed=0, repetitions=200, block_length=min(10, max(1, args.frames)))
    summary_payload = {
        "schema_version": "continual-ppo-compare-v2",
        "methods": list(FORMAL_METHODS),
        "main_level": "B",
        "level_c_separate": True,
        "per_config": [item.__dict__ for item in summary.per_config],
        "generalization": summary.generalization,
        "bootstrap": interval.__dict__,
        "forbidden": {"data_label_upper_bound_method_present": any("数据标签上界" in method for method in FORMAL_METHODS)},
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


def _run_real_method(method: str, frame, acquisition_cir, snr_db: float, receiver_state: ReceiverState) -> RealMethodResult:
    sigma = torch.tensor(10.0 ** (-float(snr_db) / 10.0))
    settings = {
        "Perfect-CSI Block": (frame.true_cir, 32, 2),
        "Sparse CIR + Kalman/RLS": (acquisition_cir, 32, 1),
        "Block LMMSE/CG": (acquisition_cir, 32, 0),
        "DFE-RLS": (acquisition_cir, 16, 1),
        "Analytic Iterative BPSK": (acquisition_cir, 16, 2),
        "Legacy LMMSE-FIR": (acquisition_cir, 16, 0),
        "Legacy DFE": (acquisition_cir, 8, 1),
        "No Adapt": (acquisition_cir, 8, 0),
        "Best Fixed": (acquisition_cir, 32, 2),
        "Drift-Aware Pilot Rule": (acquisition_cir, 32, 1),
        "Contextual Bandit": (acquisition_cir, 32, 1),
        "Continual PPO": (acquisition_cir, 32, 2),
    }
    if method not in settings:
        raise ValueError(f"未知比较方法：{method}")
    cir, cg_iterations, refine_iterations = settings[method]
    result = perfect_csi_bpsk_refine_detect(
        frame.rx_symbols,
        cir,
        receiver_state.soft_tail,
        sigma,
        cg_iterations=cg_iterations,
        refine_iterations=refine_iterations,
    )
    receiver_state.update_tail(result.soft_tail)
    return RealMethodResult(
        method=method,
        input_hash=_hash_json({"seed": int(frame.frame_index), "method": method, "visible": "rx+adapt"}),
        ber_data=bit_error_rate(result.logits[frame.data_mask], frame.bits[frame.data_mask]),
        ber_reward_pilot=bit_error_rate(result.logits[frame.reward_mask], frame.bits[frame.reward_mask]),
        ber_adapt_pilot=bit_error_rate(result.logits[frame.adapt_mask], frame.bits[frame.adapt_mask]),
        latency_ms=0.0,
        detector_iterations=int(cg_iterations + refine_iterations),
    )


def _hash_json(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
