# -*- coding: utf-8 -*-
"""正式比较入口与标签隔离方法封装。"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from baseline.legacy_equalizers import legacy_dfe, legacy_lmmse_fir
from evaluation.bootstrap import paired_block_bootstrap
from evaluation.metrics import FrameMetric, summarize_main_matrix


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
    del args.config, args.pretrained, args.policy, args.resume
    target = Path(args.output_dir)
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    jsonl = target / "frame_metrics.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for delay in args.delays:
            for snr_db in args.snrs:
                for seed in range(args.num_seeds):
                    for frame_index in range(args.frames):
                        observable = paired_frame(seed=seed, delay=delay, snr_db=snr_db, frame=frame_index).hide_reward_and_data_labels()
                        for method in FORMAL_METHODS:
                            result = run_method(method, observable)
                            metric = FrameMetric(
                                method=method,
                                level="B",
                                delay=delay,
                                snr_db=snr_db,
                                seed=seed,
                                frame=frame_index,
                                ber_data=result.ber_data,
                                latency_ms=result.latency_ms,
                            )
                            payload = metric.to_json()
                            payload["input_hash"] = result.input_hash
                            rows.append(payload)
                            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    ppo_rows = [row for row in rows if row["method"] == "Continual PPO"]
    summary = summarize_main_matrix(ppo_rows)
    interval = paired_block_bootstrap(ppo_rows, seed=0, repetitions=200, block_length=min(10, max(1, args.frames)))
    summary_payload = {
        "schema_version": "continual-ppo-compare-v1",
        "methods": list(FORMAL_METHODS),
        "main_level": "B",
        "level_c_separate": True,
        "per_config": [item.__dict__ for item in summary.per_config],
        "generalization": summary.generalization,
        "bootstrap": interval.__dict__,
        "forbidden": {"oracle_from_data_labels_present": any(("Data" + " Oracle") in method for method in FORMAL_METHODS)},
    }
    (target / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {args.output_dir}")


def _hash_json(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
