# -*- coding: utf-8 -*-
"""固定参数核查 DFE 的已知 Adapt 符号反馈，不选择或替换正式基线。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.synchronization_compensation import apply_phase_correction
from baseline.traditional_equalizers import (
    _dfe_rls, _tail_from_estimate, estimate_acquisition_cir_with_cfo, estimate_phase_residual_vector,
)
from env.comm_env import CommunicationEnvironment
from env.experiment_config import build_comm_env_config
from evaluation.pilot_transfer import paired_interval
from scripts.calibrate_dfe_rls import REFERENCE


def run_diagnostic(config, seeds, frames, output):
    target = Path(output)
    if target.exists():
        raise ValueError("诊断输出已存在，请指定新路径。")
    if (config.get("main_level") != "B" or config.get("pilot_layout") != "prefix"
            or config.get("offline_state_split") != "offline_train"):
        raise ValueError("本轮仅支持 offline_train、Level B 和前缀 Pilot。")
    if not seeds or len(set(seeds)) != len(seeds) or frames < 1:
        raise ValueError("需要非空不重复 seed 和正帧数。")
    memory, rows = int(config["max_delay"]), []
    for snr in config["main_snrs"]:
        for seed in seeds:
            env = CommunicationEnvironment(build_comm_env_config(
                config, level="B", max_delay=memory, snr_db=float(snr), seed=seed,
                total_pilot=int(config["pilot_total"]), reward_pilot_total=int(config["reward_pilot_total"]),
                pilot_layout="prefix", state_split="offline_train",
            ))
            start = env.reset_episode()
            cir, cfo = estimate_acquisition_cir_with_cfo(
                start.acquisition, memory, cfo_limit=float(config.get("acquisition_cfo_limit", 0.004)))
            tails = {flag: start.initial_soft_tail.clone() for flag in (False, True)}
            for index in range(frames):
                frame = env.next_frame()
                view = frame.receiver_view()
                estimates = {}
                for flag in (False, True):
                    phase = estimate_phase_residual_vector(view, cir, tails[flag], cfo_hint=cfo)
                    corrected = apply_phase_correction(view.rx_symbols, float(phase[0]), cfo)
                    estimate, updates = _dfe_rls(
                        corrected, view.adapt_mask, view.adapt_symbols, REFERENCE.filter_length,
                        feedback_len=REFERENCE.feedback_length, forgetting_factor=REFERENCE.forgetting_factor,
                        initial_inverse_covariance=REFERENCE.initial_inverse_covariance, known_pilot_feedback=flag)
                    tails[flag] = _tail_from_estimate(estimate, memory)
                    estimates[flag] = (estimate, updates)
                # 两种协议的接收估计均已结束；此后仅使用 Data 标签作仿真评分。
                fingerprint = hashlib.sha256(frame.rx_symbols.numpy().tobytes()).hexdigest()
                for flag, (estimate, updates) in estimates.items():
                    ber = float(((estimate.real[frame.data_mask] >= 0) != frame.bits[frame.data_mask]).float().mean())
                    rows.append({"snr_db": snr, "seed": seed, "frame": index, "known_pilot_feedback": flag,
                                 "ber_data": ber, "adapt_updates": updates, "input_sha256": fingerprint,
                                 "reward_labels_used_for_receiver": False, "data_labels_used_for_receiver": False})
            print(json.dumps({"snr_db": snr, "seed": seed, "frames_complete": frames}), flush=True)
    summary = []
    for snr in config["main_snrs"]:
        item = {"snr_db": snr}
        seed_means = {}
        for flag, name in ((False, "original"), (True, "known_feedback")):
            seed_means[flag] = [float(np.mean([row["ber_data"] for row in rows
                                if row["snr_db"] == snr and row["seed"] == seed
                                and row["known_pilot_feedback"] == flag])) for seed in seeds]
            item[name] = paired_interval(seed_means[flag])
        item["ber_improvement"] = paired_interval(np.asarray(seed_means[False]) - np.asarray(seed_means[True]))
        summary.append(item)
    from dataclasses import asdict
    payload = {"schema_version": "dfe-known-feedback-v1", "diagnostic_only": True,
               "main_aggregation_allowed": False, "config": config, "seeds": list(seeds), "frames": frames,
               "setting": asdict(REFERENCE), "state_split": "offline_train",
               "uses_neural_network": False, "uses_rl": False,
               "reward_labels_used_for_receiver": False, "data_labels_used_for_receiver": False,
               "per_snr": summary, "rows": rows}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed-start", type=int, default=43000)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_diagnostic(json.loads(Path(args.config).read_text(encoding="utf-8")),
                   list(range(args.seed_start, args.seed_start + args.num_seeds)), args.frames, args.output)


if __name__ == "__main__":
    main()
