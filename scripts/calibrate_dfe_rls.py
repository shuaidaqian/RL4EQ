# -*- coding: utf-8 -*-
"""传统 DFE-RLS 先验与遗忘因子的训练范围校准及冻结后核验。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

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


@dataclass(frozen=True, order=True)
class DFESetting:
    filter_length: int
    feedback_length: int
    forgetting_factor: float
    initial_inverse_covariance: float


REFERENCE = DFESetting(41, 5, 0.995, 10.0)
CONFIRMATION_SEEDS = list(range(59001, 59006))
CONFIRMATION_FRAMES = 8


def calibration_grid(memory):
    return [DFESetting(length, feedback, forgetting, covariance)
            for length, feedback in ((41, 5), (memory + 1, memory))
            for forgetting in (0.98, 0.995, 1.0) for covariance in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)]


def validate_selection(selection, config, seeds, frames):
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
    if selection is None or selection.get("schema_version") != "dfe-rls-frozen-selection-v1":
        raise ValueError("独立核验必须提供支持的冻结 DFE 设置。")
    if signature != selection.get("config_sha256"):
        raise ValueError("冻结 DFE 设置与信道/Pilot 配置不匹配。")
    if set(seeds).intersection(selection["calibration_seeds"]):
        raise ValueError("核验 seed 与校准 seed 重叠。")
    if list(seeds) != selection["confirmation_seeds"] or frames != selection["confirmation_frames"]:
        raise ValueError("核验 seed/帧数必须与冻结文件的预定预算一致。")
    grid = calibration_grid(int(config["max_delay"]))
    for snr in config["main_snrs"]:
        if DFESetting(**selection["by_snr"][format(float(snr), "g")]) not in grid:
            raise ValueError("冻结 DFE 参数超出预先声明的校准网格。")


def run_calibration(config, seeds, frames, output, *, state_split="offline_train", selection=None,
                    freeze_output=None):
    target = Path(output)
    freeze = Path(freeze_output) if freeze_output is not None else None
    if target.exists() or (freeze is not None and freeze.exists()):
        raise ValueError("校准输出或冻结文件已存在，请指定新路径。")
    if not seeds or len(set(seeds)) != len(seeds) or frames < 1:
        raise ValueError("校准需要非空且不重复的 seed，以及正帧数。")
    if config.get("main_level") != "B" or config.get("pilot_layout") != "prefix":
        raise ValueError("DFE 校准只支持 Level B 前缀 Pilot。")
    if state_split == "heldout_edge":
        if freeze is not None:
            raise ValueError("禁止在留出核验中重新冻结或选择参数。")
        validate_selection(selection, config, seeds, frames)
    elif state_split != "offline_train" or selection is not None:
        raise ValueError("只能在 offline_train 扫网格，或在 heldout_edge 核验冻结参数。")
    memory = int(config["max_delay"])
    grid = calibration_grid(memory)
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
    rows = []
    started = time.monotonic()
    for snr in config["main_snrs"]:
        selected = None if selection is None else DFESetting(**selection["by_snr"][format(float(snr), "g")])
        settings = grid if selected is None else list(dict.fromkeys((REFERENCE, selected)))
        for seed in seeds:
            env = CommunicationEnvironment(build_comm_env_config(
                config, level="B", max_delay=memory, snr_db=float(snr), seed=seed,
                total_pilot=int(config["pilot_total"]), reward_pilot_total=int(config["reward_pilot_total"]),
                pilot_layout="prefix", state_split=state_split,
            ))
            start = env.reset_episode()
            cir, cfo = estimate_acquisition_cir_with_cfo(
                start.acquisition, memory, cfo_limit=float(config.get("acquisition_cfo_limit", 0.004)),
            )
            tails = {setting: start.initial_soft_tail.clone() for setting in settings}
            for index in range(frames):
                frame = env.next_frame()
                view = frame.receiver_view()
                candidates = []
                for setting in settings:
                    phase = estimate_phase_residual_vector(view, cir, tails[setting], cfo_hint=cfo)
                    corrected = apply_phase_correction(view.rx_symbols, float(phase[0]), cfo)
                    estimate, updates = _dfe_rls(
                        corrected, view.adapt_mask, view.adapt_symbols, setting.filter_length,
                        feedback_len=setting.feedback_length, forgetting_factor=setting.forgetting_factor,
                        initial_inverse_covariance=setting.initial_inverse_covariance,
                    )
                    tails[setting] = _tail_from_estimate(estimate, memory)
                    candidates.append((setting, estimate, updates))
                # 全部候选估计和自身状态递推结束后，Data 标签才用于离线评分。
                fingerprint = hashlib.sha256(frame.rx_symbols.numpy().tobytes()).hexdigest()
                for setting, estimate, updates in candidates:
                    ber = float(((estimate.real[frame.data_mask] >= 0) != frame.bits[frame.data_mask]).float().mean())
                    rows.append({"snr_db": snr, "seed": seed, "frame": index, "setting": asdict(setting),
                                 "ber_data": ber, "adapt_updates": updates, "input_sha256": fingerprint,
                                 "data_labels_used_for_receiver": False, "uses_neural_network": False, "uses_rl": False})
            print(json.dumps({"snr_db": snr, "seed": seed, "frames_complete": frames,
                              "settings": len(settings)}, ensure_ascii=False), flush=True)
    summary, selected_by_snr = [], {}
    for snr in config["main_snrs"]:
        settings = grid if selection is None else list(dict.fromkeys((REFERENCE, DFESetting(
            **selection["by_snr"][format(float(snr), "g")]))))
        results = []
        for setting in settings:
            means = [np.mean([row["ber_data"] for row in rows if row["snr_db"] == snr
                              and row["seed"] == seed and row["setting"] == asdict(setting)]) for seed in seeds]
            results.append({"setting": asdict(setting), "ber_data": paired_interval(means)})
        if selection is None:
            best = min(results, key=lambda result: (result["ber_data"]["mean"],
                       result["setting"]["filter_length"] + result["setting"]["feedback_length"],
                       1 - result["setting"]["forgetting_factor"], result["setting"]["initial_inverse_covariance"]))
            selected_by_snr[format(float(snr), "g")] = best["setting"]
        item = {"snr_db": snr, "settings": results}
        if selection is not None:
            # 留出集只计算预先选中的设置与原始参考之间的配对差，不重新选择。
            fixed = selection["by_snr"][format(float(snr), "g")]
            item["selected_vs_reference_ber_improvement"] = paired_interval([
                np.mean([row["ber_data"] for row in rows if row["snr_db"] == snr and row["seed"] == seed
                         and row["setting"] == asdict(REFERENCE)])
                - np.mean([row["ber_data"] for row in rows if row["snr_db"] == snr and row["seed"] == seed
                           and row["setting"] == fixed]) for seed in seeds
            ])
        summary.append(item)
    payload = {"schema_version": "dfe-rls-calibration-v1", "diagnostic_only": True,
               "main_aggregation_allowed": False, "config": config, "config_sha256": signature,
               "state_split": state_split, "seeds": list(seeds), "frames": frames,
               "compensation": "acquisition CFO prior + current Adapt phase median",
               "frozen_selection": selection, "elapsed_seconds": time.monotonic() - started,
               "linear_operator_backend": os.environ.get("RL4EQ_LINEAR_OPERATOR_BACKEND", "direct"),
               "per_snr": summary, "rows": rows}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if freeze is not None:
        frozen = {"schema_version": "dfe-rls-frozen-selection-v1", "config_sha256": signature,
                  "calibration_seeds": list(seeds), "calibration_frames": frames,
                  "confirmation_seeds": CONFIRMATION_SEEDS, "confirmation_frames": CONFIRMATION_FRAMES,
                  "confirmation_split": "heldout_edge", "by_snr": selected_by_snr,
                  "calibration_file_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                  "selection_rule": "每 SNR 最小校准 seed 平均 Data BER；并列优先较短记忆、较小遗忘、较强先验；不使用神经结果"}
        freeze.parent.mkdir(parents=True, exist_ok=True)
        freeze.write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed-start", type=int, default=40000)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--state-split", choices=["offline_train", "heldout_edge"], default="offline_train")
    parser.add_argument("--selection")
    parser.add_argument("--freeze-output")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.num_seeds, args.frames) < 1:
        parser.error("num-seeds 和 frames 必须为正数。")
    run_calibration(json.loads(Path(args.config).read_text(encoding="utf-8")),
                    list(range(args.seed_start, args.seed_start + args.num_seeds)), args.frames, args.output,
                    state_split=args.state_split,
                    selection=json.loads(Path(args.selection).read_text(encoding="utf-8")) if args.selection else None,
                    freeze_output=args.freeze_output)


if __name__ == "__main__":
    main()
