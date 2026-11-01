# -*- coding: utf-8 -*-
"""固定信道的传统线性接收机校准诊断；不读取神经结果选择配置。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.linear_mmse import linear_block_lmmse
from baseline.synchronization_compensation import apply_phase_correction
from baseline.traditional_equalizers import (
    _dfe_rls, _lmmse_fir, _sc_fde_mmse, _tail_from_estimate,
    estimate_acquisition_cir_with_cfo, estimate_phase_residual_vector,
)
from env.comm_env import CommunicationEnvironment
from env.experiment_config import build_comm_env_config
from env.linear_operator import LinearChannelOperator
from evaluation.pilot_transfer import paired_interval


def padded_fde(rx, cir, tail, variance):
    """零填充频域 Wiener 近似；不把它误称为精确有限观测 MMSE 解。"""
    memory = cir.numel() - 1
    operator = LinearChannelOperator(rx.numel(), memory)
    target = rx - operator.forward(torch.zeros_like(rx), cir, tail)
    size = 1 << (rx.numel() + memory - 1).bit_length()
    h = torch.fft.fft(cir, n=size)
    estimate = torch.fft.ifft(torch.fft.fft(target, n=size) * h.conj() / (h.abs().square() + variance))
    return estimate[:rx.numel()]


def run_calibration(config, seeds, frames, output, *, state_split="offline_train", selection=None):
    """所有接收机只用可见输入；每帧更新结束后才读取 Data 标签。"""
    target = Path(output)
    if target.exists():
        raise ValueError("校准文件已存在，请指定新输出路径。")
    if state_split not in {"offline_train", "heldout_edge"}:
        raise ValueError("校准/独立核验只允许预先定义的信道划分。")
    config_sha256 = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
    if state_split == "heldout_edge" and selection is None:
        raise ValueError("独立核验必须传入在训练范围上冻结的 --selection，禁止在留出集重新扫网格。")
    if selection is not None and selection.get("config_sha256") != config_sha256:
        raise ValueError("冻结的传统设置与当前信道配置不匹配。")
    if state_split == "heldout_edge" and set(seeds).intersection(selection.get("calibration_seeds", [])):
        raise ValueError("独立核验 seed 不得与校准 seed 重叠。")
    memory = int(config["max_delay"])
    grid = [("fir", length, ridge) for length in sorted({41, memory + 1, 2 * memory + 1})
                for ridge in (1, 16, 224)]
    grid += [("dfe", 41, 5), ("dfe", memory + 1, memory),
                 ("circular_fde", 0, 0), ("padded_fde", 0, 0),
                 ("block_complex", 0, 0), ("block_real", 0, 0)]
    rows = []
    for snr in config["main_snrs"]:
        settings = grid if selection is None else [tuple(item) for item in selection["by_snr"][str(snr)]]
        if not settings or any(item not in grid for item in settings):
            raise ValueError("冻结设置必须来自预先声明的传统校准网格。")
        variance = torch.tensor(10 ** (-float(snr) / 10), dtype=torch.float32)
        for seed in seeds:
            env = CommunicationEnvironment(build_comm_env_config(
                config, level="B", max_delay=memory, snr_db=float(snr), seed=seed,
                total_pilot=int(config["pilot_total"]),
                reward_pilot_total=int(config["reward_pilot_total"]),
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
                    kind, length, regularizer = setting
                    tail = tails[setting]
                    # 与神经接收机相同的 acquisition CFO 先验及当前 Adapt 相位信息。
                    phase = estimate_phase_residual_vector(view, cir, tail, cfo_hint=cfo)
                    corrected = apply_phase_correction(view.rx_symbols, float(phase[0]), cfo)
                    extra = {}
                    if kind == "fir":
                        estimate, _ = _lmmse_fir(corrected, view.adapt_mask, view.adapt_symbols,
                                                length, variance * regularizer)
                    elif kind == "dfe":
                        estimate, _ = _dfe_rls(corrected, view.adapt_mask, view.adapt_symbols,
                                              length, feedback_len=regularizer)
                    elif kind == "circular_fde":
                        estimate, _ = _sc_fde_mmse(corrected, cir, variance)
                    elif kind == "padded_fde":
                        estimate = padded_fde(corrected, cir, tail, variance)
                    else:
                        estimate, extra = linear_block_lmmse(
                            corrected, cir, tail, variance, known_symbols=view.adapt_symbols,
                            known_mask=view.adapt_mask, real_bpsk=kind == "block_real",
                        )
                    tails[setting] = _tail_from_estimate(estimate, memory)
                    candidates.append((setting, estimate.detach(), extra))
                # 本帧全部接收机的估计和状态递推已经结束；Data 仅用于离线校准评分。
                for (kind, length, regularizer), estimate, extra in candidates:
                    mask = frame.data_mask
                    ber = float(((estimate.real[mask] >= 0) != frame.bits[mask]).float().mean())
                    rows.append({"kind": kind, "filter_length": length, "regularizer_or_feedback": regularizer,
                                 "snr_db": snr, "seed": seed, "frame": index, "ber_data": ber,
                                 "data_labels_used_for_receiver": False, **extra})
            print(json.dumps({"snr_db": snr, "seed": seed, "frames_complete": frames}), flush=True)
    summary = []
    for snr in config["main_snrs"]:
        settings = grid if selection is None else [tuple(item) for item in selection["by_snr"][str(snr)]]
        for kind, length, regularizer in settings:
            subset = [r for r in rows if (r["snr_db"], r["kind"], r["filter_length"], r["regularizer_or_feedback"])
                      == (snr, kind, length, regularizer)]
            means = [np.mean([r["ber_data"] for r in subset if r["seed"] == seed]) for seed in seeds]
            summary.append({"snr_db": snr, "kind": kind, "filter_length": length,
                            "regularizer_or_feedback": regularizer, "ber_data": paired_interval(means),
                            "cg_not_converged_frames": sum(not r.get("converged", True) for r in subset)})
    payload = {"diagnostic_only": True, "main_aggregation_allowed": False,
               "purpose": "传统接收机参数校准或固定设置的独立核验，不包含神经结果",
               "state_split": state_split, "config": config, "seeds": seeds, "frames": frames,
               "config_sha256": config_sha256, "frozen_selection": selection,
               "linear_operator_backend": os.environ.get("RL4EQ_LINEAR_OPERATOR_BACKEND", "direct"),
               "compensation": "acquisition CFO prior + current Adapt phase median",
               "per_snr": summary, "rows": rows}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed-start", type=int, default=36000)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--state-split", choices=["offline_train", "heldout_edge"], default="offline_train")
    parser.add_argument("--selection", help="训练范围上预先冻结的设置 JSON；heldout_edge 必填。")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.frames < 1 or args.num_seeds < 1:
        parser.error("frames 和 num-seeds 必须为正数。")
    run_calibration(json.loads(Path(args.config).read_text(encoding="utf-8")),
                    list(range(args.seed_start, args.seed_start + args.num_seeds)),
                    args.frames, args.output, state_split=args.state_split,
                    selection=json.loads(Path(args.selection).read_text(encoding="utf-8")) if args.selection else None)


if __name__ == "__main__":
    main()
