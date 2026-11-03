# -*- coding: utf-8 -*-
"""合成正规方程证明与真实历史 Adapt 坐标诊断；不更新正式接收机。"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.joint_pilot_state import (
    PilotBlock, extract_adapt_blocks, joint_posterior, prediction_loss,
    recursive_posterior, transport_block,
)


def synthetic_mechanisms(seed=110069):
    """生成真值仅供离线代数核验；这些理想坐标不传入任何在线方法。"""
    rng = torch.Generator().manual_seed(seed)
    taps, rows, count = 117, 108, 8
    truth = torch.randn(taps, dtype=torch.complex128, generator=rng) / taps ** .5
    prior = truth + .15 * torch.randn(taps, dtype=torch.complex128, generator=rng)
    covariance = torch.eye(taps, dtype=torch.complex128) * .0225
    blocks, rotated, oracle = [], [], []
    for index in range(count):
        symbols = (2 * torch.randint(0, 2, (rows + taps - 1,), generator=rng) - 1).to(torch.complex128)
        positions = torch.arange(taps - 1, symbols.numel())
        design = symbols[positions[:, None] - torch.arange(taps)[None, :]]
        target = design @ truth + .1 ** .5 * torch.randn(rows, dtype=torch.complex128, generator=rng)
        block = PilotBlock(design, target, .1, index)
        blocks.append(block)
        # 已知生成相位只用于证明坐标错误可破坏聚合，不能用作实际相位估计。
        phase = .5 * index + 2 * torch.pi * .0007 * positions
        raw = PilotBlock(design, target * torch.exp(1j * phase), .1, index)
        rotated.append(raw)
        oracle.append(PilotBlock(design, raw.target * torch.exp(-1j * phase), .1, index))
    joint, uncertainty = joint_posterior(prior, covariance, blocks)
    recursive, recursive_cov = recursive_posterior(prior, covariance, blocks)
    repeated, _ = joint_posterior(prior, covariance, blocks, prior_copies=count)
    dephased, _ = joint_posterior(prior, covariance, oracle)
    uncorrected, _ = joint_posterior(prior, covariance, rotated)
    transported, _ = joint_posterior(prior, covariance,
        [transport_block(block, torch.linspace(.1, .8, rows)) for block in blocks])
    # 同时旋转 CIR 和反向旋转公共相位，物理预测不变：必须指定 gauge。
    alpha = .7
    gauge_error = max(float((block.design @ truth -
        torch.exp(torch.tensor(-1j * alpha)) * (block.design @ (truth * torch.exp(torch.tensor(1j * alpha))))).abs().max())
        for block in blocks)
    checks = {
        "batch_recursive_mean_max_error": float((joint - recursive).abs().max()),
        "batch_recursive_cov_max_error": float((uncertainty - recursive_cov).abs().max()),
        "both_sides_coordinate_transport_max_error": float((joint - transported).abs().max()),
        "synthetic_true_phase_removal_max_error": float((joint - dephased).abs().max()),
        "gauge_prediction_max_error": gauge_error,
    }
    if max(checks.values()) > 1e-6:
        raise AssertionError(f"合成坐标或信息递推不一致：{checks}")
    return {"seed": seed, "taps": taps, "rows_per_frame": rows, "frames": count,
        "complex_joint_rank": int(torch.linalg.matrix_rank(torch.cat([b.design for b in blocks]))),
        **checks,
        "once_prior_cir_mse": float((joint - truth).abs().square().mean()),
        "repeated_prior_cir_mse": float((repeated - truth).abs().square().mean()),
        "uncorrected_phase_cir_mse": float((uncorrected - truth).abs().square().mean()),
        "repeated_vs_once_solution_norm": float((repeated - joint).norm()),
        "true_coordinates_used_only_for_offline_algebra": True,
        "online_control_executed": False}


def actual_coordinates(config, checkpoint, seeds, snrs, age, frames, window, output):
    from evaluation.pilot_transfer import _load_branch, paired_interval
    from training.pilot_transfer import frame_to_device, make_context, make_episode

    if window < 2 or age < window - 1 or frames < 1:
        raise ValueError("历史窗口至少两帧，物理历史必须足够且评分帧数为正。")
    model, payload = _load_branch(Path(checkpoint), config, "cpu")
    training = payload.get("metrics", {})
    if set(seeds).intersection(item["seed"] for item in training.get("history", [])):
        raise ValueError("诊断 seed 与离线训练 seed 重叠。")
    if int(training.get("receiver_lookahead_len", 0)) != int(config.get("receiver_lookahead_len", 0)):
        raise ValueError("训练和诊断接收窗口不一致。")
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    records = []
    raw = output / "frame_metrics.jsonl"
    with raw.open("x", encoding="utf-8") as handle:
        for snr in snrs:
            for seed in seeds:
                _, cir, cfo, tail, episode = make_episode(config, model, seed=seed, snr=snr,
                    frames=age + frames, state_split="heldout_edge", device="cpu")
                start = age - window + 1
                contexts = [make_context(frame_to_device(frame, "cpu"), cir, cfo, tail, snr)
                            for frame in episode[start:]]
                extracted = [extract_adapt_blocks(model, context) for context in contexts]
                for local in range(frames):
                    end = window + local
                    current = contexts[end-1]
                    prior = current.condition.complex_cir.flatten().to(torch.complex128)
                    noise = current.condition.noise_variance.flatten()[0]
                    variance = model.channel_state.prior_variance(prior, noise, current.frame.frame_index)
                    covariance = torch.diag(variance.to(prior))
                    scales = torch.stack([item[1] for item in extracted[end-window:end]])
                    scale_deviation = float(((scales - scales[-1]) / scales[-1]).abs().max())
                    for coordinate in ("raw", "pilot_phase", "neural_phase"):
                        blocks = [item[0][coordinate] for item in extracted[end-window:end]]
                        joint, cov = joint_posterior(prior, covariance, blocks)
                        recursive, recursive_cov = recursive_posterior(prior, covariance, blocks)
                        repeated, _ = joint_posterior(prior, covariance, blocks, prior_copies=window)
                        last, _ = joint_posterior(prior, covariance, blocks[-1:])
                        previous, _ = joint_posterior(prior, covariance, blocks[-2:-1])
                        historical, _ = joint_posterior(prior, covariance, blocks[:-1])
                        # 当前帧完全留出：历史状态只拟合前面的 Adapt，不读取当前标签。
                        row = {"seed": seed, "snr_db": snr, "frame": local,
                            "channel_frame_index": current.frame.frame_index, "coordinate": coordinate,
                            "heldout_current_loss_previous1": prediction_loss(blocks[-1], previous),
                            "heldout_current_loss_previous_history": prediction_loss(blocks[-1], historical),
                            "prior_current_loss": prediction_loss(blocks[-1], prior),
                            "in_sample_current_loss_single": prediction_loss(blocks[-1], last),
                            "in_sample_current_loss_joint": prediction_loss(blocks[-1], joint),
                            "in_sample_current_loss_repeated_prior": prediction_loss(blocks[-1], repeated),
                            "batch_recursive_mean_max_error": float((joint-recursive).abs().max()),
                            "batch_recursive_cov_max_error": float((cov-recursive_cov).abs().max()),
                            "repeated_vs_once_solution_norm": float((repeated-joint).norm()),
                            "neural_memory_scale_max_relative_change": scale_deviation,
                            "adapt_rows_per_frame": blocks[-1].design.shape[0],
                            "window": window, "data_labels_used": False, "reward_labels_used": False,
                            "generation_truth_used": False, "persistent_update": False,
                            "rx_sha256": hashlib.sha256(current.frame.rx_symbols.numpy().tobytes()).hexdigest()}
                        if max(row["batch_recursive_mean_max_error"], row["batch_recursive_cov_max_error"]) > 1e-8:
                            raise AssertionError("真实 Pilot 的共同状态批量与递推方程不一致。")
                        records.append(row)
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                print(json.dumps({"snr_db": snr, "seed": seed, "frames_complete": frames}), flush=True)
    if any(not torch.equal(value, initial[name]) for name, value in model.state_dict().items()):
        raise AssertionError("诊断不得改变模型参数。")
    summaries = []
    for snr in snrs:
        for coordinate in ("raw", "pilot_phase", "neural_phase"):
            rows = [r for r in records if r["snr_db"] == snr and r["coordinate"] == coordinate]
            values = [sum(r["heldout_current_loss_previous1"] - r["heldout_current_loss_previous_history"]
                          for r in rows if r["seed"] == seed) / frames for seed in seeds]
            keys = [key for key in rows[0] if "loss" in key or "max_error" in key or "scale_max" in key]
            summaries.append({"snr_db": snr, "coordinate": coordinate,
                **{key: sum(r[key] for r in rows) / len(rows) for key in keys},
                "heldout_history_gain": paired_interval(values)})
    return {"seeds": seeds, "snrs": snrs, "age": age, "frames": frames, "window": window,
        "records": len(records), "per_snr_coordinate": summaries, "model_unchanged": True,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(), "config": config,
        "prior_protocol": "每个对比统一当前帧龄的acquisition对角先验，静态共同CIR假设，仅诊断",
        "heldout_protocol": "前1帧或前window-1帧Adapt拟合，当前帧Adapt完全留出",
        "interpretation": "不是Data BER，不是神经/RL在线结果；相位未校正仅作坐标反例"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed-start", type=int, default=110070)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--snrs", nargs="+", type=float, default=[0, 5, 10, 15])
    parser.add_argument("--age", type=int, default=1023)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if bool(args.config) != bool(args.checkpoint):
        parser.error("真实信道诊断需同时提供配置和checkpoint。")
    torch.set_num_threads(1)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists() or (output / "frame_metrics.jsonl").exists():
        raise ValueError("诊断输出已存在，禁止覆盖。")
    started = time.perf_counter()
    result = {"schema_version": "joint-pilot-state-diagnostic-v1", "diagnostic_only": True,
        "main_aggregation_allowed": False, "synthetic": synthetic_mechanisms(),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
            ("evaluation/joint_pilot_state.py", "scripts/diagnose_joint_pilot_state.py")}}
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        result["actual"] = actual_coordinates(config, args.checkpoint,
            list(range(args.seed_start, args.seed_start + args.num_seeds)), args.snrs,
            args.age, args.frames, args.window, output)
    result["wall_seconds"] = time.perf_counter() - started
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"wall_seconds": result["wall_seconds"], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
