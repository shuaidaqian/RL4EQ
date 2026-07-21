# -*- coding: utf-8 -*-
"""极端时延信道下神经、RL 与传统均衡方法统一比较。"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from agent.adaptation_controller import OBS_DIM, AdaptationController
from agent.adaptation_policy import ACTION_TABLE, PPOPolicy
from baseline.mmse_equalizer import LMMSEFIREqualizer
from baseline.traditional_equalizers import DFERLSEqualizer
from env.comm_env import CommunicationEnv, EnvConfig, ReceivedFrame
from env.extreme_delay_channel import ExtremeDelayChannelConfig
from env.frame_structure import FrameConfig
from online_train import REQUIRED_METRICS, _configure_plot_font, evaluate_neural_data
from pretrain import load_pretrained_equalizer


METHODS = (
    "lmmse_fir",
    "dfe_rls",
    "pretrained",
    "fixed_peft",
    "pilot_rule",
    "ppo_peft",
    "data_oracle",
)

FRAME_METRICS = tuple(sorted(REQUIRED_METRICS - {"generalization"}))


def _comparison_seed(seed: int, delay: int, seed_index: int) -> int:
    """为同一时延和重复编号生成跨 SNR 复用的配对随机种子。"""
    return int(seed) + int(delay) * 10000 + int(seed_index)


def _summarize_records(
    records: list[dict[str, object]], methods: tuple[str, ...] = METHODS
) -> dict[str, dict[str, object]]:
    """先按随机种子聚合，再跨种子计算统计量。"""
    summary: dict[str, dict[str, object]] = {}
    for method in methods:
        selected = [item for item in records if item["method"] == method]
        seed_ids = sorted({int(item["seed"]) for item in selected})
        seed_means = [
            float(
                np.mean(
                    [
                        float(item["BER_data"])
                        for item in selected
                        if int(item["seed"]) == seed_id
                    ]
                )
            )
            for seed_id in seed_ids
        ]
        mean = float(np.mean(seed_means)) if seed_means else 0.0
        std = float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0
        ci95 = 1.96 * std / np.sqrt(len(seed_means)) if len(seed_means) > 1 else 0.0
        summary[method] = {
            metric: float(
                np.mean([float(item[metric]) for item in selected if metric in item])
            )
            if any(metric in item for item in selected)
            else 0.0
            for metric in FRAME_METRICS
        }
        summary[method].update({
            "mean_BER_data": mean,
            "std": std,
            "ci95": float(ci95),
            "seed_means": seed_means,
            "generalization": 0.0,
        })
    return summary


def _traditional_metrics(
    soft: torch.Tensor, received: ReceivedFrame, latency_ms: float
) -> dict[str, float | int]:
    """把传统均衡器输出转换为与在线控制器一致的帧级指标。"""
    frame = received.frame
    predictions = (soft < 0.0).float()
    logits = -soft.float()
    bits = frame.bits.float()

    def ber(mask: torch.Tensor) -> float:
        return float((predictions[mask] != bits[mask]).float().mean())

    return {
        "BER_data": ber(frame.data_mask),
        "BER_adapt_pilot": ber(frame.adapt_pilot_mask),
        "BER_reward_pilot": ber(frame.reward_pilot_mask),
        "pilot_loss": float(
            F.binary_cross_entropy_with_logits(
                logits[frame.adapt_pilot_mask], bits[frame.adapt_pilot_mask]
            ).item()
        ),
        "adapt_params": 0,
        "adapt_steps": 0,
        "latency_ms": float(latency_ms),
        "parameter_delta_norm": 0.0,
    }


def _score_without_adaptation(
    model, received: ReceivedFrame, device: torch.device
) -> dict[str, float | int]:
    """用一次前向计算无适配神经均衡器的完整指标。"""
    frame = received.frame
    start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        logits, probabilities = model(
            received.rx_symbols.unsqueeze(0).to(device),
            frame.region_ids.unsqueeze(0).to(device),
            frame.adapt_pilot_symbols.unsqueeze(0).to(device),
            frame.adapt_pilot_mask.unsqueeze(0).to(device),
        )
    latency_ms = (time.perf_counter() - start) * 1000.0
    logits = logits[0].cpu()
    probabilities = probabilities[0].cpu()
    predictions = (probabilities >= 0.5).float()

    def ber(mask: torch.Tensor) -> float:
        return float((predictions[mask] != frame.bits[mask]).float().mean())

    return {
        "BER_data": ber(frame.data_mask),
        "BER_adapt_pilot": ber(frame.adapt_pilot_mask),
        "BER_reward_pilot": ber(frame.reward_pilot_mask),
        "pilot_loss": float(
            F.binary_cross_entropy_with_logits(
                logits[frame.adapt_pilot_mask], frame.bits[frame.adapt_pilot_mask]
            ).item()
        ),
        "adapt_params": 0,
        "adapt_steps": 0,
        "latency_ms": float(latency_ms),
        "parameter_delta_norm": 0.0,
    }


def _frame_config(payload: dict[str, object]) -> FrameConfig:
    training = payload["training"]
    pilots = list(training["pilot_lengths"])
    total = 64 if 64 in pilots else int(pilots[0])
    return FrameConfig.from_total_pilot(total, frame_len=int(training["frame_len"]))


def _adapt_and_score(model, received, action, device, base_state) -> dict[str, float | int]:
    model.restore_peft_state(base_state)
    controller = AdaptationController(model, device=device)
    controller.start_episode()
    result = controller.adapt_frame(received, action)
    return {
        "BER_data": evaluate_neural_data(model, received, torch.device(device)),
        "BER_adapt_pilot": result.ber_adapt_pilot,
        "BER_reward_pilot": result.ber_reward_pilot,
        "pilot_loss": result.adapt_pilot_loss,
        "adapt_params": result.adapt_params,
        "adapt_steps": result.adapt_steps,
        "latency_ms": result.latency_ms,
        "parameter_delta_norm": result.parameter_delta_norm,
    }


def _evaluate_methods(
    model,
    policy: PPOPolicy,
    lmmse: LMMSEFIREqualizer,
    rls: DFERLSEqualizer,
    received: ReceivedFrame,
    delay: int,
    snr: float,
    device: torch.device,
) -> dict[str, dict[str, float | int]]:
    """在同一接收帧和同一 PEFT 初始状态上评估全部方法。"""
    frame = received.frame
    base_state = model.capture_peft_state()

    start = time.perf_counter()
    lmmse_soft = lmmse.equalize(
        received.rx_symbols,
        frame.adapt_pilot_symbols,
        frame.adapt_pilot_mask,
        delay,
        snr,
    )
    lmmse_metrics = _traditional_metrics(
        lmmse_soft, received, (time.perf_counter() - start) * 1000.0
    )

    start = time.perf_counter()
    rls_soft = rls.equalize(
        received.rx_symbols,
        frame.adapt_pilot_symbols,
        frame.adapt_pilot_mask,
        delay,
        snr,
    )
    rls_metrics = _traditional_metrics(
        rls_soft, received, (time.perf_counter() - start) * 1000.0
    )

    scores = {
        "lmmse_fir": lmmse_metrics,
        "dfe_rls": rls_metrics,
        "pretrained": _score_without_adaptation(model, received, device),
        "fixed_peft": _adapt_and_score(
            model, received, ACTION_TABLE[3], device, base_state
        ),
    }

    model.restore_peft_state(base_state)
    rule_controller = AdaptationController(model, device=device)
    observation = rule_controller.build_observation(received)
    rule_action = ACTION_TABLE[3] if float(observation[0]) > 0.55 else ACTION_TABLE[0]
    scores["pilot_rule"] = _adapt_and_score(
        model, received, rule_action, device, base_state
    )

    model.restore_peft_state(base_state)
    ppo_controller = AdaptationController(model, device=device)
    ppo_observation = ppo_controller.build_observation(received)
    ppo_index, _, _ = policy.sample_action(ppo_observation, deterministic=True)
    scores["ppo_peft"] = _adapt_and_score(
        model, received, ACTION_TABLE[ppo_index], device, base_state
    )

    oracle_candidates = [
        _adapt_and_score(model, received, action, device, base_state)
        for action in ACTION_TABLE[:6]
    ]
    oracle = dict(min(oracle_candidates, key=lambda item: float(item["BER_data"])))
    oracle["latency_ms"] = float(
        sum(float(item["latency_ms"]) for item in oracle_candidates)
    )
    scores["data_oracle"] = oracle
    model.restore_peft_state(base_state)
    return scores


def run_comparison(
    pretrained: str | Path,
    output_dir: str | Path = "logs/compare",
    delays: tuple[int, ...] = (20, 30, 40),
    snrs: tuple[float, ...] = (-5, 0, 5, 10, 15, 20),
    num_seeds: int = 5,
    num_frames: int = 200,
    seed: int = 42,
    device: str = "cpu",
    policy_path: str | Path | None = None,
    save_plots: bool = True,
) -> dict[str, object]:
    dev = torch.device(device)
    model, payload = load_pretrained_equalizer(pretrained, device=dev)
    frame_cfg = _frame_config(payload)
    policy = PPOPolicy(OBS_DIM, len(ACTION_TABLE)).to(dev)
    policy_loaded = False
    if policy_path and Path(policy_path).exists():
        policy.load_state_dict(torch.load(policy_path, map_location=dev, weights_only=True), strict=True)
        policy_loaded = True
    policy.eval()
    lmmse = LMMSEFIREqualizer()
    rls = DFERLSEqualizer()
    records: list[dict[str, object]] = []

    for delay in tuple(int(value) for value in delays):
        for snr in tuple(float(value) for value in snrs):
            for seed_index in range(int(num_seeds)):
                local_seed = _comparison_seed(seed, delay, seed_index)
                channel_cfg = ExtremeDelayChannelConfig(
                    max_delay_symbols=delay,
                    min_paths=min(3, delay + 1),
                    max_paths=min(7, delay + 1),
                    snr_db=snr,
                    seed=local_seed,
                )
                env = CommunicationEnv(EnvConfig(frame=frame_cfg, channel=channel_cfg, seed=local_seed))
                env.reset_episode()
                for frame_index in range(int(num_frames)):
                    received = env.next_frame()
                    scores = _evaluate_methods(
                        model, policy, lmmse, rls, received, delay, snr, dev
                    )
                    for method, metrics in scores.items():
                        records.append(
                            {
                                "delay": delay,
                                "snr": snr,
                                "seed": seed_index,
                                "frame": frame_index,
                                "method": method,
                                **metrics,
                            }
                        )
                print(
                    f"完成 delay={delay}, SNR={snr:g}, "
                    f"seed={seed_index + 1}/{num_seeds}",
                    flush=True,
                )

    generalization_records: list[dict[str, object]] = []
    generalization_delay = 50
    generalization_snr = 10.0
    for seed_index in range(int(num_seeds)):
        local_seed = _comparison_seed(seed + 900000, generalization_delay, seed_index)
        channel_cfg = ExtremeDelayChannelConfig(
            max_delay_symbols=generalization_delay,
            min_paths=8,
            max_paths=10,
            snr_db=generalization_snr,
            seed=local_seed,
        )
        env = CommunicationEnv(EnvConfig(frame=frame_cfg, channel=channel_cfg, seed=local_seed))
        env.reset_episode()
        for frame_index in range(int(num_frames)):
            received = env.next_frame()
            scores = _evaluate_methods(
                model,
                policy,
                lmmse,
                rls,
                received,
                generalization_delay,
                generalization_snr,
                dev,
            )
            for method, metrics in scores.items():
                generalization_records.append(
                    {
                        "delay": generalization_delay,
                        "snr": generalization_snr,
                        "seed": seed_index,
                        "frame": frame_index,
                        "method": method,
                        **metrics,
                    }
                )
        print(
            f"完成外推泛化 delay=50, paths=8-10, "
            f"seed={seed_index + 1}/{num_seeds}",
            flush=True,
        )

    summary = _summarize_records(records)
    generalization_summary = _summarize_records(generalization_records)
    for method in METHODS:
        summary[method]["generalization"] = generalization_summary[method]["BER_data"]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "comparison_records.json"
    generalization_records_path = out_dir / "generalization_records.json"
    summary_path = out_dir / "comparison_summary.json"
    with open(records_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    with open(generalization_records_path, "w", encoding="utf-8") as handle:
        json.dump(generalization_records, handle, ensure_ascii=False, indent=2)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {"methods": list(METHODS), "policy_loaded": policy_loaded, "summary": summary},
            handle,
            ensure_ascii=False,
            indent=2,
        )

    if save_plots and records:
        try:
            import matplotlib.pyplot as plt

            _configure_plot_font()
            figure, axis = plt.subplots(figsize=(10, 6))
            first_delay = int(tuple(delays)[0])
            for method in METHODS:
                means = []
                for snr in tuple(float(value) for value in snrs):
                    selected = [
                        item["BER_data"]
                        for item in records
                        if item["method"] == method
                        and item["delay"] == first_delay
                        and item["snr"] == snr
                    ]
                    means.append(float(np.mean(selected)))
                axis.semilogy(tuple(snrs), np.maximum(means, 1e-5), marker="o", label=method)
            axis.set_xlabel("SNR (dB)")
            axis.set_ylabel("BER_data")
            axis.set_title(f"最大时延 {first_delay} 符号")
            axis.grid(alpha=0.3)
            axis.legend()
            figure.tight_layout()
            figure.savefig(out_dir / "ber_snr_comparison.png", dpi=160)
            plt.close(figure)
        except Exception as error:
            print(f"跳过比较曲线绘制: {error}")
    return {
        "methods": list(METHODS),
        "records_path": str(records_path),
        "generalization_records_path": str(generalization_records_path),
        "summary_path": str(summary_path),
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="极端长时延均衡方法统一比较")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--policy", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="logs/compare")
    parser.add_argument("--delays", type=int, nargs="+", default=[20, 30, 40])
    parser.add_argument("--snrs", type=float, nargs="+", default=[-5, 0, 5, 10, 15, 20])
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--num_frames", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    seed = 42
    if args.config:
        with open(args.config, "r", encoding="utf-8") as handle:
            seed = int(json.load(handle).get("seed", 42))
    run_comparison(
        pretrained=args.pretrained,
        policy_path=args.policy,
        output_dir=args.output_dir,
        delays=tuple(args.delays),
        snrs=tuple(args.snrs),
        num_seeds=args.num_seeds,
        num_frames=args.num_frames,
        seed=seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
