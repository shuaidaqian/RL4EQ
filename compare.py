# -*- coding: utf-8 -*-
"""极端时延信道下神经、RL 与传统均衡方法统一比较。"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from agent.adaptation_controller import OBS_DIM, AdaptationController
from agent.adaptation_policy import ACTION_TABLE, PPOPolicy
from baseline.mmse_equalizer import LMMSEFIREqualizer
from baseline.traditional_equalizers import DFERLSEqualizer
from env.comm_env import CommunicationEnv, EnvConfig, ReceivedFrame
from env.extreme_delay_channel import ExtremeDelayChannelConfig
from env.frame_structure import FrameConfig
from online_train import _configure_plot_font, evaluate_neural_data
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
            "mean_BER_data": mean,
            "std": std,
            "ci95": float(ci95),
            "seed_means": seed_means,
        }
    return summary


def _soft_ber(soft: torch.Tensor, received: ReceivedFrame) -> float:
    predictions = (soft < 0.0).float()
    mask = received.frame.data_mask
    return float((predictions[mask] != received.frame.bits[mask]).float().mean())


def _frame_config(payload: dict[str, object]) -> FrameConfig:
    training = payload["training"]
    pilots = list(training["pilot_lengths"])
    total = 64 if 64 in pilots else int(pilots[0])
    return FrameConfig.from_total_pilot(total, frame_len=int(training["frame_len"]))


def _adapt_and_score(model, received, action, device, base_state) -> float:
    model.restore_peft_state(base_state)
    controller = AdaptationController(model, device=device)
    controller.start_episode()
    controller.adapt_frame(received, action)
    return evaluate_neural_data(model, received, torch.device(device))


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
                local_seed = seed + delay * 10000 + int(round(snr * 10)) * 100 + seed_index
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
                    frame = received.frame
                    base_state = model.capture_peft_state()
                    lmmse_soft = lmmse.equalize(
                        received.rx_symbols,
                        frame.adapt_pilot_symbols,
                        frame.adapt_pilot_mask,
                        delay,
                        snr,
                    )
                    rls_soft = rls.equalize(
                        received.rx_symbols,
                        frame.adapt_pilot_symbols,
                        frame.adapt_pilot_mask,
                        delay,
                        snr,
                    )
                    scores = {
                        "lmmse_fir": _soft_ber(lmmse_soft, received),
                        "dfe_rls": _soft_ber(rls_soft, received),
                    }
                    model.restore_peft_state(base_state)
                    scores["pretrained"] = evaluate_neural_data(model, received, dev)
                    scores["fixed_peft"] = _adapt_and_score(
                        model, received, ACTION_TABLE[3], dev, base_state
                    )
                    model.restore_peft_state(base_state)
                    rule_controller = AdaptationController(model, device=dev)
                    observation = rule_controller.build_observation(received)
                    rule_action = ACTION_TABLE[3] if float(observation[0]) > 0.55 else ACTION_TABLE[0]
                    scores["pilot_rule"] = _adapt_and_score(
                        model, received, rule_action, dev, base_state
                    )
                    model.restore_peft_state(base_state)
                    ppo_controller = AdaptationController(model, device=dev)
                    ppo_observation = ppo_controller.build_observation(received)
                    ppo_index, _, _ = policy.sample_action(ppo_observation, deterministic=True)
                    scores["ppo_peft"] = _adapt_and_score(
                        model, received, ACTION_TABLE[ppo_index], dev, base_state
                    )
                    oracle_scores = [
                        _adapt_and_score(model, received, action, dev, base_state)
                        for action in ACTION_TABLE[:6]
                    ]
                    scores["data_oracle"] = min(oracle_scores)
                    model.restore_peft_state(base_state)
                    for method, ber_data in scores.items():
                        records.append(
                            {
                                "delay": delay,
                                "snr": snr,
                                "seed": seed_index,
                                "frame": frame_index,
                                "method": method,
                                "BER_data": ber_data,
                            }
                        )

    summary = _summarize_records(records)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "comparison_records.json"
    summary_path = out_dir / "comparison_summary.json"
    with open(records_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
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
