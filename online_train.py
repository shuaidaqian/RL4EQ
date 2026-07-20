# -*- coding: utf-8 -*-
"""PPO 控制参数高效在线微调入口。"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from agent.adaptation_controller import OBS_DIM, AdaptationController
from agent.adaptation_policy import ACTION_TABLE, PPOPolicy
from env.comm_env import CommunicationEnv, EnvConfig, ReceivedFrame
from env.extreme_delay_channel import ExtremeDelayChannelConfig
from env.frame_structure import FrameConfig
from pretrain import load_pretrained_equalizer


REQUIRED_METRICS = {
    "BER_data",
    "BER_adapt_pilot",
    "BER_reward_pilot",
    "pilot_loss",
    "adapt_params",
    "adapt_steps",
    "latency_ms",
    "parameter_delta_norm",
    "generalization",
}


def _configure_plot_font() -> None:
    """注册本机中文字体，避免生成图表时出现缺字警告。"""
    import os

    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for path in (
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ):
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False


def evaluate_neural_data(model, received: ReceivedFrame, device: torch.device) -> float:
    frame = received.frame
    model.eval()
    with torch.no_grad():
        _, probabilities = model(
            received.rx_symbols.unsqueeze(0).to(device),
            frame.region_ids.unsqueeze(0).to(device),
            frame.adapt_pilot_symbols.unsqueeze(0).to(device),
            frame.adapt_pilot_mask.unsqueeze(0).to(device),
        )
    predictions = (probabilities[0].cpu() >= 0.5).float()
    return float((predictions[frame.data_mask] != frame.bits[frame.data_mask]).float().mean())


def _mean(records: list[dict[str, object]], key: str) -> float:
    values = [float(record[key]) for record in records]
    return float(np.mean(values)) if values else 0.0


def _discounted_returns(rewards: list[float], gamma: float = 0.99) -> torch.Tensor:
    running = 0.0
    output = []
    for reward in reversed(rewards):
        running = float(reward) + gamma * running
        output.append(running)
    return torch.tensor(list(reversed(output)), dtype=torch.float32)


def _default_frame_from_payload(payload: dict[str, object]) -> FrameConfig:
    training = payload["training"]
    pilot_lengths = list(training["pilot_lengths"])
    total_pilot = 64 if 64 in pilot_lengths else int(pilot_lengths[0])
    return FrameConfig.from_total_pilot(total_pilot, frame_len=int(training["frame_len"]))


def _evaluate_generalization(model, payload, seed: int, device: torch.device) -> float:
    frame_cfg = _default_frame_from_payload(payload)
    channel_cfg = ExtremeDelayChannelConfig(
        max_delay_symbols=50,
        min_paths=8,
        max_paths=10,
        snr_db=10.0,
        seed=seed + 900000,
    )
    env = CommunicationEnv(EnvConfig(frame=frame_cfg, channel=channel_cfg, seed=seed + 900000))
    env.reset_episode()
    return evaluate_neural_data(model, env.next_frame(), device)


def run_online_training(
    pretrained: str | Path,
    output_dir: str | Path = "logs/online",
    num_episodes: int = 50,
    frames_per_episode: int = 100,
    seed: int = 42,
    device: str = "cpu",
    policy_learning_rate: float = 3e-4,
    save_plots: bool = True,
) -> dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    model, payload = load_pretrained_equalizer(pretrained, device=dev)
    frame_cfg = _default_frame_from_payload(payload)
    delay_max = int(payload["training"]["delay_max"])
    policy = PPOPolicy(OBS_DIM, len(ACTION_TABLE)).to(dev)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=policy_learning_rate)
    controller = AdaptationController(model, device=dev)
    records: list[dict[str, object]] = []

    for episode in range(int(num_episodes)):
        channel_cfg = ExtremeDelayChannelConfig(
            max_delay_symbols=delay_max,
            min_paths=min(3, delay_max + 1),
            max_paths=min(7, delay_max + 1),
            snr_db=10.0,
            seed=seed + episode,
        )
        env = CommunicationEnv(EnvConfig(frame=frame_cfg, channel=channel_cfg, seed=seed + episode))
        env.reset_episode()
        controller.start_episode()
        observations = []
        actions = []
        old_log_probs = []
        values = []
        rewards = []
        for frame_index in range(int(frames_per_episode)):
            received = env.next_frame()
            observation = controller.build_observation(received)
            action_index, log_prob, value = policy.sample_action(observation, deterministic=False)
            result = controller.adapt_frame(received, ACTION_TABLE[action_index])
            ber_data = evaluate_neural_data(model, received, dev)
            records.append(
                {
                    "episode": episode,
                    "frame": frame_index,
                    "action": result.action_name,
                    "reward": result.reward,
                    "BER_data": ber_data,
                    "BER_adapt_pilot": result.ber_adapt_pilot,
                    "BER_reward_pilot": result.ber_reward_pilot,
                    "pilot_loss": result.adapt_pilot_loss,
                    "adapt_params": result.adapt_params,
                    "adapt_steps": result.adapt_steps,
                    "latency_ms": result.latency_ms,
                    "parameter_delta_norm": result.parameter_delta_norm,
                }
            )
            observations.append(observation.detach().cpu())
            actions.append(action_index)
            old_log_probs.append(log_prob.detach().cpu())
            values.append(value.detach().cpu())
            rewards.append(result.reward)

        returns = _discounted_returns(rewards)
        value_tensor = torch.stack(values).float()
        advantages = returns - value_tensor
        policy.ppo_update(
            policy_optimizer,
            torch.stack(observations).to(dev),
            torch.tensor(actions, device=dev),
            torch.stack(old_log_probs).to(dev),
            returns.to(dev),
            advantages.to(dev),
        )
        controller.end_episode()
        print(
            f"episode {episode + 1}/{num_episodes}: "
            f"BER_data={_mean(records[-frames_per_episode:], 'BER_data'):.5f}, "
            f"reward={_mean(records[-frames_per_episode:], 'reward'):.5f}"
        )

    generalization = _evaluate_generalization(model, payload, seed, dev)
    summary = {
        "BER_data": _mean(records, "BER_data"),
        "BER_adapt_pilot": _mean(records, "BER_adapt_pilot"),
        "BER_reward_pilot": _mean(records, "BER_reward_pilot"),
        "pilot_loss": _mean(records, "pilot_loss"),
        "adapt_params": _mean(records, "adapt_params"),
        "adapt_steps": _mean(records, "adapt_steps"),
        "latency_ms": _mean(records, "latency_ms"),
        "parameter_delta_norm": _mean(records, "parameter_delta_norm"),
        "generalization": generalization,
    }
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "online_records.json"
    summary_path = out_dir / "online_summary.json"
    policy_path = out_dir / "policy.pt"
    with open(records_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    torch.save(policy.state_dict(), policy_path)

    if save_plots and records:
        try:
            import matplotlib.pyplot as plt

            _configure_plot_font()
            figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            axes[0].plot([record["BER_data"] for record in records])
            axes[0].set_ylabel("BER_data")
            axes[1].plot([record["pilot_loss"] for record in records])
            axes[1].set_ylabel("Pilot loss")
            axes[2].plot([record["adapt_steps"] for record in records])
            axes[2].set_ylabel("适配步数")
            axes[2].set_xlabel("在线帧")
            for axis in axes:
                axis.grid(alpha=0.3)
            figure.tight_layout()
            figure.savefig(out_dir / "online_metrics.png", dpi=160)
            plt.close(figure)
        except Exception as error:
            print(f"跳过在线曲线绘制: {error}")
    return {
        "summary": summary,
        "records_path": str(records_path),
        "summary_path": str(summary_path),
        "policy_path": str(policy_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="极端长时延信道在线 PPO 参数高效适配")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="logs/online")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--frames_per_episode", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    seed = 42
    if args.config:
        with open(args.config, "r", encoding="utf-8") as handle:
            seed = int(json.load(handle).get("seed", 42))
    run_online_training(
        pretrained=args.pretrained,
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        frames_per_episode=args.frames_per_episode,
        seed=seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
