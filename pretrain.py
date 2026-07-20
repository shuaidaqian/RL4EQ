# -*- coding: utf-8 -*-
"""离线长记忆基础预训练与 Pilot 条件对齐训练入口。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from agent.neural_equalizer import EqualizerConfig, ExtremeDelayEqualizer
from env.comm_env import CommunicationEnv, EnvConfig, ReceivedFrame
from env.extreme_delay_channel import ExtremeDelayChannelConfig
from env.frame_structure import REGION_DATA, FrameConfig


def _configure_plot_font() -> None:
    try:
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
                break
        else:
            plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def _sample_received_frame(
    rng: np.random.Generator,
    seed: int,
    frame_len: int,
    pilot_lengths: tuple[int, ...],
    delay_min: int,
    delay_max: int,
    snr_min: float,
    snr_max: float,
    conditioned: bool,
) -> ReceivedFrame:
    total_pilot = int(rng.choice(pilot_lengths))
    frame_cfg = FrameConfig.from_total_pilot(total_pilot, frame_len=frame_len)
    max_delay = int(rng.integers(delay_min, delay_max + 1))
    channel_cfg = ExtremeDelayChannelConfig(
        max_delay_symbols=max_delay,
        min_paths=min(3, max_delay + 1),
        max_paths=min(7, max_delay + 1),
        snr_db=float(rng.uniform(snr_min, snr_max)),
        seed=seed,
    )
    env = CommunicationEnv(EnvConfig(frame=frame_cfg, channel=channel_cfg, seed=seed))
    env.reset_episode()
    received = env.next_frame()
    if not conditioned:
        received.frame.region_ids.fill_(REGION_DATA)
        received.frame.adapt_pilot_symbols.zero_()
        received.frame.adapt_pilot_mask.zero_()
        received.frame.reward_pilot_mask.zero_()
        received.frame.data_mask.fill_(True)
    return received


def _batch_loss(
    model: ExtremeDelayEqualizer,
    frames: list[ReceivedFrame],
    device: torch.device,
    conditioned: bool,
) -> tuple[torch.Tensor, float]:
    rx = torch.stack([item.rx_symbols for item in frames]).to(device)
    region_ids = torch.stack([item.frame.region_ids for item in frames]).to(device)
    adapt_symbols = torch.stack([item.frame.adapt_pilot_symbols for item in frames]).to(device)
    adapt_masks = torch.stack([item.frame.adapt_pilot_mask for item in frames]).to(device)
    bits = torch.stack([item.frame.bits for item in frames]).to(device)
    logits, probabilities = model(rx, region_ids, adapt_symbols, adapt_masks)
    if not conditioned:
        loss = F.binary_cross_entropy_with_logits(logits, bits)
    else:
        data_mask = torch.stack([item.frame.data_mask for item in frames]).to(device)
        reward_mask = torch.stack([item.frame.reward_pilot_mask for item in frames]).to(device)
        data_loss = F.binary_cross_entropy_with_logits(logits[data_mask], bits[data_mask])
        adapt_loss = F.binary_cross_entropy_with_logits(logits[adapt_masks], bits[adapt_masks])
        reward_loss = F.binary_cross_entropy_with_logits(logits[reward_mask], bits[reward_mask])
        loss = data_loss + 0.25 * adapt_loss + 0.25 * reward_loss
    ber = float(((probabilities >= 0.5).float() != bits).float().mean().item())
    return loss, ber


def _validate(
    model: ExtremeDelayEqualizer,
    device: torch.device,
    seed: int,
    frame_len: int,
    pilot_lengths: tuple[int, ...],
    delay_min: int,
    delay_max: int,
    validation_frames: int,
) -> float:
    delays = sorted({delay_min, (delay_min + delay_max) // 2, delay_max})
    bers = []
    model.eval()
    with torch.no_grad():
        for delay_index, delay in enumerate(delays):
            for frame_index in range(validation_frames):
                local_seed = seed + 100000 + delay_index * 1000 + frame_index
                total_pilot = pilot_lengths[frame_index % len(pilot_lengths)]
                frame_cfg = FrameConfig.from_total_pilot(total_pilot, frame_len=frame_len)
                channel_cfg = ExtremeDelayChannelConfig(
                    max_delay_symbols=delay,
                    min_paths=min(3, delay + 1),
                    max_paths=min(7, delay + 1),
                    snr_db=10.0,
                    seed=local_seed,
                )
                env = CommunicationEnv(EnvConfig(frame=frame_cfg, channel=channel_cfg, seed=local_seed))
                env.reset_episode()
                received = env.next_frame()
                frame = received.frame
                _, probabilities = model(
                    received.rx_symbols.unsqueeze(0).to(device),
                    frame.region_ids.unsqueeze(0).to(device),
                    frame.adapt_pilot_symbols.unsqueeze(0).to(device),
                    frame.adapt_pilot_mask.unsqueeze(0).to(device),
                )
                predictions = (probabilities[0] >= 0.5).float().cpu()
                bers.append(
                    float((predictions[frame.data_mask] != frame.bits[frame.data_mask]).float().mean())
                )
    model.train()
    return float(np.mean(bers)) if bers else 1.0


def run_offline_pretraining(
    stage_a_steps: int = 4000,
    stage_b_steps: int = 4000,
    batch_size: int = 8,
    seed: int = 42,
    save_dir: str | Path = "pretrained",
    device: str = "cpu",
    frame_len: int = 512,
    pilot_lengths: Iterable[int] = (48, 64, 80),
    delay_min: int = 12,
    delay_max: int = 40,
    snr_min: float = -5.0,
    snr_max: float = 20.0,
    validation_frames: int = 3,
    model_config: EqualizerConfig | None = None,
    learning_rate: float = 3e-4,
    save_plots: bool = True,
) -> dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device)
    pilot_lengths = tuple(int(value) for value in pilot_lengths)
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = model_config or EqualizerConfig(max_len=frame_len)
    if config.max_len < frame_len:
        raise ValueError("模型 max_len 不能小于训练帧长。")
    model = ExtremeDelayEqualizer(config).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    losses: list[float] = []
    bers: list[float] = []
    stages: list[str] = []
    validation_history: list[dict[str, object]] = []
    best_validation_ber = float("inf")
    best_path = out_dir / "model_best.pt"
    final_path = out_dir / "model_final.pt"

    for stage_name, steps, conditioned in (
        ("stage_a", int(stage_a_steps), False),
        ("stage_b", int(stage_b_steps), True),
    ):
        for step in range(steps):
            frames = [
                _sample_received_frame(
                    rng,
                    seed + len(losses) * batch_size + batch_index,
                    frame_len,
                    pilot_lengths,
                    delay_min,
                    delay_max,
                    snr_min,
                    snr_max,
                    conditioned,
                )
                for batch_index in range(batch_size)
            ]
            loss, ber = _batch_loss(model, frames, dev, conditioned)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
            bers.append(ber)
            stages.append(stage_name)
            if steps <= 10 or (step + 1) % max(1, steps // 10) == 0:
                print(f"{stage_name} {step + 1}/{steps}: loss={loss.item():.5f}, BER={ber:.5f}")
        stage_validation_ber = _validate(
            model,
            dev,
            seed + len(validation_history) * 10000,
            frame_len,
            pilot_lengths,
            delay_min,
            delay_max,
            validation_frames,
        )
        validation_history.append(
            {"stage": stage_name, "BER_data": stage_validation_ber}
        )
        if stage_validation_ber < best_validation_ber:
            best_validation_ber = stage_validation_ber
            torch.save(model.state_dict(), best_path)

    if not validation_history:
        best_validation_ber = _validate(
            model,
            dev,
            seed,
            frame_len,
            pilot_lengths,
            delay_min,
            delay_max,
            validation_frames,
        )
        validation_history.append({"stage": "initial", "BER_data": best_validation_ber})
        torch.save(model.state_dict(), best_path)
    torch.save(model.state_dict(), final_path)
    payload = {
        "model": asdict(config),
        "training": {
            "frame_len": frame_len,
            "pilot_lengths": list(pilot_lengths),
            "delay_min": delay_min,
            "delay_max": delay_max,
            "snr_min": snr_min,
            "snr_max": snr_max,
            "stage_a_steps": int(stage_a_steps),
            "stage_b_steps": int(stage_b_steps),
            "seed": seed,
        },
    }
    with open(out_dir / "model_config.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    metrics = {
        "loss": losses,
        "train_BER": bers,
        "stage": stages,
        "validation_history": validation_history,
        "best_validation_BER_data": best_validation_ber,
    }
    with open(out_dir / "pretrain_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    if save_plots and losses:
        try:
            import matplotlib.pyplot as plt

            _configure_plot_font()
            figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
            axes[0].plot(losses)
            axes[0].set_ylabel("训练损失")
            axes[0].grid(alpha=0.3)
            axes[1].plot(bers)
            axes[1].set_ylabel("训练 BER")
            axes[1].set_xlabel("更新步")
            axes[1].grid(alpha=0.3)
            figure.tight_layout()
            figure.savefig(out_dir / "pretrain_curve.png", dpi=160)
            plt.close(figure)
        except Exception as error:
            print(f"跳过预训练曲线绘制: {error}")
    return {
        "model_best": str(best_path),
        "model_final": str(final_path),
        "validation_BER_data": best_validation_ber,
    }


def load_pretrained_equalizer(
    checkpoint: str | Path, device: str | torch.device = "cpu"
) -> tuple[ExtremeDelayEqualizer, dict[str, object]]:
    checkpoint_path = Path(checkpoint)
    config_path = checkpoint_path.parent / "model_config.json"
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    model_payload = dict(payload["model"])
    model_payload["dilations"] = tuple(model_payload["dilations"])
    model = ExtremeDelayEqualizer(EqualizerConfig(**model_payload)).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    return model, payload


def _load_experiment_config(path: str | Path | None) -> dict[str, object]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="极端长时延神经均衡器离线预训练")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--stage_a_steps", type=int, default=None)
    parser.add_argument("--stage_b_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="pretrained")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    config = _load_experiment_config(args.config)
    training = config.get("pretraining", {})
    model_payload = dict(config.get("model", {}))
    if "dilations" in model_payload:
        model_payload["dilations"] = tuple(model_payload["dilations"])
    model_config = EqualizerConfig(**model_payload) if model_payload else None
    run_offline_pretraining(
        stage_a_steps=args.stage_a_steps or int(training.get("stage_a_steps", 4000)),
        stage_b_steps=args.stage_b_steps or int(training.get("stage_b_steps", 4000)),
        batch_size=args.batch_size or int(training.get("batch_size", 8)),
        seed=int(config.get("seed", 42)),
        save_dir=args.save_dir,
        device=args.device,
        frame_len=int(config.get("frame_len", 512)),
        pilot_lengths=tuple(config.get("pilot_lengths", [48, 64, 80])),
        delay_min=int(config.get("delay_train", [12, 40])[0]),
        delay_max=int(config.get("delay_train", [12, 40])[1]),
        snr_min=float(config.get("snr_train", [-5, 20])[0]),
        snr_max=float(config.get("snr_train", [-5, 20])[1]),
        validation_frames=int(training.get("validation_frames", 3)),
        model_config=model_config,
        learning_rate=float(training.get("learning_rate", 3e-4)),
    )


if __name__ == "__main__":
    main()
