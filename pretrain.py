# -*- coding: utf-8 -*-
"""离线阶段：训练与在线 PEFT 完全同构的 AdapterEqualizer。

输出：
  - pretrained/model_best.pt：可被 ``online_train.py --pretrained`` 直接加载
  - pretrained/model_final.pt：最终一步权重
  - pretrained/pretrain_curve.png：离线 BER/loss 曲线

设计原则：
  离线阶段训练完整 AdapterEqualizer；在线阶段冻结主干，只更新 Adapter/输出头。
  这样 ``θ_pre`` 和在线 PEFT 模型结构一致，不再出现旧 24 维模型无法加载的问题。
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.neural_equalizer import AdapterEqualizer, EqualizerConfig
from env.comm_env import CommunicationEnv, EnvConfig
from env.frame_structure import FrameConfig

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _configure_matplotlib_fonts() -> None:
    if not HAS_MPL:
        return
    for font_path in [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]:
        if os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)
            font_name = font_manager.FontProperties(fname=font_path).get_name()
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            return


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(np.mean(vals)) if vals else 0.0


def _sample_channel_config(rng: np.random.Generator, seed: int, snr_min: float, snr_max: float) -> dict:
    """采样离线信道族。

    当前仓库已有 Rayleigh 和 3GPP EPA/EVA/ETU；这里先用这几类构成大规模随机信道。
    后续加入 Rician/CFO 时，只需要扩展这个采样器和 env 层。
    """
    snr_db = float(rng.uniform(snr_min, snr_max))
    kind = rng.choice(["rayleigh", "epa", "eva", "etu"], p=[0.55, 0.15, 0.15, 0.15])
    if kind == "rayleigh":
        num_taps = int(rng.integers(8, 19))
        return dict(
            num_taps=num_taps,
            delay_spread=max(1, num_taps - 1),
            snr_db=snr_db,
            time_varying=bool(rng.random() < 0.25),
            doppler_hz=float(rng.choice([1.0, 5.0, 30.0, 70.0])),
            symbol_rate=1e6,
            seed=seed,
        )
    return dict(
        profile=str(kind),
        snr_db=snr_db,
        time_varying=bool(rng.random() < 0.25),
        doppler_hz=float(rng.choice([5.0, 30.0, 70.0])),
        symbol_rate=1e6,
        seed=seed,
    )


def _make_env(seed: int, channel: dict) -> CommunicationEnv:
    env = CommunicationEnv(EnvConfig(
        frame=FrameConfig(),
        channel=channel,
        window_K=10,
        seed=seed,
    ))
    env.reset()
    return env


def _evaluate_model(model: AdapterEqualizer, device: torch.device, seed: int, frames: int, snr_db: float = 10.0) -> Dict[str, float]:
    model.eval()
    frame_cfg = FrameConfig()
    pilot_mask = torch.tensor([frame_cfg.bit_type(t) == "pilot" for t in range(frame_cfg.frame_len)], dtype=torch.bool, device=device)
    data_mask = torch.tensor([frame_cfg.bit_type(t) == "data" for t in range(frame_cfg.frame_len)], dtype=torch.bool, device=device)
    records: Dict[str, List[float]] = {"rayleigh": [], "epa": [], "eva": [], "etu": []}

    with torch.no_grad():
        for idx, profile in enumerate(records):
            for frame_idx in range(frames):
                cur_seed = seed + idx * 1000 + frame_idx
                if profile == "rayleigh":
                    channel = dict(num_taps=16, delay_spread=10, snr_db=snr_db, seed=cur_seed)
                else:
                    channel = dict(profile=profile, snr_db=snr_db, seed=cur_seed)
                env = _make_env(cur_seed, channel)
                states = env.get_all_states().unsqueeze(0).to(device)
                bits = env.get_true_bits().to(device)
                _, probs = model(states)
                preds = (probs[0] > 0.5).float()
                ber_data = (preds[data_mask] != bits[data_mask]).float().mean().item()
                ber_pilot = (preds[pilot_mask] != bits[pilot_mask]).float().mean().item()
                records[profile].append((ber_data + ber_pilot) / 2.0)

    out = {name: _mean(vals) for name, vals in records.items()}
    out["mean"] = _mean(out.values())
    return out


def _save_curve(losses: List[float], train_bers: List[float], val_bers: List[float], save_dir: Path) -> None:
    if not HAS_MPL or not losses:
        return
    _configure_matplotlib_fonts()
    save_dir.mkdir(parents=True, exist_ok=True)
    xs = np.arange(1, len(losses) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(xs, losses, color="#1565C0", lw=1.5)
    axes[0].set_ylabel("BCE loss")
    axes[0].set_title("AdapterEqualizer 离线预训练")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xs, train_bers, color="#2E7D32", lw=1.5, label="train BER")
    if val_bers:
        val_x = np.linspace(1, len(losses), num=len(val_bers))
        axes[1].plot(val_x, val_bers, color="#C62828", marker="o", lw=1.5, label="val BER")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("BER")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "pretrain_curve.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_offline_pretraining(
    num_steps: int = 1000,
    batch_size: int = 4,
    seed: int = 42,
    d_model: int = 64,
    n_layers: int = 2,
    adapter_rank: int = 8,
    lr: float = 5e-4,
    snr_min: float = 0.0,
    snr_max: float = 20.0,
    val_interval: int = 100,
    val_frames: int = 3,
    save_dir: str | os.PathLike = "pretrained",
    device: str = "cpu",
    save_plots: bool = True,
) -> Dict[str, object]:
    """运行离线预训练，并保存在线阶段可加载的 AdapterEqualizer 权重。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device)
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = EqualizerConfig(
        state_dim=45,
        d_model=d_model,
        n_heads=4,
        n_layers=n_layers,
        dim_feedforward=d_model * 2,
        adapter_rank=adapter_rank,
    )
    model = AdapterEqualizer(cfg).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(num_steps, 1), eta_min=1e-6)

    config_payload = {
        "state_dim": cfg.state_dim,
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "n_layers": cfg.n_layers,
        "dim_feedforward": cfg.dim_feedforward,
        "adapter_rank": cfg.adapter_rank,
        "max_len": cfg.max_len,
    }
    with open(out_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, ensure_ascii=False, indent=2)

    losses: List[float] = []
    train_bers: List[float] = []
    val_bers: List[float] = []
    best_val = float("inf")
    best_path = out_dir / "model_best.pt"
    final_path = out_dir / "model_final.pt"
    start = time.time()

    print("离线预训练 AdapterEqualizer")
    print(f"参数量: {sum(p.numel() for p in model.parameters())}")
    print(f"输出: {best_path}")

    for step in range(1, num_steps + 1):
        model.train()
        optimizer.zero_grad()
        batch_loss = 0.0
        batch_ber = 0.0

        for batch_idx in range(batch_size):
            cur_seed = seed * 100000 + step * 100 + batch_idx
            channel = _sample_channel_config(rng, cur_seed, snr_min, snr_max)
            env = _make_env(cur_seed, channel)
            states = env.get_all_states().unsqueeze(0).to(dev)
            bits = env.get_true_bits().unsqueeze(0).to(dev)

            logits, probs = model(states)
            loss = F.binary_cross_entropy_with_logits(logits, bits)
            (loss / batch_size).backward()
            batch_loss += float(loss.item())
            preds = (probs > 0.5).float()
            batch_ber += float((preds != bits).float().mean().item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(batch_loss / batch_size)
        train_bers.append(batch_ber / batch_size)

        should_validate = step == 1 or step % val_interval == 0 or step == num_steps
        if should_validate:
            val = _evaluate_model(model, dev, seed + 900000 + step, frames=val_frames)
            val_bers.append(float(val["mean"]))
            elapsed = time.time() - start
            print(
                f"  Step {step:5d}/{num_steps} | loss={losses[-1]:.4f} "
                f"train_BER={train_bers[-1]:.4f} val_BER={val['mean']:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s"
            )
            if val["mean"] < best_val:
                best_val = float(val["mean"])
                torch.save(model.state_dict(), best_path)

    torch.save(model.state_dict(), final_path)
    if not best_path.exists():
        torch.save(model.state_dict(), best_path)
        best_val = train_bers[-1] if train_bers else 1.0

    if save_plots:
        _save_curve(losses, train_bers, val_bers, out_dir)

    return {
        "best_ber": best_val,
        "best_checkpoint": str(best_path),
        "final_checkpoint": str(final_path),
        "config_path": str(out_dir / "model_config.json"),
        "num_steps": num_steps,
        "trainable_offline_params": sum(p.numel() for p in model.parameters()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_dir", type=str, default="pretrained")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--adapter_rank", type=int, default=8)
    parser.add_argument("--snr_min", type=float, default=0.0)
    parser.add_argument("--snr_max", type=float, default=20.0)
    parser.add_argument("--val_interval", type=int, default=100)
    parser.add_argument("--val_frames", type=int, default=3)
    args = parser.parse_args()

    result = run_offline_pretraining(
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        d_model=args.d_model,
        n_layers=args.n_layers,
        adapter_rank=args.adapter_rank,
        lr=args.lr,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        val_interval=args.val_interval,
        val_frames=args.val_frames,
        save_dir=args.save_dir,
        device=args.device,
    )
    print("\n=== 离线预训练完成 ===")
    print(f"最佳验证 BER: {result['best_ber']:.5f}")
    print(f"最佳权重: {result['best_checkpoint']}")
    print(f"最终权重: {result['final_checkpoint']}")


if __name__ == "__main__":
    main()
