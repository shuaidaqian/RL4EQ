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
from env.comm_env import MMSE_FEATURE_DIM, CommunicationEnv, EnvConfig
from env.frame_structure import FrameConfig, frame_config_for_known_ratio

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


def _sample_channel_config(
    rng: np.random.Generator,
    seed: int,
    snr_min: float,
    snr_max: float,
    max_num_taps: int | None = None,
    impairment_prob: float = 0.0,
) -> dict:
    """采样离线信道族。

    当前仓库已有 Rayleigh 和 3GPP EPA/EVA/ETU；这里先用这几类构成大规模随机信道。
    后续加入 Rician/CFO 时，只需要扩展这个采样器和 env 层。
    """
    snr_db = float(rng.uniform(snr_min, snr_max))
    kind = rng.choice(["rayleigh", "rician", "epa", "eva", "etu"], p=[0.65, 0.25, 0.0334, 0.0333, 0.0333])
    rayleigh_high = max(7, int(max_num_taps or 24) + 1)
    rician_high = max(5, min(int(max_num_taps or 20), 20) + 1)
    def with_impairments(channel: dict) -> dict:
        if rng.random() >= impairment_prob:
            return channel
        impairments = {}
        mode = str(rng.choice(["cfo", "nonlinear", "cfo_nonlinear"]))
        if "cfo" in mode:
            impairments["cfo_norm"] = float(rng.uniform(-0.08, 0.08))
        if "nonlinear" in mode:
            impairments["nonlinear_alpha"] = float(rng.uniform(0.15, 0.45))
        channel["impairments"] = impairments
        if rng.random() < 0.5:
            channel["time_varying"] = True
            channel["doppler_hz"] = float(rng.choice([70.0, 150.0, 300.0]))
        return channel

    if kind == "rayleigh":
        num_taps = int(rng.integers(6, rayleigh_high))
        return with_impairments(dict(
            type="rayleigh",
            num_taps=num_taps,
            delay_spread=max(1, num_taps - 1),
            snr_db=snr_db,
            time_varying=bool(rng.random() < 0.25),
            doppler_hz=float(rng.choice([1.0, 5.0, 30.0, 70.0])),
            symbol_rate=1e6,
            seed=seed,
        ))
    if kind == "rician":
        num_taps = int(rng.integers(4, rician_high))
        return with_impairments(dict(
            type="rician",
            num_taps=num_taps,
            delay_spread=max(1, num_taps - 1),
            k_factor_db=float(rng.uniform(0.0, 12.0)),
            snr_db=snr_db,
            time_varying=bool(rng.random() < 0.25),
            doppler_hz=float(rng.choice([1.0, 5.0, 30.0, 70.0])),
            symbol_rate=1e6,
            seed=seed,
        ))
    return with_impairments(dict(
        profile=str(kind),
        snr_db=snr_db,
        time_varying=bool(rng.random() < 0.25),
        doppler_hz=float(rng.choice([5.0, 30.0, 70.0])),
        symbol_rate=1e6,
        seed=seed,
    ))


def _make_env(
    seed: int,
    channel: dict,
    window_K: int = 10,
    use_mmse_features: bool = False,
    frame_config: FrameConfig | None = None,
) -> CommunicationEnv:
    impairments = dict(channel.pop("impairments", {}))
    env = CommunicationEnv(EnvConfig(
        frame=frame_config or FrameConfig(),
        channel=channel,
        window_K=window_K,
        use_mmse_features=use_mmse_features,
        impairments=impairments,
        seed=seed,
    ))
    env.reset()
    return env


def _eval_channel_config(profile: str, seed: int, snr_db: float, window_K: int = 10) -> dict:
    rng = np.random.default_rng(seed)
    if profile == "rayleigh":
        num_taps = int(rng.integers(6, max(7, window_K + 2)))
        return dict(
            type="rayleigh",
            num_taps=num_taps,
            delay_spread=max(1, num_taps - 1),
            snr_db=snr_db,
            seed=seed,
        )
    if profile == "rician":
        num_taps = int(rng.integers(4, max(5, min(window_K + 1, 20) + 1)))
        return dict(
            type="rician",
            num_taps=num_taps,
            delay_spread=max(1, num_taps - 1),
            k_factor_db=float(rng.uniform(0.0, 12.0)),
            snr_db=snr_db,
            seed=seed,
        )
    return dict(profile=profile, snr_db=snr_db, seed=seed)


def _evaluate_model(
    model: AdapterEqualizer,
    device: torch.device,
    seed: int,
    frames: int,
    snr_db: float = 10.0,
    window_K: int = 10,
    use_mmse_features: bool = False,
) -> Dict[str, float]:
    model.eval()
    frame_cfg = FrameConfig()
    pilot_mask = torch.tensor([frame_cfg.bit_type(t) == "pilot" for t in range(frame_cfg.frame_len)], dtype=torch.bool, device=device)
    data_mask = torch.tensor([frame_cfg.bit_type(t) == "data" for t in range(frame_cfg.frame_len)], dtype=torch.bool, device=device)
    records: Dict[str, List[float]] = {"rayleigh": [], "rician": [], "epa": [], "eva": [], "etu": []}
    data_records: Dict[str, List[float]] = {"rayleigh": [], "rician": [], "epa": [], "eva": [], "etu": []}
    pilot_records: Dict[str, List[float]] = {"rayleigh": [], "rician": [], "epa": [], "eva": [], "etu": []}

    with torch.no_grad():
        for idx, profile in enumerate(records):
            for frame_idx in range(frames):
                cur_seed = seed + idx * 1000 + frame_idx
                channel = _eval_channel_config(profile, cur_seed, snr_db, window_K=window_K)
                env = _make_env(cur_seed, channel, window_K=window_K, use_mmse_features=use_mmse_features)
                states = env.get_all_states().unsqueeze(0).to(device)
                bits = env.get_true_bits().to(device)
                _, probs = model(states)
                preds = (probs[0] > 0.5).float()
                ber_data = (preds[data_mask] != bits[data_mask]).float().mean().item()
                ber_pilot = (preds[pilot_mask] != bits[pilot_mask]).float().mean().item()
                records[profile].append(ber_data)
                data_records[profile].append(ber_data)
                pilot_records[profile].append(ber_pilot)

    out = {name: _mean(vals) for name, vals in records.items()}
    out["mean"] = _mean(out.values())
    out["BER_data"] = _mean(_mean(vals) for vals in data_records.values())
    out["BER_pilot"] = _mean(_mean(vals) for vals in pilot_records.values())
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
    window_K: int = 10,
    use_channel_encoder: bool = False,
    channel_dim: int = 32,
    use_sync_head: bool = False,
    sync_dim: int = 32,
    sync_delay_bins: int = 9,
    use_mmse_features: bool = False,
    lr: float = 5e-4,
    snr_min: float = 0.0,
    snr_max: float = 20.0,
    val_interval: int = 100,
    val_frames: int = 3,
    data_loss_weight: float = 4.0,
    known_loss_weight: float = 0.5,
    init_checkpoint: str | None = None,
    train_known_ratios: Iterable[float] | None = None,
    impairment_prob: float = 0.0,
    selective_distill_weight: float = 0.0,
    use_cfo_head: bool = False,
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
    known_ratios = [float(x) for x in train_known_ratios] if train_known_ratios else [0.5]

    cfg = EqualizerConfig(
        state_dim=2 * (2 * window_K + 1) + (MMSE_FEATURE_DIM if use_mmse_features else 0) + 3,
        d_model=d_model,
        n_heads=4,
        n_layers=n_layers,
        dim_feedforward=d_model * 2,
        adapter_rank=adapter_rank,
        use_channel_encoder=use_channel_encoder,
        channel_dim=channel_dim,
        use_sync_head=use_sync_head,
        sync_dim=sync_dim,
        sync_delay_bins=sync_delay_bins,
        use_mmse_residual=use_mmse_features,
        mmse_feature_dim=MMSE_FEATURE_DIM if use_mmse_features else 0,
        use_cfo_head=use_cfo_head,
    )
    model = AdapterEqualizer(cfg).to(dev)
    if init_checkpoint:
        state = torch.load(init_checkpoint, map_location=dev, weights_only=True)
        current = model.state_dict()
        input_key = "input_proj.0.weight"
        if (
            input_key in state
            and input_key in current
            and state[input_key].shape[0] == current[input_key].shape[0]
            and state[input_key].shape[1] + MMSE_FEATURE_DIM == current[input_key].shape[1]
        ):
            expanded = current[input_key].clone()
            old_width = state[input_key].shape[1]
            raw_width = old_width - 3
            expanded[:, :raw_width] = state[input_key][:, :raw_width]
            expanded[:, raw_width:raw_width + MMSE_FEATURE_DIM] = 0.0
            expanded[:, -3:] = state[input_key][:, -3:]
            state[input_key] = expanded
        compatible = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
        skipped = sorted(k for k, v in state.items() if k in current and current[k].shape != v.shape)
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        print(
            f"加载初始化权重: {init_checkpoint} | "
            f"loaded={len(compatible)} missing={len(missing)} unexpected={len(unexpected)} skipped_shape={len(skipped)}"
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(num_steps, 1), eta_min=1e-6)

    config_payload = {
        "state_dim": cfg.state_dim,
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "n_layers": cfg.n_layers,
        "dim_feedforward": cfg.dim_feedforward,
        "adapter_rank": cfg.adapter_rank,
        "window_K": window_K,
        "max_len": cfg.max_len,
        "use_channel_encoder": cfg.use_channel_encoder,
        "channel_dim": cfg.channel_dim,
        "use_sync_head": cfg.use_sync_head,
        "sync_dim": cfg.sync_dim,
        "sync_delay_bins": cfg.sync_delay_bins,
        "use_mmse_features": use_mmse_features,
        "use_mmse_residual": cfg.use_mmse_residual,
        "mmse_feature_dim": cfg.mmse_feature_dim,
        "train_known_ratios": known_ratios,
        "impairment_prob": float(impairment_prob),
        "selective_distill_weight": float(selective_distill_weight),
        "use_cfo_head": bool(use_cfo_head),
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
            known_ratio = float(rng.choice(known_ratios))
            frame_cfg = frame_config_for_known_ratio(known_ratio)
            channel = _sample_channel_config(
                rng,
                cur_seed,
                snr_min,
                snr_max,
                max_num_taps=window_K + 1,
                impairment_prob=impairment_prob,
            )
            env = _make_env(
                cur_seed,
                channel,
                window_K=window_K,
                use_mmse_features=use_mmse_features,
                frame_config=frame_cfg,
            )
            states = env.get_all_states().unsqueeze(0).to(dev)
            bits = env.get_true_bits().unsqueeze(0).to(dev)

            logits, probs = model(states)
            data_mask = torch.tensor(
                [env.frame_cfg.bit_type(t) == "data" for t in range(env.frame_cfg.frame_len)],
                dtype=torch.bool,
                device=dev,
            ).unsqueeze(0)
            known_mask = ~data_mask
            data_loss = F.binary_cross_entropy_with_logits(logits[data_mask], bits[data_mask])
            known_loss = F.binary_cross_entropy_with_logits(logits[known_mask], bits[known_mask])
            loss = data_loss_weight * data_loss + known_loss_weight * known_loss
            if use_mmse_features and model.last_residual_correction is not None:
                loss = loss + 0.02 * model.last_residual_correction.square().mean()
                if selective_distill_weight > 0.0:
                    mmse_soft = states[..., -(MMSE_FEATURE_DIM + 3)].detach()
                    mmse_logits = -5.0 * mmse_soft
                    mmse_bits = (mmse_soft < 0.0).float()
                    correct_mask = (mmse_bits == bits).detach()
                    if correct_mask.any():
                        distill_loss = F.binary_cross_entropy_with_logits(
                            logits[correct_mask],
                            torch.sigmoid(mmse_logits[correct_mask]),
                        )
                        residual_loss = model.last_residual_correction[correct_mask].square().mean()
                        loss = loss + selective_distill_weight * (distill_loss + 0.2 * residual_loss)
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
            val = _evaluate_model(
                model,
                dev,
                seed + 900000 + step,
                frames=val_frames,
                window_K=window_K,
                use_mmse_features=use_mmse_features,
            )
            val_bers.append(float(val["mean"]))
            elapsed = time.time() - start
            print(
                f"  Step {step:5d}/{num_steps} | loss={losses[-1]:.4f} "
                f"train_BER={train_bers[-1]:.4f} "
                f"val_data={val['BER_data']:.4f} val_pilot={val['BER_pilot']:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s"
            )
            if val["BER_data"] < best_val:
                best_val = float(val["BER_data"])
                torch.save(model.state_dict(), best_path)

    torch.save(model.state_dict(), final_path)
    if not best_path.exists():
        torch.save(model.state_dict(), best_path)
        best_val = train_bers[-1] if train_bers else 1.0

    if save_plots:
        _save_curve(losses, train_bers, val_bers, out_dir)

    return {
        "best_ber": best_val,
        "best_data_ber": best_val,
        "best_checkpoint": str(best_path),
        "final_checkpoint": str(final_path),
        "config_path": str(out_dir / "model_config.json"),
        "num_steps": num_steps,
        "trainable_offline_params": sum(p.numel() for p in model.parameters()),
        "data_loss_weight": data_loss_weight,
        "known_loss_weight": known_loss_weight,
        "window_K": window_K,
        "use_channel_encoder": use_channel_encoder,
        "channel_dim": channel_dim,
        "use_sync_head": use_sync_head,
        "sync_dim": sync_dim,
        "sync_delay_bins": sync_delay_bins,
        "use_mmse_features": use_mmse_features,
        "init_checkpoint": init_checkpoint,
        "train_known_ratios": known_ratios,
        "impairment_prob": float(impairment_prob),
        "selective_distill_weight": float(selective_distill_weight),
        "use_cfo_head": bool(use_cfo_head),
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
    parser.add_argument("--window_K", type=int, default=10)
    parser.add_argument("--use_channel_encoder", action="store_true")
    parser.add_argument("--channel_dim", type=int, default=32)
    parser.add_argument("--use_sync_head", action="store_true")
    parser.add_argument("--sync_dim", type=int, default=32)
    parser.add_argument("--sync_delay_bins", type=int, default=9)
    parser.add_argument("--use_mmse_features", action="store_true")
    parser.add_argument("--snr_min", type=float, default=0.0)
    parser.add_argument("--snr_max", type=float, default=20.0)
    parser.add_argument("--val_interval", type=int, default=100)
    parser.add_argument("--val_frames", type=int, default=3)
    parser.add_argument("--data_loss_weight", type=float, default=4.0)
    parser.add_argument("--known_loss_weight", type=float, default=0.5)
    parser.add_argument("--init_checkpoint", type=str, default=None)
    parser.add_argument("--train_known_ratios", type=str, default="0.5")
    parser.add_argument("--impairment_prob", type=float, default=0.0)
    parser.add_argument("--selective_distill_weight", type=float, default=0.0)
    parser.add_argument("--use_cfo_head", action="store_true")
    args = parser.parse_args()
    train_known_ratios = [float(x.strip()) for x in args.train_known_ratios.split(",") if x.strip()]

    result = run_offline_pretraining(
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        d_model=args.d_model,
        n_layers=args.n_layers,
        adapter_rank=args.adapter_rank,
        window_K=args.window_K,
        use_channel_encoder=args.use_channel_encoder,
        channel_dim=args.channel_dim,
        use_sync_head=args.use_sync_head,
        sync_dim=args.sync_dim,
        sync_delay_bins=args.sync_delay_bins,
        use_mmse_features=args.use_mmse_features,
        lr=args.lr,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        val_interval=args.val_interval,
        val_frames=args.val_frames,
        data_loss_weight=args.data_loss_weight,
        known_loss_weight=args.known_loss_weight,
        init_checkpoint=args.init_checkpoint,
        train_known_ratios=train_known_ratios,
        impairment_prob=args.impairment_prob,
        selective_distill_weight=args.selective_distill_weight,
        use_cfo_head=args.use_cfo_head,
        save_dir=args.save_dir,
        device=args.device,
    )
    print("\n=== 离线预训练完成 ===")
    print(f"最佳验证 BER: {result['best_ber']:.5f}")
    print(f"最佳权重: {result['best_checkpoint']}")
    print(f"最终权重: {result['final_checkpoint']}")


if __name__ == "__main__":
    main()
