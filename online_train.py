# -*- coding: utf-8 -*-
"""第二阶段起：参数高效在线自适应训练与 MMSE 对比。

指标：
  - BER_data：数据段 BER，主指标
  - BER_pilot：导频段 BER，在线可观测指标
  - pilot_loss：PPO reward 来源
  - adapt_params：在线更新参数量
  - adapt_steps：每帧更新步数
  - latency_ms：每帧在线更新时间
  - generalization：未见信道类型上的 BER
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.adaptation_controller import AdaptationController, make_strategy_table
from agent.adaptation_policy import DiscretePPOPolicy
from agent.neural_equalizer import AdapterEqualizer, EqualizerConfig
from baseline.mmse_equalizer import MMSEEqualizer
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


def _mean(items: Iterable[float]) -> float:
    values = list(items)
    return float(np.mean(values)) if values else 0.0


def _capture_model_state(model: AdapterEqualizer) -> Dict[str, torch.Tensor]:
    """捕获 θ_pre 快照，用于每帧在线临时适应前恢复模型。"""
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _restore_model_state(model: AdapterEqualizer, state: Dict[str, torch.Tensor]) -> None:
    """恢复模型到 θ_pre，避免上一帧 PEFT 更新污染下一帧。"""
    model.load_state_dict(state, strict=True)


def _channel_cfg(profile: str | None, snr: float, seed: int, max_num_taps: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    if profile == "rician":
        num_taps = int(rng.integers(4, max(5, min(int(max_num_taps or 20), 20) + 1)))
        return dict(
            type="rician",
            num_taps=num_taps,
            delay_spread=max(1, num_taps - 1),
            k_factor_db=float(rng.uniform(0.0, 12.0)),
            snr_db=snr,
            seed=seed,
        )
    if profile:
        return dict(profile=profile, snr_db=snr, seed=seed)
    num_taps = int(rng.integers(6, max(7, int(max_num_taps or 24) + 1)))
    return dict(
        type="rayleigh",
        num_taps=num_taps,
        delay_spread=max(1, num_taps - 1),
        snr_db=snr,
        seed=seed,
    )


def _make_env(seed: int, snr: float, profile: str | None = None, window_K: int = 10) -> CommunicationEnv:
    env = CommunicationEnv(EnvConfig(
        frame=FrameConfig(),
        channel=_channel_cfg(profile, snr, seed, max_num_taps=window_K + 1),
        window_K=window_K,
        seed=seed,
    ))
    env.reset()
    return env


def _mask(frame_cfg: FrameConfig, kind: str) -> torch.Tensor:
    return torch.tensor([frame_cfg.bit_type(t) == kind for t in range(frame_cfg.frame_len)], dtype=torch.bool)


def evaluate_mmse(env: CommunicationEnv) -> Dict[str, float]:
    """用同一帧训练序列估计 MMSE，并分别统计 pilot/data BER。"""
    rx = env.get_rx_symbols()
    bits = env.get_true_bits()
    frame_cfg = env.frame_cfg
    num_taps = getattr(env.channel, "num_taps", getattr(env.channel, "nt", 16))
    mmse = MMSEEqualizer(num_taps=num_taps)
    soft, _ = mmse(rx, bits[:frame_cfg.train_len], rx[:frame_cfg.train_len], env.channel.snr_db)
    preds = (soft < 0).float()
    pilot_mask = _mask(frame_cfg, "pilot")
    data_mask = _mask(frame_cfg, "data")
    return {
        "BER_pilot": float((preds[pilot_mask] != bits[pilot_mask]).float().mean().item()),
        "BER_data": float((preds[data_mask] != bits[data_mask]).float().mean().item()),
    }


def _load_pretrained_if_available(model: AdapterEqualizer, pretrained: str | None, device: torch.device) -> str:
    if not pretrained:
        return "未加载预训练权重"
    path = Path(pretrained)
    if path.is_dir():
        path = path / "model_best.pt"
    if not path.exists():
        return f"预训练权重不存在，使用随机初始化: {path}"
    state = torch.load(path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return f"已加载预训练: {path} | missing={len(missing)} unexpected={len(unexpected)}"


def _summarize(records: List[dict]) -> Dict[str, float]:
    return {
        "BER_data": _mean(r["BER_data"] for r in records),
        "BER_pilot": _mean(r["BER_pilot"] for r in records),
        "pilot_loss": _mean(r.get("pilot_loss", 0.0) for r in records),
        "adapt_params": _mean(r.get("adapt_params", 0.0) for r in records),
        "adapt_steps": _mean(r.get("adapt_steps", 0.0) for r in records),
        "latency_ms": _mean(r.get("latency_ms", 0.0) for r in records),
    }


def _evaluate_generalization(model: AdapterEqualizer, device: torch.device, snr: float, seed: int) -> Dict[str, float]:
    """在未见 3GPP profile 上只评估，不在线更新。"""
    out = {}
    model.eval()
    for idx, profile in enumerate(["rician", "epa", "eva", "etu"]):
        env = _make_env(seed + 1000 + idx, snr, profile=profile, window_K=(model.config.state_dim - 5) // 4)
        controller = AdaptationController(model, env.frame_cfg, device)
        states = env.get_all_states().unsqueeze(0).to(device)
        bits = env.get_true_bits().to(device)
        with torch.no_grad():
            _, probs = model(states)
            preds = (probs[0] > 0.5).float()
            ber = (preds[controller.data_mask] != bits[controller.data_mask]).float().mean().item()
        out[profile] = float(ber)
    out["mean"] = _mean(out.values())
    return out


def _save_plots(peft_records: List[dict], mmse_records: List[dict], generalization: Dict[str, float], output_dir: Path) -> None:
    if not HAS_MPL:
        return
    _configure_matplotlib_fonts()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = np.arange(1, len(peft_records) + 1)
    eps = 1e-5

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.reshape(-1)
    axes[0].plot(frames, [max(r["BER_data"], eps) for r in peft_records], label="PEFT 数据段", color="#1565C0")
    axes[0].plot(frames, [max(r["BER_data"], eps) for r in mmse_records], label="MMSE 数据段", color="#C62828", alpha=0.8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("BER_data")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(frames, [max(r["BER_pilot"], eps) for r in peft_records], label="PEFT 导频", color="#00897B")
    axes[1].plot(frames, [max(r["BER_pilot"], eps) for r in mmse_records], label="MMSE 导频", color="#F57C00", alpha=0.8)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("BER_pilot")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(frames, [r["pilot_loss"] for r in peft_records], color="#6A1B9A")
    axes[2].set_ylabel("pilot_loss")
    axes[2].set_xlabel("Frame")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(frames, [r["latency_ms"] for r in peft_records], label="latency_ms", color="#455A64")
    axes[3].bar(frames, [r["adapt_steps"] for r in peft_records], alpha=0.25, label="adapt_steps", color="#7CB342")
    axes[3].set_xlabel("Frame")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    fig.suptitle("参数高效在线自适应 vs MMSE")
    fig.tight_layout()
    fig.savefig(output_dir / "online_adapt_metrics.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    labels = ["PEFT", "MMSE"]
    values = [max(_mean(r["BER_data"] for r in peft_records), eps), max(_mean(r["BER_data"] for r in mmse_records), eps)]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, values, color=["#1565C0", "#C62828"])
    ax.set_yscale("log")
    ax.set_ylabel("平均 BER_data")
    ax.set_title("数据段 BER 主指标对比")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "mmse_baseline_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    keys = [k for k in generalization if k != "mean"]
    ax.bar([k.upper() for k in keys], [max(generalization[k], eps) for k in keys], color="#2E7D32")
    ax.set_yscale("log")
    ax.set_ylabel("BER_data")
    ax.set_title("未见信道类型泛化")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "generalization.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_online_adaptation(
    num_frames: int = 300,
    seed: int = 42,
    snr: float = 10.0,
    d_model: int = 64,
    n_layers: int = 2,
    adapter_rank: int = 8,
    window_K: int = 10,
    use_channel_encoder: bool = False,
    channel_dim: int = 32,
    use_sync_head: bool = False,
    sync_dim: int = 32,
    sync_delay_bins: int = 9,
    device: str = "cpu",
    output_dir: str | os.PathLike = "logs",
    save_plots: bool = True,
    profile: str | None = None,
    pretrained: str | None = None,
    policy_update_interval: int = 8,
    reset_each_frame: bool = True,
) -> Dict[str, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    frame_cfg = FrameConfig()
    model = AdapterEqualizer(EqualizerConfig(
        state_dim=2 * (2 * window_K + 1) + 3,
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
    )).to(dev)
    load_msg = _load_pretrained_if_available(model, pretrained, dev)
    theta_pre = _capture_model_state(model)
    model.enable_parameter_efficient_tuning(train_adapter=True, train_output=True, train_sync=use_sync_head)

    strategies = make_strategy_table()
    policy = DiscretePPOPolicy(obs_dim=8, num_actions=len(strategies), device=dev)
    history = {"loss_ema": 0.0, "ber_ema": 0.0, "last_reward": 0.0}
    peft_records: List[dict] = []
    mmse_records: List[dict] = []

    print(load_msg)
    print(f"在线自适应: frames={num_frames}, SNR={snr}dB, profile={profile or 'rayleigh'}")
    print(f"PEFT 可训练参数: {model.trainable_parameter_count()}")

    print(f"每帧恢复 θ_pre: {reset_each_frame}")

    for frame_idx in range(num_frames):
        if reset_each_frame:
            _restore_model_state(model, theta_pre)
            model.enable_parameter_efficient_tuning(train_adapter=True, train_output=True, train_sync=use_sync_head)
        env = _make_env(seed + frame_idx, snr, profile=profile, window_K=window_K)
        mmse_metrics = evaluate_mmse(env)
        controller = AdaptationController(model, frame_cfg, dev)
        obs = controller.build_observation(env, history)
        action, log_prob, value = policy.act(obs)
        result = controller.adapt_frame(env, strategies[action])
        policy.remember(obs, action, log_prob, result.reward, value)

        history["loss_ema"] = 0.9 * history["loss_ema"] + 0.1 * result.pilot_loss
        history["ber_ema"] = 0.9 * history["ber_ema"] + 0.1 * result.ber_pilot
        history["last_reward"] = result.reward

        peft_records.append({
            "BER_data": result.ber_data,
            "BER_pilot": result.ber_pilot,
            "pilot_loss": result.pilot_loss,
            "reward": result.reward,
            "adapt_params": result.adapt_params,
            "adapt_steps": result.adapt_steps,
            "latency_ms": result.latency_ms,
            "action": action,
            "strategy": strategies[action].name,
        })
        mmse_records.append({
            "BER_data": mmse_metrics["BER_data"],
            "BER_pilot": mmse_metrics["BER_pilot"],
            "pilot_loss": 0.0,
            "adapt_params": 0,
            "adapt_steps": 0,
            "latency_ms": 0.0,
        })

        if (frame_idx + 1) % policy_update_interval == 0:
            policy.update()

        if (frame_idx + 1) == 1 or (frame_idx + 1) % max(1, min(20, num_frames)) == 0:
            print(
                f"[{frame_idx + 1:4d}/{num_frames}] "
                f"PEFT data={result.ber_data:.4f} pilot={result.ber_pilot:.4f} "
                f"loss={result.pilot_loss:.4f} action={strategies[action].name} "
                f"| MMSE data={mmse_metrics['BER_data']:.4f}"
            )

    policy.update()
    if reset_each_frame:
        _restore_model_state(model, theta_pre)
    generalization = _evaluate_generalization(model, dev, snr, seed)
    output_path = Path(output_dir)
    if save_plots:
        _save_plots(peft_records, mmse_records, generalization, output_path)

    results = {
        "peft": _summarize(peft_records),
        "mmse": _summarize(mmse_records),
        "generalization": generalization,
        "artifacts": {
            "online_metrics": str(output_path / "online_adapt_metrics.png"),
            "mmse_comparison": str(output_path / "mmse_baseline_comparison.png"),
            "generalization": str(output_path / "generalization.png"),
        },
        "reset_each_frame": reset_each_frame,
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snr", type=float, default=10.0)
    parser.add_argument("--profile", type=str, default=None, choices=[None, "rician", "epa", "eva", "etu"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--adapter_rank", type=int, default=8)
    parser.add_argument("--window_K", type=int, default=10)
    parser.add_argument("--use_channel_encoder", action="store_true")
    parser.add_argument("--channel_dim", type=int, default=32)
    parser.add_argument("--use_sync_head", action="store_true")
    parser.add_argument("--sync_dim", type=int, default=32)
    parser.add_argument("--sync_delay_bins", type=int, default=9)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="logs")
    parser.add_argument("--carry_state_across_frames", action="store_true")
    args = parser.parse_args()

    results = run_online_adaptation(
        num_frames=args.num_frames,
        seed=args.seed,
        snr=args.snr,
        d_model=args.d_model,
        n_layers=args.n_layers,
        adapter_rank=args.adapter_rank,
        window_K=args.window_K,
        use_channel_encoder=args.use_channel_encoder,
        channel_dim=args.channel_dim,
        use_sync_head=args.use_sync_head,
        sync_dim=args.sync_dim,
        sync_delay_bins=args.sync_delay_bins,
        device=args.device,
        output_dir=args.output_dir,
        profile=args.profile,
        pretrained=args.pretrained,
        reset_each_frame=not args.carry_state_across_frames,
    )

    print("\n=== 汇总指标 ===")
    for name in ["peft", "mmse"]:
        item = results[name]
        print(
            f"{name.upper():>5} | BER_data={item['BER_data']:.5f} "
            f"BER_pilot={item['BER_pilot']:.5f} pilot_loss={item['pilot_loss']:.5f} "
            f"adapt_params={item['adapt_params']:.0f} adapt_steps={item['adapt_steps']:.2f} "
            f"latency_ms={item['latency_ms']:.2f}"
        )
    print(f"泛化 BER_data: {results['generalization']}")
    print(f"可视化输出: {results['artifacts']}")


if __name__ == "__main__":
    main()
