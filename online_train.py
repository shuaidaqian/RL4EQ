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
import copy
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.adaptation_controller import OBS_DIM, AdaptationController, make_strategy_table
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


def _make_env(
    seed: int,
    snr: float,
    profile: str | None = None,
    window_K: int = 10,
    use_mmse_features: bool = False,
) -> CommunicationEnv:
    env = CommunicationEnv(EnvConfig(
        frame=FrameConfig(),
        channel=_channel_cfg(profile, snr, seed, max_num_taps=window_K + 1),
        window_K=window_K,
        use_mmse_features=use_mmse_features,
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


def _mmse_soft_output(env: CommunicationEnv) -> torch.Tensor:
    rx = env.get_rx_symbols()
    bits = env.get_true_bits()
    frame_cfg = env.frame_cfg
    num_taps = getattr(env.channel, "num_taps", getattr(env.channel, "nt", 16))
    mmse = MMSEEqualizer(num_taps=num_taps)
    soft, _ = mmse(rx, bits[:frame_cfg.train_len], rx[:frame_cfg.train_len], env.channel.snr_db)
    return soft


def _build_pseudo_labels(
    model: AdapterEqualizer,
    env: CommunicationEnv,
    device: torch.device,
    neural_threshold: float = 0.85,
    mmse_threshold: float = 0.65,
) -> tuple[torch.Tensor, torch.Tensor]:
    frame_cfg = env.frame_cfg
    data_mask = _mask(frame_cfg, "data")
    states = env.get_all_states().unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        _, probs = model(states)
        neural_probs = probs[0].detach().cpu()
    neural_bits = (neural_probs > 0.5).float()
    neural_conf = (neural_probs - 0.5).abs() * 2.0

    mmse_soft = _mmse_soft_output(env).detach().cpu()
    mmse_bits = (mmse_soft < 0).float()
    mmse_conf = torch.sigmoid(mmse_soft.abs())
    agree = neural_bits == mmse_bits
    pseudo_mask = data_mask & agree & (neural_conf >= neural_threshold) & (mmse_conf >= mmse_threshold)
    return neural_bits.to(device), pseudo_mask.to(device)


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
        "parameter_delta_norm": _mean(r.get("parameter_delta_norm", 0.0) for r in records),
    }


def _build_equalizer(
    d_model: int,
    n_layers: int,
    adapter_rank: int,
    window_K: int,
    use_channel_encoder: bool,
    channel_dim: int,
    use_sync_head: bool,
    sync_dim: int,
    sync_delay_bins: int,
    use_mmse_features: bool,
    device: torch.device,
) -> AdapterEqualizer:
    return AdapterEqualizer(EqualizerConfig(
        state_dim=2 * (2 * window_K + 1) + 3 + (2 if use_mmse_features else 0),
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
    )).to(device)


def _safe_corr(x: Iterable[float], y: Iterable[float]) -> float:
    xs = np.asarray(list(x), dtype=np.float64)
    ys = np.asarray(list(y), dtype=np.float64)
    if xs.size < 2 or ys.size < 2 or np.std(xs) < 1e-12 or np.std(ys) < 1e-12:
        return 0.0
    return float(np.corrcoef(xs, ys)[0, 1])


def _evaluate_generalization(
    model: AdapterEqualizer,
    device: torch.device,
    snr: float,
    seed: int,
    window_K: int,
    use_mmse_features: bool,
) -> Dict[str, float]:
    """在未见 3GPP profile 上只评估，不在线更新。"""
    out = {}
    model.eval()
    for idx, profile in enumerate(["rician", "epa", "eva", "etu"]):
        env = _make_env(
            seed + 1000 + idx,
            snr,
            profile=profile,
            window_K=window_K,
            use_mmse_features=use_mmse_features,
        )
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


def _save_strategy_diagnostics(results: dict, output_dir: Path) -> None:
    if not HAS_MPL:
        return
    _configure_matplotlib_fonts()
    output_dir.mkdir(parents=True, exist_ok=True)
    eps = 1e-5

    method_names = list(results["methods"].keys())
    ber_values = [max(results["methods"][name]["BER_data"], eps) for name in method_names]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(method_names, ber_values, color="#1565C0")
    ax.axhline(max(results["mmse"]["BER_data"], eps), color="#C62828", linestyle="--", label="MMSE")
    ax.set_yscale("log")
    ax.set_ylabel("BER_data")
    ax.set_title("固定策略 / PPO / Oracle 对比")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "strategy_matrix.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    points = results["correlation_records"]
    if points:
        x = [p["delta_pilot_loss"] for p in points]
        y = [p["delta_ber_data"] for p in points]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(x, y, s=16, alpha=0.55, color="#455A64")
        ax.axhline(0.0, color="#999999", linewidth=1)
        ax.axvline(0.0, color="#999999", linewidth=1)
        ax.set_xlabel("Δpilot_loss")
        ax.set_ylabel("ΔBER_data")
        ax.set_title(f"Pilot reward 与 Data BER 改善相关性 r={results['correlations']['delta_pilot_loss_vs_delta_ber_data']:.3f}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "pilot_data_correlation.png", dpi=170, bbox_inches="tight")
        plt.close(fig)


def run_strategy_diagnostics(
    num_frames: int = 50,
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
    use_mmse_features: bool = False,
    device: str = "cpu",
    output_dir: str | os.PathLike = "logs/strategy_diagnostics",
    profile: str | None = None,
    pretrained: str | None = None,
    save_plots: bool = True,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    frame_cfg = FrameConfig()
    model = _build_equalizer(
        d_model=d_model,
        n_layers=n_layers,
        adapter_rank=adapter_rank,
        window_K=window_K,
        use_channel_encoder=use_channel_encoder,
        channel_dim=channel_dim,
        use_sync_head=use_sync_head,
        sync_dim=sync_dim,
        sync_delay_bins=sync_delay_bins,
        use_mmse_features=use_mmse_features,
        device=dev,
    )
    load_msg = _load_pretrained_if_available(model, pretrained, dev)
    theta_pre = _capture_model_state(model)
    strategies = make_strategy_table()
    policy = DiscretePPOPolicy(obs_dim=OBS_DIM, num_actions=len(strategies), device=dev)
    history = {"loss_ema": 0.0, "ber_ema": 0.0, "last_reward": 0.0}

    fixed_records = {strategy.name: [] for strategy in strategies}
    ppo_records: List[dict] = []
    mmse_records: List[dict] = []
    pilot_oracle_records: List[dict] = []
    data_oracle_records: List[dict] = []
    correlation_records: List[dict] = []

    print(load_msg)
    print(f"策略诊断: frames={num_frames}, SNR={snr}dB, profile={profile or 'rayleigh'}")

    for frame_idx in range(num_frames):
        _restore_model_state(model, theta_pre)
        env = _make_env(
            seed + frame_idx,
            snr,
            profile=profile,
            window_K=window_K,
            use_mmse_features=use_mmse_features,
        )
        mmse_metrics = evaluate_mmse(env)
        pseudo_bits, pseudo_mask = _build_pseudo_labels(model, env, dev)
        mmse_records.append({
            "BER_data": mmse_metrics["BER_data"],
            "BER_pilot": mmse_metrics["BER_pilot"],
        })

        candidate_results = []
        for action_id, strategy in enumerate(strategies):
            _restore_model_state(model, theta_pre)
            model.enable_parameter_efficient_tuning(train_adapter=True, train_output=True, train_sync=use_sync_head)
            controller = AdaptationController(model, frame_cfg, dev)
            result = controller.adapt_frame(env, strategy, pseudo_bits=pseudo_bits, pseudo_mask=pseudo_mask)
            record = {
                "BER_data": result.ber_data,
                "BER_pilot": result.ber_pilot,
                "pilot_loss": result.pilot_loss,
                "reward": result.reward,
                "adapt_params": result.adapt_params,
                "adapt_steps": result.adapt_steps,
                "latency_ms": result.latency_ms,
                "parameter_delta_norm": result.parameter_delta_norm,
                "action": action_id,
                "strategy": strategy.name,
            }
            fixed_records[strategy.name].append(record)
            candidate_results.append(record)

        skip_record = candidate_results[0]
        for record in candidate_results:
            correlation_records.append({
                "frame": frame_idx,
                "strategy": record["strategy"],
                "delta_pilot_loss": skip_record["pilot_loss"] - record["pilot_loss"],
                "delta_ber_data": skip_record["BER_data"] - record["BER_data"],
                "BER_pilot": record["BER_pilot"],
                "BER_data": record["BER_data"],
            })

        pilot_best = min(candidate_results, key=lambda item: (item["pilot_loss"], item["BER_pilot"]))
        data_best = min(candidate_results, key=lambda item: item["BER_data"])
        pilot_oracle_records.append(pilot_best)
        data_oracle_records.append(data_best)

        _restore_model_state(model, theta_pre)
        model.enable_parameter_efficient_tuning(train_adapter=True, train_output=True, train_sync=use_sync_head)
        controller = AdaptationController(model, frame_cfg, dev)
        obs = controller.build_observation(env, history)
        action, log_prob, value = policy.act(obs)
        ppo_result = controller.adapt_frame(env, strategies[action], pseudo_bits=pseudo_bits, pseudo_mask=pseudo_mask)
        policy.remember(obs, action, log_prob, ppo_result.reward, value)
        ppo_records.append({
            "BER_data": ppo_result.ber_data,
            "BER_pilot": ppo_result.ber_pilot,
            "pilot_loss": ppo_result.pilot_loss,
            "reward": ppo_result.reward,
            "adapt_params": ppo_result.adapt_params,
            "adapt_steps": ppo_result.adapt_steps,
            "latency_ms": ppo_result.latency_ms,
            "parameter_delta_norm": ppo_result.parameter_delta_norm,
            "action": action,
            "strategy": strategies[action].name,
        })
        history["loss_ema"] = 0.9 * history["loss_ema"] + 0.1 * ppo_result.pilot_loss
        history["ber_ema"] = 0.9 * history["ber_ema"] + 0.1 * ppo_result.ber_pilot
        history["last_reward"] = ppo_result.reward
        history["last_latency_ms"] = ppo_result.latency_ms
        if (frame_idx + 1) % 8 == 0:
            policy.update()

    policy.update()
    methods = {name: _summarize(records) for name, records in fixed_records.items()}
    methods["ppo_policy"] = _summarize(ppo_records)
    methods["pilot_oracle_action"] = _summarize(pilot_oracle_records)
    methods["data_oracle_action"] = _summarize(data_oracle_records)
    correlations = {
        "delta_pilot_loss_vs_delta_ber_data": _safe_corr(
            (r["delta_pilot_loss"] for r in correlation_records),
            (r["delta_ber_data"] for r in correlation_records),
        ),
        "BER_pilot_vs_BER_data": _safe_corr(
            (r["BER_pilot"] for r in correlation_records),
            (r["BER_data"] for r in correlation_records),
        ),
    }
    results = {
        "methods": methods,
        "mmse": _summarize(mmse_records),
        "correlations": correlations,
        "correlation_records": correlation_records,
        "config": {
            "num_frames": num_frames,
            "seed": seed,
            "snr": snr,
            "profile": profile or "rayleigh",
            "pretrained": pretrained,
        },
        "artifacts": {
            "strategy_matrix": str(Path(output_dir) / "strategy_matrix.png"),
            "pilot_data_correlation": str(Path(output_dir) / "pilot_data_correlation.png"),
            "metrics": str(Path(output_dir) / "strategy_diagnostics.json"),
        },
    }
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "strategy_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    if save_plots:
        _save_strategy_diagnostics(results, out_dir)
    return results


def run_pilot_overhead_study(
    ratios: Iterable[float] = (0.5, 0.25, 0.125, 0.0625),
    num_frames: int = 30,
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
    use_mmse_features: bool = False,
    device: str = "cpu",
    output_dir: str | os.PathLike = "logs/pilot_overhead",
    profile: str | None = None,
    pretrained: str | None = None,
) -> Dict[str, object]:
    dev = torch.device(device)
    model = _build_equalizer(
        d_model=d_model,
        n_layers=n_layers,
        adapter_rank=adapter_rank,
        window_K=window_K,
        use_channel_encoder=use_channel_encoder,
        channel_dim=channel_dim,
        use_sync_head=use_sync_head,
        sync_dim=sync_dim,
        sync_delay_bins=sync_delay_bins,
        use_mmse_features=use_mmse_features,
        device=dev,
    )
    _load_pretrained_if_available(model, pretrained, dev)
    model.eval()
    records = []
    for ratio in ratios:
        known_total = max(4, int(round(512 * float(ratio))))
        pilot_len = max(2, known_total // 4)
        train_len = max(4, known_total - 2 * pilot_len)
        frame_cfg = FrameConfig(frame_len=512, train_len=train_len, pilot_len=pilot_len, num_pilots=2)
        data_mask = torch.tensor([frame_cfg.bit_type(t) == "data" for t in range(frame_cfg.frame_len)], dtype=torch.bool)
        pilot_mask = torch.tensor([frame_cfg.bit_type(t) == "pilot" for t in range(frame_cfg.frame_len)], dtype=torch.bool)
        neural_data = []
        neural_pilot = []
        mmse_data = []
        mmse_pilot = []
        for frame_idx in range(num_frames):
            cur_seed = seed + int(ratio * 10000) + frame_idx
            env = CommunicationEnv(EnvConfig(
                frame=frame_cfg,
                channel=_channel_cfg(profile, snr, cur_seed, max_num_taps=window_K + 1),
                window_K=window_K,
                use_mmse_features=use_mmse_features,
                seed=cur_seed,
            ))
            env.reset()
            states = env.get_all_states().unsqueeze(0).to(dev)
            bits = env.get_true_bits().to(dev)
            with torch.no_grad():
                _, probs = model(states)
                preds = (probs[0] > 0.5).float()
            neural_data.append(float((preds[data_mask.to(dev)] != bits[data_mask.to(dev)]).float().mean().item()))
            neural_pilot.append(float((preds[pilot_mask.to(dev)] != bits[pilot_mask.to(dev)]).float().mean().item()))
            mmse = evaluate_mmse(env)
            mmse_data.append(mmse["BER_data"])
            mmse_pilot.append(mmse["BER_pilot"])
        records.append({
            "known_ratio": float(ratio),
            "train_len": train_len,
            "pilot_len": pilot_len,
            "neural_BER_data": _mean(neural_data),
            "neural_BER_pilot": _mean(neural_pilot),
            "mmse_BER_data": _mean(mmse_data),
            "mmse_BER_pilot": _mean(mmse_pilot),
        })

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "records": records,
        "config": {
            "num_frames": num_frames,
            "seed": seed,
            "snr": snr,
            "profile": profile or "rayleigh",
            "pretrained": pretrained,
        },
        "artifacts": {
            "metrics": str(out_dir / "pilot_overhead.json"),
            "plot": str(out_dir / "pilot_overhead.png"),
        },
    }
    with open(out_dir / "pilot_overhead.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    if HAS_MPL:
        _configure_matplotlib_fonts()
        eps = 1e-5
        xs = [r["known_ratio"] * 100 for r in records]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(xs, [max(r["neural_BER_data"], eps) for r in records], marker="o", label="神经均衡")
        ax.plot(xs, [max(r["mmse_BER_data"], eps) for r in records], marker="s", label="MMSE")
        ax.set_yscale("log")
        ax.set_xlabel("已知位比例 (%)")
        ax.set_ylabel("BER_data")
        ax.set_title("Pilot overhead 敏感性")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "pilot_overhead.png", dpi=170, bbox_inches="tight")
        plt.close(fig)
    return results

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
    use_mmse_features: bool = False,
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
    model = _build_equalizer(
        d_model=d_model,
        n_layers=n_layers,
        adapter_rank=adapter_rank,
        window_K=window_K,
        use_channel_encoder=use_channel_encoder,
        channel_dim=channel_dim,
        use_sync_head=use_sync_head,
        sync_dim=sync_dim,
        sync_delay_bins=sync_delay_bins,
        use_mmse_features=use_mmse_features,
        device=dev,
    )
    load_msg = _load_pretrained_if_available(model, pretrained, dev)
    theta_pre = _capture_model_state(model)
    model.enable_parameter_efficient_tuning(train_adapter=True, train_output=True, train_sync=use_sync_head)

    strategies = make_strategy_table()
    policy = DiscretePPOPolicy(obs_dim=OBS_DIM, num_actions=len(strategies), device=dev)
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
        env = _make_env(
            seed + frame_idx,
            snr,
            profile=profile,
            window_K=window_K,
            use_mmse_features=use_mmse_features,
        )
        mmse_metrics = evaluate_mmse(env)
        pseudo_bits, pseudo_mask = _build_pseudo_labels(model, env, dev)
        controller = AdaptationController(model, frame_cfg, dev)
        obs = controller.build_observation(env, history)
        action, log_prob, value = policy.act(obs)
        result = controller.adapt_frame(env, strategies[action], pseudo_bits=pseudo_bits, pseudo_mask=pseudo_mask)
        policy.remember(obs, action, log_prob, result.reward, value)

        history["loss_ema"] = 0.9 * history["loss_ema"] + 0.1 * result.pilot_loss
        history["ber_ema"] = 0.9 * history["ber_ema"] + 0.1 * result.ber_pilot
        history["last_reward"] = result.reward
        history["last_latency_ms"] = result.latency_ms

        peft_records.append({
            "BER_data": result.ber_data,
            "BER_pilot": result.ber_pilot,
            "pilot_loss": result.pilot_loss,
            "reward": result.reward,
            "adapt_params": result.adapt_params,
            "adapt_steps": result.adapt_steps,
            "latency_ms": result.latency_ms,
            "parameter_delta_norm": result.parameter_delta_norm,
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
    generalization = _evaluate_generalization(model, dev, snr, seed, window_K, use_mmse_features)
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
    parser.add_argument("--use_mmse_features", action="store_true")
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
        use_mmse_features=args.use_mmse_features,
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
