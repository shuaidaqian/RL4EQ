# -*- coding: utf-8 -*-
"""PEFT 在线自适应与 MMSE baseline 的 SNR 扫描对比。"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from online_train import (
    _mean,
    _make_env,
    evaluate_traditional_baselines,
    generate_fixed_eval_seeds,
    mismatch_scenarios,
    run_online_adaptation,
)
from pretrain import run_offline_pretraining
from env.frame_structure import frame_config_for_known_ratio
from baseline.traditional_equalizers import make_traditional_equalizers

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


def _as_float_list(values: Iterable[float]) -> list[float]:
    return [float(x) for x in values]


def _save_snr_plot(results: dict, output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not HAS_MPL:
        path = output_dir / "snr_comparison.csv"
        with open(path, "w", encoding="utf-8") as f:
            f.write("snr,peft_BER_data,mmse_BER_data,peft_BER_pilot,mmse_BER_pilot\n")
            for idx, snr in enumerate(results["snr_values"]):
                f.write(
                    f"{snr},{results['peft_BER_data'][idx]},"
                    f"{results['mmse_BER_data'][idx]},"
                    f"{results['peft_BER_pilot'][idx]},"
                    f"{results['mmse_BER_pilot'][idx]}\n"
                )
        return str(path)

    _configure_matplotlib_fonts()
    path = output_dir / "snr_mmse_comparison.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results["snr_values"], results["peft_BER_data"], marker="o", lw=2, label="PEFT 在线自适应")
    ax.plot(results["snr_values"], results["mmse_BER_data"], marker="s", lw=2, label="MMSE baseline")
    ax.set_yscale("log")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER_data")
    ax.set_title("数据段 BER: PEFT 在线自适应 vs MMSE")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _save_channel_baseline_plot(results: dict, output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not HAS_MPL:
        path = output_dir / "channel_baseline_comparison.csv"
        with open(path, "w", encoding="utf-8") as f:
            f.write("channel,baseline,BER_data,BER_pilot\n")
            for channel, metrics in results["metrics"].items():
                for baseline, item in metrics.items():
                    f.write(f"{channel},{baseline},{item['BER_data']},{item['BER_pilot']}\n")
        return str(path)

    _configure_matplotlib_fonts()
    path = output_dir / "channel_baseline_comparison.png"
    channels = results["channels"]
    baselines = results["baselines"]
    x = np.arange(len(channels))
    width = 0.8 / max(1, len(baselines))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for idx, baseline in enumerate(baselines):
        values = [max(results["metrics"][channel][baseline]["BER_data"], 1e-5) for channel in channels]
        ax.bar(x + (idx - (len(baselines) - 1) / 2) * width, values, width=width, label=baseline)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in channels])
    ax.set_ylabel("BER_data")
    ax.set_title("传统均衡算法在不同信道上的 BER 对比")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def run_channel_baseline_comparison(
    profiles: Iterable[str | None] = (None, "rician", "epa", "eva", "etu"),
    num_frames: int = 50,
    seed: int = 42,
    snr: float = 10.0,
    window_K: int = 10,
    output_dir: str | os.PathLike = "logs/baseline_channels",
) -> dict:
    """在 Rayleigh/Rician/3GPP 信道上比较常用传统均衡 baseline。"""
    profile_list = list(profiles)
    seed_map = generate_fixed_eval_seeds(seed, num_frames, profile_list)
    metrics = {}
    baseline_names = []
    for profile in profile_list:
        channel_name = profile or "rayleigh"
        per_baseline = {}
        for frame_seed in seed_map[channel_name][:num_frames]:
            env = _make_env(frame_seed, snr, profile=profile, window_K=window_K, use_mmse_features=False)
            frame_metrics = evaluate_traditional_baselines(env)
            if not baseline_names:
                baseline_names = list(frame_metrics.keys())
            for baseline, item in frame_metrics.items():
                per_baseline.setdefault(baseline, []).append(item)
        metrics[channel_name] = {
            baseline: {
                "BER_data": _mean(item["BER_data"] for item in records),
                "BER_pilot": _mean(item["BER_pilot"] for item in records),
            }
            for baseline, records in per_baseline.items()
        }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "channels": [profile or "rayleigh" for profile in profile_list],
        "baselines": baseline_names,
        "metrics": metrics,
        "seed_map": seed_map,
        "snr": snr,
        "num_frames": num_frames,
    }
    with open(out_dir / "channel_baseline_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    results["artifact"] = _save_channel_baseline_plot(results, out_dir)
    results["metrics_path"] = str(out_dir / "channel_baseline_comparison.json")
    return results


def _save_mismatch_plot(results: dict, output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not HAS_MPL:
        return results["artifacts"]["metrics"]
    _configure_matplotlib_fonts()
    path = output_dir / "mismatch_scenario_comparison.png"
    scenarios = results["scenarios"]
    methods = ["peft", "mmse", "rls", "zero_forcing"]
    x = np.arange(len(scenarios))
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for idx, method in enumerate(methods):
        values = []
        for scenario in scenarios:
            if method in results["metrics"][scenario]:
                values.append(max(results["metrics"][scenario][method]["BER_data"], 1e-5))
            else:
                values.append(1e-5)
        ax.bar(x + (idx - (len(methods) - 1) / 2) * width, values, width=width, label=method)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("BER_data")
    ax.set_title("模型失配场景 BER 对比")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def run_mismatch_scenario_comparison(
    scenario_names: Iterable[str] = ("cfo", "doppler", "nonlinear", "low_pilot"),
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
    use_cfo_head: bool = False,
    pretrained: str | None = None,
    imitation_frames: int = 0,
    imitation_epochs: int = 0,
    output_dir: str | os.PathLike = "logs/mismatch_scenarios",
    device: str = "cpu",
) -> dict:
    scenarios = mismatch_scenarios(snr=snr)
    selected = list(scenario_names)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for idx, name in enumerate(selected):
        scenario = scenarios[name]
        eval_seeds = generate_fixed_eval_seeds(seed + idx * 10000, num_frames, [name])[name]
        summary = run_online_adaptation(
            num_frames=num_frames,
            seed=seed + idx * 10000,
            snr=snr,
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
            use_cfo_head=use_cfo_head,
            pretrained=pretrained,
            imitation_frames=imitation_frames,
            imitation_epochs=imitation_epochs,
            output_dir=out_dir / name,
            save_plots=False,
            device=device,
            eval_seeds=eval_seeds,
            frame_config=scenario["frame"],
            impairments=scenario["impairments"],
            channel_config=scenario["channel"],
        )
        item = {
            "peft": summary["peft"],
            "mmse": summary["mmse"],
        }
        item.update(summary.get("traditional_baselines", {}))
        metrics[name] = item
    results = {
        "scenarios": selected,
        "metrics": metrics,
        "config": {
            "num_frames": num_frames,
            "seed": seed,
            "snr": snr,
            "pretrained": pretrained,
            "imitation_frames": imitation_frames,
            "imitation_epochs": imitation_epochs,
            "use_mmse_features": use_mmse_features,
            "use_cfo_head": use_cfo_head,
        },
        "artifacts": {
            "metrics": str(out_dir / "mismatch_scenario_comparison.json"),
            "plot": str(out_dir / "mismatch_scenario_comparison.png"),
        },
    }
    with open(out_dir / "mismatch_scenario_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    results["artifacts"]["plot"] = _save_mismatch_plot(results, out_dir)
    return results


def run_low_pilot_specialized_study(
    train_ratios: Iterable[float] = (0.5, 0.25, 0.125),
    eval_ratios: Iterable[float] = (0.5, 0.25, 0.125),
    train_steps: int = 100,
    eval_frames: int = 10,
    seed: int = 42,
    snr: float = 10.0,
    d_model: int = 64,
    n_layers: int = 2,
    adapter_rank: int = 8,
    window_K: int = 10,
    pretrained_init: str | None = None,
    output_dir: str | os.PathLike = "logs/low_pilot_specialized",
    device: str = "cpu",
) -> dict:
    """分别训练低 pilot overhead 专用模型，并输出 cross-overhead 泛化表。"""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ratio_list = [float(x) for x in train_ratios]
    eval_ratio_list = [float(x) for x in eval_ratios]
    matrix = {}
    checkpoints = {}
    for idx, train_ratio in enumerate(train_ratio_list):
        train_key = f"{train_ratio:g}"
        ckpt_dir = out_dir / f"pretrained_ratio_{train_key.replace('.', '_')}"
        train_result = run_offline_pretraining(
            num_steps=train_steps,
            batch_size=1,
            seed=seed + idx * 1000,
            d_model=d_model,
            n_layers=n_layers,
            adapter_rank=adapter_rank,
            window_K=window_K,
            use_mmse_features=False,
            lr=5e-4,
            snr_min=snr,
            snr_max=snr,
            val_interval=max(1, train_steps),
            val_frames=1,
            init_checkpoint=pretrained_init,
            train_known_ratios=[train_ratio],
            save_dir=ckpt_dir,
            device=device,
            save_plots=False,
        )
        checkpoints[train_key] = train_result["best_checkpoint"]
        matrix[train_key] = {}
        for eval_idx, eval_ratio in enumerate(eval_ratio_list):
            eval_key = f"{eval_ratio:g}"
            summary = run_online_adaptation(
                num_frames=eval_frames,
                seed=seed + idx * 1000 + eval_idx * 100,
                snr=snr,
                d_model=d_model,
                n_layers=n_layers,
                adapter_rank=adapter_rank,
                window_K=window_K,
                pretrained=train_result["best_checkpoint"],
                output_dir=out_dir / f"eval_train_{train_key}_eval_{eval_key}",
                save_plots=False,
                device=device,
                frame_config=frame_config_for_known_ratio(eval_ratio),
            )
            matrix[train_key][eval_key] = {
                "peft_BER_data": summary["peft"]["BER_data"],
                "mmse_BER_data": summary["mmse"]["BER_data"],
                "rls_BER_data": summary["traditional_baselines"].get("rls", {}).get("BER_data", 0.0),
            }
    results = {
        "train_ratios": train_ratio_list,
        "eval_ratios": eval_ratio_list,
        "matrix": matrix,
        "checkpoints": checkpoints,
        "artifacts": {
            "metrics": str(out_dir / "low_pilot_specialized.json"),
        },
    }
    with open(out_dir / "low_pilot_specialized.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


def audit_rls_fairness(
    num_frames: int = 50,
    seed: int = 42,
    snr: float = 10.0,
    output_dir: str | os.PathLike = "logs/rls_fairness_audit",
) -> dict:
    """审计 RLS baseline 是否只依赖 training 段标签。"""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    deltas = []
    bers = []
    for frame_idx in range(num_frames):
        env = _make_env(seed + frame_idx, snr, profile=None, window_K=10, use_mmse_features=False)
        rx = env.get_rx_symbols()
        bits = env.get_true_bits()
        frame_cfg = env.frame_cfg
        num_taps = getattr(env.channel, "num_taps", getattr(env.channel, "nt", 16))
        rls = make_traditional_equalizers(num_taps=num_taps)["rls"]
        soft_a, _ = rls(rx, bits[:frame_cfg.train_len], rx[:frame_cfg.train_len], env.channel.snr_db)

        changed_bits = bits.clone()
        changed_bits[frame_cfg.train_len:] = 1.0 - changed_bits[frame_cfg.train_len:]
        soft_b, _ = rls(rx, changed_bits[:frame_cfg.train_len], rx[:frame_cfg.train_len], env.channel.snr_db)
        deltas.append(float((soft_a - soft_b).abs().max().item()))

        data_mask = torch.tensor([frame_cfg.bit_type(t) == "data" for t in range(frame_cfg.frame_len)], dtype=torch.bool)
        preds = (soft_a < 0).float()
        bers.append(float((preds[data_mask] != bits[data_mask]).float().mean().item()))

    result = {
        "uses_training_only": bool(max(deltas) < 1e-6),
        "max_changed_output_delta": float(max(deltas) if deltas else 0.0),
        "mean_BER_data": _mean(bers),
        "std_BER_data": float(np.std(bers)) if bers else 0.0,
        "num_frames": int(num_frames),
        "artifacts": {
            "metrics": str(out_dir / "rls_fairness_audit.json"),
        },
    }
    with open(out_dir / "rls_fairness_audit.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def run_snr_comparison(
    snr_values: Iterable[float],
    num_frames: int = 50,
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
    output_dir: str | os.PathLike = "logs",
    device: str = "cpu",
    profile: str | None = None,
    pretrained: str | None = None,
) -> dict:
    snrs = _as_float_list(snr_values)
    results = {
        "snr_values": snrs,
        "peft_BER_data": [],
        "mmse_BER_data": [],
        "peft_BER_pilot": [],
        "mmse_BER_pilot": [],
        "peft_latency_ms": [],
        "peft_adapt_params": [],
    }

    for idx, snr in enumerate(snrs):
        summary = run_online_adaptation(
            num_frames=num_frames,
            seed=seed + idx * 100,
            snr=snr,
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
            device=device,
            output_dir=Path(output_dir) / f"snr_{snr:g}",
            save_plots=False,
            profile=profile,
            pretrained=pretrained,
        )
        results["peft_BER_data"].append(summary["peft"]["BER_data"])
        results["mmse_BER_data"].append(summary["mmse"]["BER_data"])
        results["peft_BER_pilot"].append(summary["peft"]["BER_pilot"])
        results["mmse_BER_pilot"].append(summary["mmse"]["BER_pilot"])
        results["peft_latency_ms"].append(summary["peft"]["latency_ms"])
        results["peft_adapt_params"].append(summary["peft"]["adapt_params"])

    results["artifact"] = _save_snr_plot(results, Path(output_dir))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr_min", type=float, default=0.0)
    parser.add_argument("--snr_max", type=float, default=20.0)
    parser.add_argument("--snr_step", type=float, default=5.0)
    parser.add_argument("--num_frames", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--profile", type=str, default=None, choices=[None, "rician", "epa", "eva", "etu"])
    parser.add_argument("--pretrained", type=str, default=None)
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
    parser.add_argument("--output_dir", type=str, default="logs")
    args = parser.parse_args()

    snrs = np.arange(args.snr_min, args.snr_max + 1e-9, args.snr_step)
    results = run_snr_comparison(
        snr_values=snrs,
        num_frames=args.num_frames,
        seed=args.seed,
        output_dir=args.output_dir,
        device=args.device,
        profile=args.profile,
        pretrained=args.pretrained,
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
    )

    print("\n=== SNR 对比汇总 ===")
    for idx, snr in enumerate(results["snr_values"]):
        print(
            f"SNR={snr:>5.1f}dB | "
            f"PEFT data={results['peft_BER_data'][idx]:.5f} "
            f"pilot={results['peft_BER_pilot'][idx]:.5f} | "
            f"MMSE data={results['mmse_BER_data'][idx]:.5f} "
            f"pilot={results['mmse_BER_pilot'][idx]:.5f}"
        )
    print(f"可视化输出: {results['artifact']}")


if __name__ == "__main__":
    main()
