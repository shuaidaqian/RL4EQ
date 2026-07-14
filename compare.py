# -*- coding: utf-8 -*-
"""PEFT 在线自适应与 MMSE baseline 的 SNR 扫描对比。"""

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from online_train import run_online_adaptation

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
