# -*- coding: utf-8 -*-
"""传统 baseline 难度校准入口。

默认只做当前主线允许的干净 Level B 场景：

- Pilot 只在前缀；
- SNR = 0/5/10/15 dB；
- 传统方法不使用神经网络或 RL；
- CFO、额外相位扰动、非线性、编码和高阶调制默认关闭。

当前已支持 residual CFO 与慢相位扰动；非线性、编码和高阶调制仍不进入本轮主线。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline.traditional_equalizers import TRADITIONAL_BASELINES
from evaluation.research_diagnostics import run_level_b_difficulty_scan


DEFAULT_DELAYS = [20, 30, 40]
DEFAULT_SNRS = [0.0, 5.0, 10.0, 15.0]
DEFAULT_PILOT_TOTALS = [32, 48, 64, 96, 128]
DEFAULT_PILOT_LAYOUTS = ["prefix"]
DEFAULT_METHODS = ["DFE-RLS", "RLS Linear", "SC-FDE-MMSE", "LMMSE-FIR"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--delays", nargs="*", type=int, default=DEFAULT_DELAYS)
    parser.add_argument("--snrs", nargs="*", type=float, default=DEFAULT_SNRS)
    parser.add_argument("--pilot-totals", nargs="*", type=int, default=DEFAULT_PILOT_TOTALS)
    parser.add_argument("--pilot-layouts", nargs="*", default=DEFAULT_PILOT_LAYOUTS)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--output-dir", default="logs/traditional_difficulty_calibration")
    parser.add_argument("--enable-cfo", action="store_true")
    parser.add_argument("--enable-phase-perturbation", action="store_true")
    parser.add_argument("--enable-nonlinearity", action="store_true")
    parser.add_argument("--enable-coding", action="store_true")
    parser.add_argument("--enable-higher-order-modulation", action="store_true")
    parser.add_argument(
        "--impairment-profile",
        default="clean",
        choices=["clean", "cfo_tiny", "phase_tiny", "cfo_phase_tiny", "cfo_light", "phase_light", "cfo_phase_light", "cfo_phase_mid"],
    )
    args = parser.parse_args()
    if args.version:
        print("RL4EQ traditional-difficulty-calibration schema-v1")
        return

    methods = _validate_methods(args.methods)
    layouts = _validate_layouts(args.pilot_layouts)
    impairments = {
        "cfo": bool(args.enable_cfo),
        "phase_perturbation": bool(args.enable_phase_perturbation),
        "nonlinearity": bool(args.enable_nonlinearity),
        "coding": bool(args.enable_coding),
        "higher_order_modulation": bool(args.enable_higher_order_modulation),
    }
    if args.enable_nonlinearity or args.enable_coding or args.enable_higher_order_modulation:
        raise SystemExit("当前校准入口仅支持 CFO/慢相位扰动；非线性、编码和高阶调制留作后续扩展。")
    if (args.enable_cfo or args.enable_phase_perturbation) and args.impairment_profile == "clean":
        raise SystemExit("启用 CFO/相位扰动时必须选择非 clean 的 --impairment-profile。")

    target = Path(args.output_dir)
    target.mkdir(parents=True, exist_ok=True)
    difficulty = run_level_b_difficulty_scan(
        delays=[int(value) for value in args.delays],
        snrs=[float(value) for value in args.snrs],
        pilot_totals=[int(value) for value in args.pilot_totals],
        pilot_layouts=layouts,
        seeds=[int(value) for value in args.seeds],
        frames=int(args.frames),
        output_dir=target,
        methods=tuple(methods),
        impairment_profile=str(args.impairment_profile),
    )
    payload = {
        "schema_version": "traditional-difficulty-calibration-v1",
        "main_level": "B",
        "pilot_layouts": layouts,
        "snrs": [float(value) for value in args.snrs],
        "pilot_totals": [int(value) for value in args.pilot_totals],
        "delays": [int(value) for value in args.delays],
        "seeds": [int(value) for value in args.seeds],
        "frames": int(args.frames),
        "methods": methods,
        "optional_impairments": impairments,
        "impairment_profile": str(args.impairment_profile),
        "traditional_only": True,
        "level_b_difficulty": difficulty,
        "difficulty_bands": _difficulty_bands(difficulty["summary"]),
    }
    (target / "traditional_difficulty_calibration.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(target), "rows": len(difficulty["rows"])}, ensure_ascii=False))


def _validate_methods(methods: list[str]) -> list[str]:
    allowed = set(TRADITIONAL_BASELINES)
    unknown = [method for method in methods if method not in allowed]
    if unknown:
        raise ValueError(f"未知传统 baseline：{unknown}")
    deduped = []
    for method in methods:
        if method not in deduped:
            deduped.append(method)
    return deduped


def _validate_layouts(layouts: list[str]) -> list[str]:
    allowed = {"prefix", "two_block", "multi_block"}
    unknown = [layout for layout in layouts if layout not in allowed]
    if unknown:
        raise ValueError(f"未知 Pilot 布局：{unknown}")
    return list(dict.fromkeys(layouts))


def _difficulty_bands(summary: list[dict]) -> dict[str, int]:
    """统计传统 baseline 难度区间，便于快速定位论文主工作区间。"""

    bands = {
        "easy_ber_lt_0.01": 0,
        "useful_0.01_to_0.1": 0,
        "too_hard_ge_0.1": 0,
    }
    for row in summary:
        ber = float(row["mean_ber_data"])
        if ber < 0.01:
            bands["easy_ber_lt_0.01"] += 1
        elif ber < 0.1:
            bands["useful_0.01_to_0.1"] += 1
        else:
            bands["too_hard_ge_0.1"] += 1
    return bands


if __name__ == "__main__":
    main()
