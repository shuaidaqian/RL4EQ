# -*- coding: utf-8 -*-
"""EME-inspired slow-drift traditional-only 工作区校准。

该入口只运行传统非神经、非 RL baseline，用于在 Proposed 训练前冻结
slow phase/Doppler 工作区。候选选择不读取 Proposed 结果，避免事后挑场景。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline.traditional_equalizers import TRADITIONAL_BASELINES
from evaluation.research_diagnostics import run_level_b_difficulty_scan


DEFAULT_CFO_ABS_VALUES = [0.0008, 0.0015, 0.0030, 0.0050]
DEFAULT_PHASE_NOISE_STDS = [0.0008, 0.0015, 0.0030, 0.0050]
DEFAULT_RHOS = [0.995, 0.990, 0.980]
DEFAULT_DELAYS = [20, 30, 40]
DEFAULT_SNRS = [0.0, 5.0, 10.0, 15.0]
DEFAULT_METHODS = (
    "LMMSE-FIR",
    "RLS Linear",
    "DFE-RLS",
    "SC-FDE-MMSE",
    "CFO-Corrected LMMSE-FIR",
    "CFO-Corrected DFE-RLS",
    "CFO+DD-Phase LMMSE-FIR",
    "CFO+DD-Phase DFE-RLS",
)
SNR_TARGET_BANDS = {
    0.0: (0.20, 0.35),
    5.0: (0.12, 0.28),
    10.0: (0.08, 0.22),
    15.0: (0.05, 0.18),
}


def profile_cfo_limit(cfo_abs: float, multiplier: float = 1.5) -> float:
    """根据 profile-level CFO 预算生成传统补偿搜索限幅。"""

    return round(float(max(0.001, float(multiplier) * abs(float(cfo_abs)))), 10)


def build_candidates(
    cfo_abs_values: Iterable[float],
    phase_noise_stds: Iterable[float],
    rhos: Iterable[float],
    cfo_limit_multiplier: float = 1.5,
) -> list[dict]:
    """构造 slow-drift 数值化候选网格。"""

    candidates = []
    for cfo_abs in cfo_abs_values:
        for phase_std in phase_noise_stds:
            for rho in rhos:
                candidates.append(
                    {
                        "candidate_id": f"cfo{_format_id(cfo_abs)}_phase{_format_id(phase_std)}_rho{_format_id(rho)}",
                        "cfo_abs_cycles_per_symbol": float(cfo_abs),
                        "phase_noise_std": float(phase_std),
                        "rho": float(rho),
                        "residual_cfo_limit": profile_cfo_limit(float(cfo_abs), cfo_limit_multiplier),
                        "shared_profile_prior": True,
                    }
                )
    return candidates


def score_candidate_summary(summary: list[dict], target_bands: dict[float, tuple[float, float]] | None = None) -> dict:
    """按 SNR 分层，基于最强 traditional baseline 计算候选分数。"""

    bands = target_bands or SNR_TARGET_BANDS
    grouped: dict[float, list[dict]] = {}
    for row in summary:
        snr = float(row["snr_db"])
        grouped.setdefault(snr, []).append(row)

    best_by_snr = {}
    total_penalty = 0.0
    missing_layers = []
    for snr, (low, high) in sorted(bands.items()):
        rows = grouped.get(float(snr), [])
        if not rows:
            missing_layers.append(float(snr))
            total_penalty += 10.0
            continue
        best = min(rows, key=lambda item: float(item["mean_ber_data"]))
        ber = float(best["mean_ber_data"])
        if ber < low:
            penalty = low - ber
        elif ber > high:
            penalty = ber - high
        else:
            penalty = 0.0
        total_penalty += penalty
        best_by_snr[str(float(snr))] = {
            "method": str(best["method"]),
            "mean_ber_data": ber,
            "target_low": float(low),
            "target_high": float(high),
            "in_target_band": penalty == 0.0,
            "penalty": float(penalty),
        }

    return {
        "score": float(total_penalty),
        "passes_all_snr_layers": bool(total_penalty == 0.0 and not missing_layers),
        "missing_snr_layers": missing_layers,
        "best_by_snr": best_by_snr,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfo-abs-values", nargs="*", type=float, default=DEFAULT_CFO_ABS_VALUES)
    parser.add_argument("--phase-noise-stds", nargs="*", type=float, default=DEFAULT_PHASE_NOISE_STDS)
    parser.add_argument("--rhos", nargs="*", type=float, default=DEFAULT_RHOS)
    parser.add_argument("--cfo-limit-multiplier", type=float, default=1.5)
    parser.add_argument("--delays", nargs="*", type=int, default=DEFAULT_DELAYS)
    parser.add_argument("--snrs", nargs="*", type=float, default=DEFAULT_SNRS)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--pilot-total", type=int, default=128)
    parser.add_argument("--pilot-layout", default="prefix")
    parser.add_argument("--methods", nargs="*", default=list(DEFAULT_METHODS))
    parser.add_argument("--output-dir", default="logs/eme_slow_drift_traditional_grid")
    args = parser.parse_args()

    methods = _validate_methods(args.methods)
    target = Path(args.output_dir)
    target.mkdir(parents=True, exist_ok=True)
    candidates = build_candidates(
        args.cfo_abs_values,
        args.phase_noise_stds,
        args.rhos,
        cfo_limit_multiplier=float(args.cfo_limit_multiplier),
    )

    all_rows = []
    candidate_summaries = []
    for index, candidate in enumerate(candidates):
        candidate_dir = target / candidate["candidate_id"]
        result = run_level_b_difficulty_scan(
            delays=[int(value) for value in args.delays],
            snrs=[float(value) for value in args.snrs],
            pilot_totals=[int(args.pilot_total)],
            pilot_layouts=[str(args.pilot_layout)],
            seeds=[int(value) for value in args.seeds],
            frames=int(args.frames),
            output_dir=candidate_dir,
            methods=tuple(methods),
            seed_offset=80_000 + index * 1_000,
            impairment_profile="eme_slow_drift_numeric",
            cfo_abs_cycles_per_symbol=float(candidate["cfo_abs_cycles_per_symbol"]),
            phase_noise_std=float(candidate["phase_noise_std"]),
            rho=float(candidate["rho"]),
            traditional_cfo_limit=float(candidate["residual_cfo_limit"]),
        )
        rows = []
        for row in result["rows"]:
            enriched = {
                **row,
                "candidate_id": candidate["candidate_id"],
                "proposed_method": False,
            }
            rows.append(enriched)
            all_rows.append(enriched)
        summary = [{**row, "candidate_id": candidate["candidate_id"]} for row in result["summary"]]
        candidate_summaries.append(
            {
                **candidate,
                "rows_count": len(rows),
                "summary": summary,
                "score": score_candidate_summary(summary),
            }
        )

    ranked = sorted(candidate_summaries, key=lambda item: (float(item["score"]["score"]), item["candidate_id"]))
    payload = {
        "schema_version": "eme-slow-drift-traditional-grid-v1",
        "traditional_only": True,
        "proposed_methods_included": False,
        "shared_profile_prior": True,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "candidate_summaries": candidate_summaries,
        "ranked_candidates": ranked,
        "selected_candidate": ranked[0] if ranked else None,
        "snr_target_bands": {str(key): list(value) for key, value in sorted(SNR_TARGET_BANDS.items())},
        "rows": all_rows,
    }
    (target / "eme_slow_drift_grid.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(target), "candidate_count": len(candidates), "rows": len(all_rows)}, ensure_ascii=False))


def _validate_methods(methods: list[str]) -> list[str]:
    allowed = set(TRADITIONAL_BASELINES)
    unknown = [method for method in methods if method not in allowed]
    if unknown:
        raise ValueError(f"未知传统 baseline：{unknown}")
    return list(dict.fromkeys(methods))


def _format_id(value: float) -> str:
    return f"{float(value):.4f}".rstrip("0").rstrip(".").replace(".", "p")


if __name__ == "__main__":
    main()
