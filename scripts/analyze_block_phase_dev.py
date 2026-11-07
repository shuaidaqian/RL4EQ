# -*- coding: utf-8 -*-
"""汇总 block_phase 开发筛查：逐 seed 配对差、描述性区间、窗口与相对 soft 增量。

只读取 diagnose_0db_data_posterior.py 的输出 JSON；不读 Data 标签，不做新的在线
决策。描述性 95% 区间使用 session 级 t 临界值（n=10 时 2.262），仅作开发证据。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


T953 = {
    2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
    8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179,
    14: 2.160, 15: 2.145, 16: 2.131, 17: 2.120, 18: 2.110, 19: 2.101,
    20: 2.093, 21: 2.086, 22: 2.080, 23: 2.074, 24: 2.069, 25: 2.064,
    26: 2.060, 27: 2.056, 28: 2.052, 29: 2.048, 30: 2.045,
}


def _t953(df: int) -> float:
    if df in T953:
        return T953[df]
    if df > 30:
        return 2.042
    return 2.262


def _interval(values: list[float]) -> dict:
    n = len(values)
    mean = statistics.fmean(values)
    if n < 2:
        return {"mean": mean, "se": float("nan"), "lo": float("nan"), "hi": float("nan")}
    se = statistics.stdev(values) / math.sqrt(n)
    half = _t953(n - 1) * se
    return {"mean": mean, "se": se, "lo": mean - half, "hi": mean + half}


def _per_seed_gain(rows: list[dict], variant: str, seeds: list[int]) -> dict[int, float]:
    result = {}
    for seed in seeds:
        selected = [r for r in rows if r["variant"] == variant and r["seed"] == seed]
        result[seed] = statistics.fmean(r["gain"] for r in selected)
    return result


def _windows(rows: list[dict], variant: str, frames: int, window: int = 16) -> list[float]:
    selected = [r for r in rows if r["variant"] == variant]
    result = []
    for start in range(0, frames, window):
        block = [r for r in selected if start <= r["frame_index"] < start + window]
        if block:
            result.append(statistics.fmean(r["gain"] for r in block))
    return result


def _fmt(values: list[float]) -> str:
    return "[" + ", ".join(f"{v:+.6f}" for v in values) + "]"


def summarize(payload: dict) -> None:
    rows = payload["rows"]
    seeds = list(payload["seeds"])
    frames = int(payload["frames"])
    variants = list(payload["variants"])
    timing = payload.get("timing")
    print(f"# timing={timing}, seeds={seeds[0]}..{seeds[-1]} (n={len(seeds)}), frames={frames}")
    for snr in sorted({r["snr_db"] for r in rows}):
        snr_rows = [r for r in rows if r["snr_db"] == snr]
        print(f"\n## {snr} dB")
        gains = {}
        for variant in variants:
            per_seed = _per_seed_gain(snr_rows, variant, seeds)
            values = [per_seed[s] for s in seeds]
            gains[variant] = per_seed
            ci = _interval(values)
            lo_count = sum(1 for v in values if v < 0)
            win = _windows(snr_rows, variant, frames)
            ber = statistics.fmean(
                r["ber_data"] for r in snr_rows if r["variant"] == variant
            )
            frozen = statistics.fmean(
                r["frozen_ber"] for r in snr_rows if r["variant"] == variant
            )
            print(
                f"- `{variant}`: BER {frozen:.6f}→{ber:.6f}, gain {ci['mean']:+.6f} "
                f"±{_t953(len(seeds) - 1) * ci['se']:.6f} "
                f"(95% [{ci['lo']:+.6f}, {ci['hi']:+.6f}]), 负向 seed {lo_count}/{len(values)}"
            )
            print(f"  - per-seed: {[round(v, 6) for v in values]}")
            print(f"  - windows(16f): {_fmt(win)}; first={win[0]:+.6f}, last={win[-1]:+.6f}")
        if "soft" in gains and "block_phase" in gains:
            diff = [gains["block_phase"][s] - gains["soft"][s] for s in seeds]
            ci = _interval(diff)
            print(
                f"- `block_phase − soft`: {ci['mean']:+.6f} "
                f"(95% [{ci['lo']:+.6f}, {ci['hi']:+.6f}]), "
                f"同向为正 {sum(1 for v in diff if v > 0)}/{len(diff)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()
    for path in args.inputs:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        summarize(payload)


if __name__ == "__main__":
    main()
