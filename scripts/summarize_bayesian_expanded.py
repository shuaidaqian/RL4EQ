# -*- coding: utf-8 -*-
"""汇总扩样本 Bayes vs Frozen 诊断：session 级均值、配对 CI、窗口曲线与斜率。

输入为 `diagnose_bayesian_cir.py` 的 JSON。统计单位是 session（seed），不是帧或 bit。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _mean_ci(values: list[float], t_value: float) -> tuple[float, float]:
    count = len(values)
    mean = sum(values) / count
    if count < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    return mean, t_value * math.sqrt(variance / count)


def _slope(values: list[float]) -> float:
    count = len(values)
    if count < 2:
        return 0.0
    indices = list(range(count))
    mean_x = sum(indices) / count
    mean_y = sum(values) / count
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(indices, values))
    denominator = sum((x - mean_x) ** 2 for x in indices)
    return numerator / denominator if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = payload["rows"]
    seeds = sorted({int(row["seed"]) for row in rows})
    t_value = 2.045 if len(seeds) >= 30 else 2.262  # df=29 / 9 的 95% 双侧近似
    summary = {
        "input": str(Path(args.input)),
        "seeds": seeds,
        "session_count": len(seeds),
        "window": args.window,
        "per_snr": {},
    }
    for snr in sorted({int(row["snr_db"]) for row in rows}):
        snr_rows = [row for row in rows if int(row["snr_db"]) == snr]
        per_seed = {}
        for seed in seeds:
            selected = [row for row in snr_rows if int(row["seed"]) == seed]
            ordered = sorted(selected, key=lambda row: int(row["frame_index"]))
            windows = []
            for start in range(0, len(ordered), args.window):
                chunk = ordered[start:start + args.window]
                if chunk:
                    windows.append(
                        sum(row["frozen_ber"] - row["bayes_ber"] for row in chunk) / len(chunk)
                    )
            per_seed[seed] = {
                "frozen_ber": sum(row["frozen_ber"] for row in ordered) / len(ordered),
                "bayes_ber": sum(row["bayes_ber"] for row in ordered) / len(ordered),
                "gain": sum(row["frozen_ber"] - row["bayes_ber"] for row in ordered) / len(ordered),
                "frozen_nmse": sum(row["frozen_nmse"] for row in ordered) / len(ordered),
                "bayes_nmse": sum(row["bayes_nmse"] for row in ordered) / len(ordered),
                "windows": windows,
            }
        frozen = _mean_ci([item["frozen_ber"] for item in per_seed.values()], t_value)
        bayes = _mean_ci([item["bayes_ber"] for item in per_seed.values()], t_value)
        gain = _mean_ci([item["gain"] for item in per_seed.values()], t_value)
        window_count = min(len(item["windows"]) for item in per_seed.values())
        window_means = []
        for index in range(window_count):
            values = [item["windows"][index] for item in per_seed.values() if len(item["windows"]) > index]
            window_means.append(_mean_ci(values, t_value))
        summary["per_snr"][str(snr)] = {
            "frozen_ber": {"mean": frozen[0], "ci95": frozen[1]},
            "bayes_ber": {"mean": bayes[0], "ci95": bayes[1]},
            "gain": {"mean": gain[0], "ci95": gain[1]},
            "negative_seeds": sum(1 for item in per_seed.values() if item["gain"] < 0),
            "nmse": {
                "frozen": sum(item["frozen_nmse"] for item in per_seed.values()) / len(per_seed),
                "bayes": sum(item["bayes_nmse"] for item in per_seed.values()) / len(per_seed),
            },
            "windows": [{"mean": mean, "ci95": ci} for mean, ci in window_means],
            "slope": _slope([mean for mean, _ in window_means]),
            "last_minus_first": window_means[-1][0] - window_means[0][0] if window_means else 0.0,
            "per_seed": per_seed,
        }
        print(
            f"snr={snr:2d} frozen={frozen[0]:.6f}±{frozen[1]:.6f} "
            f"bayes={bayes[0]:.6f}±{bayes[1]:.6f} gain={gain[0]:+.6f}±{gain[1]:.6f} "
            f"neg={summary['per_snr'][str(snr)]['negative_seeds']}/{len(seeds)} "
            f"slope={summary['per_snr'][str(snr)]['slope']:+.6f} "
            f"last-first={summary['per_snr'][str(snr)]['last_minus_first']:+.6f}",
            flush=True,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"written {output}", flush=True)


if __name__ == "__main__":
    main()
