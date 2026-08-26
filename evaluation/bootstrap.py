# -*- coding: utf-8 -*-
"""按 seed 与连续帧块重采样的配对 bootstrap。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BootstrapInterval:
    mean: float
    low: float
    high: float
    std: float
    repetitions: int
    block_length: int
    resampling_order: tuple[str, str] = ("seed", "contiguous_frame_block")


def paired_block_bootstrap(rows: Iterable[dict], seed: int, repetitions: int = 2000, block_length: int = 10) -> BootstrapInterval:
    """先重采样 seed，再在 seed 内采样连续 frame block。"""

    materialized = list(rows)
    if not materialized:
        return BootstrapInterval(float("nan"), float("nan"), float("nan"), float("nan"), repetitions, block_length)
    rng = np.random.default_rng(seed)
    by_seed: dict[int, list[dict]] = {}
    for row in materialized:
        by_seed.setdefault(int(row["seed"]), []).append(row)
    for values in by_seed.values():
        values.sort(key=lambda item: int(item["frame"]))
    seed_keys = sorted(by_seed)
    samples = []
    for _ in range(repetitions):
        selected_values = []
        sampled_seeds = rng.choice(seed_keys, size=len(seed_keys), replace=True)
        for sampled_seed in sampled_seeds:
            frames = by_seed[int(sampled_seed)]
            if len(frames) <= block_length:
                block = frames
            else:
                start = int(rng.integers(0, len(frames) - block_length + 1))
                block = frames[start : start + block_length]
            selected_values.extend(float(row["ber_data"]) for row in block)
        samples.append(float(np.mean(selected_values)))
    values = np.asarray(samples, dtype=float)
    return BootstrapInterval(
        mean=float(np.mean(values)),
        low=float(np.quantile(values, 0.025)),
        high=float(np.quantile(values, 0.975)),
        std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        repetitions=repetitions,
        block_length=block_length,
    )
