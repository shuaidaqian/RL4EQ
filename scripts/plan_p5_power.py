# -*- coding: utf-8 -*-
"""只读功效规划：用已完成 P11 的 session 级变化估计确认阶段所需会话数。

估计量（每个 SNR、每个 seed/session）：
- own_gain  = O(平台窗) − O(早期窗)             负值表示 Online 自身改善；
- adv_gain  = G(平台窗) − G(早期窗)             正值表示相对 Frozen 优势扩大；
- adv_gain_vs_current = (B−O) 的同窗变化；
- hold_own / hold_adv = 晚窗 − 平台窗            用于饱和/保持等价检验的方差输入。
报告 session 均值、样本标准差、n=6 下的 sd 上界（80% 单侧置信），
以及给定效应量/等价边界时 80% 功效所需的会话数。只作规划，不是确认。
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]

WINDOWS = [(0, 1), (1, 4), (4, 16), (16, 64), (64, 256), (256, 1024),
           (1024, 2048), (2048, 4096), (4096, 8192), (8192, 11719)]
EARLY, PLATEAU, LATE = 4, 6, 8  # [64,256), [1024,2048), [4096,8192)
ALPHA, POWER = 0.05, 0.8


def window_of(index):
    for k, (start, stop) in enumerate(WINDOWS):
        if start <= index < stop:
            return k
    return None


def aggregate(path):
    acc = collections.defaultdict(lambda: [0, 0])
    for line in open(path, encoding='utf8'):
        row = json.loads(line)
        k = window_of(row['frame_index'])
        if k is None:
            continue
        cell = acc[(row['snr_db'], row['seed'], row['arm'], k)]
        cell[0] += row['errors']
        cell[1] += row['bits']
    return acc


def ber(acc, snr, seed, arm, k):
    cell = acc.get((snr, seed, arm, k))
    if not cell or not cell[1]:
        return None
    return cell[0] / cell[1]


def sd_upper_80(sd, n):
    """正态样本 sd 的 80% 单侧置信上界（卡方，df=n−1）。"""
    from scipy.stats import chi2
    if n < 2 or not sd:
        return None
    return sd * math.sqrt((n - 1) / chi2.ppf(0.2, n - 1))


def required_n(delta, sd, alpha=ALPHA, power=POWER, cap=100000):
    from scipy.stats import nct, t
    if not sd or not delta or delta <= 0:
        return None
    for n in range(2, cap + 1):
        crit = t.ppf(1 - alpha / 2, n - 1)
        ncp = delta / (sd / math.sqrt(n))
        got = nct.cdf(-crit, n - 1, ncp) + (1 - nct.cdf(crit, n - 1, ncp))
        if got >= power:
            return n
    return f'>{cap}'


def equivalence_n(margin, sd, alpha=ALPHA, power=POWER, cap=100000):
    """TOST 的正态近似所需会话数（每个单侧检验用 alpha）。"""
    from scipy.stats import norm
    if not sd or not margin or margin <= 0:
        return None
    n = ((norm.ppf(1 - alpha) + norm.ppf(power)) ** 2) * (sd / margin) ** 2
    return max(2, math.ceil(n))


def summary(values):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return {'n': len(values)}
    sd = statistics.stdev(values)
    return {'n': len(values), 'mean': statistics.mean(values), 'sd': sd,
            'sd_upper_80': sd_upper_80(sd, len(values)),
            'ci95_halfwidth': 1.96 * sd / math.sqrt(len(values))}


def plan(acc, arms):
    recursive, current, frozen = arms
    result = {}
    for snr in sorted({key[0] for key in acc}):
        seeds = sorted({key[1] for key in acc if key[0] == snr})
        sessions = []
        for seed in seeds:
            O = {k: ber(acc, snr, seed, recursive, k) for k in (EARLY, PLATEAU, LATE)}
            B = {k: ber(acc, snr, seed, current, k) for k in (EARLY, PLATEAU, LATE)}
            F = {k: ber(acc, snr, seed, frozen, k) for k in (EARLY, PLATEAU, LATE)}
            if any(v is None for v in (*O.values(), *B.values(), *F.values())):
                continue
            G = {k: F[k] - O[k] for k in (EARLY, PLATEAU, LATE)}
            sessions.append({
                'seed': seed,
                'O_early': O[EARLY], 'O_plateau': O[PLATEAU], 'O_late': O[LATE],
                'G_early': G[EARLY], 'G_plateau': G[PLATEAU], 'G_late': G[LATE],
                'own_gain': O[PLATEAU] - O[EARLY],
                'adv_gain': G[PLATEAU] - G[EARLY],
                'adv_gain_vs_current': (B[PLATEAU] - O[PLATEAU]) - (B[EARLY] - O[EARLY]),
                'hold_own': O[LATE] - O[PLATEAU],
                'hold_adv': G[LATE] - G[PLATEAU],
                'rel_own_gain': (O[PLATEAU] / O[EARLY] - 1) if O[EARLY] > 0 else None,
                'rel_hold_own': (O[LATE] / O[PLATEAU] - 1) if O[PLATEAU] > 0 else None,
            })
        metrics = ('own_gain', 'adv_gain', 'adv_gain_vs_current', 'hold_own', 'hold_adv',
                   'rel_own_gain', 'rel_hold_own')
        stats = {m: summary([s[m] for s in sessions]) for m in metrics}
        effects = {}
        for m, grid in (
                ('own_gain', (('abs_0.001', 0.001), ('abs_0.002', 0.002), ('abs_0.005', 0.005))),
                ('adv_gain', (('abs_0.001', 0.001), ('abs_0.002', 0.002), ('abs_0.005', 0.005))),
                ('adv_gain_vs_current', (('abs_0.001', 0.001), ('abs_0.002', 0.002), ('abs_0.005', 0.005))),
                ('rel_own_gain', (('rel_5pct', 0.05), ('rel_10pct', 0.10), ('rel_20pct', 0.20)))):
            mean = stats[m].get('mean')
            sd = stats[m].get('sd')
            candidates = {}
            if mean:
                for label, factor in (('observed', 1.0), ('half', 0.5), ('quarter', 0.25)):
                    candidates[label] = required_n(abs(mean) * factor, sd)
            for label, value in grid:
                candidates[label] = required_n(value, sd)
            effects[m] = candidates
        equivalence = {}
        for m, grid in (('hold_own', (0.001, 0.002, 0.005)),
                        ('hold_adv', (0.001, 0.002, 0.005)),
                        ('rel_hold_own', (0.10, 0.20, 0.50))):
            equivalence[m] = {f'margin_{x}': equivalence_n(x, stats[m].get('sd'))
                              for x in grid}
        result[str(snr)] = {'seeds': seeds, 'sessions': sessions, 'summary': stats,
                            'required_sessions': effects, 'equivalence_sessions': equivalence}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='docs/results/continual_v2_matched_history_long_dev_20261120/matched_long_6x11719.jsonl')
    parser.add_argument('--output', default='docs/results/continual_v2_p5_planning_20261122/power_planning.json')
    args = parser.parse_args()
    acc = aggregate(ROOT / args.input)
    report = {
        'data': args.input,
        'windows': {'early': WINDOWS[EARLY], 'plateau': WINDOWS[PLATEAU], 'late': WINDOWS[LATE]},
        'estimands': {
            'own_gain': 'O_plateau − O_early（负值表示自身改善）',
            'adv_gain': 'G_plateau − G_early，G=F−O（正值表示相对 Frozen 优势扩大）',
            'adv_gain_vs_current': 'B 相对 O 的配对优势变化（B−O）',
            'hold_own': 'O_late − O_plateau（饱和/保持等价检验的方差输入）',
            'hold_adv': 'G_late − G_plateau',
        },
        'alpha': ALPHA, 'power': POWER,
        'caveat': 'n=6 的方差估计很宽；所需会话数仅为规划参考，不是冻结样本量。',
        'per_snr': plan(acc, ('recursive_pilot_ar1', 'current_pilot_aged_ar1',
                              'acquisition_known_adapt_real_physics')),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print('wrote', output)


if __name__ == '__main__':
    main()
