# -*- coding: utf-8 -*-
"""按零均值对称漂移目标为 P5 各组匹配 seed（只用信道生成器，不看 BER）。

目标分布为均值 0、标准差等于候选池标准差的近似正态分位点；每组按目标逐个
选择最近且未被使用的候选。这样保持窗的真实 ESNR 漂移在设计上没有系统偏移，
同时保留跨轨迹的异质性。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def normal_quantile(p):
    # Acklam 近似，足够用于设计目标
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def select_group(size, candidates, used, sd):
    targets = [normal_quantile((i + 0.5) / size) * sd for i in range(size)]
    selected = []
    for target in sorted(targets, key=abs):
        best = None
        for row in candidates:
            if row['seed'] in used:
                continue
            distance = abs(row['hold_drift_db'] - target)
            if best is None or distance < best[0]:
                best = (distance, row)
        if best is None:
            raise ValueError('候选不足。')
        used.add(best[1]['seed'])
        selected.append(best[1])
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text(encoding='utf8'))
    candidates = data['all_candidates']
    sd = statistics.stdev([row['hold_drift_db'] for row in candidates])
    sizes = {snr: len(group['seeds']) for snr, group in data['groups'].items()}
    used = set()
    groups = {}
    for snr in ('0', '5', '10', '15'):
        rows = select_group(sizes[snr], candidates, used, sd)
        groups[snr] = {
            'seeds': sorted(row['seed'] for row in rows),
            'summary': {
                'n': len(rows),
                'hold_drift_mean_db': statistics.mean(row['hold_drift_db'] for row in rows),
                'hold_drift_sd_db': statistics.stdev([row['hold_drift_db'] for row in rows]),
                'late_drift_mean_db': statistics.mean(row['late_drift_db'] for row in rows),
                'late_drift_sd_db': statistics.stdev([row['late_drift_db'] for row in rows]),
            },
        }
    output = {
        'source': args.input,
        'rule': '按均值 0、sd 等于候选池 sd 的正态目标匹配 hold 窗漂移；只使用生成器。',
        'target_sd_db': sd,
        'groups': groups,
        'all_candidates': candidates,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({snr: group['summary'] for snr, group in groups.items()},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
