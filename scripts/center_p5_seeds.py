# -*- coding: utf-8 -*-
"""在保持窗分层配额内做零均值调整：交换同层候选，使每 SNR 组漂移均值接近 0。

只使用信道生成器的轨迹描述，不读取任何接收指标。
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def bins_of(candidates, count):
    order = sorted(range(len(candidates)), key=lambda i: candidates[i]['drift_db'])
    groups = [[] for _ in range(count)]
    for rank, index in enumerate(order):
        groups[min(count - 1, rank * count // len(order))].append(index)
    return groups


def center(group_seeds, candidates, bins, blocked, target=0.02, cap=5000):
    selected = set(group_seeds)
    values = {row['seed']: row['drift_db'] for row in candidates}
    bin_of = {}
    for b, indices in enumerate(bins):
        for index in indices:
            bin_of[candidates[index]['seed']] = b
    for _ in range(cap):
        mean = statistics.mean(values[seed] for seed in selected)
        if abs(mean) <= target:
            break
        chosen = min(selected, key=lambda seed: values[seed]) if mean < 0 else max(
            selected, key=lambda seed: values[seed])
        b = bin_of[chosen]
        outside = [candidates[i]['seed'] for i in bins[b]
                   if candidates[i]['seed'] not in selected
                   and candidates[i]['seed'] not in blocked]
        if not outside:
            break
        if mean < 0:
            replacement = max(outside, key=lambda seed: values[seed])
            if values[replacement] <= values[chosen]:
                break
        else:
            replacement = min(outside, key=lambda seed: values[seed])
            if values[replacement] >= values[chosen]:
                break
        selected.remove(chosen)
        selected.add(replacement)
    return sorted(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--target-db', type=float, default=0.02)
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text(encoding='utf8'))
    candidates = data['all_candidates']
    bins = bins_of(candidates, data['bins'])
    values = {row['seed']: row for row in candidates}
    all_initial = set().union(*(set(group['seeds']) for group in data['groups'].values()))
    used = set()
    groups = {}
    for snr, group in data['groups'].items():
        blocked = (all_initial - set(group['seeds'])) | used
        seeds = center(group['seeds'], candidates, bins, blocked, args.target_db)
        if used & set(seeds):
            raise ValueError(f'SNR {snr} 与其他组重复。')
        used.update(set(seeds) - set(group['seeds']))
        rows = [values[seed] for seed in seeds]
        groups[snr] = {
            'seeds': seeds,
            'summary': {
                'n': len(seeds),
                'hold_drift_mean_db': statistics.mean(row['hold_drift_db'] for row in rows),
                'hold_drift_sd_db': statistics.stdev([row['hold_drift_db'] for row in rows]),
                'late_drift_mean_db': statistics.mean(row['late_drift_db'] for row in rows),
            },
        }
    output = {
        'source': args.input,
        'rule': '在同层内交换候选，使每 SNR 组的 hold 窗漂移均值位于 ±target。',
        'target_db': args.target_db,
        'bins': data['bins'],
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
