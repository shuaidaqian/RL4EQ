# -*- coding: utf-8 -*-
"""按真实 CIR 轨迹的晚段漂移分层，为 P5 四 SNR 分配互不重叠的开发 seed。

只使用信道生成器与冻结动力学（不读取 BER、NMSE 或接收结果）。每个 seed 先在
开发阶段模拟数据帧区间的 CIR 轨迹，计算晚窗相对平台窗的有效 SNR 漂移，再按
分位数分层配额抽取。输出为待用户确认的候选 seed 清单，不自动登记。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WINDOWS = {'early': (64, 256), 'plateau': (1024, 2048),
           'late': (4096, 8192), 'hold': (8192, 11719)}


def window_of(index):
    for name, (start, stop) in WINDOWS.items():
        if start <= index < stop:
            return name
    return None


def simulate_seed(args):
    seed, frames, config_path = args
    import json as _json
    import torch
    from env.comm_env import CommunicationEnvironment
    from env.experiment_config import build_comm_env_config as _build

    config = _json.loads(Path(config_path).read_text(encoding='utf8'))
    env = CommunicationEnvironment(_build(
        config, level='B', snr_db=0, seed=seed, max_delay=config['max_delay'],
        total_pilot=config['pilot_total'], reward_pilot_total=config['reward_pilot_total'],
        pilot_layout='prefix', state_split='offline_train'))
    env.reset_episode()
    sums = {name: 0.0 for name in WINDOWS}
    counts = {name: 0 for name in WINDOWS}
    for index in range(frames):
        name = window_of(index)
        if name is not None:
            sums[name] += float(env.channel._current_cir.norm()) ** 2
            counts[name] += 1
        env.channel._evolve_eme_taps()
    means = {name: sums[name] / counts[name] if counts[name] else float('nan')
             for name in WINDOWS}
    late_drift_db = 10.0 * math.log10(means['late'] / means['plateau'])
    hold_drift_db = 10.0 * math.log10(means['hold'] / means['plateau'])
    # 保持检验使用 [8192,11719) 相对平台窗，用该漂移做分层平衡。
    return {'seed': seed, 'early_power': means['early'], 'plateau_power': means['plateau'],
            'late_power': means['late'], 'hold_power': means['hold'],
            'late_drift_db': late_drift_db, 'hold_drift_db': hold_drift_db,
            'drift_db': hold_drift_db}


def quantile_bins(values, bins):
    order = sorted(range(len(values)), key=lambda i: values[i])
    groups = [[] for _ in range(bins)]
    for rank, index in enumerate(order):
        groups[min(bins - 1, rank * bins // len(order))].append(index)
    return groups


def allocate(group_sizes, bins):
    """每个 SNR 组在每个分位层按比例取数；余数从漂移最低层开始。"""
    quotas = {}
    for name, total in group_sizes.items():
        base = [total // bins] * bins
        for i in range(total % bins):
            base[i] += 1
        quotas[name] = base
    return quotas


def pick_from_bin(indices, quota, used, values):
    """在层内按漂移排序后等间隔取数，避免聚集在层边缘。"""
    remaining = [i for i in indices if i not in used]
    if len(remaining) < quota:
        raise ValueError(f'分位层候选不足：需要 {quota}，剩余 {len(remaining)}。')
    remaining.sort(key=lambda i: values[i])
    if quota == 0:
        return []
    step = len(remaining) / quota
    return [remaining[min(len(remaining) - 1, int(k * step))] for k in range(quota)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=int, default=250001)
    parser.add_argument('--count', type=int, default=700)
    parser.add_argument('--frames', type=int, default=11719)
    parser.add_argument('--bins', type=int, default=5)
    parser.add_argument('--groups', default='0:269,5:69,10:73,15:116')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--output', default='docs/results/continual_v2_p5_seed_selection_20261123/selection.json')
    args = parser.parse_args()
    group_sizes = {}
    for part in args.groups.split(','):
        snr, size = part.split(':')
        group_sizes[snr] = int(size)
    config_path = str(ROOT / 'configs/eme_long_memory_v2_online_dd_tail_v1.json')
    seeds = list(range(args.start, args.start + args.count))
    tasks = [(seed, args.frames, config_path) for seed in seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        candidates = list(pool.map(simulate_seed, tasks))
    values = [row['drift_db'] for row in candidates]
    bins = quantile_bins(values, args.bins)
    quotas = allocate(group_sizes, args.bins)
    used = set()
    groups = {}
    for snr, total in group_sizes.items():
        picked = []
        for bin_index, quota in zip(bins, quotas[snr]):
            for index in pick_from_bin(bin_index, quota, used, values):
                used.add(index)
                picked.append(candidates[index])
        groups[snr] = sorted(picked, key=lambda row: row['seed'])
    # 每个 SNR 组的分层平衡摘要
    summary = {}
    for snr, rows in groups.items():
        summary[snr] = {
            'n': len(rows),
            'hold_drift_mean_db': statistics.mean(row['hold_drift_db'] for row in rows),
            'hold_drift_sd_db': statistics.stdev([row['hold_drift_db'] for row in rows]),
            'hold_drift_range_db': [min(row['hold_drift_db'] for row in rows),
                                    max(row['hold_drift_db'] for row in rows)],
            'late_drift_mean_db': statistics.mean(row['late_drift_db'] for row in rows),
            'late_drift_sd_db': statistics.stdev([row['late_drift_db'] for row in rows]),
        }
    result = {
        'candidates': {'start': args.start, 'count': args.count, 'frames': args.frames,
                       'drift_mean_db': statistics.mean(values),
                       'drift_sd_db': statistics.stdev(values)},
        'bins': args.bins, 'group_sizes': group_sizes, 'summary': summary,
        'groups': {snr: {'seeds': [row['seed'] for row in rows], 'descriptors': rows}
                   for snr, rows in groups.items()},
        'all_candidates': candidates,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('wrote', output)


if __name__ == '__main__':
    main()
