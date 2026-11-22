# -*- coding: utf-8 -*-
"""只读归因：递归 Pilot 后段 BER 回升是否伴随真实信道轨迹变化。

输入为已完成的 P9/P11 逐帧流（schema 3）。按预登记窗口在 session 内聚合，
比较中段与晚段的 Online BER 变化、真实信道诊断（仅事后使用）与接收诊断。
只做描述性对齐与相关，不做假设检验、不做因果声明、不生成新 seed。
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]

WINDOWS = [(0, 1), (1, 4), (4, 16), (16, 64), (64, 256), (256, 1024),
           (1024, 2048), (2048, 4096), (4096, 8192), (8192, 11719)]
METRICS = [
    'aligned_nmse_diagnostic', 'prior_covariance_trace', 'posterior_covariance_trace',
    'pilot_innovation_mse', 'pilot_innovation_variance_mean',
    'decoder_phase_data_rms_rad_diagnostic', 'decoder_phase_data_centered_rms_rad_diagnostic',
    'decoder_signal_mismatch_nmse_diagnostic', 'input_soft_tail_power',
    'true_cir_norm_diagnostic', 'true_effective_esn0_db_diagnostic',
    'true_frequency_flatness_diagnostic', 'rx_power', 'data_bce', 'data_brier',
]
COUNT_FIELDS = ['confident_wrong_data_count', 'confident_data_count']
MID, LATE = (6, 7), (8, 9)


def window_of(index):
    for k, (start, stop) in enumerate(WINDOWS):
        if start <= index < stop:
            return k
    return None


def aggregate(path, frames_limit=None):
    acc = collections.defaultdict(lambda: collections.defaultdict(float))
    for line in open(path, encoding='utf8'):
        row = json.loads(line)
        index = row['frame_index']
        if frames_limit is not None and index >= frames_limit:
            continue
        k = window_of(index)
        if k is None:
            continue
        key = (row['snr_db'], row['seed'], row['arm'], k)
        cell = acc[key]
        cell['errors'] += row['errors']
        cell['bits'] += row['bits']
        cell['_frames'] += 1
        for name in METRICS:
            value = row[name]
            if isinstance(value, (int, float)):
                cell[name] += value
        for name in COUNT_FIELDS:
            cell[name] += row[name]
    return acc


def window_stats(acc, snr, seed, arm):
    rows = []
    for k in range(len(WINDOWS)):
        cell = acc.get((snr, seed, arm, k))
        if not cell or not cell['bits']:
            rows.append(None)
            continue
        stats = {'window': k, 'frames': int(cell['_frames']),
                 'errors': int(cell['errors']), 'bits': int(cell['bits']),
                 'ber': cell['errors'] / cell['bits']}
        for name in METRICS:
            stats[name] = cell[name] / cell['_frames']
        for name in COUNT_FIELDS:
            stats[name] = int(cell[name])
        stats['confident_wrong_rate'] = (cell['confident_wrong_data_count'] / cell['confident_data_count']
                                         if cell['confident_data_count'] else None)
        rows.append(stats)
    return rows


def pooled(rows, window_ids):
    errors = sum(rows[k]['errors'] for k in window_ids if rows[k])
    bits = sum(rows[k]['bits'] for k in window_ids if rows[k])
    frames = sum(rows[k]['frames'] for k in window_ids if rows[k])
    if not bits or not frames:
        return None
    out = {'errors': errors, 'bits': bits, 'ber': errors / bits}
    for name in METRICS:
        out[name] = sum(rows[k][name] * rows[k]['frames'] for k in window_ids if rows[k]) / frames
    out['confident_wrong_rate'] = (
        sum(rows[k]['confident_wrong_data_count'] for k in window_ids if rows[k])
        / sum(rows[k]['confident_data_count'] for k in window_ids if rows[k]))
    return out


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    try:
        return statistics.correlation(xs, ys)
    except statistics.StatisticsError:
        return None


def analyze(acc, arms):
    result = {}
    for snr in sorted({key[0] for key in acc}):
        seeds = sorted({key[1] for key in acc if key[0] == snr})
        entry = {'seeds': seeds, 'arms': {}, 'sessions': {}, 'late_vs_mid': {},
                 'correlations': {}}
        for arm in arms:
            windows = {seed: window_stats(acc, snr, seed, arm) for seed in seeds}
            pooled_rows = []
            for k in range(len(WINDOWS)):
                cells = [windows[seed][k] for seed in seeds if windows[seed][k]]
                if not cells:
                    pooled_rows.append(None)
                    continue
                pooled_rows.append({
                    'window': k, 'frames': sum(c['frames'] for c in cells),
                    'errors': sum(c['errors'] for c in cells),
                    'bits': sum(c['bits'] for c in cells),
                    'ber': sum(c['errors'] for c in cells) / sum(c['bits'] for c in cells),
                    'true_effective_esn0_db_diagnostic': statistics.mean(
                        c['true_effective_esn0_db_diagnostic'] for c in cells),
                    'aligned_nmse_diagnostic': statistics.mean(
                        c['aligned_nmse_diagnostic'] for c in cells),
                })
            entry['arms'][arm] = pooled_rows
            entry['sessions'][arm] = {str(seed): [
                None if row is None else {key: value for key, value in row.items()}
                for row in windows[seed]] for seed in seeds}
            deltas = {}
            for seed in seeds:
                mid, late = pooled(windows[seed], MID), pooled(windows[seed], LATE)
                if mid is None or late is None:
                    deltas[str(seed)] = None
                    continue
                deltas[str(seed)] = {
                    'ber_mid': mid['ber'], 'ber_late': late['ber'],
                    'delta_ber': late['ber'] - mid['ber'],
                    'delta_esn0_db': late['true_effective_esn0_db_diagnostic']
                    - mid['true_effective_esn0_db_diagnostic'],
                    'delta_aligned_nmse': late['aligned_nmse_diagnostic']
                    - mid['aligned_nmse_diagnostic'],
                    'delta_phase_rms': late['decoder_phase_data_rms_rad_diagnostic']
                    - mid['decoder_phase_data_rms_rad_diagnostic'],
                    'delta_tail_power': late['input_soft_tail_power']
                    - mid['input_soft_tail_power'],
                    'delta_confident_wrong_rate': late['confident_wrong_rate']
                    - mid['confident_wrong_rate'],
                    'delta_cir_norm': late['true_cir_norm_diagnostic']
                    - mid['true_cir_norm_diagnostic'],
                    'delta_flatness': late['true_frequency_flatness_diagnostic']
                    - mid['true_frequency_flatness_diagnostic'],
                }
            entry['late_vs_mid'][arm] = deltas
            entry['correlations'][arm] = {
                metric: pearson([(deltas[str(s)] or {}).get('delta_ber') for s in seeds],
                                [(deltas[str(s)] or {}).get(metric) for s in seeds])
                for metric in ('delta_esn0_db', 'delta_aligned_nmse', 'delta_phase_rms',
                               'delta_tail_power', 'delta_confident_wrong_rate',
                               'delta_cir_norm', 'delta_flatness')
            }
        result[str(snr)] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--p11', default='docs/results/continual_v2_matched_history_long_dev_20261120/matched_long_6x11719.jsonl')
    parser.add_argument('--p9', default='docs/results/continual_v2_matched_history_dev_20261120/matched_6x1024.jsonl')
    parser.add_argument('--p8', default='docs/results/continual_v2_long_mechanism_dev_20261120/mechanism_2x11719.jsonl')
    parser.add_argument('--output', default='docs/results/continual_v2_late_rebound_attribution_20261122/late_rebound_attribution.json')
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'method': '只读描述性归因；中段窗口 [1024,4096)，晚段 [4096,11719)；真实信道诊断仅事后使用。',
              'windows': WINDOWS, 'mid_windows': MID, 'late_windows': LATE}
    p11 = aggregate(ROOT / args.p11)
    arms11 = sorted({key[2] for key in p11})
    report['p11'] = analyze(p11, arms11)
    if (ROOT / args.p9).exists():
        p9 = aggregate(ROOT / args.p9, frames_limit=1024)
        arms9 = sorted({key[2] for key in p9})
        report['p9_first_1024_frames'] = analyze(p9, arms9)
    if (ROOT / args.p8).exists():
        p8 = aggregate(ROOT / args.p8)
        arms8 = sorted({key[2] for key in p8})
        report['p8'] = analyze(p8, arms8)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print('wrote', output)


if __name__ == '__main__':
    main()
