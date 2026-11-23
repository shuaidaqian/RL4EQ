# -*- coding: utf-8 -*-
"""P5 确认分析：按预登记窗口计算 session 级自身改善/优势/保持，并执行冻结检验。

输入为 P5 协议 JSON（列出四个 SNR 实验及其 JSONL）与对应的逐帧流。任何会话
不完整、帧缺失、波形不匹配或非法字段都会中止汇总，不输出部分确认结论。
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scipy.stats import t as student_t


def window_of(index, windows):
    for name, (start, stop) in windows.items():
        if start <= index < stop:
            return name
    return None


def load_stream(path, snr, seeds, arms, frames, windows):
    """流式聚合窗口错误数并逐帧校验波形配对，避免把上千万行读入内存。"""

    allowed = set(seeds)
    counts = collections.Counter()
    sums = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    pending = {}
    problems = []
    for line in open(path, encoding='utf8'):
        row = json.loads(line)
        seed, arm, index = row['seed'], row['arm'], row['frame_index']
        if row['snr_db'] != snr or seed not in allowed or arm not in arms:
            problems.append(f'非法记录：snr={row["snr_db"]} seed={seed} arm={arm}')
            continue
        if counts[(seed, arm)] != index:
            problems.append(f'帧顺序/缺失：{seed} {arm} 期望 {counts[(seed, arm)]} 实际 {index}')
        counts[(seed, arm)] += 1
        name = window_of(index, windows)
        if name is not None:
            cell = sums[(seed, arm)][name]
            cell[0] += row['errors']
            cell[1] += row['bits']
        key = (seed, index)
        hashes = pending.setdefault(key, set())
        hashes.add(row['rx_sha256'])
        if len(hashes) == len(arms):
            if len(hashes) != 1:
                problems.append(f'波形不匹配：{seed} 帧 {index}')
            del pending[key]
    for seed in seeds:
        for arm in arms:
            if counts[(seed, arm)] != frames:
                problems.append(f'帧数不完整：{seed} {arm} {counts[(seed, arm)]}')
    if pending:
        problems.append(f'存在未配对的帧：{len(pending)}')
    if problems:
        raise ValueError('确认流完整性失败：' + '；'.join(problems[:10]))
    return sums


def session_metrics(sums, snr, seed, arms, windows):
    window_ber = {}
    for arm in arms:
        window_ber[arm] = {
            name: {'errors': sums[(seed, arm)][name][0], 'bits': sums[(seed, arm)][name][1],
                   'ber': sums[(seed, arm)][name][0] / sums[(seed, arm)][name][1]}
            for name in windows}
    for arm, values in window_ber.items():
        if not all(math.isfinite(v['ber']) for v in values.values()):
            raise ValueError(f'非法 BER：{snr} {seed} {arm}')
    O, F, B = arms
    own = window_ber[O]['plateau']['ber'] - window_ber[O]['early']['ber']
    adv_frozen = ((window_ber[F]['plateau']['ber'] - window_ber[O]['plateau']['ber'])
                  - (window_ber[F]['early']['ber'] - window_ber[O]['early']['ber']))
    adv_current = ((window_ber[B]['plateau']['ber'] - window_ber[O]['plateau']['ber'])
                   - (window_ber[B]['early']['ber'] - window_ber[O]['early']['ber']))
    hold_own = window_ber[O]['hold']['ber'] - window_ber[O]['plateau']['ber']
    hold_adv_current = ((window_ber[B]['hold']['ber'] - window_ber[O]['hold']['ber'])
                        - (window_ber[B]['plateau']['ber'] - window_ber[O]['plateau']['ber']))
    plateau = window_ber[O]['plateau']['ber']
    return {
        'seed': seed, 'windows': window_ber,
        'own_gain': own, 'adv_frozen': adv_frozen, 'adv_current': adv_current,
        'hold_own': hold_own,
        'hold_own_rel': (hold_own / plateau) if plateau > 0 else None,
        'hold_adv_current': hold_adv_current,
    }


def mean_ci(values):
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if n > 1 else float('nan')
    half = student_t.ppf(0.975, n - 1) * sd / math.sqrt(n) if n > 1 else float('nan')
    return {'n': n, 'mean': mean, 'sd': sd, 'ci95': [mean - half, mean + half]}


def one_sample(values, mu=0.0, alternative='two-sided'):
    from scipy.stats import ttest_1samp
    result = ttest_1samp(values, mu, alternative=alternative)
    return {'statistic': float(result.statistic), 'pvalue': float(result.pvalue),
            'mu': mu, 'alternative': alternative}


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    m = len(pvalues)
    adjusted = [1.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def tost(values, margin):
    upper = one_sample(values, margin, 'less')
    lower = one_sample(values, -margin, 'greater')
    pvalue = max(upper['pvalue'], lower['pvalue'])
    return {'margin': margin, 'p_upper': upper['pvalue'], 'p_lower': lower['pvalue'],
            'pvalue': pvalue, 'equivalent': pvalue < 0.05}


def analyze_experiment(protocol, snr, spec):
    arms = ['frozen_acquisition', 'current_pilot_aged_ar1', 'recursive_pilot_ar1']
    windows = protocol['windows']
    frames = protocol['frames']
    by = load_stream(ROOT / spec['stream'], snr, spec['seeds'], arms, frames)
    sessions = [session_metrics(by, snr, seed, arms, windows) for seed in spec['seeds']]
    own = [s['own_gain'] for s in sessions]
    adv_current = [s['adv_current'] for s in sessions]
    adv_frozen = [s['adv_frozen'] for s in sessions]
    hold_own = [s['hold_own'] for s in sessions]
    hold_adv = [s['hold_adv_current'] for s in sessions]
    return {
        'snr_db': snr, 'session_count': len(sessions), 'sessions': sessions,
        'own_gain': mean_ci(own), 'adv_current': mean_ci(adv_current),
        'adv_frozen': mean_ci(adv_frozen), 'hold_own': mean_ci(hold_own),
        'hold_adv_current': mean_ci(hold_adv),
        'tests': {'own_gain': one_sample(own), 'adv_current': one_sample(adv_current),
                  'adv_frozen': one_sample(adv_frozen)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    protocol = json.loads((ROOT / args.protocol).read_text(encoding='utf8'))
    results = {}
    for snr, spec in protocol['experiments'].items():
        results[snr] = analyze_experiment(protocol, int(snr), spec)
    # 多重性：主族 = 四 SNR 相对当前 Pilot 增量（Holm，双侧）；
    # 次族 = 0/5 dB 自身改善超过最小实质效应（单侧，Holm）。
    snrs = sorted(results, key=int)
    primary_p = [results[s]['tests']['adv_current']['pvalue'] for s in snrs]
    primary_adj = holm(primary_p)
    for snr, adjusted in zip(snrs, primary_adj):
        results[snr]['tests']['adv_current']['holm_pvalue'] = adjusted
    own_snrs = [s for s in snrs if int(s) in (0, 5)]
    minimum = protocol['effect_sizes']['own_abs']
    own_p = []
    for snr in own_snrs:
        values = [s['own_gain'] for s in results[snr]['sessions']]
        own_p.append(one_sample(values, -minimum, 'less')['pvalue'])
        results[snr]['tests']['own_minimum_effect'] = one_sample(values, -minimum, 'less')
    for snr, adjusted in zip(own_snrs, holm(own_p)):
        results[snr]['tests']['own_minimum_effect']['holm_pvalue'] = adjusted
    for snr in snrs:
        if int(snr) in (0, 5):
            margin = protocol['margins']['hold_own_abs']
            values = [s['hold_own'] for s in results[snr]['sessions']]
        else:
            margin = protocol['margins']['hold_own_rel']
            values = [s['hold_own_rel'] for s in results[snr]['sessions']
                      if s['hold_own_rel'] is not None]
        results[snr]['tests']['hold_tost'] = tost(values, margin)
    hold_p = [results[s]['tests']['hold_tost']['pvalue'] for s in snrs]
    for snr, adjusted in zip(snrs, holm(hold_p)):
        results[snr]['tests']['hold_tost']['holm_pvalue'] = adjusted
    summary = {
        'protocol': args.protocol, 'role': protocol.get('purpose', 'confirmation'),
        'windows': protocol['windows'], 'effect_sizes': protocol['effect_sizes'],
        'margins': protocol['margins'], 'multiplicity': protocol.get('multiplicity'),
        'results': results,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print('wrote', output)


if __name__ == '__main__':
    main()
