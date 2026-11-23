# -*- coding: utf-8 -*-
"""P5 确认流完整性审计：流式检查会话/臂/帧齐全、逐帧波形配对与有限值。"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def audit(protocol, snr, spec):
    arms = set(protocol['arms'])
    frames = protocol['frames']
    seeds = set(spec['seeds'])
    main = json.loads((ROOT / spec['output']).read_text(encoding='utf8'))
    counts = collections.Counter()
    pending = {}
    problems = []
    bad = 0
    rows_read = 0
    for line in open(ROOT / spec['stream'], encoding='utf8'):
        row = json.loads(line)
        rows_read += 1
        seed, arm, index = row['seed'], row['arm'], row['frame_index']
        if seed not in seeds or arm not in arms or row['snr_db'] != snr:
            problems.append(f'非法记录：seed={seed} arm={arm} snr={row["snr_db"]}')
            continue
        if counts[(seed, arm)] != index:
            problems.append(f'帧顺序/缺失：{seed} {arm} 期望 {counts[(seed, arm)]} 实际 {index}')
        counts[(seed, arm)] += 1
        key = (seed, index)
        hashes = pending.setdefault(key, {})
        hashes[arm] = row['rx_sha256']
        if len(hashes) == len(arms):
            if len(set(hashes.values())) != 1:
                problems.append(f'波形不匹配：{seed} 帧 {index}')
            del pending[key]
        for field in ('errors', 'bits', 'elapsed_ms', 'true_cir_norm_diagnostic'):
            value = row[field]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                bad += 1
    for seed in spec['seeds']:
        for arm in arms:
            if counts[(seed, arm)] != frames:
                problems.append(f'帧数不完整：{seed} {arm} {counts[(seed, arm)]}')
    if pending:
        problems.append(f'存在未配对的帧：{len(pending)}')
    return {
        'snr_db': snr, 'experiment': spec['name'], 'status': main.get('status'),
        'row_count': main.get('row_count'), 'rows_read': rows_read,
        'expected_rows': len(spec['seeds']) * len(arms) * frames,
        'sessions': len({seed for seed, _ in counts}),
        'expected_sessions': len(spec['seeds']),
        'nonfinite_values': bad, 'problems': problems[:20],
        'problem_count': len(problems),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    protocol = json.loads((ROOT / args.protocol).read_text(encoding='utf8'))
    results = [audit(protocol, int(snr), spec)
               for snr, spec in protocol['experiments'].items()]
    report = {'protocol': args.protocol, 'experiments': results,
              'all_complete': all(row['problem_count'] == 0 and row['status'] == 'completed'
                                  and row['rows_read'] == row['expected_rows']
                                  for row in results)}
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({row['snr_db']: {'complete': row['problem_count'] == 0,
                                      'problems': row['problem_count']}
                      for row in results}, ensure_ascii=False))
    print('all_complete', report['all_complete'])


if __name__ == '__main__':
    main()
