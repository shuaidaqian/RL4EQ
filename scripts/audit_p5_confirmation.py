# -*- coding: utf-8 -*-
"""P5 确认流完整性审计：会话/臂/帧齐全、逐帧波形配对、字段与数值合法。"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def audit(protocol, snr, spec):
    base = ROOT / spec['stream']
    main = json.loads((ROOT / spec['output']).read_text(encoding='utf8'))
    rows = [json.loads(line) for line in open(base, encoding='utf8')]
    arms = set(protocol['arms'])
    frames = protocol['frames']
    by = collections.defaultdict(lambda: collections.defaultdict(dict))
    rx = collections.defaultdict(lambda: collections.defaultdict(set))
    bad = 0
    for row in rows:
        key = (row['snr_db'], row['seed'])
        by[key][row['arm']][row['frame_index']] = row
        rx[key][row['frame_index']].add(row['rx_sha256'])
        for field in ('errors', 'bits', 'elapsed_ms', 'true_cir_norm_diagnostic'):
            value = row[field]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                bad += 1
    problems = []
    for seed in spec['seeds']:
        key = (snr, seed)
        if key not in by:
            problems.append(f'缺少 session {key}')
            continue
        if set(by[key]) != arms:
            problems.append(f'{key} 臂不完整')
        for arm in arms:
            if set(by[key].get(arm, {})) != set(range(frames)):
                problems.append(f'{key} {arm} 帧不完整')
        for index in range(frames):
            if len(rx[key][index]) != 1:
                problems.append(f'{key} 帧 {index} 波形不匹配')
                break
    return {
        'snr_db': snr, 'experiment': spec['name'], 'status': main.get('status'),
        'row_count': main.get('row_count'), 'rows_read': len(rows),
        'expected_rows': len(spec['seeds']) * len(arms) * frames,
        'sessions': len(by), 'expected_sessions': len(spec['seeds']),
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
