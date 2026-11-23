# -*- coding: utf-8 -*-
"""实验成功结束后把 seed registry 状态置为 completed；不自动释放 seed。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--result', required=True)
    args = parser.parse_args()
    result = json.loads((ROOT / args.result).read_text(encoding='utf8'))
    if result.get('status') != 'completed':
        raise SystemExit(f'结果不是 completed：{result.get("status")}')
    path = ROOT / 'docs/protocols/continual_v2_seed_registry.json'
    registry = json.loads(path.read_text(encoding='utf-8'))
    entry = next((row for row in registry['experiments']
                  if row['experiment'] == args.experiment), None)
    if entry is None:
        raise SystemExit(f'registry 中没有实验：{args.experiment}')
    entry['status'] = 'completed'
    entry['completion_note'] = f'Result status completed; row_count={result.get("row_count")}.'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)
    print('completed', args.experiment)


if __name__ == '__main__':
    main()
