# -*- coding: utf-8 -*-
"""只读分析已完成匹配验证的训练策略与递归接收交互。"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.continual_training_summary import interaction_report, summarize_training_interaction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    args = parser.parse_args()
    source = Path(args.input)
    result = json.loads(source.read_text(encoding='utf8'))
    summary = summarize_training_interaction(result)
    summary['analysis_source_sha256'] = hashlib.sha256((ROOT/'evaluation/continual_training_summary.py').read_bytes()).hexdigest()
    source.with_name(source.stem+'_interaction.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n', encoding='utf8')
    source.with_name(source.stem+'_interaction.md').write_text(interaction_report(summary), encoding='utf8')
    print(interaction_report(summary))


if __name__ == '__main__':
    main()
