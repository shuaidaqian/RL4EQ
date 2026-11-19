# -*- coding: utf-8 -*-
"""只读汇总已完成开发结果，不重新运行通信实验。"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.continual_summary import markdown_report, summarize_development


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    source = Path(args.input)
    result = json.loads(source.read_text(encoding="utf-8"))
    if result.get('rows_file'):
        rows_file = source.parent / result['rows_file']
        if not rows_file.exists():
            parser.error(f'流式帧结果文件不存在：{rows_file}')
        result['rows'] = [json.loads(line) for line in rows_file.read_text(encoding='utf-8').splitlines() if line]
    summary = summarize_development(result)
    summary['analysis_source_sha256'] = hashlib.sha256(
        (ROOT/'evaluation/continual_summary.py').read_bytes()).hexdigest()
    source.with_name(source.stem + "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    source.with_name(source.stem + "_report.md").write_text(markdown_report(result, summary), encoding="utf-8")
    print(markdown_report(result, summary))


if __name__ == "__main__":
    main()
