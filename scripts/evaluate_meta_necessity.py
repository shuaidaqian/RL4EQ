# -*- coding: utf-8 -*-
"""汇总 Frozen、普通 Pilot-SGD 与 Meta-Pilot 的配对必要性实验。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from training.online_meta_adaptation import build_meta_necessity_report


def _read_rows(path: Path) -> list[dict]:
    """读取 JSONL、含 rows 的 JSON 或单条 JSON 记录。"""

    source = path / "frame_metrics.jsonl" if path.is_dir() else path
    if source.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return [dict(row) for row in payload["rows"]]
    return [dict(payload)]


def _normalize_method(rows: Iterable[dict], method: str) -> list[dict]:
    """把历史运行名映射为必要性报告的三个固定方法名。"""

    aliases = {
        "frozen": "Frozen Offline NN",
        "pilot_sgd": "Pilot-SGD",
        "pilot": "Pilot-SGD",
        "meta": "Meta-Pilot",
        "online_meta": "Meta-Pilot",
    }
    selected = aliases.get(str(method), str(method))
    normalized = []
    for row in rows:
        item = dict(row)
        item["method"] = selected
        normalized.append(item)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总元学习在线适配必要性实验")
    parser.add_argument("--frozen", required=True, type=Path)
    parser.add_argument("--pilot-sgd", required=True, type=Path)
    parser.add_argument("--meta", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    rows.extend(_normalize_method(_read_rows(args.frozen), "frozen"))
    rows.extend(_normalize_method(_read_rows(args.pilot_sgd), "pilot_sgd"))
    rows.extend(_normalize_method(_read_rows(args.meta), "meta"))
    report = build_meta_necessity_report(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
