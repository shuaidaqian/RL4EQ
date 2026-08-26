# -*- coding: utf-8 -*-
"""JSONL 分片写出和 resume 去重工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


KEY_FIELDS = ("method", "level", "delay", "snr", "rho", "pilot", "layout", "seed", "frame")


class ShardWriter:
    """按帧追加 JSONL，并支持从已完成 key 跳过。"""

    def __init__(self, path: str | Path, flush_interval: int = 32):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_interval = flush_interval
        self._handle = self.path.open("a", encoding="utf-8")
        self._pending = 0

    def write(self, row: dict[str, Any]) -> None:
        self._handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._pending += 1
        if self._pending >= self.flush_interval:
            self.flush()

    def flush(self) -> None:
        self._handle.flush()
        self._pending = 0

    def close(self) -> None:
        self.flush()
        self._handle.close()

    def completed_keys(self) -> set[tuple[Any, ...]]:
        return completed_keys(self.path)


def completed_keys(path: str | Path) -> set[tuple[Any, ...]]:
    file_path = Path(path)
    if not file_path.exists():
        return set()
    keys = set()
    for line in file_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        keys.add(tuple(row.get(field) for field in KEY_FIELDS))
    return keys
