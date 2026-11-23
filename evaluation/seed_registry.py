# -*- coding: utf-8 -*-
"""新实验种子登记：扫描历史结果并拒绝开发/确认重复使用。"""

from __future__ import annotations

import json
from pathlib import Path


HISTORICAL_RANGES = ((80000, 99010), (100001, 100010), (130001, 130003),
                     (140001, 140002), (145001, 145010), (160001, 160201),
                     (170001, 170060))


def observed_seeds(root):
    found = set()

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"seed", "curriculum_seed"} and isinstance(item, int):
                    found.add(item)
                elif key == "seeds" and isinstance(item, list):
                    found.update(x for x in item if isinstance(x, int))
                if isinstance(item, (dict, list)):
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for path in (Path(root) / "docs" / "results").rglob("*.json"):
        try:
            visit(json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, OSError) as exc:
            raise ValueError(f"无法审计历史结果：{path}") from exc
    return found


def reserve_seeds(root, *, experiment, seeds, purpose, details):
    """实验开始前原子登记；失败或中断也不自动释放种子。"""

    root = Path(root)
    path = root / "docs/protocols/continual_v2_seed_registry.json"
    registry = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "schema_version": 1, "experiments": [],
        "historical_reserved_ranges": [list(x) for x in HISTORICAL_RANGES],
    }
    if purpose not in {"development", "confirmation", "training", "validation"}:
        raise ValueError("必须明确实验数据角色。")
    if not seeds or len(set(seeds)) != len(seeds) or any(
        not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in seeds
    ):
        raise ValueError("种子必须是非空、互异、非负整数。")
    existing = next((row for row in registry["experiments"] if row["experiment"] == experiment), None)
    if existing is not None:
        if (existing.get("status") == "reserved" and existing.get("purpose") == purpose
                and existing.get("seeds") == list(seeds)):
            existing["status"] = "running"
            existing["details"] = details
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)
            return path
        raise ValueError("实验名已登记且状态或种子不匹配，禁止隐式重跑。")
    used = observed_seeds(root)
    for row in registry["experiments"]:
        used.update(row["seeds"])
    collisions = [x for x in seeds if x in used or any(a <= x <= b for a, b in HISTORICAL_RANGES)]
    if collisions:
        raise ValueError(f"种子已使用或属于历史保留范围：{collisions}")
    registry["experiments"].append({
        "experiment": experiment, "purpose": purpose, "seeds": list(seeds),
        "status": "running", "details": details,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
