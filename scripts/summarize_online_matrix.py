"""汇总在线 PEFT 矩阵的逐 seed 配对结果。

脚本只读取 compare.py 生成的 frame_metrics.jsonl，不参与训练、更新或评估，
用于把 Frozen Offline NN 与 Pilot-Driven Online Adaptation 按相同 seed 配对。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_rows(log_dir: Path) -> list[dict[str, Any]]:
    path = log_dir / "frame_metrics.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"找不到逐帧日志：{path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_no} 行不是有效 JSON：{path}") from exc
            if isinstance(row, dict):
                rows.append(row)
    if not rows:
        raise ValueError(f"逐帧日志为空：{path}")
    return rows


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(log_dir: Path) -> dict[str, Any]:
    rows = _read_rows(log_dir)
    grouped: dict[tuple[float, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        method = str(row.get("method", ""))
        if method not in {"Frozen Offline NN", "Pilot-Driven Online Adaptation"}:
            continue
        grouped[(float(row["snr_db"]), int(row["seed"]), method)].append(row)

    per_seed: list[dict[str, Any]] = []
    for (snr_db, seed, _), online_rows in sorted(grouped.items()):
        if _ != "Pilot-Driven Online Adaptation":
            continue
        frozen_rows = grouped.get((snr_db, seed, "Frozen Offline NN"), [])
        if not frozen_rows:
            continue
        online_ber = _mean([float(row["ber_data"]) for row in online_rows])
        frozen_ber = _mean([float(row["ber_data"]) for row in frozen_rows])
        accepted = sum(bool(row.get("adaptation_accepted", False)) for row in online_rows)
        rollbacks = sum(bool(row.get("rollback", False)) for row in online_rows)
        labels_used = any(bool(row.get("data_labels_used_online", False)) for row in online_rows)
        per_seed.append(
            {
                "snr_db": snr_db,
                "seed": seed,
                "frames": len(online_rows),
                "frozen_mean_ber": frozen_ber,
                "online_mean_ber": online_ber,
                "improvement_pp": (frozen_ber - online_ber) * 100.0,
                "accepted_updates": accepted,
                "rollback_count": rollbacks,
                "data_labels_used_online": labels_used,
            }
        )

    per_snr: list[dict[str, Any]] = []
    snrs = sorted({float(item["snr_db"]) for item in per_seed})
    for snr_db in snrs:
        items = [item for item in per_seed if item["snr_db"] == snr_db]
        improvements = [float(item["improvement_pp"]) for item in items]
        per_snr.append(
            {
                "snr_db": snr_db,
                "seed_count": len(items),
                "positive_seed_count": sum(value > 0.0 for value in improvements),
                "mean_improvement_pp": _mean(improvements),
                "worst_improvement_pp": min(improvements) if improvements else None,
                "mean_frozen_ber": _mean([float(item["frozen_mean_ber"]) for item in items]),
                "mean_online_ber": _mean([float(item["online_mean_ber"]) for item in items]),
                "data_labels_used_online": any(
                    bool(item["data_labels_used_online"]) for item in items
                ),
            }
        )

    return {"log_dir": str(log_dir), "per_seed": per_seed, "per_snr": per_snr}


def _render_markdown(results: list[dict[str, Any]]) -> str:
    lines = [
        "# 在线 PEFT 矩阵自动汇总",
        "",
        "该表由 `scripts/summarize_online_matrix.py` 从 `frame_metrics.jsonl` 生成。",
        "改善值定义为 `(Frozen BER - Online BER) × 100`，单位是百分点。",
        "",
        "## SNR 汇总",
        "",
        "| 日志目录 | SNR | 正收益 seed | 平均改善 (pp) | 最差 seed 改善 (pp) | Frozen BER | Online BER | 在线使用 Data 标签 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        name = Path(result["log_dir"]).name
        for item in result["per_snr"]:
            lines.append(
                "| `{}` | {} | {}/{} | {:.4f} | {:.4f} | {:.4%} | {:.4%} | {} |".format(
                    name,
                    item["snr_db"],
                    item["positive_seed_count"],
                    item["seed_count"],
                    item["mean_improvement_pp"],
                    item["worst_improvement_pp"],
                    item["mean_frozen_ber"],
                    item["mean_online_ber"],
                    "是" if item["data_labels_used_online"] else "否",
                )
            )
    lines.extend(["", "## Seed 配对明细", "", "| 日志目录 | SNR | seed | Frozen BER | Online BER | 改善 (pp) | 接受更新 | 回滚 |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for result in results:
        name = Path(result["log_dir"]).name
        for item in result["per_seed"]:
            lines.append(
                "| `{}` | {} | {} | {:.4%} | {:.4%} | {:+.4f} | {} | {} |".format(
                    name,
                    item["snr_db"],
                    item["seed"],
                    item["frozen_mean_ber"],
                    item["online_mean_ber"],
                    item["improvement_pp"],
                    item["accepted_updates"],
                    item["rollback_count"],
                )
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总在线 PEFT 矩阵")
    parser.add_argument("log_dirs", nargs="+", type=Path, help="包含 frame_metrics.jsonl 的日志目录")
    parser.add_argument("--json", type=Path, help="可选的 JSON 输出路径")
    parser.add_argument("--markdown", type=Path, help="可选的 Markdown 输出路径")
    args = parser.parse_args()

    results = [summarize(path) for path in args.log_dirs]
    payload = {"results": results}
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    if args.json:
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.write_text(_render_markdown(results), encoding="utf-8")


if __name__ == "__main__":
    main()
