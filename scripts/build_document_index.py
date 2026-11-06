# -*- coding: utf-8 -*-
"""生成全仓文档索引与取代关系。

索引只读取文件路径，不改写任何历史文档。当前规范以 research_route_v4
为准；所有旧文档保留原语义，由本索引标注状态。
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUTPUT = DOCS / "document_index_20261106.md"

CURRENT = {
    "docs/research_route_v4_20261106.md",
    "docs/document_index_20261106.md",
    "docs/protocols/research_route_v4_20261106.json",
    "docs/protocols/confirmation_criteria_v4_draft_20261106.md",
}

SUPERSEDED = {
    "docs/research_route_v3_20261103.md",
    "docs/protocols/research_route_v3_20261103.json",
    "docs/protocols/confirmation_criteria_draft_20261105.md",
}

CURATED = [
    ("当前规范", "docs/research_route_v4_20261106.md", "唯一当前路线、自由度、方法与阶段"),
    ("当前规范", "docs/protocols/research_route_v4_20261106.json", "机器可读路线状态"),
    ("当前规范", "docs/protocols/confirmation_criteria_v4_draft_20261106.md", "确认判据草案（未冻结）"),
    ("当前规范", "docs/document_index_20261106.md", "本索引"),
    ("进展总表", "docs/online_adaptation_progress_20261106.md", "在线微调阶段进展与失败路线"),
    ("离线基线", "docs/offline_equalizer_training_and_baseline_20261105.md", "离线模型与传统基线主矩阵"),
    ("P0 结果", "docs/results/tail_alignment_v1_20261106/README.md", "soft-tail 训练/评估对齐"),
    ("统计功效", "docs/results/power_error_composition_20261106/README.md", "MDE、session 数与错误构成"),
    ("方法调研", "docs/results/method_survey_20261106/README.md", "在线适配机制与文献"),
    ("离线饱和", "docs/results/offline_saturation_20261106/README.md", "oracle 上界与饱和验证（子 agent，待归档）"),
    ("近期机制", "docs/results/rank1_state_filter_dev_v1_20261106/README.md", "rank-1 状态滤波开发筛查"),
    ("近期机制", "docs/results/full_head_softber_train32_v1_20261106/README.md", "soft-BER 全 head trajectory"),
    ("近期机制", "docs/results/subspace_v27_checkpoint_replay_v1_20261105/README.md", "v27 冻结 checkpoint 复核"),
]

SUPERSESSION = [
    ("docs/research_route_v3_20261103.md", "docs/research_route_v4_20261106.md", "当前路线"),
    ("docs/protocols/research_route_v3_20261103.json", "docs/protocols/research_route_v4_20261106.json", "机器可读状态"),
    ("docs/protocols/confirmation_criteria_draft_20261105.md", "docs/protocols/confirmation_criteria_v4_draft_20261106.md", "确认判据草案"),
    ("docs/results/**/README.md（v14–v27）", "docs/research_route_v4_20261106.md", "保留原语义；不按 v4 追认成功"),
    ("pilot_only tail 协议", "configs/eme_long_memory_v2_online_dd_tail_v1.json（dd_soft）", "新实验用 dd_soft；旧结果用 pilot_only 复现"),
    ("旧 PPO/RL、短/中记忆 profile 计划", "docs/research_route_v4_20261106.md", "历史记录；不是当前默认下一步"),
]


def _classify(rel: str) -> tuple[str, str, str]:
    if rel in CURRENT:
        return ("当前规范", "当前", "唯一事实源/索引")
    if rel in SUPERSEDED:
        return ("已被取代", "取代", "v4 取代；保留原语义")
    if rel.startswith("docs/archive/"):
        return ("历史归档", "历史", "归档快照")
    if rel.startswith("docs/superpowers/"):
        return ("流程文档", "流程", "过程/设计记录")
    if rel.startswith("docs/results/"):
        return ("结果归档", "结果", "原语义；不按 v4 追认")
    if rel.startswith("docs/protocols/"):
        return ("协议/判据", "历史", "旧协议；确认前以 v4 草案为准")
    name = Path(rel).name
    if any(token in name for token in (
        "_plan_", "_stage_", "audit", "progress", "route", "baseline",
        "implementation", "validation", "criteria", "findings", "report",
    )):
        return ("路线/计划/进展", "历史", "开发记录；当前规范以 v4 为准")
    return ("其他文档", "历史", "保留原语义")


def main() -> None:
    files = sorted(
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in DOCS.rglob("*")
        if path.is_file()
        and (path.suffix == ".md" or (path.parent == DOCS / "protocols" and path.suffix == ".json"))
    )
    grouped: dict[str, list[tuple[str, str, str]]] = {}
    for rel in files:
        category, status, note = _classify(rel)
        grouped.setdefault(category, []).append((rel, status, note))

    lines = [
        "# 全仓文档索引（2026-11-06）",
        "",
        "本索引由 `scripts/build_document_index.py` 生成。它只读取路径，不改写历史文档。",
        "",
        "- 当前规范：[research_route_v4_20261106.md](research_route_v4_20261106.md)。",
        "- 旧文档保留原语义；被取代关系见下表，不按 v4 追认旧实验成功。",
        "- 结果归档的数值与结论以各自 README 为准；本索引只标注状态。",
        "",
        "## 1. 当前规范与关键入口",
        "",
        "| 类别 | 文档 | 用途 |",
        "|---|---|---|",
    ]
    for category, rel, note in CURATED:
        if (ROOT / rel).exists():
            lines.append(f"| {category} | [{rel}]({rel.replace('docs/', '')}) | {note} |")
        else:
            lines.append(f"| {category} | {rel}（待归档） | {note} |")

    lines += [
        "",
        "## 2. 取代关系",
        "",
        "| 旧文档/旧协议 | 当前替代 | 说明 |",
        "|---|---|---|",
    ]
    for old, new, note in SUPERSESSION:
        lines.append(f"| {old} | {new} | {note} |")

    lines += ["", "## 3. 全部文档（按类别）", ""]
    order = ["当前规范", "已被取代", "路线/计划/进展", "协议/判据", "结果归档", "历史归档", "流程文档", "其他文档"]
    for category in order:
        entries = grouped.get(category, [])
        if not entries:
            continue
        lines += [f"### {category}（{len(entries)}）", "", "| 文档 | 状态 | 说明 |", "|---|---|---|"]
        for rel, status, note in entries:
            lines.append(f"| [{rel}]({rel.replace('docs/', '')}) | {status} | {note} |")
        lines.append("")

    lines += [
        "## 4. 维护规则",
        "",
        "1. 新增结果归档使用 `docs/results/<name>_<date>/README.md`，写目的、协议、命令、数值、边界与指纹。",
        "2. 新的路线/判据变更先改 v4 单一事实源，再在结果归档中引用；不静默改写旧结果。",
        "3. 旧文档只加取代标注，不删除、不改写数值；索引由脚本重新生成。",
        "4. 正式确认前冻结判据与 checkpoint；开发集不得反复当作确认集。",
        "",
    ]
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUTPUT} with {len(files)} entries")


if __name__ == "__main__":
    main()
