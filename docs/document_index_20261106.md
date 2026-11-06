# 全仓文档索引（2026-11-06）

本索引由 `scripts/build_document_index.py` 生成。它只读取路径，不改写历史文档。

- 当前规范：[research_route_physical_v1_20261106.md](research_route_physical_v1_20261106.md)。
- 旧文档保留原语义；被取代关系见下表，不按 v4 追认旧实验成功。
- 结果归档的数值与结论以各自 README 为准；本索引只标注状态。

## 1. 当前规范与关键入口

| 类别 | 文档 | 用途 |
|---|---|---|
| 当前规范 | [docs/research_route_physical_v1_20261106.md](research_route_physical_v1_20261106.md) | 物理版唯一事实源 |
| 当前规范 | [docs/protocols/confirmation_physical_v1_20261106.md](protocols/confirmation_physical_v1_20261106.md) | 独立确认协议与结果 |
| 当前规范 | [docs/protocols/confirmation_criteria_v4_draft_20261106.md](protocols/confirmation_criteria_v4_draft_20261106.md) | 确认判据草案 |
| 当前规范 | [docs/protocols/research_route_v4_20261106.json](protocols/research_route_v4_20261106.json) | 机器可读状态 |
| 当前规范 | [docs/document_index_20261106.md](document_index_20261106.md) | 本索引 |
| 离线基线 | [docs/offline_equalizer_training_and_baseline_20261105.md](offline_equalizer_training_and_baseline_20261105.md) | 离线模型与传统基线主矩阵 |
| 物理确认 | [docs/results/bayesian_cir_confirmation_v1_20261106/README.md](results/bayesian_cir_confirmation_v1_20261106/README.md) | H1/H2 通过；H3 0 dB 不通过 |
| 物理开发 | [docs/results/bayesian_cir_diag_v1_20261106/README.md](results/bayesian_cir_diag_v1_20261106/README.md) | 3×16 至 10×128 开发筛查 |
| 物理基础 | [docs/results/tail_alignment_v1_20261106/README.md](results/tail_alignment_v1_20261106/README.md) | dd_soft tail 训练/评估对齐 |
| 物理基础 | [docs/results/offline_saturation_20261106/README.md](results/offline_saturation_20261106/README.md) | 训练预算/宽度接近饱和；CIR 是主要残差 |
| 统计功效 | [docs/results/power_error_composition_20261106/README.md](results/power_error_composition_20261106/README.md) | MDE、session 数与错误构成 |
| 方法调研 | [docs/results/method_survey_20261106/README.md](results/method_survey_20261106/README.md) | 在线适配机制与文献 |

## 2. 取代关系

| 旧文档/旧协议 | 当前替代 | 说明 |
|---|---|---|
| 旧 PEFT/RL/子空间实验 | docs/research_route_physical_v1_20261106.md | 保留在 codex/online-peft 与 docs/route-v4-20261106 分支；本快照不保留 |
| pilot_only tail 协议 | configs/eme_long_memory_v2_online_dd_tail_v1.json（dd_soft） | 物理管线使用 dd_soft |
| acquisition CIR only | Bayes CIR 后验 | 物理主张的 Frozen 对照为 acquisition CIR |

## 3. 全部文档（按类别）

### 当前规范（5）

| 文档 | 状态 | 说明 |
|---|---|---|
| [docs/document_index_20261106.md](document_index_20261106.md) | 当前 | 唯一事实源/索引 |
| [docs/protocols/confirmation_criteria_v4_draft_20261106.md](protocols/confirmation_criteria_v4_draft_20261106.md) | 当前 | 唯一事实源/索引 |
| [docs/protocols/confirmation_physical_v1_20261106.md](protocols/confirmation_physical_v1_20261106.md) | 当前 | 唯一事实源/索引 |
| [docs/protocols/research_route_v4_20261106.json](protocols/research_route_v4_20261106.json) | 当前 | 唯一事实源/索引 |
| [docs/research_route_physical_v1_20261106.md](research_route_physical_v1_20261106.md) | 当前 | 唯一事实源/索引 |

### 路线/计划/进展（1）

| 文档 | 状态 | 说明 |
|---|---|---|
| [docs/offline_equalizer_training_and_baseline_20261105.md](offline_equalizer_training_and_baseline_20261105.md) | 历史 | 开发记录；当前规范以 v4 为准 |

### 结果归档（6）

| 文档 | 状态 | 说明 |
|---|---|---|
| [docs/results/bayesian_cir_confirmation_v1_20261106/README.md](results/bayesian_cir_confirmation_v1_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/bayesian_cir_diag_v1_20261106/README.md](results/bayesian_cir_diag_v1_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/method_survey_20261106/README.md](results/method_survey_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/offline_saturation_20261106/README.md](results/offline_saturation_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/power_error_composition_20261106/README.md](results/power_error_composition_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/tail_alignment_v1_20261106/README.md](results/tail_alignment_v1_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |

## 4. 维护规则

1. 新增结果归档使用 `docs/results/<name>_<date>/README.md`，写目的、协议、命令、数值、边界与指纹。
2. 新的路线/判据变更先改 v4 单一事实源，再在结果归档中引用；不静默改写旧结果。
3. 旧文档只加取代标注，不删除、不改写数值；索引由脚本重新生成。
4. 正式确认前冻结判据与 checkpoint；开发集不得反复当作确认集。
