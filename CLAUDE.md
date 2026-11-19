# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

本仓库 `bayes-cir` 分支是**物理算法快照**：离线神经均衡器 + 在线贝叶斯 CIR 适配，对照同源 acquisition-CIR Frozen。历史 PEFT/RL/子空间实验在 `peft-experiments` / `route-v4-docs` 分支，本分支不保留。

- 唯一事实源：[研究路线（物理版 v1）](docs/research_route_physical_v1_20261106.md)；工作约定见 [AGENTS.md](AGENTS.md)。
- 开发语境为中文：对话与代码注释使用中文，文档亦为中文。
- 本快照**不宣称在线神经参数微调**；该子主张在开发中未成立。

## 常用命令

所有命令在仓库根目录执行。Windows PowerShell 先设 UTF-8：

```powershell
$env:PYTHONUTF8=1
```

### 环境

```powershell
# Python 3.12 + PyTorch cu128 + requirements-gpu.txt → .venv-gpu/
powershell -ExecutionPolicy Bypass -File scripts/setup_gpu_env.ps1
```

`requirements-gpu.txt` 只含 numpy/scipy/matplotlib/pytest；torch 由 setup 脚本单独从 cu128 源安装。仓库无 `pyproject.toml`/`setup.py`，模块按仓库根目录顶层包导入（`env.`、`training.`、`evaluation.` 等），pytest 必须从根目录运行。

### 测试

```powershell
# 全量（项目约定）
python -m pytest -q -p no:cacheprovider

# 单文件
python -m pytest tests/test_bayesian_cir.py -q -p no:cacheprovider

# 单个用例（node id）
python -m pytest tests/test_bayesian_cir.py::test_data_posterior_is_label_free_and_shrinks_covariance -q -p no:cacheprovider
```

测试多为协议/信息边界契约测试（如 Data 标签隔离、跨帧时序、checkpoint 配置一致性），改动 `training/bayesian_cir.py` 或 `training/pilot_transfer.py` 后必须全量跑。

### 物理确认（独立 seed，本分支配置已冻结）

```powershell
python scripts/diagnose_bayesian_cir.py --config configs/eme_long_memory_v2_online_dd_tail_v1.json --checkpoint D:/Research/02_研究点二_神经均衡/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --seeds 100001 100002 100003 100004 100005 100006 100007 100008 100009 100010 --frames 128 --process-noise 1e-4 --ridge 1e-2 --device cuda --output docs/results/bayesian_cir_confirmation_v1_20261106/confirmation.json
```

**确认集只运行一次**；已用 seed 不得重测或并入确认（完整清单见 AGENTS.md「证据边界」）。

### 其他诊断脚本

| 脚本 | 用途 |
|---|---|
| `scripts/diagnose_0db_data_posterior.py` | Data 后验变体（soft/oracle/EM2/EM3/block_phase）；`--timing {same_frame,causal}` 必须显式选择 |
| `scripts/summarize_bayesian_expanded.py` | 汇总 `diagnose_bayesian_cir.py` 输出；以 session（seed）为统计单位，输出配对 CI、窗口斜率 |
| `scripts/analyze_block_phase_dev.py` | 汇总 block_phase 开发筛查 JSON（只读） |
| `scripts/compare_traditional_baselines.py` | Frozen/Bayes 与传统基线同 episode 公平对照 |
| `scripts/profile_bayes_cost.py` / `profile_bayes_cost_sweep.py` | 逐帧成本分解 / warm-start+AMP 扫描 |
| `scripts/diagnose_tail_protocol_alignment.py` | pilot_only vs dd_soft tail 协议对齐 |
| `scripts/diagnose_offline_saturation.py` | 离线网络饱和与可修复空间 |
| `scripts/build_document_index.py` | 重建 `docs/document_index_20261106.md`（只读路径） |

所有诊断脚本都是只读的：Data 标签只用于事后 BER，真实 CIR 只用于事后 NMSE。

### 清理脚本（危险）

`scripts/clean_local_artifacts.ps1` 会删除 `pretrained/`、`logs/`、`artifacts/`、`.pytest_cache` 等。**`pretrained/` 是 gitignore 的 checkpoint 目录且不可从仓库恢复**，除非确认可重训，不要运行。

## 架构总览

### 数据流（物理管线）

```
configs/eme_long_memory_v2_online_dd_tail_v1.json
  └─ env/experiment_config.build_comm_env_config
       └─ env/comm_env.CommunicationEnvironment
            ├─ env/extreme_delay_channel.ExtremeDelayChannel  (EME 长记忆物理信道)
            └─ env/frame_structure.FrameGenerator             (整帧 BPSK + Pilot mask)
  ↓  episode: acquisition 帧 → 逐帧 Frame
training/pilot_transfer.make_episode  ── acquisition CIR/CFO 估计（只用已知 acquisition 符号）
  ↓
每帧循环（scripts/diagnose_bayesian_cir.py 等）：
  frozen_context = make_context(frame, frozen_cir, ...)   # 固定 acquisition CIR
  bayes_context  = make_context(frame, bayes_cir, ...)    # 上一帧后验 CIR
  logits ← context_logits(model, context, visible="data")
  BER ← Data mask 上的硬判决（仅事后评分）
  tail ← tail_from_logits(logits, tail)                   # dd_soft：末尾 max_delay 个软判决
  bayes_cir, cov ← pilot_posterior(frame, ...)            # Adapt Pilot 已知符号
  bayes_cir, cov ← data_posterior(frame, logits, ...)     # Data soft 判决 + 符号方差传播
  cov ← add_process_noise(cov)
```

- 两臂共享同一 checkpoint、同一 dd_soft tail 规则、同一波形；唯一差异是 CIR 是否递归更新。
- `training/bayesian_cir.py` 是 Bayes 递推核心：`acquisition_prior` → `pilot_posterior` → `data_posterior` → `add_process_noise`。Data 行噪声为 `sigma^2 + Σ_l |h_l|^2 Var_{t-l}`，低 SNR 下错误符号自动降权。
- 帧结构：1024 符号、`max_delay=116`、prefix Pilot 256（Adapt 224 / Reward 32）、BPSK、SNR 0/5/10/15 dB。

### 模型

- `agent/unfolded_equalizer.UnfoldedEqualizer`：物理 H/Hᴴ 展开迭代 + 3 层非因果 Transformer denoiser + CIR/pilot conditioner；`agent/` 下还有 adapter 与 PEFT 注册表。
- 离线权重冻结；在线适配器（phase 等）只在动态挂载时存在，不进入离线 checkpoint。本快照的在线动作只到 CIR/tail/相位估计，不微调神经参数。

### 关键模块职责

- `env/`：信道 profile、帧结构、在线状态划分（`offline_train`/`heldout_edge`/`drift`）、配置构建与维度校验。
- `training/`：`bayesian_cir.py`（Bayes 递推）、`pilot_transfer.py`（episode/context/tail 公共工具 + 历史训练器）、`curriculum.py`（离线课程预训练）。
- `evaluation/`：`pilot_transfer._load_branch` 是加载 checkpoint 的规范入口，会拒绝与配置不一致的模型结构；其余为指标、bootstrap、协议审计。
- `baseline/`：传统均衡器（LMMSE-FIR、DFE-RLS、SC-FDE-MMSE）。
- `scripts/`：所有实验/诊断的 CLI 入口。

### 不可放松的信息边界

在线只允许读取：已接收波形、Adapt Pilot 已知符号、模型 soft 输出、由它们驱动的 CIR/CFO/相位/tail 估计。

禁止：Data 标签、真实 CIR/相位/CFO、Reward Pilot 回流内循环、未来帧标签。Data 标签只事后评分；Reward Pilot 只做动作后验收。改动管线时若引入上述任一量，测试会失败且结论作废。

## 开发与结果约定

- 结果写入 `docs/results/<主题>_<日期>/`，附 `README.md` 说明协议、seed、配置与状态（开发筛查 / 确认 / 负结果），并归档原始 JSON 与 run.log。
- **开发结果不等于确认**；文档中不得把开发筛查写成确认。已确认的独立确认集只跑一次。
- 两种时序不得混合：`causal`（本帧判决早于本帧后验更新，与已确认结论一致）与 `same_frame`（同帧重解码，BER 使用同帧更新后的 CIR）。same_frame 结果必须单独标注。
- 0 dB 的判据是「早期收敛 / 后段饱和 + 晚窗为正」；只有 5/10/15 dB 保留增长假设。不要用旧增长判据重写 0 dB 结论。
- 新增/修改统计汇总时以 **session（seed）** 为统计单位，不是帧或 bit。
- 成本现状：模型前向 ≈400 ms/帧（tail 复用后总 ≈435 ms/帧），102.4 ms 帧预算下**不实时**；不要声称实时。
- 当前所有确认命令中的 checkpoint 路径是绝对路径 `D:/Research/02_研究点二_神经均衡/pretrained/...`；`pretrained/` 不在 git 中，迁移仓库后需同步更新文档与命令。

## Agent skills

### Issue tracker

Issue 和 spec 作为 GitHub issues 存放在 `shuaidaqian/RL4EQ`，通过 `gh` CLI 操作。见 `docs/agents/issue-tracker.md`。

### Domain docs

单上下文（single-context）：根目录 `GLOSSARY.md` + `docs/adr/`。见 `docs/agents/domain.md`。
