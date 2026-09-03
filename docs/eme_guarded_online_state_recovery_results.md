# Level B 安全在线状态恢复统一矩阵

## 目的与结论范围

本实验是对条件边界修正后在线接收机的完整复核，而不是把历史结果重新命名。它检验以下三个可以分离的来源：

1. 当前帧 prefix Adapt Pilot 相位条件；
2. Pilot-only 稀疏 CIR 更新；
3. Adapt Pilot 驱动、Reward Pilot 决定接受或回滚的 PEFT 更新。

结论必须分开表述：当前 Proposed 在所有主 SNR 配置都显著优于公平传统基线；当前帧 Pilot 相位条件是主要神经收益；本矩阵不支持把现有 PEFT 候选选择写成稳定的长期 Data BER 增益。最后一点是本轮最重要的负结论，决定后续工作必须先改进在线目标的代表性，而不是继续堆叠更新步数或动作数。

## 不可混用的历史协议

本文只报告下列目录的结果：

```text
logs/eme_guarded_unified_main_60f_3s/
```

它与下列历史目录不能合并成主表：

- `logs/eme_unified_boundary_main_60f_3s/`：已修正神经条件边界，但尚未加入 CIR Reward Pilot guard、PEFT 最小改善门限和低置信度 CFO 隔离；
- `logs/eme_snr_freeze_main_60f_3s/` 及更早矩阵：Frozen 的当前 Pilot 信息边界与本轮定义不同，只保留为路线演进诊断。

## 完整实验协议

| 项目 | 固定值 |
|---|---|
| 信道 | `eme_long_memory_v2`，Level B，`max_delay=116`（11.6 ms，符号率 10 ksym/s） |
| 长回波结构 | 12--24 条强径，diffuse energy ratio 0.20--0.35 |
| 跨帧过程 | 相干时间 120 s；acquisition 到首数据帧老化 30 s；60 连续帧 |
| 扰动 | `cfo_phase_tiny` residual CFO 与慢相位扰动 |
| Pilot | prefix 128，Adapt Pilot 96，Reward Pilot 32 |
| SNR | 0 / 5 / 10 / 15 dB |
| 重复 | 每个 SNR 3 个独立 seed，每个方法 180 帧 |
| checkpoint | `pretrained/eme_meta_from_offline_32/model_best.pt` |
| 在线 CIR | acquisition support 上的 `pilot_sparse` 候选，`alpha=0.2` |
| 在线 PEFT | 当前正式配置为 `phase` 组，只由 Adapt Pilot 求梯度；`adapter_lora + conditioner_film` 仅保留为历史兼容方案 |
| CIR 接受规则 | 候选 CIR 不得恶化 Reward Pilot BCE |
| PEFT 接受规则 | Reward Pilot BCE 至少降低 `0.001` |
| 物理 CFO 注入 | 仅 Pilot physical-state confidence `>=0.5` |
| 标签边界 | 在线 observation、CIR、梯度和动作均不读取 Data 标签；Reward Pilot 仅用于候选验收 |

运行命令：

```powershell
$methods=@(
  'Offline NN only',
  'Pilot-conditioned frozen NN',
  'Pilot CIR only',
  'Pilot-Driven Online Adaptation',
  'CFO+DD-Phase LMMSE-FIR',
  'CFO+DD-Phase DFE-RLS'
)

.\.venv-gpu\Scripts\python.exe compare.py `
  --config configs/eme_long_memory_v2.json `
  --methods $methods `
  --pretrained pretrained/eme_meta_from_offline_32/model_best.pt `
  --delays 116 --snrs 0 5 10 15 --num-seeds 3 --frames 60 `
  --pilot-total 128 --pilot-layout prefix `
  --impairment-profile cfo_phase_tiny `
  --cir-update pilot_sparse --cir-alpha 0.2 --update-interval 8 `
  --device cuda --output-dir logs/eme_guarded_unified_main_60f_3s
```

输出共 `6 x 4 x 3 x 60 = 4320` 条逐帧记录，已核验 `frame_metrics.jsonl` 行数为 4320。

## 方法信息边界

| 方法 | 当前 Adapt Pilot 相位 | 当前 Pilot CIR | PEFT | 作用 |
|---|---:|---:|---:|---|
| Offline NN only | 否，使用 acquisition 相位 | 否 | 否 | acquisition 老化压力基线 |
| Pilot-conditioned frozen NN | 是 | 否 | 否 | 当前相位状态恢复的纯贡献 |
| Pilot CIR only | 是 | 裸 `pilot_sparse` 候选 | 否 | 未受 CIR guard 保护的物理更新消融 |
| Pilot-Driven Online Adaptation | 是 | 仅 Reward Pilot 接受 | 仅 Reward Pilot 改善至少 0.001 时接受 | Proposed 安全在线链路 |
| CFO+DD-Phase LMMSE-FIR / DFE-RLS | Pilot 驱动的传统补偿 | 传统规则 | 否 | 公平、非神经、非 RL 基线 |

`Pilot CIR only` 在本次运行中与 `Pilot-conditioned frozen NN` 数值一致。这不是把两条路径合并，而是当前 0.5 物理置信度门限下所有候选 CIR 更新均未被允许执行；完整 Proposed 仍会做受约束 PEFT 候选评估。

## 主结果

表中是 3 seed、60 帧合并的 Data BER；前 5 / 后 5 分别反映初始化段和长期尾段。

| SNR | 方法 | 全 60 帧 | 前 5 帧 | 后 5 帧 |
|---:|---|---:|---:|---:|
| 0 dB | Offline NN only | 50.00% | 51.41% | 48.47% |
|  | Pilot-conditioned frozen NN | **31.89%** | 31.13% | **29.68%** |
|  | Pilot-Driven Online Adaptation | 32.79% | **30.54%** | 31.93% |
|  | 最优传统：DFE-RLS | 45.88% | 47.60% | 41.54% |
| 5 dB | Offline NN only | 50.34% | 53.62% | 50.78% |
|  | Pilot-conditioned frozen NN | 23.20% | 21.65% | 21.87% |
|  | Pilot-Driven Online Adaptation | **22.89%** | **19.78%** | **21.80%** |
|  | 最优传统：DFE-RLS | 40.04% | 32.03% | 35.09% |
| 10 dB | Offline NN only | 50.50% | 54.49% | 51.21% |
|  | Pilot-conditioned frozen NN | **16.55%** | 16.38% | **14.22%** |
|  | Pilot-Driven Online Adaptation | 16.74% | **14.58%** | 18.03% |
|  | 最优传统：LMMSE-FIR | 23.71% | 28.39% | 23.40% |
| 15 dB | Offline NN only | 50.49% | 54.32% | 51.24% |
|  | Pilot-conditioned frozen NN | 14.03% | 12.44% | 15.70% |
|  | Pilot-Driven Online Adaptation | **14.02%** | **12.00%** | **14.82%** |
|  | 最优传统：LMMSE-FIR | 18.25% | 18.83% | 18.78% |

相对该 SNR 下的最优传统基线，完整 Proposed 的 60 帧 BER 降幅分别为：0 dB `28.52%`、5 dB `42.82%`、10 dB `29.37%`、15 dB `23.21%`。因此“在 Level B 极端稀疏、跨越 116 个符号的长回波信道中，Pilot-only 神经接收机比公平传统均衡器更优”已由完整多 seed 主矩阵支持。

严格 Offline NN only 接近 50% 不是神经网络失效的泛化结论。该压力基线禁止读取当前帧 Pilot，而 acquisition 后的 30 s 老化使其冻结相位条件失配。`Pilot-conditioned frozen NN` 恢复当前帧相位条件后，立刻降至 31.89% / 23.20% / 16.55% / 14.03%，这清楚地分离了当前 Pilot 状态恢复的贡献。

## 在线增量的审计结果

相对 `Pilot-conditioned frozen NN` 的逐帧配对差定义为 `BER_frozen - BER_online`，正数代表完整 Online 较优。

| SNR | 60 帧平均差 | Online 较优帧比例 | 前 5 帧差 | 后 5 帧差 |
|---:|---:|---:|---:|---:|
| 0 dB | -0.91 pp | 43.89% | +0.60 pp | -2.25 pp |
| 5 dB | +0.30 pp | 40.56% | +1.88 pp | +0.07 pp |
| 10 dB | -0.19 pp | 45.00% | +1.80 pp | -3.81 pp |
| 15 dB | +0.02 pp | 41.11% | +0.44 pp | +0.88 pp |

这意味着目前安全 Online 链路可以作为 Proposed 的有效完整接收机，但不能声称其现有 PEFT 在每个 SNR、每个时间段均带来独立、稳定收益。

| SNR | physical-state tracking | 接受 CIR 更新 | 接受 PEFT 更新 | physical-state confidence 中位数 / P90 |
|---:|---:|---:|---:|---:|
| 0 dB | 0 / 180 | 0 / 180 | 0 / 180 | 0.000 / 0.000 |
| 5 dB | 180 / 180 | 0 / 180 | 43 / 180 | 0.000 / 0.109 |
| 10 dB | 180 / 180 | 0 / 180 | 47 / 180 | 0.077 / 0.269 |
| 15 dB | 180 / 180 | 0 / 180 | 74 / 180 | 0.115 / 0.372 |

所有置信度均低于 0.5，故这轮矩阵没有把跟踪 CFO 注入神经 conditioner，也没有接受任何 CIR 候选。这证实 0.5 门限确实阻断了先前低置信度状态污染的故障模式，但也说明当前 physical-state estimator 在 96 符号 Adapt Pilot、116 符号长记忆下过于保守。

对于已接受 PEFT 的帧，Reward Pilot loss 的改善与同帧 Data BER 改善的 Pearson 相关系数为 `-0.407`、`-0.425`、`-0.348`（5/10/15 dB）。负相关表明 32 符号 Reward Pilot 对当前候选的排序没有可靠代表整段数据；直接增加 PEFT 学习率、步数或放宽回滚门限都不能解决这个目标错配。

## 论文可写与不可写的表述

可以写：

- 在冻结的 Level B 长回波 profile、30 s acquisition 老化、residual CFO 和慢相位扰动下，Pilot-conditioned neural block equalizer 在所有 0/5/10/15 dB 主配置优于两种传统非神经、非 RL 基线；
- 当前 prefix Pilot 提供的相位条件是维持神经均衡性能的关键在线可见状态；
- Proposed 的 Pilot-only 约束、Reward Pilot 独立验收、低 SNR 冻结和低置信度物理状态隔离均有逐帧审计证据，且在线过程不读取 Data 标签。

不能写：

- 现有 PEFT 微调在该矩阵中稳定超过 Pilot-conditioned frozen NN；
- 当前 sparse CIR 更新是性能来源；本轮为零次接受；
- 32 符号 Reward Pilot 已经是足够可靠的在线代理目标。

## 下一阶段：解决在线研究点的主矛盾

下一步不是加入信道编码、提高调制阶数或继续把 PPO 当作均衡器主体。这些都会掩盖“Pilot-only 在线适配是否泛化到后续数据段”的问题。应以冻结 `Pilot-conditioned frozen NN` 为强对照，完成以下受控路线：

1. 在不改变总 prefix Pilot=128 和信道 profile 的前提下，对 96/32 Adapt/Reward 切分做因果消融，例如 64/64 和 80/48；只要 Reward Pilot 增大后 PEFT 的 Data BER 配对增益稳定，才能将其作为主协议候选。
2. 将 PEFT 的候选动作收缩为离散、安全的更新强度与跨帧更新率，而不是持续使用单一每帧梯度步。动作和接受仍完全由 Pilot 构成，符合 `RL-Modulated Neural Block Equalizer` 契约；Contextual Bandit 是比 PPO 更适合该低维、短时窗口选择问题的首选调度器。
3. 将 Reward 从单帧 32 符号 BCE 扩展为短窗口的留出 Pilot 累积改善，并使用置信区间或一致性门控。这要先做 Pilot-only 的离线 replay 验证：候选选择排序与同帧/后续数据 BER 的关联必须为正，才进入完整在线主矩阵。
4. 物理状态层单独校准：不因提高/降低阈值而改变主表，而是在相同 channel seed 上测量 `phase0/CFO` 误差、置信度校准曲线和 CIR 候选的 Reward Pilot 接受率。若 96 符号无法可靠恢复 116-symbol memory 的 sparse tap，在线主动作应先只调神经 Adapter/FiLM，而不宣称 CIR tracking 成果。

这条路线保留第一研究点的长时延、跨符号稀疏回波信道承接，也把第二研究点聚焦为可验证的 Pilot-only 在线决策与安全更新，而不是把所有当前 Pilot 条件化都泛化为“微调收益”。
