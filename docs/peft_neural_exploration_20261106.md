# 神经 PEFT 探索计划（2026-11-06）

本文件是 `codex/online-peft` 分支上"在线神经参数微调"子主张的当前计划；物理 Bayes CIR 的确认结果在 `physical-bayes-cir-v1` 快照分支，不混入本计划。

## 0. 用户对齐的约束

- **两臂共享 acquisition CIR + dd_soft tail**；Bayes CIR 的物理增益不进入神经 PEFT 比较。
- 允许：**不确定性条件化 FiLM/LoRA**、**更大参数集/算力**。
- 不允许：同帧 TTT（同一帧内适配后重新判决）、离线重训/增广。
- 成功判据：5/10 dB 出现**可重复正增量**（例如 ≥1e-4，10 session × 128 帧），0/15 dB 不退化；先 smoke 再扩。
- 聚焦两个机制：**A 不确定性条件化 FiLM/LoRA**、**B ELBO 跨帧低秩 Adapter**。

## 1. 卡点（为什么当前 PEFT 不能实现）

1. CIR 相关头room 已被 Bayes CIR 吃掉；共享 Bayes CIR 后 Adapt-Pilot head 只增加约 1 比特。
2. Adapt Pilot 信息不足：10/15 dB 的 BCE 梯度趋零；Data 软似然 ELBO 方向正相关但幅度小。
3. 同帧 oracle head 是逐帧标签拟合，留出帧上消失；持久静态 head 不迁移。
4. 65 维 head 的 Fisher 有效秩约 1；输出由物理 warm-start 与 CIR 条件主导。
5. 10/15 dB 错误数少，神经增量低于统计分辨率。
6. 神经 head 只能重加权同一接收波形的特征；离线模型在训练/宽度上接近饱和。
7. 内循环 Pilot 观测不能预测 Data 改善；硬门控阻断外层梯度。

**一句话**：会话特定信息主要是 CIR，而 CIR 已被物理跟踪近最优地估计；当前 head 作用点没有剩下可学的会话特定残差。

## 2. 文献结论（v2）

报告：`results/neural_peft_survey_v2_20261106/README.md`。

在"同分布、准静态、线性 FIR、无编码、Bayes CIR 近 oracle"下，神经在线增量的先验接近零：Bayes gap、BCRB、BCJR 最优性、adaptation=offset recovery、校准无出口。最值得尝试的是 query-aligned 不确定性条件化低秩残差、ELBO+可靠符号门控、学习式 R_t/Q/R；建议先做只读诊断。

## 3. 三个只读诊断

报告：`results/peft_query_alignment_v1_20261106/README.md`。

- **Query-vs-prompt**：Adapt Pilot 在 0/5 dB 有可用方向（cos `+0.71/+0.49`，捕获 `81.5%/73.3%`）；10/15 dB 基本没有（cos `+0.002/+0.015`，捕获 `35.7%/27.8%`，Data 方向跨帧稳定性 `0.14/−0.09`）。
- **ELBO**：4 维子空间在四 SNR 捕获 **96%–99%** 的 Data 梯度能量；梯度范数在 10/15 dB 是 Adapt 的约 14×/77×；但逐帧方向不稳定（0/5 dB 稳定性为负），需要低秩持久子空间。
- **0 dB R_t oracle**：frozen `0.1829`、soft `0.1675`、oracle `0.1628`；soft 已接近 oracle，R_t 不是主要瓶颈。

结论：**Adapt Pilot 驱动在 10/15 dB 不可行；ELBO 是唯一在 10/15 dB 有可用 Data 方向的信号；必须用低秩持久子空间，不能逐帧单方向。**

## 4. 机制设计

### 4.1 B：query-aligned 低秩 Adapter + ELBO 跨帧更新

- 两臂共享 acquisition CIR + dd_soft tail；Frozen 的 Adapter 系数 `c=0`。
- 低秩基 `B ∈ R^{65×K}`（K=4–8）由**开发 seed** 的 ELBO 梯度 top-K 子空间得到；不重训离线模型。
- 在线：用当前帧 ELBO 对 `c` 求梯度，Adam/归一化步进 + 信赖域 + Reward Pilot 门控；更新后的 `c` 用于**下一帧**（跨帧，不做同帧 TTT）。
- 成功判据：5/10 dB Frozen/Online 配对正增量，0/15 dB 不退化。

### 4.2 A：不确定性条件化 FiLM/LoRA

- 把 CIR/Data 置信度、符号方差、ELBO 残差统计作为 FiLM/LoRA 的 context；context 缩放 B 的步长或调制基。
- 在 10/15 dB，Adapt Pilot 的方向不可用；A 的更新信号应与 B 共用 ELBO，而不是单独用 Adapt BCE。
- 四臂：Frozen/Online × 固定基/条件化基；先做 A 的 smoke，再与 B 比较。

## 5. 当前状态

- B 的 2 seed × 16 帧 smoke 运行中：基来自 3 开发 seed × 8 帧的 ELBO 梯度，K=4，`lr=0.01`、半径 `0.05`、Reward BCE 门控。
- 1 seed × 4 帧的预 smoke 没有 BER 增益（5 dB 略差 1 比特）；完整 smoke 决定是否扩到 10×128。
- A 尚未实现；若 B 的完整 smoke 无正增量，A 的 FiLM 仍值得一次 smoke，但文献先验不支持大增益。

## 6. 证据边界

- smoke 是开发机制筛查，不是确认；eval seed 120001–120002 与基 seed 110001–110003 隔离。
- 诊断在 65 维 head 空间；更大的 Adapter/LoRA 参数空间可能不同，需在 smoke 中复测。
- ELBO 子空间捕获高不等于 BER 增益；必须看 Frozen/Online 配对 BER。
- 0 dB 的物理 H3 已不通过；本计划不改变物理确认结论。
