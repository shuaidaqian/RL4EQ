# RL-Modulated Neural Block Equalizer 研究合同 Spec

日期：2026-08-01  
适用仓库：`D:\Research\RL4EQ`  
状态：已由用户确认的研究方向重对齐 spec  
目标：把 RL4EQ 从“Continual PPO 必须超过 Best Fixed”重对齐为“神经网络 + RL 在线均衡超过传统非神经、非 RL 均衡器”

## 1. 背景与重对齐原因

前一轮正式矩阵显示，修复公平性后的 `Best Fixed` 明显强于当前 Continual PPO。进一步分析表明，`Best Fixed` 并不是普通传统均衡器，而是：

```text
CG block detector
+ BPSK local refinement
+ decision-directed CIR tracking
+ 整帧非因果序列检测
```

因此它应被视为强模型驱动诊断参考，而不是主论文中的传统均衡 baseline。用户明确目标是：

```text
baseline 不加入神经网络；
baseline 不使用 RL；
只有 proposed 方法使用神经网络；
只有 proposed 方法使用 RL 做在线均衡；
论文目标是超过传统均衡器。
```

本 spec 取代旧合同中“PPO 必须超过 Best Fixed / Rule / Bandit”的成功门槛。

## 2. 研究目标

主目标：

```text
在 EME-inspired 极端稀疏长回波单载波 BPSK 信道中，
提出一个离线监督预训练 + 在线 RL 调制的神经块均衡器，
在 0–20 dB 主 SNR 范围内超过传统非神经、非 RL 均衡器，
并在每个主配置达到 BER_data < 0.01。
```

论文贡献聚焦：

1. 极端长时延扩展下传统均衡器的退化分析。
2. Pilot 条件神经块均衡器。
3. RL 在线低维调制神经均衡器。
4. 严格 Adapt / Reward / Data 标签隔离。
5. 与传统非神经、非 RL baseline 的公平比较。

本阶段不声称：

```text
Proposed 方法超过所有可能的非神经强模型驱动序列检测器。
```

## 3. Baseline 分类

### 3.1 主 baseline：传统非神经、非 RL 均衡器

正式主表只包含以下传统 baseline：

```text
LMMSE-FIR
LMS
NLMS
RLS Linear Equalizer
DFE-RLS
SC-FDE-MMSE
```

这些方法必须满足：

```text
不使用神经网络；
不使用 RL；
不使用离线数据驱动训练；
不使用 Data 标签；
只使用 acquisition / Adapt Pilot / 接收信号 / 传统自适应规则。
```

### 3.2 Proposed 内部消融

这些方法不是传统 baseline，而是 proposed 方法族内的消融：

```text
Offline NN only
NN + Fixed Modulation
NN + Rule Modulation
NN + Discrete PEFT Scheduler
RL-Modulated Neural Block Equalizer
```

消融目的：

```text
证明神经网络本身有用；
证明在线调制有用；
证明 RL 比固定/规则调制更有用。
```

### 3.3 Diagnostic reference

以下方法不参与主胜负：

```text
Perfect-CSI Block
Fixed CG-BPSK-DD Block Detector
```

其中：

```text
Fixed CG-BPSK-DD Block Detector = 原 Best Fixed
```

该方法只作为 strong model-based diagnostic reference：

```text
不作为传统 baseline；
不纳入主成功门槛；
不参与主论文胜负判断；
可放入 diagnostic 表、appendix 或方法边界分析。
```

### 3.4 删除或降级项

```text
Contextual Bandit 不作为主 baseline；
Drift-Aware Rule 不作为主 baseline，只可作为 NN + Rule Modulation 消融；
Data Oracle 不恢复；
Best Fixed 名称不再用于主结果。
```

## 4. 信道、SNR 与 Pilot 设置

### 4.1 主信道

第一阶段继续使用干净 Level B，不升级 Level B+。

保留：

```text
Level B 极端稀疏长回波
max_delay = 20 / 30 / 40
单载波 BPSK
线性复多径
AWGN
Gauss-Markov 慢漂移
跨帧 ISI
路径随机相位
episode 内 support 固定
整帧缓冲、非因果块均衡
```

第一阶段不加入：

```text
CFO
额外公共相位扰动
硬件非线性
信道编码
强制强远端回波
support birth/death
delay jitter
传统均衡器额外复杂度限制
```

### 4.2 Future Work / Later Extensions

以下内容写入后续扩展，不进入第一阶段主实验：

```text
CFO：同步误差 / 载波频偏鲁棒性；
额外相位扰动：相位噪声 / 公共相位漂移；
硬件非线性：PA / ADC / IQ imbalance / clipping；
信道编码：LDPC / Turbo / Polar，coded BER / BLER；
support birth/death：非平稳路径结构；
delay jitter：路径时延漂移；
强远端回波：更极端的 late echo profile。
```

这些因素当前不加入的原因是保持论文主线聚焦在：

```text
极端长时延线性复多径下的在线神经/RL 均衡。
```

### 4.3 SNR

主 SNR：

```text
0, 5, 10, 15, 20 dB
```

压力测试：

```text
-5 dB
```

成功门槛：

```text
0–20 dB 主配置要求 BER_data < 0.01；
-5 dB 单独报告，不要求 <0.01。
```

### 4.4 Pilot sweep

第一轮 pilot sweep：

```text
pilot_total = 64, 96, 128, 160, 192
layout = multi_block
Adapt : Reward = 3 : 1
```

对应帧结构：

| pilot_total | Adapt | Reward | Data |
|---:|---:|---:|---:|
| 64 | 48 | 16 | 448 |
| 96 | 72 | 24 | 416 |
| 128 | 96 | 32 | 384 |
| 160 | 120 | 40 | 352 |
| 192 | 144 | 48 | 320 |

选择正式 pilot 的指标：

1. Proposed 是否超过所有传统 baseline。
2. Proposed 是否达到 `BER_data < 0.01`。
3. `effective_goodput`。
4. Reward/Data 相关性。
5. 最坏配置和最坏 seed。

第二轮再做布局消融：

```text
prefix
two_block
multi_block
```

## 5. Proposed 方法架构

正式方法名称：

```text
RL-Modulated Neural Block Equalizer
```

### 5.1 离线神经块均衡器

输入：

```text
rx frame
region ids
Adapt Pilot symbols / mask
pilot-conditioned features
```

输出：

```text
全帧 bit logits / probabilities
```

模型内部需要预留可调制模块：

```text
Adapter gates
FiLM residual correction
LoRA scales
Head temperature / bias
confidence gate
```

离线训练：

```text
Level A -> Level B curriculum supervised pretraining
全程 Pilot 条件监督训练
Data 标签只用于离线监督
```

### 5.2 在线 RL 调制

在线每帧流程：

```text
1. 接收 Adapt Pilot + Reward Pilot + Data。
2. 神经均衡器用当前调制向量 z_old 输出 logits_before。
3. 构造 RL observation。
4. PPO 输出低维连续调制向量 z_new。
5. z_new 直接调制神经均衡器 Adapter / FiLM / LoRA / Head。
6. 调制后的神经均衡器输出 logits_after。
7. Reward Pilot loss_before - loss_after 作为 reward。
8. Data BER 只用于评估。
9. PPO 在部署期间持续更新 policy。
```

### 5.3 RL 动作向量 z

`z` 是低维连续向量，不超过 16–32 维。

候选维度：

```text
adapter_gate_1 ... adapter_gate_L
film_residual_scale
lora_scale_1 ... lora_scale_K
head_temperature
head_bias
confidence_threshold
rollback_gate
```

约束示例：

```text
adapter_gate ∈ [0, 2]
film_residual_scale ∈ [-0.5, 0.5]
lora_scale ∈ [0, 2]
temperature ∈ [0.5, 2.0]
bias ∈ [-1, 1]
```

明确不做：

```text
RL 不逐 bit 判决；
RL 不直接输出完整高维 Δθ；
第一阶段 RL 不控制 CG / BPSK refinement / Fixed CG-BPSK-DD 类检测器。
```

### 5.4 可选 hybrid / 消融

第一阶段主方法是直接低维调制。

可选消融：

```text
NN + Discrete PEFT Scheduler
NN + direct modulation + 1-step Adapt Pilot gradient update
```

这些不是主方法。

## 6. 数据隔离与 reward

### 6.1 标签隔离

严格规则：

```text
Adapt Pilot 标签：
  可用于 conditioner、observation、在线更新。

Reward Pilot 标签：
  不进入动作前 observation；
  不用于梯度更新；
  只用于动作后 reward、last-good、rollback 判定。

Data 标签：
  只用于离线监督和仿真 BER 评估；
  不进入在线 observation / reward / action / 参数更新。
```

### 6.2 Reward

基础 reward：

```text
reward =
  RewardPilotLoss_before - RewardPilotLoss_after
  - λ_action * ||z_new - z_old||²
  - λ_instability * instability_penalty
```

可记录但不用于 reward：

```text
BER_data
Data loss
Data labels
```

### 6.3 Last-good / rollback

允许使用：

```text
Reward Pilot 改善；
非有限 loss / NaN / Inf；
动作幅度过大；
logits 崩溃。
```

禁止使用：

```text
Data BER 改善；
Data label loss。
```

## 7. 实验矩阵

### 7.1 Pilot sweep

目的：

```text
选择正式 pilot_total。
```

建议矩阵：

```text
delay = 20, 40
SNR = 0, 10, 20
pilot_total = 64, 96, 128, 160, 192
layout = multi_block
seeds = 3
frames = 200 或 300
```

方法：

```text
Traditional baselines:
  LMMSE-FIR
  LMS
  NLMS
  RLS Linear
  DFE-RLS
  SC-FDE-MMSE

Proposed family:
  Offline NN only
  NN + Fixed Modulation
  NN + Rule Modulation
  RL-Modulated Neural Block Equalizer
```

### 7.2 正式主矩阵

选择 1–2 个 pilot 设置后：

```text
delay = 20, 30, 40
SNR = 0, 5, 10, 15, 20
seeds = 10
frames = 1000
layout = selected layout
```

方法：

```text
LMMSE-FIR
LMS
NLMS
RLS Linear
DFE-RLS
SC-FDE-MMSE
Offline NN only
NN + Fixed Modulation
NN + Rule Modulation
RL-Modulated Neural Block Equalizer
```

### 7.3 压力测试

```text
SNR = -5 dB
delay = 20, 30, 40
seeds = 5 或 10
frames = 1000
```

不纳入主成功门槛。

### 7.4 Diagnostic

单独报告：

```text
Perfect-CSI Block
Fixed CG-BPSK-DD Block Detector
```

不参与主胜负。

## 8. 成功标准

### 8.1 主目标成立

每个主配置：

```text
BER_data(RL-Modulated NN)
<
min BER_data(所有传统 baseline)
```

传统 baseline：

```text
LMMSE-FIR
LMS
NLMS
RLS Linear
DFE-RLS
SC-FDE-MMSE
```

### 8.2 绝对性能达标

每个 0–20 dB 主配置：

```text
BER_data(RL-Modulated NN) < 0.01
```

### 8.3 RL 贡献成立

做配对统计：

```text
RL-Modulated NN
vs Offline NN only
vs NN + Fixed Modulation
vs NN + Rule Modulation
```

目标：

```text
总体显著改善；
关键困难配置改善；
不要求每个配置都超过所有神经消融。
```

### 8.4 统计方式

必须输出：

```text
逐配置 mean / std / 95% CI
按 seed 配对比较
最坏 seed
最坏配置
effective_goodput
Reward/Data correlation
```

不能只报告总体平均。

## 9. 文档和代码需要同步修改

### 9.1 文档

需要更新：

```text
AGENTS.md
开发框架.md
RL信道均衡研究分析.md
README.md
后续迭代记录
```

必须删除或改写：

```text
Best Fixed 是主成功门槛；
PPO 必须超过 Bandit / Rule；
Best Fixed gate 是正式前置门槛；
Contextual Bandit 是主 baseline。
```

### 9.2 代码

需要调整：

```text
agent/continual_policy.py
agent/unfolded_equalizer.py
agent/adaptation_controller.py
training/continual_ppo.py
compare.py
baseline/
configs/
tests/
```

新增或重构：

```text
agent/rl_modulator.py
agent/modulated_equalizer.py 或在 unfolded_equalizer 中增加 modulation 接口
training/rl_modulated_online.py
baseline/traditional_equalizers.py
evaluation/pilot_sweep.py
tests/test_rl_modulated_equalizer.py
```

### 9.3 Compare 输出

主 compare 应支持：

```text
--method-group traditional
--method-group proposed
--method-group diagnostic
--pilot-total
--pilot-layout
--snrs -5 0 5 10 15 20
```

## 10. 不做项

第一阶段不做：

```text
CFO
额外相位扰动
硬件非线性
信道编码
MIMO
RIS
多调制
Data oracle
逐 bit RL 判决
完整高维 Δθ 生成
Fixed CG-BPSK-DD 作为主 baseline
Bandit 作为主 baseline
```

## 11. Spec 落地顺序

本 spec 只定义研究合同，不直接实施代码。后续实施应单独写计划，建议顺序：

1. 更新项目文档，统一新研究合同。
2. 重命名并重分层 baseline。
3. 实现传统 baseline method group。
4. 给神经均衡器增加 modulation 接口。
5. 实现 RL continuous modulation policy。
6. 实现 pilot sweep。
7. 跑 smoke 与小矩阵。
8. 冻结 pilot 设置。
9. 跑正式主矩阵。
10. 再决定是否进入 future work。

