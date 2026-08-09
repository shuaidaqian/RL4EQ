# 迭代记录：reward surrogate、动作空间与长 episode CIR 跟踪

时间戳：2026-08-09 22:10:09

## 本轮目标

用户要求“持续修改直到达到目标”。本轮按门槛式推进，目标拆成三层：

1. reward surrogate 更贴近 Data BER 改善；
2. 动作空间能产生正向 Data BER 改善；
3. 在线 proposed 在 hard Level B 配置中向 `BER_data < 0.01` 推进。

## 本轮主要修改

### 1. 新增窗口级 reward/Data 诊断

新增函数：

- `summarize_windowed_reward_data_correlation()`
- `summarize_windowed_selected_actions()`
- `evaluate_peft_window_candidates()`

关键变化：

- 不再只看单帧 candidate；
- 支持同一 PEFT 动作在窗口内持续更新模型和 soft tail；
- Data BER 仍只用于离线诊断，不进入在线 observation、reward 或动作选择。

### 2. 新增带参数范数惩罚的安全 surrogate

新增 surrogate：

```text
loss_minus_0.005_delta =
  reward_loss_improvement - 0.005 * peft_delta_norm
```

原因：

- 单纯 `reward_loss_delta` 会偏向更大参数扰动；
- 大参数扰动在 Reward Pilot 上 loss 下降更明显，但未必改善 Data BER；
- 加入 `peft_delta_norm` 后，surrogate 更偏向小扰动安全动作。

### 3. 扩展但收缩安全动作表

保留的正式离散动作：

```text
identity
peft_head_light
peft_head_fast
peft_adapter_lora_conservative
peft_adapter_lora_light
peft_adapter_lora_head_light
rollback_identity
```

仍然没有恢复：

```text
deep adapter update
大步数 adapter update
连续高维参数增量
Data Oracle
旧逐 bit PPO / A2C
```

其中新增动作：

- `peft_head_fast`: head 参数，`lr=5e-4, steps=1`
- `peft_adapter_lora_conservative`: adapter/attention_lora/ffn_lora，`lr=5e-5, steps=1`

### 4. 在线窗口 reward 与诊断 surrogate 对齐

窗口 reward 从旧公式：

```text
reward = reward_ber_delta + 0.1 * reward_loss_delta - 0.001 * action_delta_norm
```

改为：

```text
reward = reward_ber_delta + reward_loss_delta - 0.005 * action_delta_norm
```

这样在线 reward 与诊断中的 `loss_minus_0.005_delta` 一致。

### 5. 增加参数范数门控

单帧 safe guard 现在不是只看 Reward loss 是否下降，而是看：

```text
safe_loss_delta =
  reward_loss_before - reward_loss_after
  - 0.005 * peft_delta_norm
```

只有 `safe_loss_delta > 1e-6` 才接受 loss-only PEFT 动作。

### 6. 增加窗口级 rollback

窗口开始时保存 PEFT 参数快照。

窗口结束时如果：

```text
window_reward <= 0
```

则恢复窗口开始的 PEFT 参数，避免坏窗口的 PEFT 漂移污染后续帧。

### 7. 调整 PPO 初始策略先验

旧策略先验偏向 `identity`。

本轮改为偏向安全 PEFT 探索，尤其是：

```text
peft_head_fast
peft_adapter_lora_conservative
```

原因：

- 诊断显示 `peft_head_fast` 在 d30/s10/p64、d40/s10/p128 上更容易产生正向 Data BER 改善；
- adapter 动作保留，但作为次级探索。

### 8. 支持在线 decision-directed CIR update 与 alpha 搜索

`online_train.py` 新增：

```bash
--cir-update fixed|decision_directed
--cir-alpha <float>
```

原因：

- 300 帧长 episode 中 BER 明显漂移；
- 根因是在线 runner 默认 fixed CIR，无法跟踪 episode 内信道增益缓慢变化；
- decision-directed CIR update 能显著降低长 episode BER。

## 测试结果

完整测试通过：

```text
133 passed in 277.16s
```

说明：

- 新增窗口诊断接口通过；
- 新 reward surrogate 通过；
- 新动作表通过；
- online runner、compare 相关契约未破坏；
- Data 标签仍未进入在线 reward/action/PPO。

## 关键实验结果

### 1. 真实窗口诊断：安全 surrogate 有效但 Spearman 仍不稳定

#### d30/s10/p64

使用：

```text
loss_minus_0.005_delta
```

窗口诊断结果：

```text
Spearman = 0.6504
gate_pass = True
selected_mean_data_ber_improvement = 1.395e-4
```

说明该配置上 reward surrogate 具备可用预测性。

#### d40/s10/p128

同样 surrogate：

```text
Spearman = 0.4599
gate_pass = False
selected_mean_data_ber_improvement = 1.628e-4
```

说明动作选择有正收益，但排序相关性仍未达到 `0.6`。

### 2. 扩展动作后，selected action 平均 Data BER 改善为正

扩展动作表后：

#### d30/s10/p64，16 窗口

```text
selected_mean_data_ber_improvement = 6.98e-5
fraction_data_improved = 0.25
```

其中 `peft_head_fast` 的平均 Data BER 改善最高：

```text
mean_data_improvement = 2.79e-4
```

#### d40/s10/p128，16 窗口

```text
selected_mean_data_ber_improvement = 8.14e-5
fraction_data_improved = 0.3125
```

说明当前动作空间不再是“完全无有效动作”，但正收益仍然偏小。

### 3. 32 帧 online smoke

#### d30/s10/p64

```text
Offline-like BER = 0.0237165
Proposed BER     = 0.0236468
Improvement      = 6.98e-5
```

#### d40/s10/p128

```text
Offline-like BER = 0.0283203
Proposed BER     = 0.0276693
Improvement      = 6.51e-4
```

结论：

- safe prior + safe reward + rollback 能产生正向改善；
- 但绝对 BER 仍远高于 `<0.01`。

### 4. 300 帧 long episode：fixed CIR 明显失败

fixed CIR 下：

#### d30/s10/p64

```text
Proposed BER = 0.0611
Offline-like = 0.0615
```

#### d40/s10/p128

```text
Proposed BER = 0.0597
Offline-like = 0.0601
```

解释：

- Proposed 有小幅正改善；
- 但绝对 BER 漂到 0.06；
- 主要原因不是 reward/action，而是 fixed CIR 无法跟踪长 episode 信道增益变化。

### 5. 300 帧 long episode：decision-directed CIR 明显改善但仍未达标

使用：

```text
--cir-update decision_directed
--cir-alpha 0.2
```

#### d30/s10/p64

```text
Proposed BER = 0.01515
Offline-like = 0.01526
Improvement  = 1.12e-4
```

#### d40/s10/p128

```text
Proposed BER = 0.01677
Offline-like = 0.01684
Improvement  = 6.94e-5
```

结论：

- DD-CIR 把 BER 从约 0.06 压到 0.015–0.017；
- 但仍未达到 `<0.01`。

### 6. alpha 搜索

100 帧短实验：

#### d30/s10/p64

```text
alpha=0.05  BER=0.01792
alpha=0.10  BER=0.01516
alpha=0.30  BER=0.01299
alpha=0.40  BER=0.01288
alpha=0.60  BER=0.01261
alpha=0.80  BER=0.01266
```

最佳约为：

```text
alpha = 0.6
BER   = 0.01261
```

#### d40/s10/p128

```text
alpha=0.05  BER=0.01927
alpha=0.10  BER=0.01714
alpha=0.30  BER=0.01563
alpha=0.40  BER=0.01510
alpha=0.60  BER=0.01552
alpha=0.80  BER=0.01534
```

最佳约为：

```text
alpha = 0.4
BER   = 0.01510
```

### 7. pilot_total=160 没有解决问题

100 帧短实验：

```text
d30/s10/p160, alpha=0.6  BER=0.01395
d40/s10/p160, alpha=0.4  BER=0.01625
d40/s10/p160, alpha=0.8  BER=0.01651
```

结论：

- pilot 加到 160 没有把 BER 压到 `<0.01`；
- 当前瓶颈不是 pilot 长度，而是接收机 / CIR 条件在 SNR=10 长 episode 下仍不够强。

## 当前是否达到目标

没有。

本轮达成：

- reward/action 不再完全无效；
- online proposed 对 Offline-like 有小幅正改善；
- fixed CIR 长 episode 失败原因被定位；
- DD-CIR 明显降低 BER；
- 代码测试全通过。

本轮未达成：

- `BER_data < 0.01`
- 稳定 Spearman >= 0.6 覆盖所有 hard 配置
- proposed 大幅超过传统 baseline

## 当前判断

继续只调 PPO reward / 动作空间，已经不是最高收益方向。

目前最强证据是：

1. fixed CIR 下 BER 漂到 0.06；
2. DD-CIR 可降到 0.015 左右；
3. alpha 调优可把 d30 降到约 0.0126；
4. 但 d40 仍卡在约 0.015；
5. pilot=160 无法解决；
6. PEFT/RL 只能在 Offline-like 基础上带来 `1e-4 ~ 6e-4` 量级改善。

因此，下一阶段如果仍坚持 `BER_data < 0.01`，应优先改：

```text
Offline NN / unfolded block detector / CIR conditioner
```

而不是继续堆 PPO。

## 下一步建议

建议下一轮按这个顺序：

1. 把 DD-CIR alpha 纳入正式配置搜索，默认 hard SNR=10 使用 `alpha=0.4~0.6`。
2. 回到 offline 接收机训练：
   - 加强 d30/d40、SNR=10 的 curriculum；
   - 训练时混合 fixed CIR / DD-CIR / noisy CIR；
   - 显式让 conditioner 适配 DD-CIR 误差分布。
3. 若 offline + DD-CIR 能稳定到 `<0.01`，再继续 PPO。
4. 若 offline 仍卡在 `0.012~0.016`，应增强 unfolded detector 本体，而不是继续调 reward。

## 本轮修改文件

- `evaluation/research_diagnostics.py`
- `scripts/diagnose_research_assumptions.py`
- `agent/discrete_safe_policy.py`
- `training/windowed_discrete_ppo.py`
- `online_train.py`
- `tests/test_research_diagnostics.py`
- `tests/test_windowed_discrete_ppo.py`

