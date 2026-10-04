# Pilot 驱动的均衡器内部 PEFT 在线微调路线

## 目的

本路线保持已有离线模型和 checkpoint 不变。在线阶段只使用前缀中的 Adapt Pilot
计算自监督 BCE 梯度，更新均衡器内部的受限参数；Reward Pilot 不参与梯度计算，
只负责更新后验收和失败回滚；Data 标签只在最终 BER 统计时读取。

当前建议的第一候选是：

```text
OnlinePhaseTrendAdapter + head
```

其中 `OnlinePhaseTrendAdapter` 是运行时挂载的两个有界参数，用于补偿接收 IQ 的
公共相位和线性相位趋势。它不修改 CIR，不替换 `CIRCondition`，也不读取真实的
Data 符号。`head` 是已有均衡器输出头的参数组。两组都由
`PilotDrivenOnlineAdapter` 用 Adapt Pilot 更新。

## 与状态恢复的边界

纯 PEFT 配置使用 `online_condition_source=acquisition` 和 `cir-update=fixed`。
Frozen 与 Online 因此共享同一个 acquisition 条件；Online 只多出
`phase_trend/head` 参数更新。`cir_update_applied` 必须为 `false`，否则该结果应
被标记为状态恢复或联合消融，不能写成纯参数微调收益。

需要使用当前帧 Pilot 估计 CIR 或 phase 条件时，单独运行 `pilot_sparse` 配置，
并将其标记为 `State + PEFT` 消融。该消融的收益不能归因给 Adapter/head。

## Reward Pilot 守门

固定调度器下每个候选配置只有一个更新候选，Reward Pilot 执行以下检查：

1. 更新后窗口 BCE 达到最小改善阈值；
2. 每个 Reward Pilot 子窗口都没有恶化；
3. Reward Pilot 硬判决错误数不增加；
4. 跨帧发现上一轮更新造成恶化时恢复上一轮参数快照。

任一检查失败都恢复更新前的 Adapter/LoRA/head 参数。结果字段
`reward_pilot_guard_only=true`、`data_labels_used_online=false` 和
`adapt_pilot_only=true` 用于审计信息边界。

## 当前配置

配置文件为 [`configs/online_peft_phase_trend_head.json`](/D:/Research/RL4EQ/configs/online_peft_phase_trend_head.json)。
它只改变在线字段，模型维度和离线 checkpoint 仍由 checkpoint 的
`model_config.json` 严格决定。推荐先用以下命令做短 replay：

```powershell
.\.venv-gpu\Scripts\python.exe compare.py `
  --config configs/online_peft_phase_trend_head.json `
  --methods "Frozen Offline NN" "Pilot-Driven Online Adaptation" `
  --pretrained pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt `
  --delays 116 --snrs 0 5 10 15 --num-seeds 3 --frames 12 `
  --pilot-total 256 --reward-pilot-total 32 --pilot-layout prefix `
  --scheduler fixed --cir-update fixed --online-freeze-below-snr-db -1 `
  --update-interval 1 `
  --output-dir logs/online_peft_phase_trend_head_3s12f
```

这里的 `-1` 只是筛选阶段让 0 dB 也进入同一套 Reward Pilot 守门；正式主矩阵可以
保留配置中的 `5 dB` 低 SNR 保护，并单独报告 0 dB 的冻结行为。

正式矩阵前必须检查：

- 每个 SNR 的 `peft_update_applied` 和 `online_parameter_delta_norm` 是否真实出现；
- `cir_update_applied` 是否始终为 `false`；
- `data_labels_used_online` 是否始终为 `false`；
- Frozen 与 Online 是否使用同一 checkpoint、轨迹、tail 更新和 acquisition 条件；
- 配对收益是否跨 seed 同方向，而不是只在单个 seed 或少数帧出现。

## 当前统一 replay 结果

在同一 checkpoint、同一 acquisition 条件、固定调度、3 seeds × 12 frames 的统一
replay 中，`phase_trend + head` 相对 Frozen 的配对结果为：

| SNR | Frozen BER | Online BER | 配对收益 | seed 方向 | 保留更新 |
|---:|---:|---:|---:|---:|---:|
| 0 dB | 50.8500% | 49.7938% | +1.0561 pp | 3/3 | 32/36 |
| 5 dB | 53.6314% | 48.8680% | +4.7635 pp | 3/3 | 33/36 |
| 10 dB | 55.0998% | 50.1953% | +4.9045 pp | 3/3 | 36/36 |
| 15 dB | 56.1994% | 49.3236% | +6.8757 pp | 3/3 | 34/36 |

这是 acquisition 状态失配下的参数微调增量证据：CIR 更新数为 0，三个 seed 在每个
SNR 都同方向改善，且每帧参数组固定为 `head + phase_trend`、共 67 个参数。绝对
BER 仍接近随机猜测，说明该设置适合验证“在线参数是否超过同一离线模型”的因果
问题，不足以作为系统性能结论。正式结论仍需要至少 5 seeds × 60 frames，并同时
报告配对置信区间、最坏 seed、更新接受率和回滚率。

此前 `adapter + head`、LoRA 和 RLS 残差探针出现过“参数确实变化但 Data BER
不变”的情况。原因是该 checkpoint 的物理 warm-start 分支占主导，普通 Adapter
和 LoRA 主要改变很小的神经残差；因此不能只看 Adapt loss 降低或参数范数增加，
必须以 Reward Pilot 守门后的 Data 配对 BER 作为最终判断。

## 5 seeds × 60 frames 正式结果

在同一 checkpoint、同一 acquisition 条件、固定调度和 `cir-update=fixed` 下，完成了
Level B、116-symbol 延迟、prefix Pilot=256、Reward Pilot=32 的 5 seeds × 60 frames
矩阵。收益定义为 `BER_frozen - BER_online`，正值表示在线更好。95% 区间按 5 个 seed
重采样；每个 seed 内保留连续帧序列。

| SNR | Frozen BER | Online BER | 配对收益 | 95% CI | seed 方向 | 保留更新 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 dB | 50.3160% | 49.0907% | +1.2253 pp | [+1.0694, +1.3811] pp | 5/5 | 272/300 |
| 5 dB | 50.5356% | 46.3294% | +4.2062 pp | [+3.9071, +4.5052] pp | 5/5 | 266/300 |
| 10 dB | 50.6324% | 46.1918% | +4.4405 pp | [+3.9449, +4.9371] pp | 5/5 | 276/300 |
| 15 dB | 50.7018% | 47.0100% | +3.6918 pp | [+2.6992, +4.7062] pp | 5/5 | 260/300 |

全矩阵共 1200 个 Online 帧，其中 1074 帧保留了 PEFT 更新，682 帧触发了跨帧回滚，
1105 帧的 Data BER 判决发生变化。所有记录均满足：

```text
condition_source = acquisition
cir_update_applied = false
online_condition_update_applied = false
online_parameter_groups = [head, phase_trend]
online_parameter_count = 67
adapt_pilot_only = true
reward_pilot_guard_only = true
data_labels_used_online = false
```

前 30 帧与后 30 帧配对收益分别为：0 dB `+1.2786 -> +1.1719 pp`、5 dB
`+4.3837 -> +4.0286 pp`、10 dB `+4.6102 -> +4.2708 pp`、15 dB
`+4.6736 -> +2.7101 pp`。因此目前可以确认长期窗口内在线 PEFT 稳定超过同一
Frozen Offline NN，但尚未证明收益随帧数增加而扩大。

本实验的绝对 BER 仍约为 46%–49%，因为纯 PEFT 对照刻意固定了老化的 acquisition
条件；它验证的是“参数微调能否在不更新 CIR/phase 状态的前提下恢复部分性能”，不是
最终系统的最佳 BER。要改善绝对 BER，需要另行报告 Pilot 状态恢复或联合 State+PEFT
消融，不能把那部分收益写成纯 Adapter/head 收益。

## 结论边界

当前已完成一个符合信息隔离契约、并在 5 seeds × 60 frames 上稳定超过同一 Frozen
Offline NN 的在线 PEFT 主路径。尚未完成的是“同时明显超过传统非神经 baseline 且达到
较低绝对 BER”的系统目标；也未证明收益会随在线帧数扩大。下一阶段应保持本路线不变，
分别做传统 baseline 对照和 `State + PEFT` 消融，把参数微调增量与状态恢复增量分开报告。
