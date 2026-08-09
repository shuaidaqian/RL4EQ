# RL 信道均衡研究分析

## 1. 研究方向重新对齐

在常规无线信道中，LMMSE、DFE、RLS、Kalman tracking 和频域 MMSE 等传统均衡方法已经非常强。若只在短时延或温和多径上使用神经网络，很难形成稳定、可辩护的创新优势。

当前研究转向 EME 启发的极端稀疏长时延扩展场景。这里的 EME 是启发来源，不是完整物理 EME 仿真；研究重点是 20–40 符号相对多径时延、跨帧 ISI 和稀疏长回波，而不是同步前约 2.5 秒的绝对地月传播时延。

## 2. 当前唯一方法

当前唯一 proposed 方法是：

```text
RL-Modulated Neural Block Equalizer
```

它由三部分组成：

- 离线真实信道 Pilot 条件课程监督预训练。
- 整帧缓冲、非因果神经块均衡器。
- 部署期间 PPO 选择离散安全动作，并用窗口级 Reward Pilot 反馈更新策略。

PPO 的职责不是直接判决 Data bit，也不是直接输出完整高维参数增量，而是根据 Adapt/Reward Pilot 反馈选择更合适的神经接收机安全动作。当前不再采用逐帧连续 modulation 作为主路线，因为前期诊断表明逐帧 Reward Pilot 改善与 Data BER 改善相关性不足，连续动作容易扰动已学好的 Offline NN。

## 3. Baseline 边界

用户目标是超过传统均衡器。因此主 baseline 必须是传统非神经、非 RL 方法：

```text
LMMSE-FIR
LMS
NLMS
RLS Linear
DFE-RLS
SC-FDE-MMSE
```

`Perfect-CSI Block` 与 `Fixed CG-BPSK-DD Block Detector` 属于强模型驱动诊断参考，不是主 baseline。它们可以帮助判断信道/检测可达性，但不能作为“传统 baseline”压制 proposed，也不能作为论文主成功门槛。

## 4. Pilot 与标签隔离

全程采用 Pilot 条件监督预训练，原因是在线部署一定依赖 Pilot 条件；离线若长期无 Pilot，会造成训练/部署分布错位。

标签隔离规则固定：

- Adapt Pilot：可用于 conditioner、observation 和在线动作前可观测状态。
- Reward Pilot：只用于动作执行后的 reward 和留出评估。
- Data 标签：只用于离线监督和仿真评估 `BER_data`。
- 在线 observation、reward、动作选择、调制更新不使用 Data 标签，并且不使用数据标签上界。
- Data Oracle 不恢复。

## 5. 成功标准

主论文成功标准：

```text
BER_data(RL-Modulated Neural Block Equalizer) < 0.01
即每个主配置 BER_data < 0.01
并且
BER_data(RL-Modulated Neural Block Equalizer)
  < min BER_data(所有传统非神经、非 RL baseline)
```

该标准必须在 Level B 主配置逐配置成立，不能只看总体平均。

## 6. 当前执行结果与局限

本轮已经补齐工程链路：

- 真实传统 baseline：`LMMSE-FIR/LMS/NLMS/RLS Linear/DFE-RLS/SC-FDE-MMSE`。
- 神经均衡器低维调制接口。
- 离散安全动作 PPO 策略。
- `training/windowed_discrete_ppo.py` 在线 runner。
- `compare.py --method-group main/proposed/traditional/diagnostic`。
- 真实信道 Pilot 条件预训练数据流。
- pretrained checkpoint strict-load。
- append-safe compare smoke。

早期 2-step smoke 只验证链路，不代表训练完成。该 checkpoint 下 proposed 神经 BER 接近随机；后续更长离线训练已经证明 Offline NN 能进入可用区间，但在 9 个 Level B 主配置上仍未稳定达到 `<0.01`。

```text
SC-FDE-MMSE BER_data ≈ 0.00446
DFE-RLS BER_data = 0.0
RL-Modulated Neural Block Equalizer BER_data ≈ 0.509
```

因此当前执行顺序调整为：先把 Offline NN 稳定打到 `<0.01`，再训练窗口级离散 PPO，只有 `RL-Modulated Neural Block Equalizer < Offline NN only` 后才进入正式 pilot sweep 或主矩阵。

## 7. 下一阶段优先级

下一阶段不应优先扩大矩阵，而应优先解决 proposed 接收机可训练性：

1. 提高离线预训练规模，并记录 validation BER。
2. 检查神经输出 bit/logit 符号约定是否与 `bit_error_rate()` 完全一致。
3. 在 Level A 单配置上先要求 Offline NN 明显低于随机 BER。
4. 再进入 Level B 单配置训练。
5. 只有 Offline NN 具备可用性能后，再分析离散安全动作 PPO 是否进一步降低 Reward/Data BER。

在上述条件未满足前，论文不能声称“显著优于传统算法”或“RL 显著优于固定调制”。
