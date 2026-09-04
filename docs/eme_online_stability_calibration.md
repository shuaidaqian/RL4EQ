# EME 在线更新稳定性校准记录

## 1. 校准目的

当前在线研究的主要风险不是离线模型无法工作，而是 Adapt Pilot 上的梯度更新可能只在当前帧降低损失，却在后续帧造成退化。因此本轮固定 Level B 信道、checkpoint、前缀 Pilot 划分和 seed 协议，单独检查更新强度、proximal 约束和物理状态置信度门控。

所有在线更新仍只使用 Adapt Pilot。Reward Pilot 只用于更新后的验收和跨帧回滚，Data 标签只用于实验结束后的统计。

## 2. 新增在线约束

`training/online_adaptation.py` 中的 `PilotDrivenOnlineAdapter` 增加了归一化 proximal trust-region 正则：

```text
loss_adapt = BCE(Adapt Pilot) + proximal_weight * mean((theta - theta_before)^2)
```

正则只作用于本次选中的 PEFT 参数组，不改变普通网络参数，也不改变 Reward Pilot 的验收边界。

`compare.py` 增加了可复现实验参数：

- `--online-learning-rate`
- `--online-steps`
- `--online-max-delta-norm`
- `--online-proximal-weight`
- `--online-min-reward-improvement`
- `--online-cross-frame-tolerance`
- `--online-phase-smoothing`
- `--online-phase-min-confidence`

## 3. 物理状态门控校准

15 dB、64 帧、单 seed 的门限筛选结果如下。Frozen NN 使用相同 checkpoint 作为参照。

| phase 置信度门限 | CIR alpha | Online 平均 BER | 前 16 帧 BER | 后 16 帧 BER | 说明 |
|---:|---:|---:|---:|---:|---|
| 0.15 | 0.20 | 0.119332 | 0.100098 | 0.137207 | 错误 CIR 更新过多，明显退化 |
| 0.30 | 0.05 | 0.082223 | 0.081543 | 0.080636 | 仍有不可信 CIR 写回 |
| 0.40 | 0.05 | 0.074027 | 0.093890 | 0.058943 | 前期退化，末段接近 Frozen NN |
| 0.50 | 0.20 | 0.063825 | 0.069545 | 0.057408 | 保守门控，CIR 基本不写回 |
| Frozen NN | - | 0.063843 | 0.069196 | 0.058594 | 参照 |

因此最终配置保持 `online_phase_tracking_min_confidence=0.5`。当前证据不支持放宽门限；低置信度 Pilot 状态估计虽然数量更多，但估计误差会污染极端稀疏长回波的主径增益。

## 4. 最终在线配置

```text
online_adaptation_groups = ["phase"]
online_adaptation_learning_rate = 0.0004
online_adaptation_steps = 1
online_adaptation_max_delta_norm = 1.0
online_adaptation_proximal_weight = 0.1
online_adaptation_min_reward_improvement = 0.0005
online_adaptation_freeze_below_snr_db = 5.0
online_phase_tracking_min_confidence = 0.5
online_phase_tracking_smoothing = 0.5
update_interval = 8 frames
```

`online_adaptation_candidates` 使用 `lora_conservative`，因此实际候选学习率和范数上限还会乘以候选的 `0.5` 缩放因子。Contextual Bandit 不参与最终配置。

## 5. 修正版 60 帧正式矩阵

协议为 Level B、`max_delay=116`、`0/5/10/15 dB`、3 seeds、60 帧、前缀 Pilot 128、Adapt Pilot 96、Reward Pilot 32、每 8 帧调度一次。结果文件为 `logs/eme_formal_main_60f_phase_stable/`。

| SNR | CFO+DD-Phase LMMSE-FIR | CFO+DD-Phase DFE-RLS | Frozen NN | Online phase |
|---:|---:|---:|---:|---:|
| 0 dB | 0.459536 | 0.458761 | 0.318862 | 0.327933 |
| 5 dB | 0.407676 | 0.400372 | 0.231988 | 0.229136 |
| 10 dB | 0.237054 | 0.260534 | 0.165532 | 0.165625 |
| 15 dB | 0.182515 | 0.187872 | 0.140309 | 0.146323 |

Online phase 在四个主 SNR 层均优于传统非神经均衡器，但仅在 5 dB 和 10 dB 接近或略优于 Frozen NN；不能表述为在线更新在所有 SNR 上带来额外 BER 增益。

## 6. 修正版 200 帧长期诊断

协议为 15 dB、3 seeds、200 帧，其余设置与正式矩阵相同。结果文件为 `logs/eme_long_formal_200f_phase_stable/`。

| 方法 | 前 16 帧 BER | 后 16 帧 BER | 全 200 帧 BER |
|---|---:|---:|---:|
| CFO+DD-Phase LMMSE-FIR | 0.156715 | 0.226539 | 0.209753 |
| CFO+DD-Phase DFE-RLS | 0.183501 | 0.220703 | 0.208890 |
| Frozen NN | 0.126511 | 0.208380 | 0.163631 |
| Online phase | 0.122652 | 0.193522 | 0.169470 |

Online phase 的前段和末段均优于 Frozen NN，表明保守在线更新可以改善初始适配并抑制后期退化；但全程平均 BER 仍略高于 Frozen NN。在线更新共接受 36 次，跨帧回滚 34 次，说明当前机制仍有较多试探性更新，尚不能作为“稳定提升平均性能”的充分证据。

## 7. 研究结论

本轮校准后，第二研究点的可信结论应限定为：在 EME 类极端长记忆、稀疏长回波和慢状态漂移信道下，离线神经均衡器在主配置上显著优于传统均衡器；Pilot 驱动的 phase 在线适配在部分 SNR 和长期末段能够改善或抑制退化，但其相对于 Frozen NN 的全程平均增益尚未稳定成立。

因此当前不引入 Contextual Bandit。后续只有在 Reward Pilot 对跨帧 Data 性能的排序关系稳定通过 replay 门槛后，才有理由把动作调度器作为新的研究变量。

## 8. BER 根因修正后的复核

前述 60/200 帧结果使用了 `acquisition_to_data_gap_seconds=30`，但当前配置的相干时间为
`120 s`。这相当于在首个数据帧前主动推进一次约 `exp(-30/120)=0.7788` 的复 tap
状态相关过程，产生的状态失配明显强于“EME 时变不严重”的主研究假设。该设置会把
acquisition-CIR 老化、在线状态恢复和神经参数适配混在一起，不能直接作为主论文结果。

同一 checkpoint、同一 seed 和同一 `cfo_phase_tiny` 轨迹的诊断如下：

| acquisition 空档 | 15 dB、4 帧 Frozen NN BER（3 个 seed） |
|---:|---:|
| 0 s | 4.88% / 2.93% / 3.24% |
| 30 s | 6.70% / 9.93% / 20.37% |

另一个确定性问题是长记忆 CG warm-start 只运行了 2 次。保持权重不变，只把物理初值
求解迭代改为 4/8/16 次后，0 s 空档、15 dB 的三 seed 平均 BER 分别为 `0.91%`、
`0.24%`、`0.16%`。因此当前主配置固定为：

```text
acquisition_to_data_gap_seconds = 0.0
physics_warm_start_iterations = 8
```

旧 checkpoint 的网络结构和权重仍可复用，`compare.py` 只覆盖这类非结构性推理参数，
避免把求解器精度改善误写成重新训练收益。新 checkpoint 为
`pretrained/eme_offline_physics8_gap0/model_best.pt`。

使用该 checkpoint 的 15 dB、3 seed、8 帧公平复核结果为：

| 方法 | 平均 Data BER |
|---|---:|
| Pilot-conditioned frozen NN | 0.265% |
| Pilot-Driven Online Adaptation（只做 phase/PEFT，CIR fixed） | 0.256% |

这组结果已经满足“高 SNR 离线低于 10%、在线低于 1%”的阶段性目标。在线主实验必须
使用 `--cir-update fixed`，因为 `pilot_sparse` CIR 更新属于独立消融；它在 Reward Pilot
小窗口上变好时仍可能伤害 Data，不能与 phase PEFT 一起默认启用。其余 `0/5/10/15 dB`
和 60/200 帧结果应在该修正配置下重新生成后再写入论文主表。
