> 状态：历史诊断/已被当前主路线取代。原始结果保留用于追溯，不代表当前默认方法。
> 当前规范：docs/state_tracking_receiver_design.md；验证：docs/state_tracking_validation_protocol.md。

# 状态 embedding 条件化 PEFT 小样本 replay（2026-10-11）

## 目的

当前路线的卡点是：Adapt Pilot 上的改善不一定能迁移到 Data 段。之前的
`PilotStateEmbedding` 只写入日志，没有进入 Adapter 的前向路径，因此它不能
帮助在线参数区分不同的信道状态。

本阶段保持以下内容不变：

- 离线 checkpoint：`pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`；
- Level B、delay=116、prefix Pilot、Pilot 总长度 256、Reward Pilot 32；
- acquisition 到首个数据帧间隔 600 s；
- 在线更新对象仍为 `phase_trend` Adapter；
- Adapt Pilot 负责参数梯度更新，Reward Pilot 负责验收、拒绝和回滚；
- Data 标签只用于 replay 结束后的评估，`data_labels_used_online=false`。

## 实现

新增的条件化路径是：

```text
Adapt Pilot 可见状态
  CIR residual、phase0、CFO、噪声、confidence、reconstruction error
             ↓
       6 维 PilotStateEmbedding
             ↓
  phase_trend Adapter 的 phase/CFO 调制
             ↓
    Adapt Pilot 重构损失更新 Adapter 参数
             ↓
       Reward Pilot 选择或回滚候选
```

`OnlinePhaseTrendAdapter.state_to_phase_cfo` 零初始化，所以状态条件化打开时仍
从冻结离线模型的恒等映射开始；只有在线梯度更新后，它才会学到“状态到相位/CFO
参数”的映射。默认主配置明确设置
`online_state_embedding_conditioned=false`，因此本阶段不会偷偷改变主实验。

## 小样本 A/B replay

两组实验使用同一批 Level B 信道轨迹和同一 checkpoint：

- 对照：`configs/online_phase_signal_multiscale.json`，状态条件化关闭；
- 条件化：`configs/online_phase_signal_state_embedding_replay.json`，状态条件化打开；
- 3 个 seed × 6 帧 × 5/10 dB，固定调度器，gap=600 s。

汇总脚本：

```powershell
python scripts\summarize_online_matrix.py `
  logs\state_embedding_replay_baseline_3s6f_5_10db_20261011 `
  logs\state_embedding_replay_conditioned_3s6f_5_10db_20261011 `
  --markdown logs\state_embedding_replay_summary_20261011.md
```

## 正式 5 seed × 60 帧矩阵

在首版正式矩阵之后，代码又补上了一个必要的观测修正：即使 `cir_update_mode`
固定为 acquisition，Adapt Pilot 仍会独立拟合当前 Pilot 的 CIR residual、phase/CFO
和 confidence，只把它们送入 embedding，不写回均衡器的主干条件。首版正式表格仍
对应修正前的日志；修正后的正式矩阵需要重新运行后才能作为最终统计。

小样本阶段确认 `phase_signal_moderate` 是唯一实际通过 Reward 守门并被选中的
候选后，正式矩阵只保留这个候选，以避免把没有被选中的分支重复计算。正式运行
使用主路线的参数：不指定额外 state split、`pilot_cir_phase`、AdamW、CIR 漂移
门限 0.7、Reward Signal 守门、每帧调度。

结果日志目录：
`logs/state_embedding_formal_moderate_5s60f_0_5_10_15db_20261011`。

| SNR | Frozen BER | Online BER | 平均改善 | 正收益 seed | 最差 seed 改善 |
|---:|---:|---:|---:|---:|---:|
| 0 dB | 38.9644% | 38.9644% | +0.0000 pp | 0/5 | 0.0000 pp |
| 5 dB | 29.0881% | 28.0269% | +1.0612 pp | 4/5 | -0.0629 pp |
| 10 dB | 26.6133% | 25.6415% | +0.9718 pp | 5/5 | +0.0716 pp |
| 15 dB | 25.0955% | 25.0712% | +0.0243 pp | 3/5 | -1.0677 pp |

这份正式结果达到两个阶段性结论：

1. 10 dB 已经满足“5/5 seed 稳定超过冻结离线模型”；
2. 5 dB 平均收益明显，但仍有 1 个轻微负收益 seed；15 dB 平均几乎持平，且有
   2 个负收益 seed；0 dB 继续冻结更安全。

因此状态条件化 PEFT 已经证明能把一部分收益稳定转移到在线阶段，但还不能把
当前配置宣称为四个 SNR 全部稳定超过离线模型。下一轮应专门处理 15 dB 的误放行
更新，再决定是否把条件化打开写入主配置。

## Pilot 状态观测修正后的 replay

修正后使用同一 checkpoint、Level B、gap=600 s、3 seed × 6 帧，四个 SNR 的
初步结果为：

| SNR | 平均改善 | 正收益 seed | 最差 seed |
|---:|---:|---:|---:|
| 0 dB | +0.0000 pp | 0/3 | 0.0000 pp |
| 5 dB | +0.8898 pp | 2/3 | 0.0000 pp |
| 10 dB | +0.7885 pp | 2/3 | 0.0000 pp |
| 15 dB | +3.9424 pp | 3/3 | +0.3906 pp |

这说明把真正的 Pilot 状态观测接入 embedding 后，15 dB 的短 replay 已经明显
改善；但它仍然不是 5 seed × 60 帧结论，正式矩阵必须在该修正版代码上重新运行。

## 修正版正式 5 seed × 60 帧矩阵（2026-10-12）

修正版正式矩阵已经完成，固定参数为：

- Level B、delay=116、prefix Pilot 总长度 256、Reward Pilot 32；
- acquisition 到首个数据帧间隔 600 s；
- 5 个 seed、每个 seed 60 帧，SNR 为 0/5/10/15 dB；
- `phase_signal_moderate`、AdamW、每帧更新、Reward Signal 重构守门；
- 离线 checkpoint 仍为 `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`；
- 在线更新只作用于 `phase_trend` Adapter，`data_labels_used_online=false`。

汇总日志：
`logs/state_embedding_observed_formal_5s60f_0_5_10_15db_20261012`。

| SNR | Frozen BER | Online BER | 平均改善 | 正收益 seed | 最差 seed 改善 |
|---:|---:|---:|---:|---:|---:|
| 0 dB | 38.9644% | 38.9644% | +0.0000 pp | 0/5 | 0.0000 pp |
| 5 dB | 29.0881% | 28.5326% | +0.5556 pp | 4/5 | -0.0304 pp |
| 10 dB | 26.6133% | 25.8576% | +0.7556 pp | 4/5 | -0.3559 pp |
| 15 dB | 25.0955% | 24.5933% | +0.5022 pp | 4/5 | -0.1497 pp |

这次矩阵确认了两点：

1. Adapt Pilot 驱动的 Adapter 微调确实能把平均收益转移到 Data 段，5/10/15 dB
   的平均 BER 均低于冻结离线模型；0 dB 继续冻结，避免低信噪比 Pilot 误更新。
2. 当前 moderate 配置还不能称为“逐 seed 稳定超过离线模型”。每个可更新 SNR
   都有一个负收益 seed，主要问题是 Reward Pilot 通过的候选仍可能在后续 Data
   段产生小幅反向变化。因此下一步应优先收紧候选更新幅度和 Reward 接受阈值，
   而不是修改离线 checkpoint。

审计检查：2400 条帧记录中 `data_labels_used_online` 全部为 `false`。当前正式
日志里的 `pilot_state_cir_residual_norm` 仍为 0，说明物理 CIR residual 在低置信度
时被安全门控，当前有效的状态条件主要来自 Pilot 估计的 CFO、噪声和置信度；这项
观测路径需要在后续小样本 replay 中单独修正和验证，不能把它误写成已经稳定利用了
CIR residual。

## 稳定性候选对比（2026-10-12）

针对正式矩阵中每个可更新 SNR 都出现一个负收益 seed 的问题，又做了同一 checkpoint、
同一 Level B、gap=600 s 的小样本候选对比。结果如下，改善均为相对冻结离线模型的
Data BER 百分点：

| 候选策略 | 5 dB | 10 dB | 15 dB | 结论 |
|---|---:|---:|---:|---|
| moderate，正式 5×60 帧 | +0.556，最差 -0.030 | +0.756，最差 -0.356 | +0.502，最差 -0.150 | 平均收益最好，但不是逐 seed 稳定 |
| conservative，正式 5×60 帧 | +0.335，最差 -0.063 | +0.424，最差 -0.002 | +0.092，最差 -0.421 | 缩小步长不能解决 15 dB 误更新 |
| 延迟一帧提交，3×12 帧 | +0.061 | +0.166 | +0.101 | 基本安全，但只提交极少候选，收益太小 |
| Reward Pilot=64，3×12 帧 | +2.080 | -0.322 | +1.060 | 增大 Reward Pilot 仍会出现严重负收益 seed |
| CIR residual 直接进入 embedding，3×12 帧 | -0.242 | +1.845 | +1.736 | 低置信度 CIR 估计会放大错误更新，已撤回该激进路径 |
| Reward logits 改变量≤0.1/0.2 | 0 或接近冻结 | 0 或接近冻结 | 0 或接近冻结 | 守门过严，失去在线收益 |

这组对比说明当前真正的卡点是“Adapt/Reward Pilot 的改善方向不能稳定代表 Data
段改善”。只收紧步长、增加 Reward 长度或直接放入低置信度 CIR residual，都不能同时
得到明显收益和逐 seed 非负。当前可以复现的最好平均结果是 moderate 正式矩阵；若优先
保证安全，则使用延迟提交，但它只提供很小的增益。

因此当前阶段仍不能把 `online_state_embedding_conditioned=true` 写入默认主配置，
也不能宣称“在线稳定超过离线”已经完成。下一步需要新的 Pilot-only 可迁移性方法，
例如离线校准 Pilot 连续指标到 Data 收益的可靠性模型，或重新设计能更好覆盖 Data
状态的 Reward 采样；这属于在线目标/验收设计，仍不需要修改离线 checkpoint。

### Data BER 结果

改善定义为 `(Frozen BER - Online BER) × 100`，单位为百分点。

| 设置 | SNR | 正收益 seed | 平均改善 | 最差 seed 改善 |
|---|---:|---:|---:|---:|
| 未条件化对照 | 5 dB | 2/3 | +0.1157 pp | 0.0000 pp |
| 状态条件化 | 5 dB | 3/3 | +0.3038 pp | +0.0217 pp |
| 未条件化对照 | 10 dB | 2/3 | +0.1374 pp | 0.0000 pp |
| 状态条件化 | 10 dB | 3/3 | +0.2604 pp | +0.0434 pp |

这说明在当前小样本中，状态条件化让 5/10 dB 的每个 seed 都有非负收益，并提高了平均收益。
随后补做的 0/15 dB 结果为：0 dB 平均改善 0.0000 pp（按安全策略冻结），15 dB
平均改善 +0.3400 pp，15 dB 为 3/3 seed 正收益。它仍然不是正式统计结论，因为
尚未完成 5 seed × 60 帧。

### 连续输出和梯度路径检查

在同一条 10 dB、seed=0、2 帧轨迹上：

| 指标 | 未条件化 | 状态条件化 |
|---|---:|---:|
| 首帧 Data BER | 79.95% | 79.69% |
| 首帧 Data logits 平均绝对变化 | 0.0337 | 0.0923 |
| 首帧 Reward Pilot BCE | 10.0042 | 9.9873 |
| 在线参数变化范数 | 0.00121 | 0.00200 |

因此 Adapter 更新确实改变了 Data 段输出，且状态条件化不是只改日志。单元测试还
验证了：零初始化恒等映射、状态到参数的梯度路径和错误状态维度检查。

## 当前结论

本阶段已经完成“状态 embedding 真正调制 PEFT Adapter”的工程闭环，并通过了
小样本 replay 的初步验证。当前结果支持继续扩大统计矩阵，但还不能把状态条件化
配置替换成主配置，也不能宣称所有 SNR 和所有 seed 都稳定超过离线模型。

## 下一步

正式矩阵需要在状态观测修正版上重跑。固定这版代码和 checkpoint，不重新设计
离线训练。重点检查：

1. 0 dB 是否因 Pilot 噪声而需要继续冻结；
2. 5/10/15 dB 是否保持每个 seed 非负收益；
3. Reward Pilot 的候选排序是否仍能带来 Data 连续指标改善；
4. 是否仍满足 `data_labels_used_online=false`。

诊断重点：

1. 15 dB 是否能通过更严格的 Reward Pilot 窗口或更小的 moderate 更新幅度消除
   负收益 seed；
2. 5 dB 的轻微负收益是否能在不牺牲平均收益的情况下消除；
3. 只有 0/5/10/15 dB 都达到逐 seed 非负且平均明显改善，才讨论把
   `online_state_embedding_conditioned=true` 纳入主配置。
