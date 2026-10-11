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

固定这版代码和 checkpoint，运行正式的 5 seed × 60 帧矩阵，SNR 使用主场景的
0/5/10/15 dB。重点检查：

1. 0 dB 是否因 Pilot 噪声而需要继续冻结；
2. 5/10/15 dB 是否保持每个 seed 非负收益；
3. Reward Pilot 的候选排序是否仍能带来 Data 连续指标改善；
4. 是否仍满足 `data_labels_used_online=false`。

目前四个主 SNR 的小样本检查已经完成。只有正式矩阵通过后，才讨论把
`online_state_embedding_conditioned=true` 纳入主配置，并继续评估是否需要更换
Adapter 对象或引入其他调度策略。
