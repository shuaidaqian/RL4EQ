# Pilot-only 在线适配阶段性路线

日期：2026-10-06

## 目标

保持离线 checkpoint、Level B 信道和 prefix Pilot 总长度不变，寻找一种真正修改
均衡器内部 PEFT 参数、且能把 Adapt Pilot 的自监督改善迁移到 Data 段 BER 的在线方法。
在线过程继续遵守以下边界：Adapt Pilot 用于参数更新，Reward Pilot 只用于验收和回滚，
Data 标签只用于离线最终评估。

## 固定协议

- checkpoint：`pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`
- profile：`eme_long_memory_v2`，Level B，最大记忆 `116`
- 主 SNR：`0/5/10/15 dB`
- Pilot：prefix，总长度 `256`，Adapt `224`，Reward `32`
- 在线条件和信道轨迹必须在 Frozen/Online 之间配对
- 离线 checkpoint 不重训
- 主配置 `gap=0` 与状态老化诊断分开报告

## 已完成的 replay 事实

### 1. 单帧 BCE 与普通 PEFT

`phase_trend + head` 在主配置 5 seeds × 60 frames 上只有小幅收益，且 10 dB 区间跨零。
RLS、input/logit affine、input FIR、physics residual 等候选要么不改变 Data 硬判决，
要么在低 SNR 或状态失配下退化。继续调学习率、步数或参数范数没有足够依据。

### 2. Reward 代理失配

已有 Pilot replay 诊断显示，单帧 Reward Pilot loss 对后续 Data BER 的 Spearman 相关在窗口
2/4/8 上分别为 `0.037/-0.227/-0.151`；只看已接受 PEFT 事件仍为
`-0.116/-0.104/-0.170`。因此当前不能接入 Contextual Bandit：动作选择器无法修复
不能稳定排序候选好坏的 reward。

### 3. 物理重构目标

`channel_residual + pilot_reconstruction` 是唯一已经在 Data 硬判决上产生可重复变化的
候选。保持 checkpoint 不变，在 `heldout_edge + acquisition_to_data_gap=120 s` 下的
5 seeds × 60 frames 结果为：

| SNR | Frozen BER | Online BER | 配对收益 | seed 方向 |
|---:|---:|---:|---:|---:|
| 0 dB | 26.0885% | 26.0885% | 0.0000 pp | 0/5，按协议冻结 |
| 5 dB | 11.2062% | 11.1237% | +0.0825 pp | 5/5 |
| 10 dB | 6.0373% | 5.9796% | +0.0577 pp | 5/5 |
| 15 dB | 4.4709% | 4.3954% | +0.0755 pp | 5/5 |

120 s、5 dB 扩展到 120 帧后，前 60 帧收益约 `+0.0825 pp`，后 60 帧约 `+0.2318 pp`，
说明该候选在明显 acquisition 状态老化时具有随帧数积累的收益。

但这不是主配置已完成的证明：`gap=0` 仍基本持平，0 dB 长时间更新会退化，300 s 过强
失配也不稳定。因此当前只能把它定义为“状态老化诊断工作区的候选主线”。

## 决策门槛

每个候选必须同时满足以下条件，才允许进入正式 5 seeds × 60 frames 主矩阵：

1. Pilot-only replay 的候选排序在多个窗口上对后续 Data BER 呈正相关，目标门槛为
   Spearman `>=0.6`；
2. 至少 4/5 seed 的 Data 配对收益为正；
3. Data 硬判决确实发生变化，而不是只有连续 logit 变化；
4. 低 SNR 不出现系统性退化；
5. 在线审计确认 `data_labels_used_online=false`；
6. 参数更新受单步和累计 trust-region 约束，并可由 Reward Pilot 回滚。

当前 `channel_residual + pilot_reconstruction` 通过了第 2、3、5、6 条的部分诊断，
但尚未通过主配置和 Pilot-only 排序门槛，因此暂不引入 Bandit。

## 后续实施顺序

### 阶段 A：统一 Pilot-only replay

在相同的候选、相同帧窗口和相同初始模型快照下比较：

- 单帧 Adapt Pilot BCE；
- 多帧 Adapt/Reward Pilot 累积目标；
- Adapt Pilot 复数信道重构目标；
- 重构目标与软判决目标的联合目标；
- 参数 proximal/trust-region 惩罚。

Data 只能用于事后计算排序相关性和 BER，不能进入候选选择过程。

当前进度：已有窗口 replay 和 `channel_residual + pilot_reconstruction` 诊断结果；统一
三目标的排序报告尚未完成。现有 `evaluate_peft_window_candidates` 在长记忆物理 warm-start
下计算成本较高，后续 replay 必须使用固定候选、较短窗口和可复现的分层统计，不能直接扩大
到大矩阵后再解释排序失败。

2026-10-07 阶段诊断：在已有主配置 `phase_trend + head` 的 5 seeds × 60 frames 日志上，
新增离线排序分析得到单帧 BCE 在窗口 2/4/8 的 Spearman 为
`0.160/0.036/-0.005`；多帧累计目标与当前联合代理几乎相同，也全部低于 `0.6` 门槛。
这证明简单改变 Reward 聚合窗口不能解决 Pilot 到 Data 的目标失配。由于旧的
`channel_residual + pilot_reconstruction` 日志没有保存所有候选的未采用分数，不能据此
伪造三目标的公平排序结论；下一轮必须生成带完整候选分数的短 replay。

2026-10-07 实现进展：新增 `objective` 选项支持 `bce`、`pilot_reconstruction` 和 `joint`
三种 Adapt Pilot 更新目标；诊断窗口会从相同模型快照分别评估 identity 与
`channel_residual` 候选，并记录目标名称、Reward 指标、参数增量和事后 Data 指标。
动态 `channel_residual` Adapter 的 PEFT 分组恢复也已补齐。单帧三目标接口探针已成功，
但 1 seed × 4 SNR × 4 帧的窗口探针因完整物理 warm-start 和模型复制开销过大被中止，
没有产生正式统计，不能据此判断哪种目标通过排序门槛。下一步先优化为单 SNR、单窗口的
最小 replay，再逐步扩样。

2026-10-07 短 replay 复核：在固定 Level B、delay=116、SNR=10 dB、Pilot=256、
3 seeds、2 帧窗口下，对 `bce`、`pilot_reconstruction`、`joint` 分别扫描
`channel_residual` 的保守/标准/快速三档学习率。三种目标均没有产生 Data 硬判决改善，
因此每个窗口的 Data 改善序列为常数，Spearman 排序相关性不可定义；不能把该结果记为
通过排序门槛。Pilot reward 偶尔偏好快速更新，但 Data 三档候选仍持平，说明当前
`channel_residual` Adapter 的更新幅度或作用位置不足以在短窗口改变均衡输出。新增
`scripts/summarize_pilot_objective_replay.py` 按目标独立汇总结果，避免把逐帧 head/LoRA
候选的相关性混入 channel residual 窗口结论。下一步应先记录连续 Data loss/logit 改变量，
确认更新是否实际影响数据段，再决定是否增加步数或调整 Adapter 调制位置；在此之前不进入
正式 5 seeds × 60 frames，也不引入 Contextual Bandit。

2026-10-07 主路径 PEFT 探针：新增并测试了已有的 `physics_residual` 与
`physics_blend` 在线 Adapter。`physics_blend` 仍只产生约 `1e-7` 级 Data logits 变化；
`physics_residual` 位于最终 logits 主路径，3 seeds × 2 帧探针中产生约 `3.7e-4` 的
平均 logits 变化、约 3.7% 的方向改变，并出现约 1.15 pp 的 Data BER 改善，证明
Adapter 已真正影响数据段输出。随后固定 5 seeds、4 个主 SNR、4 帧、窗口 2 的短统计：
`physics_residual` 标准档在 0/5/10/15 dB 的平均 Data BER 改善约为
`0.92/1.84/1.80/2.01 pp`，Data BCE、soft output 和 logits 均同步发生可观测变化。
但 Reward Pilot loss 在所有候选上几乎都下降，无法区分 Data 好坏更新；Reward loss 单指标
不能直接作为接受条件。下一步应实现多帧 Reward 验收、最大退化约束和 trust-region 回滚，
再在更长帧数上确认在线收益是否稳定。

2026-10-08 多帧 Reward 验收原型：新增 `RewardWindowGate` 和
`scripts/replay_reward_gate.py`。replay 时首帧只用 Adapt Pilot 更新一次
`physics_residual`，后续两帧冻结参数，只用 Reward Pilot 的 loss/BER 做累计验收；
Data 标签只做事后评估。4 个主 SNR × 5 seed 的短矩阵中，三档更新的接受率约为 60%。
门控能够拒绝一部分出现单帧 Reward 退化的候选，但接受组的 Data BER 改善仍接近 0，
只有 Data BCE 出现小幅正向变化，说明当前门控主要提供稳定性保护，还没有把 Reward
验收稳定转化为 BER 收益。该结果不满足正式 5 seed × 60 frame 的成功门槛，暂不引入
Contextual Bandit；下一步需在更长 Reward 窗口和 acquisition 状态老化场景中验证，
并比较接受/回滚后的连续指标与 Frozen 基线。

2026-10-08 状态老化长窗口复核：在 `acquisition_to_data_gap_seconds=120`、Level B、
4 个主 SNR、5 seeds、首帧 Adapt 更新后连续 4 帧 Reward 验收的 replay 中，三档
`physics_residual` 接受率约为 45%。10 dB 候选全部因单帧 Reward 退化被回滚，说明
长窗口确实提高了保护强度；接受组 Data BCE 有小幅正向变化，但 Data BER 改善仍接近 0，
只有 0 dB 标准档出现极小收益。当前结论是“多帧门控能够减少不稳定更新，但 Reward
验收尚未稳定转化为 BER 收益”，不满足正式 5 seed × 60 frame 成功门槛。下一步应优化
Reward 统计与候选更新方向（例如使用 Reward loss + margin/BER 的联合验收、限制快速档），
再决定是否进入正式长矩阵；暂不引入 Contextual Bandit。

### 阶段 B：状态条件化 PEFT

由 Adapt Pilot 产生低维状态 embedding，至少包含：

- CIR residual 或主要 tap 变化；
- residual CFO 与慢相位趋势；
- 噪声水平；
- Pilot 重构残差；
- 状态置信度。

embedding 只调制 `channel_residual`、Adapter、FiLM 或 LoRA 等受限参数，不能退化成
只恢复 CIR/phase 而不修改均衡器参数。

当前进度：新增了只读的 `agent/pilot_state.py`，可生成 6 维状态摘要/有界 embedding，
并记录到 compare 审计字段。该 embedding 目前不改变在线动作；其中 CIR residual 和重构误差
仍是审计占位值，尚未接入 PEFT 调制，不能视为状态条件化已经完成。

### 阶段 C：漂移检测与异步更新

使用 Pilot-only 统计量进行 CUSUM、Page-Hinkley、Hotelling 或等价变点检测。未检测到
状态老化时保持参数；检测到老化时才产生候选更新。检测器不能读取 Data BER。

当前进度：已实现 `PilotDriftDetector` 的无标签基础模块，但尚未接入主在线更新门控。
在 Pilot-only replay 通过排序门槛前，不启用该门控，也不以其审计字段宣称在线收益。

### 阶段 D：多帧 Reward 验收

Reward Pilot 采用连续短窗口的一致性验收：累计改善、单帧最大退化、参数距离和置信度
同时满足条件才接受，否则回滚。低 SNR 需要更严格门控或冻结策略。

### 阶段 E：正式矩阵

只有 A-D 的短 replay 通过门槛后，才在固定 Level B 主配置执行 5 seeds × 60 frames。
若主配置仍无增益，则如实保留“在线状态老化诊断有效、gap=0 主配置 PEFT 未完成”的结论，
不使用更强失配结果替代主配置结论。

## 当前结论

现在不应继续扩大普通 BCE、phase/head、RLS 或输入/输出仿射候选，也不应直接引入
Contextual Bandit。当前最有根据的路线是：

```text
Adapt Pilot
  -> Pilot 重构得到 CIR residual / CFO / phase / noise / confidence
  -> 状态条件化 channel-residual PEFT
  -> 漂移检测决定是否更新
  -> 多帧 Reward Pilot 验收与回滚
```

这条路线已经在状态老化诊断区显示小幅、跨 seed、随帧数增强的收益，但尚未证明在
`gap=0` 主配置中稳定超过 Frozen Offline NN，因此下一阶段必须先完成 Pilot-only replay
排序门槛，再决定是否进入正式主矩阵。

2026-10-08 联合 Reward 验收复核：在 gap=120s、4 个主 SNR、5 seed、4 帧 Reward
窗口上加入 Reward BER 和 margin 的联合条件后，接受/拒绝结果与 loss-only 门控完全一致。
逐帧统计显示 BER/margin 改善与 loss 基本同向，15 dB 还会因 Pilot 量化出现大量 0，
因此简单增加指标没有提升 Data 收益筛选能力。当前将多帧 Gate 定位为“防止明显退化的
回滚保护”，不再把它当作 Data BER 收益选择器；默认主候选移除快速档，只保留保守和标准
`physics_residual`，快速档仅在显式压力诊断时启用。下一阶段应在更长状态老化轨迹上比较
接受但 Data 未改善的比例，并继续寻找能预测 Data 变化的 Pilot-only 状态特征。

2026-10-08 长轨迹敏感性：在 gap=120s、10 dB、5 seeds、20 帧轨迹中，首帧更新后
冻结参数并用 2/4/8 帧 Reward 窗口验收时，`physics_residual` 保守档和标准档均全部
被回滚；将学习率从 `1e-4` 降到 `1e-5` 仍全部回滚。说明问题来自“首帧 Adapt 梯度在
状态持续漂移下长期保持”的目标失配，而不是单纯更新幅度或窗口长度。下一步改为滚动
短窗口：每个窗口先用当前 Adapt Pilot 小步更新，再用后续 Reward Pilot 验收，接受后
进入下一窗口，拒绝则回滚并重新估计状态。该协议更接近真实在线微调，仍不使用 Data 标签。

2026-10-08 滚动短窗口 replay：新增 `scripts/replay_rolling_reward_gate.py`，每个窗口
先用当前 Adapt Pilot 小步更新，再用 2 个 Reward 帧验收；接受的参数进入下一窗口，
拒绝则恢复窗口快照。在 gap=120s、4 个主 SNR、5 seeds、20 帧条件下，两个候选的窗口
接受率约为 54% 到 55%。标准 `physics_residual` 的整体 Data BCE 改善为正，0 dB 下
接受窗口的 Data BCE 全部改善并出现小幅 BER 收益；5/10/15 dB 的 Data BCE 多数改善，
硬 BER 基本持平。保守档整体 BER 略有下降，标准档略有上升，说明滚动窗口比一次冻结
更接近在线微调目标，但收益仍不足以进入正式 5 seed × 60 frame。下一步应围绕标准档
优化状态条件化和窗口接受策略，并继续报告 Data BCE 与 BER 的配对结果。

2026-10-09 漂移门控接入滚动 replay：使用 `PilotStateEmbedding` 和
`PilotDriftDetector` 对 CIR residual、噪声和置信度构造无标签状态距离；首窗口只建立
基线，后续只有检测到漂移才尝试 `physics_residual` 更新。修正候选之间共享 CIR、模型
和 soft-tail 的隔离问题后，在 gap=120s、4 个主 SNR、5 seeds、20 帧矩阵中，两档候选
的漂移触发率均为约 45%，标准档接受率约 63%。标准档接受窗口的 Data BCE 在 0/5/10 dB
均为正，硬 BER 基本持平；保守档整体 BER 略有下降。在线审计显示所有窗口均未使用 Data
标签。该结果证明“漂移检测 + 滚动 Reward 回滚”链路可运行并能减少无漂移更新，但仍未
达到稳定 BER 优势，下一步应把真实 Adapt Pilot 重构误差和 phase/CFO 变化接入状态摘要，
再验证状态条件化 PEFT 是否能把连续收益转成硬 BER 收益。

2026-10-09 真实 Pilot 状态与条件化步长：滚动 replay 已接入当前 Adapt Pilot 的实际
相位残差向量、CIR 线性重构相对误差和由重构误差修正后的置信度。状态条件化选项
`--state-conditioned` 只根据漂移距离、Pilot 重构误差和置信度缩放
`physics_residual` 的 PEFT 学习率，仍不读取 Reward/Data 标签；Reward Pilot 继续只负责
验收、拒绝和回滚。固定 gap=120 s、Level B、window=2、5/10 dB、3 seeds、8 帧的短矩阵
中，5 dB 标准档接受率约 16.7%、10 dB 全部回滚，Data BCE 变化为正但硬 BER 仍为 0；
随后在相同配置下扩展到 5/10 dB、3 seeds、20 帧，对照固定步长与条件化步长：两者的
漂移触发率、接受率和 Data 硬 BER 完全相同，条件化平均学习率约为固定值的 0.74/0.80，
但 Data BCE 略低。因此该阶段验证了真实状态接入和审计链路，没有通过“稳定 BER 超过
Frozen”的门槛，也不能替代正式 5 seeds × 60 frames 主矩阵。下一步应优先寻找能在
Pilot-only 上区分 Data 收益的更新方向或验收特征；在此之前不扩大正式矩阵，也不引入
Contextual Bandit。

2026-10-10 选择性边界目标复核：在 Adapt Pilot BCE 的基础上加入只对低绝对值 logit
施加主要梯度的 `selective_boundary` 目标，并同步接入实际
`PilotDrivenOnlineAdapter`（默认不启用）。该目标包含有界边界样本权重和有限 margin
惩罚，高置信度 Pilot 基本保持冻结。固定 gap=120 s、Level B、3 seeds × 2 SNR × 4
帧候选 replay 中，`selective_boundary` 与 Data BER 的 Spearman 约为 `0.024`，仍低于
`0.6`；各候选的 Data 硬 BER 也没有稳定改善。由此确认仅重新加权 Adapt Pilot 判决
边界仍不能解决 Pilot 到 Data 的目标失配。该目标保留为可复现实验选项，但不进入主
在线策略，也不启动正式 5 seed × 60 帧矩阵。下一步应转向“状态/时间一致性约束”或
显式的 Data 无标签物理一致性，先证明 Pilot-only 代理能够排序后再继续。

2026-10-10 Pilot-only 候选方向排序 replay：在固定 checkpoint、Level B、delay=116、
prefix Pilot=256、gap=120 s 的同一轨迹上，对 `physics_residual`、`channel_residual`、
`head`、identity 和低维 `PilotResidualRLSAdapter` 进行候选扫描。候选更新只读取
Adapt Pilot；Data 只在 replay 输出中用于事后验证。3 seeds × 2 SNR × 4 帧结果中，
Adapt BCE、Reward loss、Pilot 重构改善与 Data BER 的 Spearman 分别为约
`-0.113`、`0.043`、`-0.117`，均未达到 `0.6` 门槛。RLS smoke 中 Adapt BCE 改善约
`1.6e-2`，但 Data BCE 退化约 `-1.06e-1`，说明“Pilot 拟合更好”不代表数据均衡更好；
RLS 当前不能作为主路线。`channel_residual` 和 `physics_residual` 能稳定改变 Data
连续输出，但尚未稳定改变硬 BER。因此当前瓶颈确认是更新方向与 Data 目标失配，不是
单纯学习率或 Reward 窗口长度问题。下一步应优先设计能约束均衡器判决边界的 Pilot-only
更新目标，再重新做排序验证；不引入 Contextual Bandit，也不扩大正式 5 seed × 60 帧矩阵。
