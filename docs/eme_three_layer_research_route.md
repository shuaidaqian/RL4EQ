# EME 在线均衡三层研究路线

## 研究问题

第一层回答“神经均衡器能否利用整帧接收波形和 EME 极端长记忆结构学到比传统均衡更强的离线基础能力”。第二层回答“在相同离线基础和相同 Pilot 更新预算下，元学习初始化是否比普通 Pilot-SGD 更能适应未见信道状态”。第三层回答“在线更新不是每帧都盲目执行时，安全 Contextual Bandit 能否根据 Pilot 观测选择合适的更新对象、强度和保持时间”。

三层必须使用同一 Level B 信道协议：最大多符号相对时延、跨帧 ISI、稀疏长回波，以及逐级加入 residual CFO 和慢相位扰动。主 SNR 固定为 `0/5/10/15 dB`，Pilot 只放在前缀，数据标签只在离线监督和最终仿真评估中使用。

## 第一层：离线整帧监督均衡器

默认损失已经固定为：

```text
BCE_all = mean(BCE(logits_all, bits_all))
```

网络输出整帧 logits，但模型输入的已知符号来自 `receiver_view().adapt_symbols`，只有 Adapt Pilot 可见。这样离线阶段可以用整帧 Data 标签学习，在线阶段仍不能把 Data 真值当输入。checkpoint 仍按独立验证集 Data BER 选择，并同时报告整帧 BCE、Adapt Pilot BER、Reward Pilot BER 和 Data BER。

已完成代码和测试：

- `training/curriculum.py` 提供 `full_frame_bce_loss`；
- `configs/continual_ppo.json` 与 `configs/eme_long_memory_v2.json` 显式设置 `offline_loss=bce_all`；
- `tests/test_meta_adaptation.py` 验证整帧逐符号等权和混合 delay batch；
- 离线 smoke 已成功生成 `pretrained/final_smoke`，但 2 步 smoke 不是性能结果。

## 第二层：元学习在线适配器

元学习不是默认主方法。当前可行比较为：

```text
Frozen Offline NN
Pilot-SGD：Adapt Pilot 上普通受限 PEFT 更新
Meta-Pilot：经过跨信道状态任务训练的快速初始化/内循环
```

三者必须共享初始 checkpoint、信道轨迹、Pilot 切分、更新参数组、梯度步数和最大参数增量。`build_meta_necessity_report` 按配置、seed、帧号配对，并分别统计 early/middle/late 与 `heldout_edge`。只有 Meta-Pilot 在后期和 heldout edge 的配对 Data BER 都稳定优于 Pilot-SGD，才把元学习纳入主系统；否则元学习仅作为消融，普通 Pilot-SGD 保留为更简洁的在线基线。

当前状态：必要性评估接口和测试已完成，但还没有用新的 `BCE_all` checkpoint 生成正式 Meta-Pilot 配对矩阵。因此当前不能声称元学习已经必要，也不能以 Reward Pilot loss 的下降替代 Data BER 证据。

用新的 32 步 `BCE_all` EME checkpoint 做的 8 帧、10 dB 短对照中，固定 Pilot-SGD
和 Bandit 的 Data BER 均为 `0.00683594`。这说明当前 Bandit 尚未显示独立 BER
增益，后续正式实验必须扩大帧数、seed 和状态失配范围后再判断调度收益。

## 第三层：安全 Contextual Bandit

Bandit 的职责是调度，不是生成网络参数。安全动作集合为 `skip`、弱/正常 head 更新、弱 phase 更新、FiLM 更新和联合更新；动作必须属于当前模型真实存在的 PEFT 组，并在短窗口内保持不变。上下文包含 Adapt Pilot loss/confidence、residual CFO、phase slope、CIR drift、SNR、Reward 趋势、回滚率、连续拒绝数和上次参数变化范数。

在线路径为：

```text
当前帧 Adapt Pilot 估计状态
 -> Bandit 选择安全离散动作
 -> Adapt Pilot 产生候选 PEFT 更新
 -> Reward Pilot 计算独立改善
 -> 成本/回滚安全门控
 -> 接受或恢复参数
```

reward 为 Reward Pilot loss 改善减去更新成本、动作成本和回滚惩罚，绝不使用 Data BER。已完成 Bandit 接口、动作可用性过滤、记录字段和跨帧诊断器。真实 EME 8 帧 smoke 输出 `delayed_effect_detected=false`，并推荐继续使用 Contextual Bandit；该样本规模不足以成为正式论文结论。

## 是否升级 Recurrent Double DQN

先使用 `scripts/diagnose_action_delay.py` 在多 seed、多个主 SNR、`offline_train/heldout_edge/drift` 状态上统计 horizon `0/1/2/4/8`。只有动作收益稳定在后续帧出现，且即时 reward 无法解释收益，才建立安全约束 Recurrent Double DQN。若延迟收益不显著，Contextual Bandit 已经是更符合问题结构的最终调度器。

## 当前可以写入论文的程度

可以写方法、问题定义、信道假设、信息边界、实验协议和算法设计；不能把旧 checkpoint 的正式 BER 直接标成新 `BCE_all` 路线结果，也不能声称元学习或 Bandit 已经带来独立 BER 增益。论文结果章还需要完成新的 BCE_all 离线 checkpoint、Frozen/Pilot-SGD/Meta-Pilot 配对矩阵，以及多 seed 的 Bandit 与延迟诊断。
