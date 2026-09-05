# 整帧离线训练与安全在线适配设计

## 目标

建立 Level B 极端稀疏长回波信道下的三层接收机路线：离线整帧监督均衡器、可验证必要性的元学习在线适配器，以及选择安全更新动作的 Contextual Bandit。

## 研究定义

离线阶段使用完整接收帧和完整发送标签训练基础神经块均衡器。Pilot、Reward Pilot 和 Data 的物理位置仍保留，但不再把它们作为三个独立的离线训练任务。主离线损失为：

```text
BCE_all = mean(BCE(logits_all, bits_all))
```

离线验证和 checkpoint 选择只使用未知 Data 区域的 BER，避免已知 Pilot 的低错误率掩盖 Data 均衡失败。

在线阶段加载离线 checkpoint。整帧接收波形输入非因果块均衡器，当前帧已知 Adapt Pilot 用于物理状态估计和在线候选更新，Reward Pilot 不参与梯度更新，只用于估计动作对未参与更新样本的改善，并执行接受、拒绝和回滚。

## 三层结构

### 第一层：离线整帧监督均衡器

输入为完整接收帧、只在 Adapt Pilot 位置可见的 Pilot 上下文、接收端可获得的 CIR/phase 条件和跨帧 soft-tail。监督标签为完整帧 bits，损失为 BCE_all。网络输出完整帧 logits，Data 区域是主性能指标。

### 第二层：元学习在线适配器

元学习不是默认加入主系统，而是一个需要通过实验判定的候选层。离线阶段构造多个信道状态任务，每个任务包含 Adapt Pilot support 和 Data/query。优化模型初始化或少量 PEFT 参数，使其在 1--3 个 Pilot 梯度步后能改善独立 query 的 Data BER。

必须与普通 Pilot-SGD 在同一 checkpoint、同一信道轨迹、同一 Pilot 切分和同一更新预算下配对比较。只有在后期帧、heldout_edge 和多 seed 上稳定超过普通 Pilot-SGD，元学习才进入最终主线；否则保留为消融，不用额外复杂度掩盖在线适配器本身的问题。

### 第三层：安全 Contextual Bandit

Bandit 观察当前 Adapt Pilot 的 loss、Pilot confidence、residual CFO、phase slope、CIR 变化代理量、SNR、前一窗口 reward、连续拒绝次数和参数变化范数等无 Data 标签信息。动作只从有限安全集合选择：skip、phase 轻量更新、phase 正常更新、FiLM/Adapter 轻量更新、CIR 状态更新和 rollback。动作在短窗口内保持不变。

Reward 由独立 Reward Pilot 的 loss/BER 改善减去参数变化和动作成本构成，不使用 Data 标签。先使用安全的 Thompson Sampling 或 LinUCB 作为 Contextual Bandit；若动作收益明确跨越多个帧才显现，再升级为带历史状态的安全 Recurrent Double DQN。PPO 只保留为比较算法，不作为默认实现。

## 信息边界

| 阶段 | Adapt Pilot | Reward Pilot | Data |
|---|---|---|---|
| 离线训练 | 输入和标签可用 | 标签可用 | 标签可用并参与 BCE_all |
| 在线状态估计 | 可用 | 不用于产生候选 | 不可用 |
| 在线参数更新 | 用于梯度 | 不参与梯度 | 不可用 |
| 在线动作 reward | 可作为状态趋势 | 只用于验收/回滚 | 不可用 |
| 最终仿真评估 | 可统计 | 可统计 | 只用于离线评估指标 |

## 主实验顺序

1. 重新训练并验证 BCE_all 离线 checkpoint。
2. 在相同轨迹上比较 Frozen、普通 Pilot-SGD 和元学习适配器。
3. 统计元学习相对普通 Pilot-SGD 的后期帧、heldout_edge、多 seed 配对增益。
4. 固定已经验证的适配器后，训练和评估安全 Contextual Bandit。
5. 检查动作 reward 是否存在跨帧延迟；若存在，再比较安全 Recurrent Double DQN。
6. 与传统均衡器比较，所有主结果保持 Level B、0/5/10/15 dB、prefix Pilot、Data BER 指标一致。

## 风险和处理

- BCE_all 可能让模型利用已知 Pilot 形成容易的局部预测，因此 checkpoint 必须按 Data BER 选择，且需要报告 Pilot BER 与 Data BER 的分离结果。
- Reward Pilot 过短时 reward 方差较大，使用窗口聚合、跨帧配对和回滚，不能直接放大更新步长。
- 元学习若只改善 Adapt Pilot 而不改善 Data/query，应判定为无效，而不是保留。
- Bandit 若只能根据当前窗口 reward 排序动作，则不应升级到完整 RL；只有出现明确长期 credit assignment 问题才引入 Recurrent Double DQN。

## 验收标准

- 离线 checkpoint 的验证指标明确记录 `BCE_all` 和 Data BER。
- 在线路径没有使用 Data 标签进行梯度、动作或回滚。
- 元学习有普通 Pilot-SGD 对照，并报告是否必要，不以复杂度替代证据。
- Bandit 动作、状态和 reward 可审计，且所有更新有安全边界。
- 所有新增行为有先失败后通过的测试。
