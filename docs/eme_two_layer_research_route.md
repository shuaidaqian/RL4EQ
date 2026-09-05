# EME 在线均衡两层研究路线

## 核心路线

当前研究只保留两个必要对象：

```text
离线整帧 BCE_all 神经均衡器
    -> 安全 Contextual Bandit 在线调度
```

离线阶段用完整帧标签训练基础网络，在线阶段不再引入元学习层，也不再为元学习
增加独立入口或必要性实验。在线研究的核心问题是：在极端稀疏长回波、跨帧 ISI、
residual CFO 和慢相位扰动下，已训练网络能否根据当前已知 Pilot 选择安全更新，
并改善未知 Data 区域的均衡结果。

## 第一层：离线基础网络

模型输出完整帧 logits，默认损失为：

```text
BCE_all = mean(BCE(logits_all, bits_all))
```

输入中的已知发送符号仍来自 `receiver_view().adapt_symbols`，只有 Adapt Pilot
可见，Reward Pilot 和 Data 符号不会作为输入。离线训练可以使用整帧 Data 标签；
checkpoint 用独立验证帧的 Data BER 选择，最终测试集只用于报告性能。

## 第二层：安全 Contextual Bandit 在线调度

在线每帧执行：

```text
Adapt Pilot 估计当前状态
    -> Bandit 选择离散安全动作
    -> Adapt Pilot 产生受限 PEFT 候选更新
    -> Reward Pilot 独立验收
    -> 接受或回滚
    -> 处理未知 Data
```

Bandit 只调度，不直接生成高维网络参数。动作包括 `skip`、弱/正常 head 更新、
弱 phase 更新、FiLM 更新和联合更新，但必须过滤掉当前模型不存在的 PEFT 组。动作
在短窗口内保持一致。上下文包括 Adapt Pilot loss/confidence、residual CFO、
phase slope、CIR drift、SNR、Reward 趋势、回滚率、连续拒绝数和参数变化范数。

reward 只由 Reward Pilot 的 loss 改善、更新成本和回滚惩罚构成，Data 标签不能参与
在线 observation、动作选择、更新或回滚。

## 何时升级 Recurrent Double DQN

Contextual Bandit 是第一选择。只有在真实主实验中，动作收益经过至少多个 seed、
多个 SNR 和不同状态 split 的验证，并且主要收益稳定出现在动作后的多个帧，才能把
同一个调度位置替换为安全动作约束的 Recurrent Double DQN。否则不增加循环 Q 网络。

`scripts/diagnose_action_delay.py` 只承担这一项判断，统计 horizon `0/1/2/4/8`
的 Reward Pilot reward，并要求至少覆盖 2 个 seed 和 2 个 SNR 才允许标记延迟效应。

## 研究对照

主对照只保留：

```text
传统非神经、非 RL 均衡器
Frozen Offline NN
Pilot-Driven Online Adaptation（由安全 Contextual Bandit 调度）
```

Frozen 和 Online 使用相同离线 checkpoint、相同 Level B 信道轨迹、相同前缀 Pilot
和相同跨帧 soft-tail。Frozen 冻结网络参数；Online 只利用 Adapt Pilot 产生受限
候选更新，并由 Reward Pilot 验收或回滚。

## 当前证据边界

新的 `BCE_all` EME 32 步 smoke 在 10 dB、8 帧上约为 `0.6836%` Data BER；固定
Pilot-SGD 与 Bandit 在同一短实验中均为 `0.6836%`，因此目前只证明在线安全门控
链路可运行，尚未证明 Bandit 已带来独立 BER 增益。正式结论必须来自 Level B、
0/5/10/15 dB、多 seed、长帧矩阵，不能用 smoke 替代。
