# EME 长记忆信道与在线均衡研究设计

## 1. 研究命题

第二研究点不把问题表述为“神经网络在任意线性信道上必然优于传统均衡器”。对于已知 CIR、线性时不变、AWGN 信道，最优线性 MMSE 或 MAP 检测器不存在可由神经网络保证超越的理论理由。可检验且与工程接收机一致的命题是：

> 在只提供前缀 Pilot、存在跨帧 ISI、长稀疏回波、残余 CFO/慢相位和慢 CIR 状态漂移的整帧接收机中，在线 Pilot 驱动适配能否降低固定离线模型和标准有限记忆自适应均衡器的稳态误码率。

因此，在线适配是研究贡献，PPO 不是研究贡献的前提。PPO 只保留为动作调度消融；主线先使用可解释的 Pilot 驱动受限 PEFT 更新，证明在线更新本身有效。

## 2. 信道证据与建模边界

公开的 Evans 月球雷达资料（DOI `10.6028/jres.069d.195`）支持以下事实：月球回波是分布式目标回波；短脉冲可以在相对月面前沿的时延上分辨回波功率；月球完整雷达深度约为 `11.6 ms`；回波功率随相对时延衰减，并同时包含准镜面与漫散射行为。公开资料不直接给出某一通信体制下的离散 CIR、路径相位或帧间统计过程。

第一研究点给出的宽带 EME 论文进一步说明了工程问题：在 `1 ksymbol/s` 和约 `10 ms` 的建模设置中，长时延已跨越多个符号，且采用了大量路径和分段式功率分布。第二研究点不复制第一点的公式和代码，只承接其物理方向。

`eme_long_memory_v2` 的取值为：

- 雷达深度：`11.6 ms`，来自公开回波资料；
- 符号率/采样率：`10 ksymbol/s`，作为宽带通信离散化假设；
- 最大离散记忆：`ceil(11.6 ms * 10 ksymbol/s) = 116` 个符号；
- 帧长：`1024` 个符号，使一帧内部包含长记忆和可观测的前缀至数据过渡；
- CIR 结构：早期准镜面强径加按观测功率包络抽样的长尾漫散射；
- 强径数：`12..24`，漫散射能量比例：`0.20..0.35`；
- support 在 episode 内固定，复路径增益跨帧按 `120 s` 相关时间缓慢漂移；
- 当前主扰动只加入小 residual CFO 与慢相位随机游走，不把大 Doppler 当作 EME 主结论。

该 profile 是“文献包络约束的 EME-inspired long-memory channel class”，不是精确测量得到的 EME 离散 CIR。旧 `eme_measurement_v1` 保留用于兼容和对照，不与新 profile 混合平均。

## 3. 传统均衡的失效机制

新信道的困难必须来自可解释的接收机约束，而不是故意损坏代码：

1. `116` 个符号记忆远大于常见短 FIR/DFE 的有限记忆；
2. `128` 个前缀 Pilot 中只有 `96` 个 Adapt Pilot，无法以很大的冗余同时精确估计长 CIR、相位斜率和跨帧 tail；
3. 第一帧后的接收波形包含上一帧软尾，前缀 Pilot 不能消除帧边界历史；
4. 相位/CFO 估计发生在前缀，数据段的相位状态只能依靠模型或递推跟踪；
5. 状态随帧缓慢漂移，获取帧 CIR 会逐渐过时，但 support 本身不应每帧随机重采样。

必须同时报告 acquisition/Adapt Pilot 估计、soft tail、CFO/相位补偿和计算复杂度，禁止用真实 CIR、真实相位或 Data 标签给传统方法或 proposed 方法作在线上界。

## 4. 在线方法

主方法采用两时间尺度的 Pilot 驱动受限在线适配：

```text
当前帧前缀 Adapt Pilot
  -> 估计可见的相位/CFO统计和 CIR/tail状态
  -> 只在 Adapter/FiLM/head 上做一步小梯度更新
  -> 整帧非因果神经块均衡
  -> 留出 Reward Pilot 做 guard/reward
  -> 变差则回滚 PEFT，保留真实时间顺序下的 soft tail
  -> 下一帧继续
```

在线更新不逐 bit 输出动作，不读 Reward Pilot 标签以外的 Data 标签，也不直接输出高维完整参数增量。第一版使用固定安全规则，之后再比较 PPO 调度 correction strength、PEFT group 和更新率是否带来额外收益。

## 5. 实验顺序

1. 信道可达性：报告 PDP、有效记忆、跨帧 tail 能量、帧间 CIR 相关和相位漂移统计。
2. Traditional-only：冻结 profile 后运行 CFO+DD-Phase LMMSE-FIR、CFO+DD-Phase DFE-RLS、SC-FDE-MMSE，并按 `0/5/10/15 dB` 分层报告。
3. Offline：验证 Pilot-conditioned 神经块均衡器能否从长记忆波形中学习；不能把这一步的失败隐藏在在线更新中。
4. Online：比较 `Offline NN only`、固定规则 Pilot 更新和可选 PPO 调度，画出前 5 帧、后 5 帧、全程均值以及 Reward Pilot 与 Data BER 的配对关系。
5. 统计：至少 5 个独立 channel seed；主结论按每个 SNR 和配置给出 paired block bootstrap 置信区间，不只报告总体平均。

## 6. 成功与失败判据

成功不是“某一次随机实验 BER 很低”，而是：

- 新 profile 的传统-only 困难来自可解释结构并可复现；
- Offline 模型至少在固定实验配置上进入可用区间；
- 在线适配相对 Offline NN only 的收益随着帧数增加而持续或稳定；
- 与最强传统 baseline 的比较逐 SNR 报告，并说明哪些配置没有超过；
- Reward Pilot 的改善和 Data BER 改善具有稳定统计关系。

如果纯线性长记忆信道上的强传统块检测器仍然优于 proposed，应保留该结果，说明研究问题需要引入接收机模型失配或复杂度约束，而不是继续调参制造神经网络优势。

## 7. 参考资料

- Evans, J. V., “Radar Studies of the Moon,” *Journal of Research of the National Bureau of Standards*, 1965, DOI: `10.6028/jres.069d.195`。
- Senior, T., Siegel, K. and Weil, H., “The influence of radar reflection characteristics of the moon on specifications for earth-moon-earth communication systems,” WESCON, 1958, DOI: `10.1109/WESCON.1958.1150219`。
- 《宽带 EME 通信信道的建模与特性分析-拟录用》，第一研究点论文，仅作为研究承接背景，不作为新 profile 的直接离散实现。
