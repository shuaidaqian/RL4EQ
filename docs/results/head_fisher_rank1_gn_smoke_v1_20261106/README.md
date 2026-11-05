# 固定 Fisher rank-1 的阻尼 Gauss–Newton smoke

本次在强离线 checkpoint 上只比较开发机制，信道/Pilot 固定为既有 Level B `eme_long_memory_v2`、116 符号长回波、前缀 Pilot=256（Adapt=224、Reward=32），四个主要 SNR 均报告。方向由训练角色 seed 63000 的 Adapt Fisher 估计；评估只用新 session seed 63014，每 SNR 一帧。该 seed 仅作开发 smoke。

新增分支在固定 rank-1 方向系数上执行一次阻尼 logistic Gauss–Newton 更新，最大单步范数0.05；Reward 32符号按既有硬规则独立守门，Data 只在动作结束后评分。对照包括 Frozen、全 head Adam、Fisher rank-1 标量 Adam、Fisher rank-1 阻尼 Gauss–Newton。完整逐帧数据、Adapt 梯度/Fisher、Reward BCE/BER、参数步长及 checkpoint/代码指纹见 `result.json`。

| SNR | GN Reward 接受 | GN Data BER | Frozen Data BER | 配对 BER 增益 |
|---:|---:|---:|---:|---:|
| 0 dB | 拒绝 | 14.1927% | 14.1927% | 0 |
| 5 dB | 接受（BER持平、BCE下降） | 4.2969% | 4.2969% | 0 |
| 10 dB | 接受（BER持平、BCE下降） | 0.3906% | 0.3906% | 0 |
| 15 dB | 拒绝 | 0% | 0% | 0 |

结论严格限于一次候选烟测：阻尼 GN 在两个 SNR 上给出通过 Reward BCE 的候选，但没有改变单帧 Data BER。每 SNR 一个会话、一个 Data 帧无法证明在线增益或随帧增长；门控接受率也不是研究目标本身。当前该算法只是候选求解器，不是已匹配的离线外层训练方法。

## 后续

不扩展单帧统计。实现与评估共享同一系数更新器的连续轨迹训练：离线外层学习 rank-1 方向以提升未来 Data 检测，同时固定在线 Adapt-only GN、Reward 硬守门和单步/累计信赖域；比较固定 Fisher 方向、匹配预算的普通监督继续训练、外层学习方向与 Frozen。开发通过后，再以全新独立 session、多帧窗口评估全程配对收益、晚早窗口差、会话趋势及成本。不得使用 Data 标签做在线动作选择。
