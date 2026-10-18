# Pilot-only 状态跟踪验证协议

状态：当前主验证协议。所有方法使用相同 checkpoint、信道轨迹、prefix Pilot、数据帧和初始 tail；Data 仅用于事后硬 BER。

## 分层基线

| 代号 | 方法 | 在线状态/参数更新 | 用途 |
|---|---|---|---|
| F0 | `F0 Acquisition Frozen NN` | 固定 acquisition CIR/phase；冻结网络 | 冻结模型在在线信道的基线 |
| F1 | `F1 Pilot-Conditioned Frozen NN` | 当前 Adapt Pilot 只生成条件；冻结 CIR 更新和网络 | 当前 Pilot 条件化的收益 |
| F2 | `F2 Pilot State Tracking Frozen NN` | Adapt Pilot 更新物理后验；冻结网络 | 当前 proposed 接收机 |
| T-MMSE | `SC-FDE-MMSE` 与 `LMMSE-FIR` | 传统 Pilot 驱动估计/补偿 | 非神经传统基线 |
| T-DFE | `DFE-RLS` 与 `CFO+DD-Phase DFE-RLS` | 传统递推 | 长记忆和同步误差比较 |
| Oracle | Perfect CIR/tail 诊断 | 使用仿真上界 | 只定位可达性，不进主成功门槛 |

主论文 SNR 为 0/5/10/15 dB，主信道 Level B、最大记忆 116，prefix Pilot。另按 acquisition gap、tap drift、CFO 和慢相位分桶报告压力测试。

## 变量隔离

1. 先固定 checkpoint 与信道轨迹，比较 F0/F1/F2 和传统 MMSE，判断模型泛化、Pilot 条件和物理状态跟踪分别贡献多少。
2. 对 F2 逐次只改一个因素：仅 CFO/phase 跟踪、仅 sparse CIR 更新、tail 点估计/均值方差 belief、置信度门控、不同 Pilot 几何留出。其余因素固定。
3. 固定状态候选集，分别评估 Pilot BCE、复数重构、末端边界残差、innovation whiteness、后验不确定度和 posterior predictive risk，不允许同时改候选生成和代理公式。
4. 最后才比较状态跟踪 + 固定神经调制、有限 PEFT 和离散 RL 调度。任何动作只能在 Data 标签不可见时选择。

## Pilot 代理排序门槛

使用多个独立 seed/连续轨迹，冻结每个候选的 Pilot-only 分数；事后计算未来 H 帧 Data 硬 BER。按 SNR、delay、drift、CFO、phase noise、acquisition gap 单独报告：

- Spearman 与 Kendall 排序相关及 seed/轨迹 bootstrap 置信区间；
- top-1 候选误选率和最坏 BER regret；
- 分数置信区间覆盖率、OOD 检出率和 abstain 率；
- 当前帧/下一帧 prefix 留出对照，判断几何及时序偏移。

通过条件：每个主配置的相关方向均为正、置信区间不支持系统性负相关，且最坏 regret 不高于 Frozen；不能用跨配置总体平均抵消单配置负相关。未通过时代理不得控制更新，接收机保持 F1/Frozen。

## 泄漏检查

把 Data 发射标签替换为随机标签、把 Data 接收波形替换为随机复数值，Adapt Pilot 状态后验、Pilot 风险、动作与提交决策必须逐项保持相同。冻结接收机的事后 BER 可以改变。Reward Pilot 只用于代理留出或动作后 reward；不得读取其 Data 真值。
