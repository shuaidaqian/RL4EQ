# RL 信道均衡研究分析

## 当前判定

当前失效根因属于 Level B 三层叠加：

1. **物理状态层**：acquisition-to-data 老化、residual CFO、慢相位和稀疏 tap 漂移没有被稳定表示；长记忆 CIR 的小误差会通过 soft-tail 跨帧递推。
2. **代理目标层**：前缀 Reward Pilot BCE 与后续 Data 的位置、卷积边界和错误传播不同，已有 replay 已出现负相关；简单延长窗口或调学习率不能修复可辨识性。
3. **在线机制层**：PEFT/PPO 在冻结主干已获得大部分收益后继续改参数，容易破坏离线长多径反演；候选提交、tail 回滚和状态不确定度没有完全原子化。

因此当前主范式改为“冻结主干 + Pilot-only 物理状态后验跟踪”。PEFT、Bandit、PPO 只能在物理状态跟踪和 Pilot 风险排序单独通过后作为调度消融。

## 主接收机

离线阶段在在线 soft-tail 条件下做整帧 BCE 预训练。在线阶段只用 Adapt Pilot 更新 CIR/CFO/phase 后验，用 Pilot-only 风险检查状态一致性和不确定度；低置信度则 abstain。冻结均衡器使用全帧接收波形完成检测，随后从自身 soft logits 更新下一帧 tail belief。Data 标签只在评估阶段读取。

状态包括：稀疏 CIR 均值/方差、residual CFO、phase0、innovation variance、soft-tail 均值/方差/置信度。所有状态用 `PhysicalStatePosterior` 统一 clone、commit 和 rollback，避免只回滚网络参数而遗留错误尾状态。

## 代理边界

`agent/pilot_risk.py` 的默认风险由 Pilot 复数重构 NLL、创新白化、prefix 末端边界残差、tail 传播风险和后验不确定度组成。它不能保证任意未知时变信道下与未来 Data BER 正相关；正式使用前必须在独立轨迹逐配置检查 Spearman/Kendall、top-1 误选率、置信区间和 abstain 率。未通过时保持 Frozen。

## 验证顺序

先跑 F0 acquisition 冻结、F1 当前 Pilot 条件冻结、F2 Pilot 物理状态跟踪，再比较传统 MMSE/LMMSE/DFE-RLS。固定 checkpoint、轨迹和候选集合，分别消融 Pilot 几何留出、物理残差、后验不确定度、tail belief 和未来 rollout。只有 F2 和代理排序通过后，才允许研究有限 PEFT 或 RL 调度。

## 历史结论

旧 `Pilot-Driven Online Adaptation`、Reward Pilot BCE、Contextual Bandit 和窗口 PPO 结果保留为历史诊断/已被当前主路线取代。它们证明了“代理错配、状态传播和参数破坏”是根因，不能继续写成当前 proposed。旧实验文件的数值不删除，但必须关联 [接收机设计](docs/state_tracking_receiver_design.md) 和 [验证协议](docs/state_tracking_validation_protocol.md)。

历史路线名称为 `RL-Modulated Neural Block Equalizer`。所有比较仍以整帧缓冲、非因果接收机为边界，在 CFO、相位扰动下要求传统方法合理补偿，并以 proposed 明显超过传统为目标；在线不使用数据标签上界。
