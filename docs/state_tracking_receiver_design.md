# Pilot-only 状态跟踪接收机设计

状态：当前主路线规范。实现状态：冻结主干、Adapt Pilot 物理状态候选、门控提交和 soft-tail 均值/方差已落地；当前 Pilot 风险为物理一致性代理，尚未证明与未来 Data 硬 BER 稳定正相关。

分支操作约定见 [研究路线隔离说明](route_isolation.md)，当前阶段结果见 [阶段一诊断](state_tracking_stage1_diagnostic.md)。

## 研究目标

在 Level B `eme_long_memory_v2` 上离线训练长记忆 ISI 均衡器，在线仅通过帧前缀 Pilot 跟踪慢 tap 漂移、residual CFO、慢相位噪声和 acquisition 状态老化，持续降低 Data 段硬 BER。在线不通过 Data 标签更新状态、参数、动作或提交决策。

## 接收机状态

帧间维护一个 `PhysicalStatePosterior`：

| 状态 | 内容 | 在线来源 |
|---|---|---|
| CIR | 稀疏 tap 后验均值和逐 tap 方差代理 | acquisition 先验 + Adapt Pilot |
| CFO/phase | residual CFO、phase0、拟合置信度 | Adapt Pilot 相位残差 |
| innovation | Pilot 预测残差功率 | Adapt Pilot |
| tail belief | soft-tail 均值、方差、置信度 | 冻结接收机自身的 soft logits |

当前 `cir_variance` 是由 Pilot 创新构造的保守不确定度代理，尚不是经校准的完整 Kalman 协方差。研究报告必须保留这一实现边界。

## 单帧流程

1. acquisition 只初始化 CIR、CFO 和已知历史 tail。
2. 用上一帧状态及其 tail belief 建立先验。
3. 当前帧只用 Adapt Pilot 估计 phase/CFO，再在 acquisition 稀疏 support 上做 CIR 更新。
4. 置信度低、数值非有限或有效 Pilot 不足时，保留先验；成功时原子提交 CIR、phase/CFO 和 posterior 元数据。
5. 用当前后验构造神经条件，冻结模型对整帧接收波形做块均衡。
6. 从冻结均衡器自身的 soft 输出更新下一帧 tail 均值和方差。该步骤不访问真实 Data 发射符号或标签。
7. 最后仅在评估侧读取 Data 标签，计算硬 BER。

完整 Data 接收波形只用于冻结均衡器做检测；它不会流入物理状态估计或 Pilot 风险。Reward Pilot 的已知符号和接收波形仅能用于留出物理一致性评价，不能用 BCE 直接接受 PEFT/PPO 参数更新。

## Pilot 风险代理

`compute_pilot_risk` 默认只读取留出的 Reward Pilot，记录复数重构 NLL、连续采样点上的残差创新白化、留出段边界残差、tail 风险、后验不确定度和置信度。该分数当前仅记录与事后排序审计，不参与状态提交、参数更新或动作选择；`pilot_subset="adapt"` 只用于单独对照，不能再称为留出代理。`rank_pilot_candidates` 在候选区间无法区分或置信度低时 abstain。

要把该分数用于候选排序，必须在独立信道轨迹、未见 seed 上验证其对未来 H 帧 Data 硬 BER 的方向和校准。Pilot 位于帧前缀，无法对其后的任意突变提供保证；适用条件必须明确为 Pilot 可辨识且 Data horizon 内状态满足已验证的慢漂移模型。

## 可扩展方向

状态跟踪通过后，可对状态后验做 Monte Carlo 未来 rollout：从 CIR/CFO/phase/tail 后验及已标定的动态模型采样未来信道和虚拟 BPSK，通过冻结均衡器估计后验预测 Data BER。仅当独立数据上分桶排序通过，才允许它选择有限状态校正或离散调制动作。RL 只能在此后作为窗口调度器，不直接输出高维参数增量。
