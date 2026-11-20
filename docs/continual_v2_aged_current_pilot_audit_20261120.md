# 当前 Pilot 对照的 acquisition 先验老化审查

本实现最初位于隔离分支 `continual-aged-prior-control-20261120`，提交5dd01dc。P8全部完成并归档为a6e6d38后，已整合为7e9d080；主工作树全量 **324 passed in 31.62s**。下文保留原定义与隔离验证；不将旧P8结果回写为新对照结果。

新预登记P9a使用seed237001、四SNR四臂8帧，共128行，只核验真实入口、首帧匹配与schema3链路。协议见 `docs/protocols/continual_v2_aged_prior_smoke_dev_20261120.json`，短会话不证明研究性能目标。

## 原对照的解释边界

现有 `current_pilot` 每帧重新使用 acquisition 的均值和协方差，仅加入当前 Pilot。它不累计普通帧 CIR 后验，但也不计 acquisition 到当前时刻的物理老化。递归 AR(1) 则使用冻结的转移和过程协方差。因此长程两臂差值同时包含历史 CIR 观测与先验模型差异，不能全部归因于新增普通帧 Pilot 信息。

这不否定旧对照的可复现性，也不意味着新对照一定改善 BER。旧 `current_pilot` 保留为历史锚点。问题来自源码定义，不是看到中途表现后的筛选。

## 新对照定义

为 `arm=current_pilot` 增加显式选项 `current_pilot_prior=aged_acquisition`。默认仍为 `acquisition`，其他臂禁止使用老化单帧选项。

设首帧转移间隔为 s₀，当前已处理 k 个普通帧；每次重新从原始 acquisition 的 m₀、P₀ 计算累计间隔 s=s₀+k 的先验。随机游走与 AR(1) 都复用与递归臂相同的冻结转移函数。AR(1) 为

\[
m^-_k=\rho^s m_0,\qquad
P^-_k=\rho^{2s}P_0+(1-\rho^{2s})D.
\]

然后仅使用当前合法 Adapt 观测更新一次、解码一次。上一普通帧的 CIR 后验不作为下一帧先验。这里仍是与递归臂匹配的近似状态模型，不宣称精确条件贝叶斯后验或最优单帧接收机。

| 臂 | 当前 Pilot 前的 CIR 先验 | 当前 Pilot 更新 |
|---|---|---|
| acquisition Frozen | 固定 acquisition | 无 |
| 历史 current_pilot | 固定 acquisition | 一次 |
| 老化先验 current_pilot | 从 acquisition 按累计物理时间预测 | 一次 |
| recursive_pilot | 从上一普通帧后验预测一个间隔 | 一次 |

所有臂继续使用合法 acquisition 残差估噪、相同 checkpoint、同类 Pilot 同步与 soft-tail 规则。CFO 不读取真值，神经权重冻结。各自 soft-tail 与同步轨迹可能随解码而改变；如果后续要将整个递归策略的增量进一步拆分到单一模块，仍需共享上下文等专门消融。

## 实现与测试

- 接收机仅在显式单帧老化选项下增加累计预测；历史默认路径及现有递归逻辑保留。
- 比较清单验证选项与臂兼容，协议显式记录实际选项，真实入口将其传入接收机。
- 定向检查直接截取 Pilot 更新实际收到的先验，核验 RW/AR(1)、0/1/3.5 首帧间隔及后续累计时序，强制改变上一后验以确认不会带入下一帧。
- 实际接收路径验证：首帧匹配老化对照与递归臂的 logits、CIR、先验迹及调用/更新次数逐位一致。

相关定向测试 **75 passed in 4.23s**；隔离工作树全套 **302 passed in 33.90s**。这些仅验证定义、时序与实现，不证明新对照的 BER 或新增观测收益。

整合发生于P8完成并归档之后；真实入口和匹配比较采用新协议、新seed。已完成的P8仍采用原current_pilot，它与新对照不能混为同一结果。
