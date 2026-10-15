# 严格 Pilot-only 在线排序复核（2026-10-16）

本轮目标是继续寻找只依赖 Pilot 的候选排序方法。离线 checkpoint、Level B
信道、delay=116、prefix Pilot=256（Adapt=224、Reward=32）均保持不变。
Data 波形和 Data 标签不参与在线更新、候选排序、Reward 守门、接受、拒绝、
回滚、策略状态或跨帧 soft-tail/CIR 状态。

## 在线边界

- Adapt Pilot：产生 Adapter/PEFT 候选参数更新。
- Reward Pilot：计算 BCE、硬判错数、复数信号重构误差和软符号边际，负责候选排序、
  接受、拒绝和回滚。
- Data：所有在线决策完成后才做完整帧前向，只记录 BER、BCE 和 logits 变化等事后指标。
- CIR 决策导向更新：卷积输入只保留已知 Adapt Pilot 和历史已知 soft-tail；未知区域
  不再用模型硬判决填充。
- 神经、传统和历史 RL 入口：在线状态更新先使用 Data 清零的 Pilot-only 前向，
  完整接收帧只用于最后评估输出。

## 新排序方法

新增 CLI 值：

```text
--online-reward-ranking pilot_margin_consensus
```

它针对 Reward Pilot 只有 32 个符号时硬 BER 过于离散的问题，要求三个连续证据
同时支持候选：

1. Reward Pilot BCE 子窗口改善的稳健下置信界；
2. Reward Pilot 复数信号重构相对改善；
3. 已知 Pilot 符号对应的软 logits 边际改善。

第 3 项定义为 `target_sign * logit` 的平均变化，并按 Reward Pilot 子窗口计算。
排序分数取三项稳健改善的最小值，再乘 BCE 子窗口稳定性。Reward Pilot 硬判错数
不增加仍是独立硬门控条件。任何一项没有改善，候选就不会因为另一项分数很高而被选中。

## 小样本 replay

命令固定为 Level B、delay=116、SNR=5/10/15 dB、seed=0、4 帧、每帧尝试更新、
固定调度器。配置加入三个同一 `phase_trend` PEFT 组候选：保守、中等和激进。

日志：

- `logs/replay_pilot_ranking_stable_20261016d/`
- `logs/replay_pilot_ranking_margin_20261016d/`

在 12 个在线帧中，两种排序都保留候选 11 帧，跳过 1 帧；两者最终选择相同的
激进候选，事后 Data BER 完全相同。新增软边际证据改变了部分候选的分数，但在
这一条 seed 轨迹上没有改变最终选择，也没有形成“新排序优于旧排序”的证据。

这组 replay 只说明：

- Adapter 参数确实改变了 Data logits；
- 在线日志的 `data_waveform_used_online` 全部为 `false`；
- 新排序可以在真实入口运行，并且候选记录包含 BCE、信号重构和软边际指标。

这组 replay 不能说明在线微调已经稳定超过离线模型。当前配置下 Data BER 仍受
checkpoint 与 acquisition 条件影响，且只有一个独立 seed；后续必须在新的 seed、
多个 SNR 和更长帧数上验证候选排序是否能稳定保留有益更新。

## 验证

已通过的局部测试包括在线更新边界、CIR unknown-region 不变性、排序函数、窗口
Reward 和历史 RL runner。完整测试仍需在本轮所有修改完成后重新运行。

当前明确不做：decision-directed Data 伪标签、Data 驱动 Reward、Contextual Bandit
扩展和离线 checkpoint 重训。
