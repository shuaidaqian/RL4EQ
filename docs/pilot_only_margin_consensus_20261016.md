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

## 候选级 Data 事后诊断

本轮把完整 Data 前向严格移动到当前帧的 Pilot 排序、接受/拒绝、回滚和 Pilot
状态更新全部结束之后。每个候选的 PEFT 快照只在这个阶段临时恢复，记录以下指标，
然后恢复最终接受或冻结的 PEFT 参数：

- `data_bce_improvement_eval_only`、`data_ber_improvement_eval_only`；
- `data_logits_abs_delta_mean_eval_only`；
- `data_logits_direction_change_rate_eval_only`；
- `data_soft_output_delta_mean_eval_only`。

这些字段只进入 `reward_pilot_candidate_ranking` 日志，不能进入当前帧的候选排序、
Reward、接受/拒绝、回滚、CIR、soft-tail 或跨帧状态。最终帧级字段也使用同一
`_eval_only` 语义。新增的 `scripts/summarize_pilot_ranking_replay.py` 只做离线
统计，并在发现任何在线 Data 标志时直接报错。

## 独立 seed replay 结果

固定 checkpoint、Level B、delay=116、prefix Pilot=256（Adapt=224、Reward=32）、
SNR=5/10/15 dB、seed=2/3/4、每个配置 4 帧，比较
`pilot_stable_consensus` 和 `pilot_margin_consensus`。两组各有 36 个在线帧、108
个候选记录，所有在线行的 `data_waveform_used_online` 和 `data_labels_used_online`
均为 `false`。

| 排序 | Pilot 分数与 Data BCE Spearman | Pilot 分数与 Data BER Spearman | top-1 命中 Data BCE 最优 | top-1 命中 Data BER 最优 | 实际选中候选平均 Data BER 改善 | Data BER 改善比例 |
|---|---:|---:|---:|---:|---:|---:|
| `pilot_stable_consensus` | 0.457 | 0.673 | 88.9% | 88.9% | 0.0815 | 94.4% |
| `pilot_margin_consensus` | 0.597 | 0.743 | 88.9% | 88.9% | 0.0815 | 94.4% |

两种排序在全部 36 个帧上选择了同一个候选（32 次激进、4 次保守），所以这轮
margin 证据提高了排序分数与 Data BER 的离线相关性，但没有改变实际在线动作或
收益。实际选中候选的平均 Data BCE 改善为 1.8006，Data BCE 改善比例为 100%；
Data BER 改善在 5/10/15 dB 分别为 0.0795/0.0845/0.0804。

这说明当前 Pilot-only 方法已经能较稳定地挑出有益更新，尤其对硬 BER；但 BCE
相关性仍低于预设的 0.6 门槛，且当前 replay 只有 4 帧。正式 5 seed × 60 帧之前，
还需要完成更长窗口的稳定性复核，并把候选排序与冻结离线模型的逐配置 BER 差异
一起报告，不能只根据候选级 Data 诊断宣称目标已经完成。

本轮结果文件：

- `logs/replay_pilot_ranking_stable_20261016f/pilot_ranking_summary.json`
- `logs/replay_pilot_ranking_margin_20261016f/pilot_ranking_summary.json`
