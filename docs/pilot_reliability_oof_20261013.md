# Pilot-only 可靠性校准留出 replay

> **2026-10-14 协议更正：**本页源自旧 `replay_pilot_candidate_ranking.py`
> 的排序日志及其派生留出校准结果，实际使用 legacy 默认信道、512 符号帧和
> clean 扰动，不能作为真实 EME Level B 或 600 秒老化的结论。历史数值保留作
> 诊断记录；修正协议后需要重新评估，见
> [Pilot replay 协议复核](pilot_replay_protocol_audit_20261014.md)。

日期：2026-10-13

## 目的

当前在线方法的关键问题是：Adapt Pilot 或 Reward Pilot 变好，不一定表示 Data 段会变好。
本轮先不修改离线 checkpoint，也不修改在线主流程，只验证一个候选方向：能否用在线可见的
Pilot/状态指标，在离线仿真中校准出一个候选可靠性分数，再预测候选更新是否会改善 Data。

Data 标签只作为离线仿真的目标和事后评估，绝不进入在线更新、在线 observation、Reward
验收或回滚。

## 实验设置

- Level B，`delay=116`，prefix Pilot，总长度 256；
- 固定 checkpoint：`pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`；
- acquisition 到数据帧间隔 600 秒；
- SNR：5/10/15 dB；seed：0/1/2；每条轨迹 4 帧；
- 候选：冻结、`phase_trend` Adapt Pilot BCE 更新、`phase_trend` Pilot 信号重构更新；
- 每个候选从同一帧初始模型出发，记录 Pilot 损失、Pilot 重构、状态、参数变化、Data 连续
  输出变化和 Data BER；
- 校准器：只使用 Pilot/状态/候选类型特征的岭回归；按完整 seed 留一验证；
- 输出：`logs/pilot_candidate_ranking_state_embedding_3s3snr4f_20261012.json`；
- 留出结果：`logs/pilot_candidate_ranking_state_embedding_3s3snr4f_20261012_oof.json`。

复现命令：

```powershell
.\.venv-gpu\Scripts\python.exe scripts/replay_pilot_candidate_ranking.py `
  --config configs/online_phase_signal_state_embedding_formal_moderate.json `
  --pretrained pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt `
  --output logs/pilot_candidate_ranking_state_embedding_3s3snr4f_20261012.json `
  --snrs 5 10 15 --seeds 0 1 2 --frames 4 --delay 116 --pilot-total 256 `
  --gap-seconds 600 --lr 0.005 --steps 5 `
  --candidates identity phase_trend_bce phase_trend_signal_reconstruction

.\.venv-gpu\Scripts\python.exe scripts/evaluate_pilot_reliability_oof.py `
  --input logs/pilot_candidate_ranking_state_embedding_3s3snr4f_20261012.json `
  --output logs/pilot_candidate_ranking_state_embedding_3s3snr4f_20261012_oof.json
```

## 结果

| 排序方式 | 与 Data BER 改善的 Spearman | 样本数 | 结论 |
|---|---:|---:|---|
| 现有 Reward loss 改善，包含冻结候选 | 0.093 | 108 | 不能排序 |
| 现有 Reward loss 改善，只看真实更新候选 | 0.149 | 72 | 不能排序 |
| Pilot-only 校准器，包含冻结候选 | -0.232 | 108 | 留出 seed 反而反向 |
| Pilot-only 校准器，只看真实更新候选 | -0.275 | 72 | 留出 seed 反而反向 |
| 预设通过线 | 0.600 | — | 未达到 |

策略级比较如下，改善单位为 Data BER 的绝对比例：

| 策略 | 更新覆盖率 | 平均 Data BER 改善 | 正收益试验比例 |
|---|---:|---:|---:|
| 始终冻结 | 0% | 0.0000 | 0.0% |
| 现有 Reward 排序 | 66.7% | +0.000217 | 13.9% |
| 留出可靠性排序 | 55.6% | -0.000217 | 11.1% |
| 事后 Data BCE 最优参考 | 72.2% | +0.000868 | 22.2% |

可靠性模型没有在留出 seed 上提供可用的排序能力。它的 Data BER 和 BCE 排序相关都为负，
策略平均 Data BER 还略差于冻结模型；因此不能把它接入在线 gate，也不能据此宣称在线微调
已经稳定超过离线模型。

## 当前判断

现有证据把卡点进一步缩小为：候选参数更新作用在 Data 段时，Data 的真实变化仍主要由未被
Adapt/Reward Pilot 充分观测的状态决定。简单增加 Reward 窗口、调门限、换一个离线可靠性
回归器，不能解决 Pilot 到 Data 的可迁移性。

这条实验仍然完成了一个必要的排除：在接入更复杂的 RL 或 Contextual Bandit 前，已经验证
“离线用 Data 标签训练一个 Pilot 可靠性分数”也不能在未见 seed 上稳定工作。

## 子窗口分离 replay 补充

随后把同一帧 192 个 Adapt Pilot 符号拆成前 96 个更新窗和后 96 个 holdout 窗。前窗
用于真实 Adapter 梯度更新，后窗只用于候选验收诊断；设置仍为 3 seed × 3 SNR × 4 帧。
结果文件为 `logs/pilot_subwindow_split_3s3snr4f_20261013.json`：

| Pilot-only 分数 | 与 Data BER 改善的 Spearman |
|---|---:|
| 后半 Pilot holdout BCE 改善 | -0.093 |
| 前半 Adapt BCE 改善 | 0.038 |
| Pilot 重构改善 | -0.007 |
| Data BCE（仅事后诊断） | 0.310 |

后半 Pilot 的独立验收没有达到 0.6 门槛，因此暂不接入主流程。它排除了“只要把同一 Pilot
拆成更新段和验收段就能自动恢复 Data 可迁移性”的简单解释，但也说明当前后半 Pilot 的
长度和位置仍不足以代表 Data 状态。

## 下一步

下一阶段应优先改变候选的可观测性，而不是继续拟合 gate：

1. 在固定 checkpoint 下，把同一帧的 Adapt Pilot 划分为多个互不重叠的 Pilot 子窗口；
2. 用前一子窗口生成参数候选，用后一子窗口做候选筛选，避免同一 Pilot 噪声同时负责更新和验收；
3. 只保留在多个子窗口和多个 seed 上都能改善连续输出的 Adapter 更新；
4. 再用完整 Reward Pilot 做最终接受/回滚；
5. 仍然先做 3 seed 小 replay，只有候选排序达到 Spearman 0.6 且最差 seed 不退化，才扩大到
   5 seed × 60 帧；
6. 如果子窗口分离仍不能预测 Data，必须如实保留当前结论：当前 Pilot 长度和模型结构不足以
   支撑“在线稳定超过离线”的研究声明，需要增加 Pilot 覆盖或改变均衡器可适配位置。

## 状态条件化范围更正（2026-10-14）

代码审计发现，上述候选 replay 虽然记录了 Pilot 状态特征，但候选模型前向和
`apply_adapt_only_peft_update` 没有收到状态 embedding。因此这些 OOF 结果测试的是
无状态条件化的 `phase_trend` 候选；它们不能作为“状态条件化 Adapter 已被 Pilot-only
排序验证”的证据。上述排序器泛化失败仍然有效，但只对其实际运行的候选和目标成立。

现在诊断路径支持显式传入 Adapt Pilot 状态 embedding，并新增了同条件有/无状态对照。
该对照仍未通过硬 BER 稳定性门槛，详见
[`pilot_peft_group_screen_20261013.md`](pilot_peft_group_screen_20261013.md) 的
“状态条件化 replay 的因果复核”。
