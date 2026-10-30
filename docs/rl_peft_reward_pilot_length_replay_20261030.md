# Reward Pilot 长度对 Pilot-only PEFT 验收的影响（2026-10-30）

## 结论

在固定离线 checkpoint、Level B `heldout_edge` / `cfo_phase_mid` 信道、Pilot 总长 256 和相同 PEFT 更新方法后，把 Reward Pilot 从 32 增到 64（Adapt 从 224 减到 192），**没有证明在线更新能稳定优于冻结模型，也没有可靠解决低 SNR 有害更新**。

初始 5 个 seed 中，64 符号 Reward 拒绝了原先一条 0 dB 被错误接受、Data BER 恶化 8.9 个百分点的更新；但在独立的 5 个新 seed 中，它仍接受了两条 0 dB 有害更新，其中一条 BER 恶化 13.7 个百分点。其 Reward BCE 改善约 0.0134，Reward 硬错误数没有增加。也就是说，更多 Reward 符号有时能挡住坏更新，却不能可靠识别坏更新。

独立新 seed 上，四个 SNR 的 BER 改善 95% 区间都跨零；把两批共 10 个 seed 合并做描述性分析，四个 SNR 区间仍全部跨零。当前不应把 64 符号版本称为有效策略，也不应扩大正式矩阵或接入 Bandit。

## 实验设置

- 固定 checkpoint：`pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`，SHA256 `8CD3912D1C957C9689C51730AFF9D27A8F74942186C802A7483E5781AF65AEC4`；未重训、未修改。
- 信道：Level B `eme_long_memory_v2`，`heldout_edge`、`cfo_phase_mid`、最大时延 116、acquisition gap=0。
- 所有条件 Pilot 总长 256、前缀布局；32 组为 Adapt/Reward=224/32，64 组为 192/64。
- 在线更新固定为 16 个 `state-mapping + phase/CFO` PEFT 参数，Adapt 信号重构损失、学习率 0.0025、5 步、最大增量范数 0.05。
- 每条轨迹 5 帧：首帧 Adapt Pilot 更新；后两帧 Reward Pilot 用 BCE 与硬错误数门控；最后两帧用于事后 Data 评估。
- 比较了初始 seed 1091–1095 和独立确认 seed 2001–2005，每种配置均为 5 seed × 4 SNR。Reward 32/64 在每个 seed/SNR 上成对运行，信道状态元数据与 Data 测试帧索引匹配。
- 所有在线更新与门控均未使用 Data 波形或标签；Data 只用于决策后的评估。离线 checkpoint 在两组间完全相同。

需要注意：Pilot 总长固定时，增加 Reward 必然减少 Adapt；并且 Pilot-conditioned 的物理状态条件也会由更短的 Adapt 窗估计。因此这项实验评估的是固定预算下“更多验收、较少更新数据”的整体取舍，不是单独改变 Reward 样本数的纯消融。主比较使用每种分配各自的 Frozen 输出作基线。

## 独立确认 seed 结果

下表单位为 BER 百分点，正值代表在线方法优于同一 Pilot 分配下的 Frozen。置信区间以 5 个 seed 为统计单位；每个 seed/SNR 对最后两帧 Data 的改善取平均。

| SNR | Reward=32：平均改善（95% CI） | 正收益 seed；接受/有害接受 | Reward=64：平均改善（95% CI） | 正收益 seed；接受/有害接受 |
|---:|---:|---:|---:|---:|
| 0 dB | -0.313 pp（-2.939, +2.314） | 1/5；2/5 接受，1 条有害 | -2.786 pp（-10.839, +5.266） | 1/5；3/5 接受，2 条有害 |
| 5 dB | -0.013 pp（-0.158, +0.132） | 1/5；2/5 接受，1 条有害 | -0.052 pp（-0.197, +0.093） | 0/5；1/5 接受，1 条有害 |
| 10 dB | +0.234 pp（-0.419, +0.888） | 2/5；3/5 接受，1 条有害 | +0.273 pp（-0.488, +1.035） | 2/5；3/5 接受，1 条有害 |
| 15 dB | +0.065 pp（-0.164, +0.294） | 1/5；2/5 接受，1 条有害 | +0.078 pp（-0.186, +0.343） | 1/5；2/5 接受，1 条有害 |

64 相对 32 的配对改善在四个 SNR 上也都不确定，95% 区间跨零。初始 5 个 seed 上，64 在 0 dB 拒绝了全部候选，避免了一次 -8.9 pp 的退化；但中高 SNR 看似增益主要由单个 seed 拉动。初始 seed 用于方案筛选，因此 10 seed 合并结果仅作描述，不替代独立确认结论。

Reward BCE 与候选 Data BER 改善的 Spearman 相关在 64 符号组仍不稳定：按 10 个 seed 汇总，0/5/10/15 dB 分别为 -0.358、0.309、0.122、0.153。它说明验收分数和 Data 改善仍没有稳定的同向排序关系。

## 对路线的影响

这一步排除了“单纯增加 Reward Pilot 长度就能解决 Pilot 到 Data 迁移”的简单解释。Reward 64 在 0 dB 仍接受 seed 2005 的有害候选：Data BER 恶化 13.7 pp，候选 logits 平均变化 0.381；两帧 Reward BCE 的平均改善为 0.0134，硬错误数变化均为 0。把门控阈值稍作提高未必能识别这种情况，因为 Reward 指标确实明显改善。

观察到的 0 dB 有害候选有接近更新半径上限的参数偏移：该 seed 的 `state_mapping_delta_norm=0.0468`、`direct_phase_cfo_delta_norm=0.0177`，候选 logits 有明显变化。因此随后在新 seed 上比较了 0.01、0.025、0.05 三种更新半径；结果见[PEFT 更新半径筛查](rl_peft_update_radius_replay_20261030.md)。较小半径没有显示稳定优势，四个 SNR 的区间仍跨零。

这说明只在现有固定 checkpoint 上继续调整 Reward 长度或梯度更新幅度，尚未找到稳定有效的在线策略。若要继续追求“离线训练适应信道，在线用 Pilot 微调 PEFT 参数并稳定提升”，下一步需要讨论是否允许离线训练一个专门适配 Pilot 更新的 checkpoint；原 checkpoint 保留作 Frozen 对照。在线阶段仍可以维持 Data 隔离。未经这项路线对齐，不应把当前结果包装成成功。

## 原始回放与审计

原始逐帧数据位于 Git 忽略的 `logs/`：

- 初始 seed，Reward=32：`pilot_update_scope_mid_gap0_mapping_phase_bce_reward_5s4snr5f_20261030.json`
- 初始 seed，Reward=64：`pilot_update_scope_mid_gap0_mapping_phase_bce_reward64_5s4snr5f_20261030.json`
- 确认 seed，Reward=32：`pilot_update_scope_mid_gap0_mapping_phase_bce_reward32_confirm_5s4snr5f_20261030.json`
- 确认 seed，Reward=64：`pilot_update_scope_mid_gap0_mapping_phase_bce_reward64_confirm_5s4snr5f_20261030.json`

机器可读摘要见 [`docs/results/rl_peft_reward_pilot_length_replay_20261030.json`](results/rl_peft_reward_pilot_length_replay_20261030.json)。
