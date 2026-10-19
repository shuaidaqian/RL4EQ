# 物理状态跟踪阶段一诊断

状态：迭代诊断，不是正式结果。测试日期：2026-10-19。旧版本 F2 的整帧 BER 受 Data-logit tail 回流影响，已判为不符合目标数据边界；下方“旧 joint 矩阵”只保留为缺陷定位证据。

## 固定条件

- 配置：`configs/level_b_state_tracking.json`
- 离线 checkpoint：`pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`
- 信道：Level B `eme_long_memory_v2`，最大记忆 116，`cfo_phase_tiny`
- Pilot：prefix，总数 256，其中 Adapt 224、Reward 留出 32
- 每个 SNR：2 个 seed，每条 8 帧；F0/F1/F2 使用相同数据轨迹和 checkpoint
- 在线冻结神经参数，Data 标签只用于事后 Data BER

## 旧 joint 矩阵（作废）

| SNR | F0 Acquisition | F1 Pilot-conditioned | F2 State tracking | F2-F1 |
|---:|---:|---:|---:|---:|
| 0 dB | 0.52791 | 0.23088 | 0.20492 | -0.02596 |
| 5 dB | 0.58211 | 0.05868 | 0.05949 | +0.00081 |
| 10 dB | 0.61320 | 0.00846 | 0.00895 | +0.00049 |
| 15 dB | 0.63737 | 0.00163 | 0.00130 | -0.00033 |

这组每格 5 seed×60 帧的结果曾显示 F2 在不同 SNR 有正有负增量，但旧实现把本帧整帧 logits 的尾部写回下一帧 tail belief。由于 logits 受到 Data 接收样本影响，违反“在线仅利用前缀 Pilot”的硬约束；F2 数字整体作废，不用于方法结论。F0/F1 不使用该 tail 回流，可继续作为对应旧轨迹上的诊断参考。F0 接近随机 BER 且随 SNR 不变，表明 acquisition 条件路径不适合作为性能基线，不解释为信道物理规律。

## Pilot-only tail 修复

`agent.state_tracking.update_tail_from_adapt_pilot` 现在只用前缀 Adapt Pilot 估计跨帧 tail：在先验 tail 与当前已知 Pilot 的卷积模型下做复数 MAP；只保留不含未知当前帧符号的观测行；当有效观测不足或求解失败时保留旧先验。协方差对角线设 `0.05` 的保守下限，当前置信度仍是近似量而非校准概率。`compare.py` 与 `online_train.py --method state` 均在检测前提交该状态，不再用检测 logits 回写。

数据隔离测试将 Data 接收波形替换为大幅随机复数值，并翻转 Data 标签；Adapt Pilot tail 后验的均值与方差保持完全相同。

## 合规组件短消融

固定 checkpoint、相同随机轨迹，F1 与各 F2 模式使用 3 seeds×12 帧；`phase_only`、`cir_only` 冻结 tail，`joint` 使用 Pilot-only tail MAP。每格包含 36 帧，仅用于方向诊断。

| SNR | F1 | F2 phase-only | F2 CIR-only | F2 joint |
|---:|---:|---:|---:|---:|
| 5 dB | 0.061089 | 0.057798 | 0.060872 | 0.062753 |
| 10 dB | 0.010236 | 0.010272 | 0.011393 | 0.010634 |

当前信号指向：5 dB 的有限收益来自 phase/CFO 条件，CIR 更新没有显示稳定增量；10 dB 三种模式均未超过 F1。不能据此宣称持续 BER 改善。下一步先检查相位后验 calibration 与 CIR innovation 门控，再跑主 SNR/多 seed 长窗口。

Pilot-only tail smoke（10 dB、1 seed×8 帧）中 F1 BER 为 `0.009440`、合规 joint 为 `0.009115`。tail 后验方差均值约 `0.05`、置信度约 `0.95`，仍需覆盖率校准；单 seed smoke 不构成收益证据。

## 状态更新诊断

- 0 dB：16 帧均因 `low_pilot_confidence` 拒绝 CIR/phase 候选；后验保持 acquisition/F1 对应状态。
- 5 dB：14/16 帧提交，2 帧因置信度不足冻结。
- 10 dB 和 15 dB：各 16/16 帧提交物理状态候选。
- 代理 abstain：0 dB 为 4/16 帧；其余配置为 0/16。代理没有参与状态提交或动作选择。

因此当前门控在低 SNR 有效阻止不可靠更新；在中高 SNR 下，状态更新对 BER 的边际方向仍依配置变化。下一轮应将状态估计贡献拆为 CFO/phase-only、CIR-only、joint，并按相同帧做配对 seed bootstrap；同时记录候选置信度、CIR 更新前后 innovation 和过期 acquisition gap。

## 运行问题

1. `eme_long_memory_v2_offline120_robustphase` checkpoint 使用 8 段 phase conditioner，而规范配置要求 4 段，compare 正确拒绝该组合。后续必须使用匹配配置，或显式训练并验证 4 段模型。
2. 传统 10 dB、1 seed×8 帧校准中 LMMSE-FIR BER 为 `0.293`、SC-FDE-MMSE 为 `0.504`、CFO-only LMMSE-FIR 为 `0.293`、CFO+decision-directed phase LMMSE-FIR 为 `0.068`。传统基线尚未满足可信主比较要求，但相位跟踪版本明显可用，后续应校验 phase tracker 与 tail 初始化并扩种子。
3. Pilot risk 仍只作为 Reward Pilot 留出指标记录，没有完成未来 Data BER 排序验收。`pilot_risk_is_validated_for_data_ber` 仍必须为 false。

完整逐帧输出保存在本机忽略目录 `logs/state_tracking_*_20261019/`，不作为源码提交。新一轮正式 F2 评估必须基于修复后的 Pilot-only tail 路径。
