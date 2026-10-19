# 物理状态跟踪阶段一诊断

状态：小样本诊断，不是正式结果。测试日期：2026-10-19。

## 固定条件

- 配置：`configs/level_b_state_tracking.json`
- 离线 checkpoint：`pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`
- 信道：Level B `eme_long_memory_v2`，最大记忆 116，`cfo_phase_tiny`
- Pilot：prefix，总数 256，其中 Adapt 224、Reward 留出 32
- 每个 SNR：2 个 seed，每条 8 帧；F0/F1/F2 使用相同数据轨迹和 checkpoint
- 在线冻结神经参数，Data 标签只用于事后 Data BER

## 结果

| SNR | F0 Acquisition | F1 Pilot-conditioned | F2 State tracking | F2-F1 |
|---:|---:|---:|---:|---:|
| 0 dB | 0.52791 | 0.23088 | 0.20492 | -0.02596 |
| 5 dB | 0.58211 | 0.05868 | 0.05949 | +0.00081 |
| 10 dB | 0.61320 | 0.00846 | 0.00895 | +0.00049 |
| 15 dB | 0.63737 | 0.00163 | 0.00130 | -0.00033 |

表内 BER 是每个配置 16 帧的均值，不应当作稳定总体估计。F2 在本轮不同 SNR 上有正有负的增量，尚未证明稳定优于 F1。F0 随 SNR 增加而更差是异常信号，优先检查 acquisition CIR/phase 条件进入神经 conditioner 的缩放和语义，不解释为信道物理规律。

## 状态更新诊断

- 0 dB：16 帧均因 `low_pilot_confidence` 拒绝 CIR/phase 候选；后验保持 acquisition/F1 对应状态。
- 5 dB：14/16 帧提交，2 帧因置信度不足冻结。
- 10 dB 和 15 dB：各 16/16 帧提交物理状态候选。
- 代理 abstain：0 dB 为 4/16 帧；其余配置为 0/16。代理没有参与状态提交或动作选择。

因此当前门控在低 SNR 有效阻止不可靠更新；在中高 SNR 下，状态更新对 BER 的边际方向仍依配置变化。下一轮应将状态估计贡献拆为 CFO/phase-only、CIR-only、joint，并按相同帧做配对 seed bootstrap；同时记录候选置信度、CIR 更新前后 innovation 和过期 acquisition gap。

## 运行问题

1. `eme_long_memory_v2_offline120_robustphase` checkpoint 使用 8 段 phase conditioner，而规范配置要求 4 段，compare 正确拒绝该组合。后续必须使用匹配配置，或显式训练并验证 4 段模型。
2. 试跑 `SC-FDE-MMSE` 和 `LMMSE-FIR` 时两者 Data BER 约为 0.48-0.50，未形成可接受的传统基线。该结果只表明当前 baseline 实现/调用尚待校准，不能用于宣称 proposed 优势。
3. Pilot risk 在本轮仅作为 Reward Pilot 留出指标记录；本轮没有做候选排序或未来 Data BER 相关性验收。`pilot_risk_is_validated_for_data_ber` 仍必须为 false。

完整逐帧输出保存在本机忽略目录 `logs/state_tracking_main_snr/`，不作为源码提交。
