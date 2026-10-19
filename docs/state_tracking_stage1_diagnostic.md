# 物理状态跟踪阶段一诊断

状态：迭代诊断，不是正式成功结论。数据日期：2026-10-19。当前 F2 尚未在所有主 SNR 配置上稳定降低 Data 硬 BER，也尚未证明超过合格的传统接收机基线。

## 当前合规主矩阵

- 配置：`configs/level_b_state_tracking.json`
- 离线 checkpoint：`pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`
- 信道：Level B `eme_long_memory_v2`，最大记忆 116，`cfo_phase_tiny`
- Pilot：prefix，总数 256，其中 Adapt 224、Reward 留出 32
- 方法：F0 acquisition frozen、F1 Pilot-conditioned frozen、F2 Pilot state tracking frozen；相同 checkpoint 和信道轨迹
- 每个方法/SNR：5 seeds × 60 帧；每格 300 帧
- 在线神经参数冻结；物理状态只由 Adapt Pilot 更新；Data 标签只用于事后硬 BER
- 逐帧日志：`logs/state_tracking_compliant_main_5s60f_20261019/frame_metrics.jsonl`（本机忽略目录，不随源码提交）

| SNR | F0 Acquisition | F1 Pilot-conditioned | 合规 F2 joint | F2-F1 配对差值（seed bootstrap 95% CI） |
|---:|---:|---:|---:|---:|
| 0 dB | 0.503151 | 0.223924 | 0.215825 | -0.008099 [-0.015807, -0.000391] |
| 5 dB | 0.505404 | 0.057309 | 0.060829 | +0.003520 [+0.000221, +0.007977] |
| 10 dB | 0.506254 | 0.009566 | 0.009931 | +0.000365 [-0.000417, +0.001141] |
| 15 dB | 0.506966 | 0.001615 | 0.001189 | -0.000425 [-0.000985, -0.000061] |

负差值表示 F2 BER 较低。F2 在 0 dB 和 15 dB 有统计方向上的改善，5 dB 明显退化，10 dB 与 F1 接近。因此目前不能声称 F2 稳定降低 BER，更不能声称已满足主目标。F0 近似随机 BER，说明 acquisition-only 条件不是有效性能基线；F1 才是判断物理跟踪增量的关键神经对照。

## 数据边界修复

旧版 F2 曾用整帧检测 logits 更新下一帧 tail belief。因为检测波形含 Data 接收样本，这条结果不符合在线信息边界，数字作废，仅保留作缺陷定位证据。

当前 `agent.state_tracking.update_tail_from_adapt_pilot` 仅使用前缀 Adapt Pilot 与上一帧 tail 先验做复数 MAP；卷积观测行只在输入完全由当前已知 Pilot 或先验 tail 解释时保留。有效行不足或求解失败时保持先验。Data 波形替换和 Data 标签翻转测试确认 tail 后验不变。该实现满足数据隔离，但其方差当前有 `0.05` 下限，主矩阵所有 SNR 的均值都贴近该下限，不能把对应 `0.95` 左右置信度视为已校准概率。

## 组件短消融

固定 checkpoint 和相同轨迹，3 seeds × 12 帧；phase_only 与 cir_only 冻结 tail，joint 使用 Adapt Pilot MAP tail。每格 36 帧，只能作方向诊断。

| SNR | F1 | phase-only | CIR-only | joint |
|---:|---:|---:|---:|---:|
| 5 dB | 0.061089 | 0.057798 | 0.060872 | 0.062753 |
| 10 dB | 0.010236 | 0.010272 | 0.011393 | 0.010634 |

短消融暗示 phase/CFO 可能有条件性收益，CIR 与 joint tail 更新没有显示稳定增量。样本量不足以确定哪一项导致正式矩阵 5 dB 退化。

## 状态提交与代理

主矩阵中物理状态提交统计：0 dB 0/300 次提交，全部因 `low_pilot_confidence` abstain；5 dB 212/300 次提交、88 次拒绝；10 dB 300/300 次提交；15 dB 299/300 次提交、1 次拒绝。0 dB 下 F2 与 F1 的差异主要不是已提交 CIR/phase 更新造成，后续要先核对 F1/F2 的 tail 与状态初始化等路径是否完全一致。5 dB 的状态更新接受率高但 BER 反而恶化，说明当前置信度门控不能代表状态候选对均衡器有益。

实现审计发现 CIR 与 phase 候选当前共用 phase confidence 门槛；这会把不同可辨识度的状态量绑在同一个接受条件上。下一轮应分别记录和门控 CIR innovation、phase/CFO 残差和 tail posterior predictive residual，并先做固定轨迹的 replay 归因，再改接受逻辑。

Reward Pilot 风险仍只作留出物理一致性记录，`pilot_risk_validated_for_data_ber` 为 false；不得用其控制在线状态更新。完成独立轨迹排序审计前，不能声称代理与未来 Data BER 正相关。

## 传统基线边界

10 dB、1 seed × 8 帧初步校准中，LMMSE-FIR 为 0.293、SC-FDE-MMSE 为 0.504、CFO-only LMMSE-FIR 为 0.293、CFO+decision-directed phase LMMSE-FIR 为 0.068。该样本不足以比较优劣，但裸传统基线显然缺少公平的相位补偿；完成各传统方法的 Pilot CFO/phase 补偿和多 seed 校准前，不能据此宣称 Proposed 超过传统基线。

## 下一阶段

1. 对逐帧日志按相同 seed/frame 对齐 F1 与 F2，拆解状态候选提交前后的 CIR、phase/CFO、tail 变化和 BER 差值；重点定位 5 dB 接受更新为何有害。
2. 独立校准 tail MAP 后验：记录已知仿真 tail 真值下的均值误差、区间覆盖率、跨帧相关误差和冻结/更新对照；移除或重新解释固定方差下限前，不把其作为风险置信度。
3. 解耦 CIR、phase/CFO 的估计置信度和提交门槛，再进行单变量 replay；同时保留严格 F1 fallback。
4. 校准带合理 Pilot CFO/phase 补偿的 SC-FDE-MMSE、LMMSE-FIR、DFE/RLS 传统基线，并在相同 Level B 轨迹上扩展 seeds 和帧数。
5. 在独立连续轨迹上完成 Pilot 代理对未来窗口 Data BER 的分桶排序与 bootstrap；未通过正向关联门槛时，代理继续只记录、不控制。

完整逐帧输出保存在本机忽略目录 `logs/state_tracking_compliant_main_5s60f_20261019/`。任何正式主结果都必须使用 Pilot-only tail 路径和上述隔离协议。
