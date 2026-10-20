# 物理状态跟踪阶段一诊断

状态：迭代诊断，不是正式成功结论。数据日期：2026-10-20。F2 已在本 checkpoint 和本 Level B 主矩阵上超过带 CFO/相位补偿的传统 DFE-RLS，但尚未在所有主 SNR 配置上稳定超过 F1 冻结神经对照。

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

## 传统基线正式复核

在同一 checkpoint、同一 Level B 配置、同一 5 seeds × 60 帧轨迹上，单独复核了带 Pilot CFO 线性拟合和 decision-directed phase tracker 的 `CFO+DD-Phase DFE-RLS`。结果来自 `logs/compare_baseline_key_5s60f_20261020/summary.json`，每格 300 帧：

| SNR | F2 Pilot State Tracking | CFO+DD-Phase DFE-RLS | F2 - 基线（seed bootstrap 95% CI） |
|---:|---:|---:|---:|
| 0 dB | 0.215825 | 0.439926 | -0.224102 [-0.250139, -0.199965] |
| 5 dB | 0.060829 | 0.157821 | -0.096992 [-0.120933, -0.073668] |
| 10 dB | 0.009931 | 0.054427 | -0.044497 [-0.063876, -0.027465] |
| 15 dB | 0.001189 | 0.032839 | -0.031649 [-0.051020, -0.016367] |

负差值表示 F2 BER 更低。区间按 5 个 seed 的逐 seed 平均差做 100,000 次 percentile bootstrap；只有 5 个 seed，区间仍是有限证据。该结果满足“先超过合理传统非神经基线”的阶段门槛，但不等于在线状态跟踪已经稳定带来神经模型增益：5 dB F2 比 F1 高约 0.35 个百分点，10 dB 基本持平。传统基线逐帧日志位于 `logs/compare_baseline_key_5s60f_20261020/frame_metrics.jsonl`，不会提交到源码仓库。

## 数据边界修复

旧版 F2 曾用整帧检测 logits 更新下一帧 tail belief。因为检测波形含 Data 接收样本，这条结果不符合在线信息边界，数字作废，仅保留作缺陷定位证据。

当前 `agent.state_tracking.update_tail_from_adapt_pilot` 仅使用前缀 Adapt Pilot 与上一帧 tail 先验做复数 MAP；卷积观测行只在输入完全由当前已知 Pilot 或先验 tail 解释时保留。有效行不足或求解失败时保持先验。Data 波形替换和 Data 标签翻转测试确认 tail 后验不变。该实现满足数据隔离，但其方差当前有 `0.05` 下限，主矩阵所有 SNR 的均值都贴近该下限，不能把对应 `0.95` 左右置信度视为已校准概率。

## 组件短消融

固定 checkpoint 和相同轨迹，先做了 3 seeds × 12 帧方向诊断，随后按同一 5 seeds × 60 帧主矩阵扩展 phase_only 与 cir_only；三种模式都遵守 Pilot-only tail 边界。

| SNR | F1 | phase-only | CIR-only | joint |
|---:|---:|---:|---:|---:|
| 5 dB | 0.061089 | 0.057798 | 0.060872 | 0.062753 |
| 10 dB | 0.010236 | 0.010272 | 0.011393 | 0.010634 |

短消融暗示 phase/CFO 可能有条件性收益，CIR 与 joint tail 更新没有显示稳定增量。样本量不足以确定哪一项导致正式矩阵 5 dB 退化。

5 seeds × 60 帧扩展结果如下，使用 `logs/compare_phase_only_5s60f_20261020/summary.json` 和 `logs/compare_cir_only_5s60f_20261020/summary.json`：

| SNR | F1 | phase_only | cir_only | joint |
|---:|---:|---:|---:|---:|
| 0 dB | 0.223924 | 0.223924 | 0.223924 | 0.215825 |
| 5 dB | 0.057309 | 0.060530 | 0.057591 | 0.060829 |
| 10 dB | 0.009566 | 0.009531 | 0.011332 | 0.009931 |
| 15 dB | 0.001615 | 0.001385 | 0.003446 | 0.001189 |

phase_only 在 10/15 dB 有小幅改善，在 5 dB 仍退化；cir_only 在 5 dB 接近 F1，但在 10/15 dB 退化明显；joint 在 15 dB 最好、5 dB 退化。说明状态更新收益依赖 SNR 和状态可辨识度，不能把当前联合更新直接宣称为稳定在线增益。下一步应只用 Adapt Pilot 的置信度、创新和状态候选变化量做门控 replay，并保留 F1 作为安全 fallback。

## Pilot-only 排序审计

对主矩阵逐帧计算 F2 相对 F1 的事后 BER 差值，并只用 F2 日志中在线可见的 Pilot 代理字段做相关性审计。5 dB 下 `pilot_risk_score` 与该差值的 Spearman 相关为 `0.140`（p=`0.015`），`residual_cfo` 为 `0.197`；10 dB 分别为 `0.122` 和 `0.085`；0 dB 因所有状态更新均 abstain，15 dB 的风险分数相关接近 0。相关性很弱，且离线按 Data BER 选择阈值会产生选择偏差，不能直接部署成在线门控。

因此当前 Pilot-only 代理尚未通过独立轨迹的排序准入门槛。Reward Pilot 风险继续只记录、不控制状态提交；后续必须在未参与阈值选择的连续信道轨迹上做预注册阈值和 bootstrap 验证。

## heldout_edge 压力测试

为检验强状态失配，固定同一离线 checkpoint、Level B、delay=116、prefix Pilot 和 5 seeds × 60 帧，在 `state_split=heldout_edge` 上比较 F1、F2 和 `CFO+DD-Phase DFE-RLS`。结果来自 `logs/compare_heldout_edge_5s60f_20261020/summary.json`，每格 300 帧：

| SNR | F1 | F2 | CFO+DD-Phase DFE-RLS | F2 - F1（seed bootstrap 95% CI） |
|---:|---:|---:|---:|---:|
| 0 dB | 0.218142 | 0.218941 | 0.446719 | +0.000799 [-0.008155, +0.011285] |
| 5 dB | 0.055881 | 0.063902 | 0.211589 | +0.008021 [+0.002378, +0.012109] |
| 10 dB | 0.009462 | 0.011293 | 0.089089 | +0.001832 [-0.000260, +0.003815] |
| 15 dB | 0.002014 | 0.002183 | 0.062977 | +0.000169 [-0.000126, +0.000464] |

压力测试下 F2 仍明显优于传统基线，但四个 SNR 都没有优于 F1；5 dB 的退化约 0.80 个百分点，且 bootstrap 区间不跨 0。F2 物理状态提交 732/1200 次，Data 标签使用次数为 0，神经参数更新次数为 0。这个结果否定了“状态失配越强，当前 joint 跟踪越能稳定带来在线收益”的假设，说明 Pilot 估计误差会被传给冻结网络，必须先解决候选状态的安全筛选或回退。

## 状态提交与代理

主矩阵中物理状态提交统计：0 dB 0/300 次提交，全部因 `low_pilot_confidence` abstain；5 dB 212/300 次提交、88 次拒绝；10 dB 300/300 次提交；15 dB 299/300 次提交、1 次拒绝。0 dB 下 F2 与 F1 的差异主要不是已提交 CIR/phase 更新造成，后续要先核对 F1/F2 的 tail 与状态初始化等路径是否完全一致。5 dB 的状态更新接受率高但 BER 反而恶化，说明当前置信度门控不能代表状态候选对均衡器有益。

实现审计发现 CIR 与 phase 候选当前共用 phase confidence 门槛；这会把不同可辨识度的状态量绑在同一个接受条件上。下一轮应分别记录和门控 CIR innovation、phase/CFO 残差和 tail posterior predictive residual，并先做固定轨迹的 replay 归因，再改接受逻辑。

Reward Pilot 风险仍只作留出物理一致性记录，`pilot_risk_validated_for_data_ber` 为 false；不得用其控制在线状态更新。完成独立轨迹排序审计前，不能声称代理与未来 Data BER 正相关。

## 原子提交复核与修复

2026-10-20 复核发现，旧实现的 `joint` 路径在 CIR/phase 候选被拒时仍会基于先验 CIR/phase 计算并提交 Adapt Pilot tail 更新。heldout 旧日志中 468 次状态拒绝全部伴随 tail 更新，因此旧 F2 矩阵不符合“CIR、phase/CFO、tail 整体接受或整体冻结”的状态提交契约，不能作为修复后方案的性能结果。

当前实现统一使用 `is_safe_to_commit` 判断状态候选是否通过原有置信度门槛；只有通过时才继续估计并提交 joint tail。拒绝时保留完整先验，逐帧日志的 `physical_state_update` 和 `tail_update_source` 表示实际接受结果。Adapt Pilot 仍是唯一更新输入，Reward Pilot 与 Data 标签不参与提交决策。

修复后 smoke（heldout_edge、5 dB、seed 0、4 帧）中 2 帧候选被拒，两帧均保持 `frozen_prior` tail；契约测试通过。随后固定 checkpoint 的 heldout_edge 5 seeds × 60 帧全 SNR F1/F2 重跑完成，日志为 `logs/compare_heldout_edge_atomic_f1f2_5s60f_20261020/frame_metrics.jsonl`（本机忽略目录）。

修复后结果如下，差值为 F2-F1，负值才表示 F2 更好：

| SNR | F1 | 原子提交 F2 | F2-F1 | F2 接受数 |
|---:|---:|---:|---:|---:|
| 0 dB | 0.218142 | 0.218142 | +0.000000 | 0/300 |
| 5 dB | 0.055881 | 0.059371 | +0.003490 | 126/300 |
| 10 dB | 0.009462 | 0.011124 | +0.001662 | 291/300 |
| 15 dB | 0.002014 | 0.002231 | +0.000217 | 291/300 |

1200 个 F2 逐帧记录中，所有被拒绝帧都记录为 `tail_update_source=frozen_prior`。原子修复消除了数据边界/提交契约错误，但没有带来在线超过 F1 的收益；在强失配 heldout_edge 上，5/10/15 dB 仍退化，0 dB 因全部 abstain 与 F1 相同。因此下一步不能扩大矩阵或引入 PEFT，而应先针对通过门槛的候选做 Pilot-only 事后归因和更严格的安全回退设计。

另增加仿真事后 tail 诊断：tail NMSE、按后验方差归一化误差，以及复高斯 50/90/95% 覆盖率。这些字段只使用仿真真实 tail 做评估，不输入在线接收机或候选门控。当前 joint tail 方差存在 0.05 下限，覆盖率需结合正式重跑解释，不能直接当作校准证明。

### 原子提交后的 tail 校准结果

在修复后的 `online_train.py --method state` 路径运行 Level B 主配置 5 seeds × 60 帧，完整输出位于 `logs/tail_calibration_atomic_5s60f_20261020/online_metrics.json` 和对应 `frame_metrics.jsonl`。指标只在仿真评估阶段读取 `frame.tail_symbols`；接收机状态更新前后均没有使用该真值。

| SNR | 接受帧数 | 接受帧 tail NMSE | 标准化平方误差均值 | 50% 覆盖率 | 90% 覆盖率 | 95% 覆盖率 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 dB | 0/300 | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 |
| 5 dB | 152/300 | 1.657 | 33.14 | 0.375 | 0.456 | 0.470 |
| 10 dB | 295/300 | 1.416 | 28.31 | 0.189 | 0.271 | 0.294 |
| 15 dB | 299/300 | 1.301 | 26.01 | 0.143 | 0.213 | 0.236 |

50/90/95% 覆盖率均按复高斯标量方差定义。接受帧上尾状态的均方误差明显大于其报告方差，且覆盖率随 SNR 升高反而下降；`0.05` 下限导致不同帧方差贴边，不能解释成可靠的后验协方差。0 dB 全部候选被拒，故该 SNR 没有可评价的更新后验，拒绝后保留旧 tail 是原子冻结行为，不应把旧先验的归一化误差当成新后验校准率。

这项结果否定了当前“连续复数 MAP 均值 + 线性协方差”作为 BPSK 跨帧 tail 后验的校准模型。下一版应显式建模符号离散性、CIR/CFO/phase 不确定度传入的模型误差，以及 acquisition 状态老化；训练/回放只用 Adapt Pilot 形成候选，仿真真实 tail 仅用于独立校准。新模型通过覆盖率与 BER 配对评估前，不得用 tail 方差控制状态接受或声称 tail belief 已校准。

## 下一阶段

1. 对逐帧日志按相同 seed/frame 对齐 F1 与 F2，拆解状态候选提交前后的 CIR、phase/CFO、tail 变化和 BER 差值；重点定位 5 dB 接受更新为何有害。
2. 重建 tail 后验：将 BPSK 符号先验、Pilot-only 似然和 CIR/CFO/phase 不确定度合并，使用独立轨迹校准覆盖率；真实 tail 仍只做事后校准。
3. 解耦 CIR、phase/CFO 的估计置信度和提交门槛，再进行单变量 replay；同时保留严格 F1 fallback。
4. 扩大同一传统基线对照的独立 seed 数，确认目前四个 SNR 上的优势区间在更多轨迹上仍成立。
5. 在独立连续轨迹上完成 Pilot 代理对未来窗口 Data BER 的分桶排序与 bootstrap；未通过正向关联门槛时，代理继续只记录、不控制。

完整逐帧输出保存在本机忽略目录 `logs/state_tracking_compliant_main_5s60f_20261019/` 和 `logs/compare_baseline_key_5s60f_20261020/`。任何正式主结果都必须使用 Pilot-only tail 路径和上述隔离协议。
