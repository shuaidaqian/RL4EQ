# RLS 在线 PEFT 阶段复核（2026-10-02）

> 最新正式证据为下方的 5 seeds × 60 帧矩阵。此前 3 × 12 帧结果只保留为调试过程，
> 不再作为方法有效性的主要依据。

## 目标和边界

本阶段保持现有离线 checkpoint 和 Level B 信道结构不变，检查不使用 RL 的 Pilot 驱动
残差 Adapter 是否能在在线阶段改善均衡。离线 checkpoint 为
`pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`。在线更新只读取前缀
Adapt Pilot；Reward Pilot 只做候选验收和回滚；Data 标签只在实验结束后计算 BER。

RLS Adapter 是动态挂载的零初始化线性 logit 残差层。离线主干、head 和 checkpoint
结构不变。在线参数更新确实发生时，`peft_update_applied=true`；CIR 固定，因此 BER
差异不能归因于 CIR 更新。所有结果均为配对 replay，使用同一 seed、信道轨迹和 Frozen
Offline NN checkpoint。

## 已完成的实现保护

1. 零参数变化不再计为一次 PEFT 更新。
2. 单帧 Adapter 参数增量受 `online_adaptation_max_delta_norm` 限制。
3. 新增相对零初始化离线 Adapter 的累计参数距离限制
   `online_rls_max_total_delta_norm`，默认 Level B 配置为 `0.5`。
4. Reward Pilot 拒绝候选时恢复 Adapter 参数和 RLS covariance；审计记录报告回滚后真实
   的累计参数距离。
5. 汇总审计包含每帧是否真正保留 PEFT 更新及累计参数距离；所有测试结果中的
   `data_labels_used_online` 均为 `false`。

## 结果

### Level B 正式矩阵：5 seeds × 60 帧

日志目录：`logs/rls_heldout_edge_formal_5s60f_20261002/`。每个 SNR 含 5 个 seed、
每个 seed 连续 60 帧；共 1200 个 Frozen/Online 配对帧。两种方法的每个配对帧具有
相同的 `state_instance`，状态条件来源均为 `pilot_cir_phase`，CIR 参数固定。比较量为
`BER_frozen - BER_online`，正数代表在线更好。95% 区间按 seed 重采样，再在每个抽中
seed 内随机抽取一个连续 10 帧块，重复 10,000 次。

| SNR | Frozen BER | Online BER | 配对收益 | 配对收益 95% CI | seed 方向 | 保留 PEFT 更新 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 dB | 21.9944% | 23.1328% | -1.1385 pp | [-2.9871, +0.0652] pp | 0/5 改善，1 持平，4 退化 | 10/300 |
| 5 dB | 5.5447% | 5.7700% | -0.2253 pp | [-0.6146, +0.0755] pp | 2/5 改善，3/5 退化 | 21/300 |
| 10 dB | 0.9462% | 0.9379% | +0.0082 pp | [-0.0052, +0.0260] pp | 4/5 改善，1 基本持平 | 73/300 |
| 15 dB | 0.2027% | 0.2031% | -0.0004 pp | [-0.0026, 0.0000] pp | 4/5 持平，1 轻微退化 | 23/300 |

所有区间都包含零。低 SNR 没有稳定增益，0 dB 的平均退化较大；10 dB 的正向均值
只有 `0.0082` 个百分点，统计区间仍跨零；15 dB 基本持平。当前实验不能证明 RLS
PEFT 稳定超过 Frozen Offline NN。

RLS 的有效 Adapt Pilot 候选通过次数在 0/5/10/15 dB 分别为
`112/66/156/94`；Reward Pilot 最终放行并保留的参数更新分别只有
`10/21/73/23` 次。所有逐帧记录均为 `data_labels_used_online=false`，Frozen 与 Online
配对状态实例不一致的记录数为 0。逐 SNR 区间由原始 `frame_metrics.jsonl` 计算，
不能用 `summary.json` 中针对 Online 原始 BER 的全局 bootstrap 代替。

### 结果解释

更新器每帧使用 108 个有效 Adapt Pilot 样本，为 64 维隐藏特征加偏置拟合 65 个残差
参数。低 SNR 时冻结主模型的 Pilot 判决不可靠，错误 Pilot 会产生方向错误的残差目标；
而 Reward Pilot 总共只有 32 个符号，即使分成窗口也只能检查少量 Pilot 上的 loss/BER，
无法保证候选对 Data 段有泛化收益。高 SNR 下错误较少，但残差改善空间也很小。这与
观察到的“低 SNR 有退化、高 SNR 收益接近零”一致。

这是一种基于现有记录和更新目标的机制解释，不是已完成的因果分解。跨帧比较同时包含
PEFT 参数变化及其后续 soft-tail 递推影响。下一步继续测试更强的 Level B 状态失配时，
应将其作为单独标注的诊断条件；当前冻结的 `eme_long_memory_v2/cfo_phase_tiny` 主配置
和离线 checkpoint 均保持不变，并先用传统非神经 baseline 检查诊断工作区是否可用。

下表 BER 使用百分数。`W1/W2` 表示将 Reward Pilot 分成 1/2 个窗口，所有对照为
3 seed × 12 帧，SNR 为 10/15 dB，Level B，`heldout_edge`，prefix Pilot 总数 256，
checkpoint 和状态条件相同。单帧参数增量上限均为 `0.5`。

| 设置 | Reward Pilot | 累计参数上限 | 10 dB Frozen -> Online | 15 dB Frozen -> Online | 观察 |
|---|---:|---:|---:|---:|---|
| 漂移 split，W1 | 32 | 未启用 | 1.0742% -> 1.0417% | 0.2098% -> 0.2098% | 小幅总体下降；10 dB 收益只出现在 seed 0、2，seed 1 持平 |
| heldout edge，W1 | 32 | 未启用 | 0.8319% -> 0.8247% | 0.1591% -> 0.1591% | 10 dB 净收益约 0.0072 个百分点，只来自 seed 2；逐帧仍有反向变化 |
| heldout edge，W2 | 32 | 未启用 | 0.8319% -> 0.8717% | 0.1591% -> 0.1591% | seed 2 的 10 dB 累积参数漂移造成明显退化 |
| heldout edge，W1 | 32 | 0.5 | 0.8319% -> 0.8247% | 0.1591% -> 0.1591% | 与未启用累计上限的 W1 完全相同；该轨迹未触发上限 |
| heldout edge，W2 | 32 | 0.5 | 0.8319% -> 0.8355% | 0.1591% -> 0.1591% | 限制了累积退化，但 10 dB 仍略差于 Frozen |
| heldout edge，W2 | 64 | 0.5 | 0.8355% -> 0.8355% | 0.1591% -> 0.1591% | 10 dB 没有 PEFT 更新被保留；更多验收 Pilot 使方法退回 Frozen |

总体上，Reward Pilot=32、W1 时在线在部分帧确实改变 Data 判决，但平均收益很小，
且没有跨 seed 稳定复现。W2 的放宽设置暴露了跨帧参数累计漂移：单帧更新范数不能
约束多帧后离线初始参数与在线参数的距离。加入累计上限后，最大累计距离符合 0.5，
原先的明显退化消失，但没有转化为稳定 BER 增益。Reward Pilot=64 的配置则没有留下
10 dB 更新，也没有损害 Frozen 性能。

本轮日志目录：

- `logs/rls_drift_relaxed_3s12f_20261002/`
- `logs/rls_heldout_edge_relaxed_3s12f_20261002/`
- `logs/rls_heldout_edge_relaxed_w2_3s12f_20261002/`
- `logs/rls_heldout_edge_trustregion_3s12f_20261002/`
- `logs/rls_heldout_edge_trustregion_w2_3s12f_20261002/`
- `logs/rls_heldout_edge_reward64_w2_3s12f_20261002/`

## PEFT-only 条件消融

修正 RLS 条件来源覆盖后，额外运行了相同 Level B heldout 轨迹、3 seed × 12 帧，
但 Frozen 与 Online 都固定使用 acquisition 条件。Online 仍然只通过 Adapt Pilot 更新
Adapter。这是诊断消融，不是主结果：acquisition 条件没有跟踪当前 Pilot 相位，Frozen
BER 在 10/15 dB 分别达到 `54.67%` / `55.78%`。在线微调只把 10 dB 降到 `54.60%`，
15 dB 完全不变。仅更新残差 Adapter 无法替代当前帧物理相位条件恢复，因此论文主比较
仍需让 Frozen 和 Online 使用同一、合理的 Pilot 条件，并单独报告 PEFT 的配对增量。

日志：`logs/rls_heldout_edge_peft_only_3s12f_20261002/`。

## 结论

- 离线模型和在线 RLS PEFT 都能正常运行，在线确实只改动神经 Adapter 参数。
- 已观察到少数 seed、10 dB 帧上的真实 BER 改善，但量级很小，尚不能称为稳定收益。
- 更保守的留出验收能保证不做危险更新，却经常完全不更新；放宽验收则会受小样本
  Reward Pilot 和跨帧累计偏移影响。
- 当前在线参数微调目标尚未完成。不能将 CIR/phase 状态恢复、更新接受率或单个 seed
  的改善表述为在线微调稳定超过离线训练。
- 下一步应先在更强但仍满足冻结 Level B 物理范围的失配下复核，再针对在线 Adapter
  的可辨识目标和跨帧验证设计做算法调整；不需要重新设计离线训练。

## 验证

完整测试：`499 passed`（包含参数累计保护、条件来源覆盖和配对条件一致性的回归测试）。

## 更强状态失配筛选（2026-10-03）

为继续验证 PEFT 是否能利用更大的状态失配，沿用冻结 Level B 的长记忆 profile、
116-symbol 延迟、prefix Pilot=256 和同一离线 checkpoint。以下条件均单独标记为诊断，
不进入 `cfo_phase_tiny` 主平均。

### CFO 与相位同时增强：工作区过难

`cfo_phase_light` 使用命名 profile 产生更大的 CFO/慢相位扰动，传统校准为 5 seeds ×
60 帧。两种带 CFO/Decision-Directed phase 补偿的传统方法 BER 为：

| SNR | CFO+DD LMMSE-FIR | CFO+DD DFE-RLS |
|---:|---:|---:|
| 0 dB | 49.3181% | 49.7287% |
| 5 dB | 32.9371% | 35.2257% |
| 10 dB | 13.2587% | 11.1593% |
| 15 dB | 9.8481% | 8.6740% |

这说明 CFO 与相位同时增强时，0 dB 接近随机猜测，不能作为有意义的 PEFT 主诊断区。
同一组合条件下，单 seed × 8 帧 Frozen BER 在 10/15 dB 为 `2.1322%/0.7975%`；
当前 RLS 和加大步数后的 phase+head SGD 候选都没有得到 Reward Pilot 放行，或在线
Data BER 与 Frozen 完全一致，因此没有扩大组合失配矩阵。

### 单独增强慢相位：SGD PEFT 短矩阵

拆分后，phase-only 工作区的传统 baseline 在一个 seed、12 帧探针中的 BER 为：
5/10/15 dB 下最优传统方法分别为 `16.6341%/7.4870%/4.8069%`，仍有明显误差但
没有塌到随机猜测。随后用 `phase_light`、同一 checkpoint、固定 Pilot 条件来源、
Adapt Pilot 驱动 `phase+head` SGD、Reward Pilot 双窗口验收，运行 3 seeds × 12 帧：

| SNR | Frozen BER | Online BER | 配对收益 | seed 配对收益 | 保留更新 |
|---:|---:|---:|---:|---|---:|
| 5 dB | 5.0637% | 5.0637% | 约 0.0000 pp | seed 0: -0.0109 pp；seed 1: 0；seed 2: +0.0109 pp | 15/36 |
| 10 dB | 0.8066% | 0.8066% | 0.0000 pp | 三个 seed 均为 0 | 10/36 |
| 15 dB | 0.1374% | 0.1374% | 0.0000 pp | 三个 seed 均为 0 | 15/36 |

所有数据标签在线标记仍为 `false`。此筛选实验表明“确实更新了参数”仍未转化为
Data BER 优势，且当前 12 帧也未显示后期收益。样本量只适合淘汰候选，不能据此给出
最终显著性结论；由于没有出现稳定正向信号，暂不把该候选扩大成正式 5 × 60 矩阵。

### 当前判断

1. 正式 `heldout_edge` 5 × 60 结果仍是 PEFT 机制有效性的主证据：在线参数微调没有
   稳定超过 Frozen，0 dB 退化，10 dB 收益很小且区间跨零。
2. 同时加大 CFO 与相位会把低 SNR 工作点推到近随机猜测，不能单凭这种过难场景
   宣称 PEFT 有价值或无价值。
3. 单独加大慢相位后，当前 SGD Adapter 更新能通过 Pilot 验收，但没有带来稳定的
   Data BER 变化。需要改变可辨识的参数对象、Adapt 目标或 reward 到 Data 的迁移，
   而不是只增加更新次数、放宽门控或继续扩大同一设置。
4. 当前离线 checkpoint 和 `cfo_phase_tiny` 主配置未改变。在线参数微调创新点仍未
   完成；此阶段完成的是正式评估、失败机制定位和更强失配筛选。

本节日志位于 `logs/rls_cfo_phase_light_heldout_5s60f_20261002/`，包括传统 5 × 60
组合扰动校准、组合扰动 Frozen/Online 探针、CFO-only 与 phase-only 单因素探针，以及
phase-only SGD PEFT 的 1 × 12 和 3 × 12 结果。主要组合失配和 phase-only SGD 配置已
归档在 `configs/diagnostics/`，均标记为 diagnostic，不进入主平均。
