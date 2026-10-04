# RLS 在线 PEFT 阶段复核（2026-10-02）

> 最新诊断证据见文末：0/5/10 dB 为 5 seeds × 60 帧，15 dB 已扩为 30 seeds × 60 帧。
> 文中的较早矩阵保留为实验过程记录，不代表最终样本规模。

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

完整测试：`505 passed`（包含参数累计保护、条件来源覆盖、配对条件一致性和本轮新增
PEFT 分组隔离/在线更新回归测试）。

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

### acquisition 到数据段的长间隔诊断

为检查“采集得到的状态已经老化，但网络参数仍可用 Pilot 修正”的可能性，在保持
Level B 长记忆信道、116-symbol 延迟、prefix Pilot=256 和离线 checkpoint 不变的前提下，
额外推进 acquisition 与首个数据帧之间的时间间隔。该实验只改变诊断配置中的
`acquisition_to_data_gap_seconds`，不改变主配置的统计口径。

| 间隔 | SNR | Frozen BER | Online BER | 保留更新 | 结论 |
|---:|---:|---:|---:|---:|---|
| 30 s | 10 dB | 1.4106% | 1.4106% | 3/12 | 参数更新没有改变数据判决 |
| 30 s | 15 dB | 0.3147% | 0.3147% | 6/12 | 参数更新没有改变数据判决 |
| 120 s | 10 dB | 3.5916% | 3.5916% | 3/12 | 状态老化使 Frozen 变差，但 PEFT 没有转化为收益 |
| 120 s | 15 dB | 1.7904% | 1.8012% | 5/12 | 在线略有退化，不能视为收益 |

日志目录分别为 `logs/rls_gap30_peft_1s12f_20261003/` 和
`logs/rls_gap120_peft_1s12f_20261003/`。这组结果说明，仅增加 acquisition 与数据段的
时间失配，不会自动使当前 Pilot BCE 更新目标变得可迁移；长间隔确实降低了 Frozen
性能，但当前可调参数没有学到对应的 Data 残差。

### 强更新探针：参数变化仍未转化为 BER 变化

针对“之前的更新量太小”的假设，使用 `conditioner_film+head`，学习率 `1e-2`、每帧
8 步、单步范数上限 `0.05` 和 `proximal_weight=0.01`，在 120 s gap、单 seed × 8 帧、
10/15 dB 下进行强更新探针。结果如下：

| SNR | Frozen BER | Online BER | 保留更新 | 平均参数增量 | 最大参数增量 |
|---:|---:|---:|---:|---:|---:|
| 10 dB | 3.6133% | 3.6133% | 2/8 | 1.50e-4 | 9.69e-4 |
| 15 dB | 1.8066% | 1.8066% | 7/8 | 6.62e-4 | 1.42e-3 |

即使放大在线优化超参数，Data BER 仍与 Frozen 每帧完全相同。日志目录为
`logs/rls_gap120_film_head_strong_1s8f_20261003/`。该探针排除了“完全没有更新”这一
解释，但也显示当前参数对象主要改变 logit 的连续置信度，尚未改变足够多的符号判决。
结合模型结构，离线 checkpoint 的物理 warm-start 占据主导，`neural_residual_scale=0.1`，
而离线 `head` 范数约为 `1.19e-2`；因此单纯继续增加学习率、步数或放宽 Reward Pilot
门控没有明确的收益机制，反而可能放大 Pilot 噪声造成的错误更新。

### 阶段性结论更新

1. acquisition gap 诊断确认了状态失配可以让 Frozen 性能明显下降，但当前 PEFT 不会因
   失配变强而自然获得收益。
2. 强更新探针确认“参数确实变化”与“Data BER 下降”是两件事；当前更新主要改变连续
   logit，不足以改变 Data 硬判决。
3. 当前 Pilot BCE/Reward Pilot 门控加低维 `conditioner_film/head` 或 RLS 残差 Adapter
   不能作为稳定在线微调创新点。主目标仍未完成，现阶段不应扩大同一更新对象的学习率
   或帧数扫描来包装收益。
4. 后续若继续推进，必须改变在线可辨识目标或参数对象，并保持离线训练和主 Level B
   配置不变；新方案需要先在小样本配对 replay 中同时证明参数确实改变 Data 判决、跨
   seed 方向一致，再进入 5 seeds × 60 帧正式统计。

## 更多在线参数对象筛选（2026-10-03）

为区分“参数对象不合适”和“参数没有更新”这两个问题，在不改变离线 checkpoint、
Pilot 划分、Reward Pilot 门控或在线标签约束的前提下，增加了五类动态 PEFT Adapter，
并对已有的最终 logit 仿射候选补充了诊断配置。所有 Adapter 都是零初始化或恒等
初始化，运行时才挂载，不改变离线 checkpoint 的结构；在线梯度只使用 Adapt Pilot，
Reward Pilot 只用于候选验收和回滚。以下结果全部是诊断 replay，不进入 Level B 主平均。

### 候选定义

| 分组 | 在线参数对象 | 初始化和作用 |
|---|---|---|
| `logit_affine` | 最终 logit 的增益和偏置 | 零初始化；只改变最终软判决的全局尺度和偏移 |
| `input_affine` | 接收 IQ 的实值 2×2 仿射变换 | 恒等初始化；补偿全局幅度、IQ 混合和偏置失配 |
| `input_trend` | 随帧内位置变化的一阶 IQ 仿射趋势 | 恒等初始化；尝试表达前缀到数据段的缓慢变化 |
| `input_fir` | 接收 IQ 的短残差 FIR | 中心抽头为恒等、其余抽头为零；补偿局部回波/ISI 失配 |
| `logit_fir` | 最终 logit 的短残差 FIR | 零初始化；补偿判决序列的局部时域偏差 |
| `pilot_encoder` | 已有 Pilot 条件编码器 | 不新增模块；只微调 Pilot 到帧条件的映射 |

### 小样本 replay 结果

各项均使用 3 seeds × 12 帧、Level B、prefix Pilot=256、Adapt=224、Reward=32、固定
Pilot 条件来源和相同 Frozen/Online 轨迹；表中 BER 使用百分数。`保留更新`表示经过
Reward Pilot 验收后真正保留的帧级参数更新，不代表算法已经取得收益。

| 候选 | 诊断条件 | SNR | Frozen BER | Online BER | 保留更新 | 结果 |
|---|---|---:|---:|---:|---:|---|
| `input_affine` | acquisition gap=120 s | 10 dB | 5.4941% | 5.4977% | 19/72 | 轻微退化 |
| `input_affine` | acquisition gap=120 s | 15 dB | 3.8086% | 3.8158% | 19/72 | 轻微退化 |
| `input_affine` | phase-only | 10 dB | 0.8066% | 0.8066% | 17/72 | 无硬判决变化 |
| `input_affine` | phase-only | 15 dB | 0.1374% | 0.1374% | 17/72 | 无硬判决变化 |
| `input_trend` | phase-only | 10 dB | 0.8066% | 0.8102% | 23/72 | 轻微退化 |
| `input_trend` | phase-only | 15 dB | 0.1374% | 0.1374% | 23/72 | 无收益 |
| `input_fir` | phase-only | 10 dB | 0.8066% | 0.8066% | 25/72 | 无硬判决变化 |
| `input_fir` | phase-only | 15 dB | 0.1374% | 0.1374% | 25/72 | 无硬判决变化 |
| `pilot_encoder` | phase-only | 10 dB | 0.8066% | 0.8066% | 41/72 | 参数更新但输出不变 |
| `pilot_encoder` | phase-only | 15 dB | 0.1374% | 0.1374% | 41/72 | 参数更新但输出不变 |
| `logit_fir` | phase-only | 10 dB | 0.8066% | 0.8066% | 31/72 | 0 帧硬判决改变 |
| `logit_fir` | phase-only | 15 dB | 0.1374% | 0.1374% | 31/72 | 0 帧硬判决改变 |

`logit_affine` 的首轮运行使用了 compare 的默认 bandit 和较大的更新间隔，大多数帧
选择 `skip`，因此不能作为有效的算法比较；它被保留为可复现实验接口，不据此宣称
收益。上述候选中，`input_affine` 在 gap=120 s 下的 Data logit 平均绝对变化约为
`8.05e-3`，`input_trend` 和 `input_fir` 约为 `5e-4`，`logit_fir` 约为 `1.60e-3`；
这些连续输出变化没有稳定跨过 Data 硬判决阈值。`pilot_encoder` 的 Data logit 平均
变化约为 `5.39e-11`，说明该条件编码器在当前前向路径中几乎没有可迁移的影响。

### 解释和路线影响

这轮筛选排除了“只要把可调参数移到输入端、输出端或 Pilot 编码器，就会自然得到
在线收益”的假设。各候选均能在 Adapt Pilot 上更新，且在线审计仍报告
`data_labels_used_online=false`；但参数变化主要表现为连续 logit 的微调，不能稳定
改变 Data 硬判决。gap=120 s 的 `input_affine` 还出现了轻微退化，说明在状态老化
条件下，当前 Pilot BCE 的局部目标可能与数据段残差方向不一致。

因此，继续扫描同一批 Adapter 的学习率、步数或门控阈值没有充分的收益依据。下一候选
必须改变“Pilot 监督如何产生可外推到 Data 的残差”或“在线参数如何控制物理 warm-start
与神经残差的组合”，并先在小样本配对 replay 中同时满足：参数确实改变 Data 判决、
跨 seed 方向一致、Reward Pilot 不依赖 Data 标签。离线训练、Level B 主配置和正式
5 seeds × 60 帧矩阵在新候选出现稳定正向信号前保持不变。

## 物理分支与结构化 PEFT 诊断（2026-10-03）

为直接检验“主导物理 warm-start 使神经 Adapter 只能改变置信度”的判断，增加了三个
零初始化、动态挂载的参数组。它们都通过 `PEFTRegistry` 管理，在线梯度只使用 Adapt
Pilot，Reward Pilot 只用于验收和回滚；没有修改离线 checkpoint，也没有把 CIR/phase
估计器的输出写回在线参数。

| 参数组 | 可训练对象 | 诊断结果 |
|---|---|---|
| `physics_blend` | 物理 warm-start 与神经 residual 两个分支增益 | 3 seeds × 12 帧、gap=120 s、10/15 dB 中有 12/72 帧改变 Data 连续 logit，但 0 帧改变 Data BER |
| `physics_residual` | `[physics logit, proposal, residual, position]` 到 logit 的 5 维残差头 | 10 dB 有 12/36 帧真正保留更新，全部改变连续 logit，但 0 帧改变 Data BER；15 dB 有 9/36 帧更新，同样 0 帧改变 Data BER |
| `phase_trend` | 输入端公共相位和线性相位趋势两个 PEFT 参数 | phase-only 单 seed × 12 帧中 5 dB 退化约 0.0109 pp，10/15 dB 无 Data BER 变化 |
| `channel_residual` | 117 个 CIR tap 的复数残差参数 | 1 seed 探针在 10 dB 有 +0.0543 pp；扩大到 3 seeds 后平均仅 +0.0181 pp，只有 1/3 seed 改善，15 dB 无更新被保留 |

`channel_residual` 是这轮最接近目标的候选，因为它首次在 Data 硬判决上产生了真实变化，
但 3 seeds 结果仍不足以证明稳定收益，不能进入 Level B 主平均，也不能写成“在线稳定
超过离线”。当前结果更准确的表述是：在线参数确实可以通过 Pilot 修正部分 acquisition
信道失配，但小样本 Reward Pilot 的验收在高 SNR 下经常拒绝更新，且跨 seed 方向不一致。

在更强的 `heldout_edge` 状态范围上复核了同一候选：10 dB 的 Frozen/Online 平均为
`6.8757% -> 6.8685%`，配对收益约 `+0.0072 pp`，只有 1/3 seed 改善；15 dB 为
`5.4253% -> 5.4253%`，没有更新被保留。该结果与 gap=120 s 的筛选方向一致，说明
扩大状态失配没有把当前参数组变成稳定收益方法。

本轮可复现实验命令（均为诊断，不进入主统计）：

```powershell
.\.venv-gpu\Scripts\python.exe compare.py --config configs/diagnostics/eme_long_memory_v2_gap120_input_affine.json --method-group main --pretrained pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --delays 116 --snrs 10 15 --num-seeds 3 --frames 12 --pilot-total 256 --pilot-layout prefix --online-groups channel_residual --scheduler fixed --update-interval 1 --resume --output-dir logs/gap120_channel_residual_fixed_3s12f_20261003
```

当前判断没有改变：仅继续调学习率、步数、门控阈值或同类低维 Adapter，缺少能产生稳定
收益的机制依据。若要继续追求目标，下一步必须改变 Pilot 监督到 Data 的迁移结构，并
在扩大矩阵前同时满足：至少多数 seed 改善、Data 硬判决确实变化、Reward Pilot 不使用
Data 标签、且低 SNR 不出现系统性退化。

## Pilot 重构目标探针（2026-10-03）

上面的 `channel_residual` 默认使用 Adapt Pilot BCE 更新。为区分“参数对象不合适”和
“BCE 梯度不适合物理 tap”，新增可选在线目标 `pilot_reconstruction`：用已知 Adapt Pilot
符号、接收 IQ 和当前可训练 channel residual 的复数重构误差更新参数；未知 Data 区域
的符号保持为零，不参与目标。默认目标仍为 `bce`，离线训练和已有实验不受影响。

小样本结果如下，均为固定调度、Reward Pilot 验收、3 seeds × 12 帧诊断 replay：

| 状态范围 | SNR | Frozen BER | Online BER | 配对收益 | seed 方向 | 保留更新 |
|---|---:|---:|---:|---:|---|---:|
| gap=120 s | 10 dB | 5.4941% | 5.4905% | +0.0036 pp | 1/3 改善，2/3 退化 | 11/36 |
| gap=120 s | 15 dB | 3.8086% | 3.7941% | +0.0145 pp | 2/3 改善，1/3 持平 | 11/36 |
| heldout_edge + gap=120 s | 5 dB | 11.2739% | 11.2486% | +0.0253 pp | 2/3 改善，1/3 持平 | 15/36 |
| heldout_edge + gap=120 s | 10 dB | 6.8757% | 6.8649% | +0.0109 pp | 1/3 改善，2/3 持平 | 9/36 |
| heldout_edge + gap=120 s | 15 dB | 5.4253% | 5.3928% | +0.0326 pp | 3/3 改善 | 12/36 |

这是一条值得保留的候选路线，但目前只在强状态失配的 15 dB 小样本上同时满足“多数
seed 同方向”和“Data 硬判决改变”。10 dB 的收益量级很小，0 dB 仍按现有安全策略冻结
在线更新；因此尚未达到全 SNR 稳定、明确超过 Frozen 的目标，也没有进入主平均。

可复现实验命令：

```powershell
.\.venv-gpu\Scripts\python.exe compare.py --config configs/diagnostics/eme_long_memory_v2_gap120_input_affine.json --method-group main --pretrained pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --delays 116 --snrs 5 10 15 --num-seeds 3 --frames 12 --pilot-total 256 --pilot-layout prefix --state-split heldout_edge --online-groups channel_residual --online-objective pilot_reconstruction --scheduler fixed --update-interval 1 --resume --output-dir logs/gap120_channel_recon_heldout_edge_fixed_3s12f_20261003
```

后续只有在该目标先通过 5 seeds × 60 帧的独立诊断矩阵后，才考虑把它接入主比较；
主配置、离线 checkpoint 和 0 dB 的冻结策略保持不变。

## 5 seeds × 60 帧复核与主配置边界（2026-10-03）

在 `heldout_edge + acquisition_to_data_gap=120 s` 上扩大到 5 seeds × 60 帧后，
`channel_residual + pilot_reconstruction` 的配对结果为：

| SNR | Frozen BER | Online BER | 配对收益 | seed 方向 | 保留更新 |
|---:|---:|---:|---:|---|---:|
| 0 dB | 26.0885% | 26.0885% | 0.0000 pp | 0/5 改善，5/5 持平 | 0/300（按 SNR 冻结） |
| 5 dB | 11.2062% | 11.1237% | +0.0825 pp | 5/5 改善 | 109/300 |
| 10 dB | 6.0373% | 5.9796% | +0.0577 pp | 5/5 改善 | 96/300 |
| 15 dB | 4.4709% | 4.3954% | +0.0755 pp | 5/5 改善 | 96/300 |

按 seed 重采样的收益近似 95% 区间为：5 dB `[+0.0241, +0.1408]` pp，10 dB
`[+0.0021, +0.1134]` pp；0 dB 没有在线更新，15 dB 的逐 seed 收益也全部为正。
所有 1200 条 Online 记录的 `data_labels_used_online` 均为 `false`，且每个真正保留的
更新都改变了 Data 连续 logit。这是目前最强的可复现证据，但状态范围是诊断用的
`heldout_edge + 120 s gap`，不能直接替换论文主平均。

随后用冻结的主配置 `eme_long_memory_v2/cfo_phase_tiny`、gap=0、同一 checkpoint、
同样的 5 seeds × 60 帧重新运行，并显式覆盖 `--online-algorithm sgd`，避免配置中的
旧 RLS 路线混入：

| SNR | Frozen BER | Online BER | 配对收益 | seed 方向 | 保留更新 |
|---:|---:|---:|---:|---|---:|
| 0 dB | 21.3841% | 21.3841% | 0.0000 pp | 0/5 改善，5/5 持平 | 0/300（按 SNR 冻结） |
| 5 dB | 5.7066% | 5.7075% | -0.0009 pp | 0/5 改善，2/5 轻微退化 | 92/300 |
| 10 dB | 0.9549% | 0.9549% | 0.0000 pp | 0/5 改善，5/5 持平 | 103/300 |
| 15 dB | 0.1641% | 0.1641% | 0.0000 pp | 0/5 改善，5/5 持平 | 162/300 |

因此，当前创新点已经在“明显状态老化/边界失配”条件下形成了稳定的参数微调收益，
但在现有主配置的 gap=0 条件下没有收益。不能把诊断收益描述为主配置已完成；主论文
若采用该路线，应把“Pilot 重构驱动的 CIR-PEFT 用于 acquisition 状态失配”作为明确
适用条件，并单独报告主配置持平结果。

为拆分两个因素，又保持主 `cfo_phase_tiny` 配置不变、只切换到 `heldout_edge` 且
gap=0 做了 5 seeds × 60 帧负对照。结果为：0 dB `0.0000 pp`（5/5 持平），5 dB
`0.0000 pp`（5/5 持平），10 dB `+0.0004 pp`（仅 1/5 seed 改善），15 dB
`0.0000 pp`（5/5 持平）。这说明当前收益不是由边界信道范围单独造成，而是来自
“acquisition 到数据段的状态老化”与边界失配共同形成的可辨识 CIR 残差。

该负对照命令：

```powershell
.\.venv-gpu\Scripts\python.exe compare.py --config configs/eme_long_memory_v2.json --method-group proposed --pretrained pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --delays 116 --snrs 0 5 10 15 --num-seeds 5 --frames 60 --pilot-total 256 --pilot-layout prefix --state-split heldout_edge --online-groups channel_residual --online-algorithm sgd --online-objective pilot_reconstruction --scheduler fixed --update-interval 1 --resume --output-dir logs/main_channel_recon_heldout_edge_gap0_sgd_5s60f_20261003
```

主配置复核命令：

```powershell
.\.venv-gpu\Scripts\python.exe compare.py --config configs/eme_long_memory_v2.json --method-group proposed --pretrained pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --delays 116 --snrs 0 5 10 15 --num-seeds 5 --frames 60 --pilot-total 256 --pilot-layout prefix --online-groups channel_residual --online-algorithm sgd --online-objective pilot_reconstruction --scheduler fixed --update-interval 1 --resume --output-dir logs/main_channel_recon_cfo_phase_tiny_sgd_5s60f_20261003
```

本轮代码回归覆盖动态挂载、PEFT 分组隔离、恒等初始化和 Adapt Pilot-only 更新；
诊断配置位于 `configs/diagnostics/eme_long_memory_v2_gap120_input_affine.json` 与
`configs/diagnostics/eme_long_memory_v2_gap120_logit_affine.json`，均设置
`diagnostic_only=true` 和 `main_aggregation_allowed=false`。

## 与传统均衡器的同条件配对复核（2026-10-03）

为检验这条 PEFT 候选的系统价值，固定上节 `heldout_edge + acquisition_to_data_gap=120 s`
诊断条件、Level B、delay=116、prefix Pilot=256、`cfo_phase_tiny`，并用同一组 5 seeds ×
60 帧运行三个传统方法。在线/Frozen 结果来自前述同条件矩阵。环境配置和局部随机数生成器
按 seed 固定；逐 SNR、seed、frame 对照检查的 `state_instance` 不一致数为 0，因此 BER
差异是同一接收轨迹上的配对比较。每个传统方法每个 SNR 均有 300 帧。

其中 `CFO+DD-Phase LMMSE-FIR` 和 `CFO+DD-Phase DFE-RLS` 每帧用 Adapt Pilot 拟合线性
相位/CFO，并作跨帧判决导向跟踪；日志中的相位拟合平均使用 112 个可靠 Pilot 点，估计
CFO 平均绝对值约为 `5.8e-4` cycles/symbol。两种方法均不使用神经网络、RL、Reward Pilot
标签或 Data 标签。`SC-FDE-MMSE` 没有启用同等 Pilot 相位/CFO 补偿，单独列作参考，不纳入
“最强公平传统基线”的选择。

| SNR | Frozen BER | Online BER | 最低 BER 的 Pilot 补偿传统法 | Online 对 Frozen 收益（95% CI） | Online 对传统法收益（95% CI） | 传统比较 seed 方向 |
|---:|---:|---:|---|---:|---:|---:|
| 0 dB | 26.0885% | 26.0885% | DFE-RLS 46.5846% | 0.0000 pp `[0.0000, 0.0000]` | +20.4961 pp `[+13.8568, +25.9609]` | 5/5 改善 |
| 5 dB | 11.2062% | 11.1237% | DFE-RLS 31.4965% | +0.0825 pp `[+0.0286, +0.1536]` | +20.3728 pp `[+12.1276, +29.3151]` | 5/5 改善 |
| 10 dB | 6.0373% | 5.9796% | LMMSE-FIR 13.1250% | +0.0577 pp `[+0.0130, +0.1146]` | +7.1454 pp `[+1.7109, +12.5964]` | 5/5 改善 |
| 15 dB | 4.4709% | 4.3954% | LMMSE-FIR 8.2600% | +0.0755 pp `[+0.0286, +0.1302]` | +3.8646 pp `[-0.3438, +9.4349]` | 4/5 改善 |

区间按 10,000 次分层配对 bootstrap 估计：先重采样 seed，再在每个抽中的 seed 内抽取一个
连续 10 帧块。Online 对 Frozen 的 5/10/15 dB 区间均为正且 5/5 seeds 改善；0 dB 按冻结
策略不更新。Online 对最强传统法在 0/5/10 dB 的均值和区间均显示优势；15 dB 均值较低，
但区间跨零且只有 4/5 seeds 改善，证据不足以称为稳定显著优势。

本矩阵记录到 301 次被 Reward Pilot 接受的 PEFT 更新；1,200 条 Online 记录均为
`data_labels_used_online=false`。这支持“Pilot 重构驱动 CIR-PEFT 在明显 acquisition 状态
老化时能稳定小幅提升 Frozen 均衡，并在 0/5/10 dB 胜过当前传统实现”的阶段性结论；
它不改变主配置 gap=0 下几乎持平的事实，也没有证明所有 SNR 和状态条件都能获益。
传统基线中 15 dB 的不确定性，以及传统实现本身与神经模型的 BER 差距较大，仍需在最终
主比较前复核，不能只凭本诊断矩阵宣称论文目标已全部完成。

复现实验命令：

```powershell
.\.venv-gpu\Scripts\python.exe compare.py --config configs/diagnostics/eme_long_memory_v2_gap120_input_affine.json --methods "CFO+DD-Phase LMMSE-FIR" "CFO+DD-Phase DFE-RLS" SC-FDE-MMSE --delays 116 --snrs 0 5 10 15 --num-seeds 5 --frames 60 --pilot-total 256 --pilot-layout prefix --state-split heldout_edge --scheduler fixed --resume --output-dir logs/gap120_traditional_heldout_edge_5s60f_20261003
```

## 低 SNR 更新与 15 dB 扩样（2026-10-04）

上一节 0 dB 使用了 `freeze_below_snr_db=5`，因此没有测试参数微调本身能否改善低 SNR。
本轮只在诊断命令中把阈值覆盖为 `-1 dB`，离线 checkpoint、主配置和在线目标保持不变，
完成 0 dB 5 seeds × 60 帧。随后把 15 dB 在线/Frozen 和传统基线扩至 30 seeds × 60 帧；
其余 SNR 保持 5 seeds × 60 帧。比较仍为同 seed、同帧的配对 replay，各 SNR 的
`state_instance` 不一致数均为 0。

| SNR | seeds × 帧 | Frozen BER | Online BER | Online 对 Frozen 收益（95% CI） | 最强 Pilot 补偿传统方法 BER | Online 对传统方法收益（95% CI） | 收益方向 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 dB | 5 × 60 | 26.0885% | 26.0556% | +0.0330 pp `[+0.0052, +0.0703]` | DFE-RLS 46.5846% | +20.5291 pp `[+13.8880, +26.2865]` | Frozen 5/5；传统 5/5 改善 |
| 5 dB | 5 × 60 | 11.2062% | 11.1237% | +0.0825 pp `[+0.0286, +0.1510]` | DFE-RLS 31.4965% | +20.3728 pp `[+12.0990, +29.3802]` | Frozen 5/5；传统 5/5 改善 |
| 10 dB | 5 × 60 | 6.0373% | 5.9796% | +0.0577 pp `[+0.0130, +0.1146]` | LMMSE-FIR 13.1250% | +7.1454 pp `[+1.7005, +12.4922]` | Frozen 5/5；传统 5/5 改善 |
| 15 dB | 30 × 60 | 4.7399% | 4.6774% | +0.0625 pp `[+0.0434, +0.0829]` | LMMSE-FIR 8.1230% | +3.4456 pp `[+0.7079, +4.6771]` | Frozen 29/30 改善、1 持平；传统 25/30 改善 |

区间按 30,000 次分层配对 bootstrap 计算：重采样 seed 后，在每个抽中的 seed 内抽一个
连续 10 帧块。15 dB 的另一个公平 Pilot 补偿基线 DFE-RLS BER 为 `8.4094%`，Online
相对它的收益为 `+3.7320 pp`，95% CI `[+0.8411, +5.2331]`，27/30 seeds 改善。
传统方法是配置中预先列出的非神经方法；`CFO+DD-Phase LMMSE-FIR` 和
`CFO+DD-Phase DFE-RLS` 都用 Adapt Pilot 估计相位/CFO。无 Pilot 相位/CFO 补偿的
`SC-FDE-MMSE` 不作为公平传统基线。

解除 0 dB 冻结后，Pilot 重构目标在 300 帧中保留了 98 次 PEFT 更新，所有 5 个 seed
均比 Frozen 略好；5/10/15 dB 分别保留 109/96/577 次更新。所有在线记录仍满足
`data_labels_used_online=false`。这些结果支持一个明确但有限的结论：在
`heldout_edge + 120 s acquisition-to-data gap` 下，Pilot 重构驱动的 CIR-PEFT 可在
所有测试 SNR 稳定超过 Frozen，并在扩大到 30 seeds 后也能超过两个 Pilot 补偿传统基线。

将每个 seed 的 60 帧再按 1–20、21–40、41–60 分段后，配对收益随在线帧数增加而扩大：

| SNR | 帧 1–20 收益（95% CI） | 帧 21–40 收益（95% CI） | 帧 41–60 收益（95% CI） |
|---:|---:|---:|---:|
| 0 dB | +0.0182 pp `[+0.0052, +0.0495]` | +0.0287 pp `[-0.0104, +0.0625]` | +0.0521 pp `[+0.0156, +0.1016]` |
| 5 dB | +0.0273 pp `[+0.0052, +0.0547]` | +0.0729 pp `[+0.0339, +0.1302]` | +0.1471 pp `[+0.0755, +0.2083]` |
| 10 dB | +0.0117 pp `[0.0000, +0.0234]` | +0.0742 pp `[+0.0208, +0.1354]` | +0.0872 pp `[+0.0365, +0.1406]` |
| 15 dB | +0.0224 pp `[+0.0122, +0.0295]` | +0.0640 pp `[+0.0464, +0.0803]` | +0.1011 pp `[+0.0768, +0.1237]` |

窗口区间按 20,000 次分层配对 bootstrap 计算：先重采样 seed，再在对应的 20 帧窗口内
抽取连续 10 帧块。15 dB 三个窗口区间都为正；0 dB 中间窗口跨零，其他窗口区间为正。

但 Online 相对 Frozen 的绝对收益只有 `0.033–0.083 pp`，属于稳定的小幅提升，不是大幅
提升；15 dB 传统比较使用了更多 seeds，其他 SNR 的传统区间仍基于 5 seeds。主配置
`gap=0` 的既有结果依然基本持平，所以不能把这里的诊断收益推广成普通主配置下均有增益。
目前已证明的是“明显 acquisition 状态老化时有效”，尚未达到不依赖状态失配的通用在线
微调方案。

0 dB 解冻探针复现命令：

```powershell
.\.venv-gpu\Scripts\python.exe compare.py --config configs/diagnostics/eme_long_memory_v2_gap120_input_affine.json --methods "Frozen Offline NN" "Pilot-Driven Online Adaptation" --pretrained pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --delays 116 --snrs 0 --num-seeds 5 --frames 60 --pilot-total 256 --pilot-layout prefix --state-split heldout_edge --online-groups channel_residual --online-algorithm sgd --online-objective pilot_reconstruction --online-freeze-below-snr-db -1 --scheduler fixed --update-interval 1 --resume --output-dir logs/gap120_channel_recon_heldout_edge_0db_unfrozen_5s60f_20261003
```

`compare.py` 现支持 `--seed-start`，便于在 `--resume` 输出目录中追加非重叠 seed。15 dB
最后一次扩样命令使用 `--num-seeds 10 --seed-start 20 --snrs 15`；更早的 seed 0–19
已保留在同一在线与传统日志目录中。

## acquisition gap 范围复核（2026-10-04）

为判断 120 s 诊断收益是否能在较短状态老化下复现、以及更强失配能否带来更大收益，
保持同一离线 checkpoint、Level B `heldout_edge`、116-symbol 延迟、prefix Pilot=256、
`channel_residual + pilot_reconstruction`、固定调度与 Reward Pilot 验收，仅覆盖本次
运行的 acquisition 到数据段间隔。所有在线记录的 `data_labels_used_online` 均为 `false`，
Frozen/Online 对应帧的 `state_instance` 不一致数均为 0。这些间隔属于单独诊断，不进入
gap=0 主配置平均。

### 5 dB 间隔筛查与复核

| Gap | 样本 | Frozen BER | Online BER | 配对收益 | seed 收益（pp） | 结论 |
|---:|---:|---:|---:|---:|---|---|
| 30 s | 3 × 12 帧 | 6.7274% | 6.7202% | +0.0072 pp | +0.0000, +0.0217, +0.0000 | 仅筛查，近乎持平 |
| 60 s | 3 × 12 帧 | 8.1489% | 8.1199% | +0.0289 pp | +0.0543, +0.0217, +0.0109 | 3/3 同向，进入扩样 |
| 60 s | 5 × 60 帧 | 8.2990% | 8.2639% | +0.0352 pp `[+0.0078, +0.0703]` | +0.0629, +0.0391, +0.0043, +0.0543, +0.0152 | 5/5 同向，区间为正 |
| 120 s | 5 × 60 帧 | 11.2062% | 11.1237% | +0.0825 pp `[+0.0286, +0.1510]` | 5/5 改善 | 已有正式诊断结果 |
| 300 s | 3 × 12 帧 | 21.9582% | 21.9437% | +0.0145 pp `[-0.0347, +0.0477]` | +0.0651, +0.0217, -0.0434 | 2/3 同向，未通过扩样门槛 |

60 s 的 5 × 60 区间由逐帧配对差 `BER_frozen - BER_online` 经 30,000 次分层
bootstrap 计算：先重采样 seed，再在每个 seed 内抽连续 10 帧块。该结果确认 5 dB 下
在线 PEFT 有跨 seed 的小幅收益，但绝对改善约为 `0.035 pp`，不属于大幅提升。

30/60 s 的全 SNR 3 × 12 帧筛查中，30 s 的 0 dB 平均看似改善 `0.7053 pp`，但 seed
收益为 `-0.0109、+2.1484、-0.0217 pp`，完全由一个 seed 拉动，不能视为稳定结果；其余
档位均接近持平。60 s 下 0/10/15 dB 的平均差为 `+0.0109/-0.0036/+0.0109 pp`，样本
不足以支持稳定跨 SNR 收益。300 s 时 Frozen BER 已大幅升高，但 PEFT 收益没有同步扩大，
95% 区间跨零，说明“失配越强、在线微调收益越大”不成立；过强状态老化也会超出当前
适配能力。

综合现有证据，`heldout_edge + 60–120 s acquisition gap` 是当前值得保留的诊断工作区：
在 5 dB 下已复现小幅、跨 seed 的收益；120 s 正式矩阵还显示 0/5/10/15 dB 均优于
Frozen。`gap=0` 仍基本持平，因此这个结论限定在 acquisition 状态有一定老化的条件，
不能包装成通用在线 PEFT 已解决，也不能声称收益很大。下一步应检查在线帧数增加是否能
继续扩大 60–120 s 条件下的收益，并同时确认参数更新没有跨帧累积退化。

本轮日志：

- `logs/acqgap30_channel_recon_heldout_edge_3s12f_20261004/`
- `logs/acqgap60_channel_recon_heldout_edge_3s12f_20261004/`
- `logs/acqgap60_channel_recon_heldout_edge_5db_5s60f_20261004/`
- `logs/acqgap300_channel_recon_heldout_edge_5db_3s12f_20261004/`

正式 60 s 复现命令：

```powershell
.\.venv-gpu\Scripts\python.exe compare.py --config configs/diagnostics/eme_long_memory_v2_gap120_input_affine.json --method-group proposed --pretrained pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --delays 116 --snrs 5 --num-seeds 5 --frames 60 --pilot-total 256 --pilot-layout prefix --state-split heldout_edge --online-groups channel_residual --online-algorithm sgd --online-objective pilot_reconstruction --online-freeze-below-snr-db -1 --scheduler fixed --update-interval 1 --acquisition-to-data-gap-seconds 60 --output-dir logs/acqgap60_channel_recon_heldout_edge_5db_5s60f_20261004
```

`compare.py` 新增 `--acquisition-to-data-gap-seconds` 本次运行覆盖参数，并在摘要中将该
运行标记为 `diagnostic_only=true`、`main_aggregation_allowed=false`；resume 会拒绝把不同
gap 的帧记录混入同一输出目录。

### 0/5/10/15 dB 长序列统一复核（120 s gap，5 seeds × 120 帧，2026-10-04）

在完成 5 dB 长序列后，补齐同一 `heldout_edge + 120 s acquisition gap` 条件下的 0、10、
15 dB；5 dB 直接使用上一节的同配置结果。四个 SNR 共 2,400 条配对记录，Frozen/Online
对应帧的 `state_instance` 不一致数为 0，所有 Online 记录的 `data_labels_used_online` 均为
`false`。

| SNR | 帧区间 | Frozen BER | Online BER | 配对收益（95% CI） | seed 方向 |
|---:|---|---:|---:|---:|---:|
| 0 dB | 1–60 | 26.0885% | 26.0556% | +0.0330 pp `[+0.0143, +0.0560]` | 5/5 |
| 0 dB | 61–120 | 26.6536% | 26.9071% | -0.2535 pp `[-1.4588, +0.1528]` | 4/5 |
| 0 dB | 1–120 | 26.3711% | 26.4813% | -0.1102 pp `[-0.6723, +0.0907]` | 4/5 |
| 5 dB | 1–60 | 11.2062% | 11.1237% | +0.0825 pp `[+0.0430, +0.1259]` | 5/5 |
| 5 dB | 61–120 | 11.8832% | 11.6515% | +0.2318 pp `[+0.1454, +0.3898]` | 5/5 |
| 5 dB | 1–120 | 11.5447% | 11.3876% | +0.1571 pp `[+0.0927, +0.2418]` | 5/5 |
| 10 dB | 1–60 | 6.0373% | 5.9796% | +0.0577 pp `[+0.0234, +0.0998]` | 5/5 |
| 10 dB | 61–120 | 6.2027% | 5.9939% | +0.2088 pp `[+0.1311, +0.2808]` | 5/5 |
| 10 dB | 1–120 | 6.1200% | 5.9868% | +0.1332 pp `[+0.0760, +0.1936]` | 5/5 |
| 15 dB | 1–60 | 4.4709% | 4.3954% | +0.0755 pp `[+0.0421, +0.1107]` | 5/5 |
| 15 dB | 61–120 | 4.5938% | 4.3954% | +0.1984 pp `[+0.1298, +0.2665]` | 5/5 |
| 15 dB | 1–120 | 4.5323% | 4.3954% | +0.1369 pp `[+0.0859, +0.1921]` | 5/5 |

区间仍使用逐帧配对差的分层 block bootstrap：先重采样 seed，再在每个 seed 内抽取连续 10
帧块。5、10、15 dB 的后半段收益分别约为前半段的 2.8、3.6、2.6 倍，且每个 seed 都为
正；这说明在中高 SNR 下，Pilot 重构驱动的 channel-residual PEFT 确实能随帧数积累收益。

0 dB 呈现相反趋势：前 60 帧 5/5 seed 改善，后 60 帧总体退化 `0.2535 pp`，其中 seed 3
为 `-1.5755 pp`，其余四个 seed 仍为正。因此 0 dB 的 120 帧结果不能视为稳定在线增益，
也说明“Reward Pilot 守门”在低 SNR 下仍不足以阻止错误参数逐步积累。各 SNR 全程保留更新数
分别为 196、207、192、191；回滚数为 114、85、77、73，更新候选未被保留的记录数为
404、393、408、409。被保留更新的参数范数均约为 `0.010–0.013`，最大约 `0.01997`，
所以 0 dB 退化不是更新范数突然失控，而更像是低 SNR 下 Pilot 目标的辨识噪声累积。

当前可成立的阶段性结论是：在 `heldout_edge + 120 s acquisition gap` 下，5/10/15 dB
存在跨 seed、随在线帧数扩大的小幅收益；0 dB 只能在较短在线窗口内获得收益，长时间更新会
出现不稳定。主配置 `gap=0` 仍不能由这些诊断结果替代。后续在线策略应优先解决低 SNR 的
更新可信度（例如更严格的连续窗口验收或低 SNR 冻结），同时保持中高 SNR 的已验证收益；
不应通过放大学习率或 Adapter 容量来掩盖该问题。

本轮日志：`logs/gap120_channel_recon_heldout_edge_0_10_15db_5s120f_20261004/`。

### 120 s gap 的长序列复核（5 seeds × 120 帧，2026-10-04）

为确认 120 s 状态老化下的收益是否会随在线帧数增加而扩大，继续使用上一节完全相同的
离线 checkpoint、Level B `heldout_edge`、5 dB、116-symbol 延迟、prefix Pilot=256、
`channel_residual + pilot_reconstruction` 和固定调度，仅把在线长度从 60 帧扩展到 120 帧。
Frozen 与 Online 使用同一 seed、同一帧的接收轨迹；逐帧 `state_instance` 不一致数为 0，
600 条 Online 记录的 `data_labels_used_online` 均为 `false`。

| 帧区间 | Frozen BER | Online BER | 配对收益（95% CI） | 5 个 seed 方向 |
|---|---:|---:|---:|---:|
| 1–60 | 11.2062% | 11.1237% | +0.0825 pp `[+0.0430, +0.1259]` | 5/5 改善 |
| 61–120 | 11.8832% | 11.6515% | +0.2318 pp `[+0.1454, +0.3898]` | 5/5 改善 |
| 1–120 | 11.5447% | 11.3876% | +0.1571 pp `[+0.0927, +0.2418]` | 5/5 改善 |

区间采用逐帧配对差的分层 block bootstrap：先重采样 seed，再在每个 seed 内抽取连续 10 帧
块。后半段收益约为前半段的 2.8 倍，支持“在线参数逐步吸收 acquisition-to-data 状态
老化信息”的现象；这不是单纯由某一个 seed 拉动，后半段每个 seed 的收益均为正，分别为
`+0.4123、+0.2756、+0.1345、+0.2192、+0.1172 pp`。

长序列中共保留 207 次 PEFT 更新，85 次记录触发回滚，393 个候选未被保留；被保留更新的
`parameter_delta_norm` 平均为 `0.01134`、最大为 `0.01993`。前后半段保留更新次数分别为
109 和 98，说明后半段收益扩大并不是因为更新次数增加，而更可能来自已保留的参数逐步积累。
同时，回滚比例并不低（85/600），因此现有“Reward Pilot 通过才保留”的验收机制仍需在
更大矩阵中检查其稳定性，不能把该结果表述成无条件的在线增益。

这组结果把当前证据边界进一步收窄并变得清楚：在 `heldout_edge + 120 s acquisition gap`
这一诊断工作区，Pilot 重构驱动的 CIR-PEFT 能随在线帧数增加稳定超过冻结离线模型，且收益
从约 `0.08 pp` 增加到约 `0.23 pp`；但该结论仍不适用于 `gap=0` 主配置，也没有证明在
300 s 过强失配下同样有效。下一步应优先分析回滚与参数漂移的关系，并评估是否能把可辨识的
在线目标限制在 CIR/phase residual 等少量物理相关参数上；在此之前不应继续扩大同类
Adapter 的学习率或容量来追求更大的数字。

本轮日志：

- `logs/gap120_channel_recon_heldout_edge_5db_5s120f_20261004/`

复现实验命令：

```powershell
.\.venv-gpu\Scripts\python.exe compare.py --config configs/diagnostics/eme_long_memory_v2_gap120_input_affine.json --methods "Frozen Offline NN" "Pilot-Driven Online Adaptation" --pretrained pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --delays 116 --snrs 5 --num-seeds 5 --frames 120 --pilot-total 256 --pilot-layout prefix --state-split heldout_edge --online-groups channel_residual --online-algorithm sgd --online-objective pilot_reconstruction --online-freeze-below-snr-db -1 --scheduler fixed --update-interval 1 --acquisition-to-data-gap-seconds 120 --output-dir logs/gap120_channel_recon_heldout_edge_5db_5s120f_20261004
```
