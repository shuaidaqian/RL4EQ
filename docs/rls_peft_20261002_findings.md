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

本轮代码回归覆盖动态挂载、PEFT 分组隔离、恒等初始化和 Adapt Pilot-only 更新；
诊断配置位于 `configs/diagnostics/eme_long_memory_v2_gap120_input_affine.json` 与
`configs/diagnostics/eme_long_memory_v2_gap120_logit_affine.json`，均设置
`diagnostic_only=true` 和 `main_aggregation_allowed=false`。
