# Pilot replay 协议复核与累计参数微调（2026-10-14）

本阶段保持离线 checkpoint 不变，检查“Adapt Pilot 更新均衡器内部参数，Reward
Pilot 决定是否保留，Data 标签只做事后评估”能否带来额外均衡收益。先修正诊断协议
和训练目标的顺序，再验证累计多帧目标与独立未来 Reward 验收。

短窗口有小幅正收益，但独立 seed 的长窗口验证未通过稳定性门槛：10 dB 验收后
出现负收益，不能写成已经稳定超过离线模型。核心配置、逐 seed 验收决定和结果保存在
[精简结果 JSON](results/pilot_replay_protocol_audit_20261014.json)，完整逐帧日志在
`logs/replay_protocol_audit_20261014/`。

精简证据包含 121 条逐 seed 对照、各次运行的 CLI 参数与源日志 SHA256、固定
配置快照、真实错 bit 数、候选保留比例和回滚帧位置。可从本地原始日志重新生成：

```powershell
.\.venv-gpu\Scripts\python.exe -B scripts/build_pilot_replay_audit_artifact.py
```

两组持续复查日志还保存一次验收持有的事后对照；它仅从原始候选输出重建，不是
新的独立实验。较早监测日志的 `test_reward_role` 字样未随运行中的元数据修正；
精简证据根据 CLI 的监测窗口和实际接受/回滚字段明确其作用，不改写源日志。

## 本轮修正

1. 旧 `replay_pilot_candidate_ranking.py` 和
   `replay_phase_state_mapping_sequence.py` 直接创建 `CommEnvConfig`，实际退回
   `legacy_sparse_v1`、512 符号帧、clean 扰动。它们的历史数值不能用于正式 EME
   Level B 或 acquisition 老化结论。现在统一使用正式配置构造器、严格模型维度
   校验及正式 `compare.py` 的 acquisition/相位条件流程。
2. 原诊断配置中的 `online_condition_source=acquisition` 属于消融。本轮主场景诊断
   显式使用 `--condition-source pilot_cir_phase`，让 Frozen 和候选共享当前帧 Adapt
   Pilot 相位条件。固定 acquisition 相位而忽略后续 CFO 会产生很差的冻结参照，
   不能将其用于夸大参数更新收益。
3. 模型实际先做基础相位补偿，再执行 `phase_trend` Adapter；旧信号重构目标
   却直接将原始接收信号交给 Adapter。现在 Adapt 训练、Reward 信号验收和诊断
   都遵守同一前向顺序，Adapter 学的是已有补偿后的残差。诊断 Adapt 损失复用
   正式训练函数，避免再次形成两套实现。

已有补偿准确的合成用例中，重构误差接近零，不再驱动重复补偿。真实轨迹 smoke
中，正式 `compare.py` 与累计 replay 的第 4 帧 Frozen BER 均为
`0.006510416977107525`（约 0.651%）；这次核对确认了 BER 一致，没有声称全量
logits 或不同诊断尾状态都一致。

## 固定协议

- checkpoint：`pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`
- SHA256：`8CD3912D1C957C9689C51730AFF9D27A8F74942186C802A7483E5781AF65AEC4`
- 配置：`configs/online_phase_signal_state_embedding_formal_moderate.json`
- 信道：`eme_long_memory_v2` Level B，最大记忆 116，帧长 1024
- 扰动：配置中的 `cfo_phase_tiny`，gap=0，coherence time=1200 s
- SNR：0/5/10/15 dB
- prefix Pilot：总长 256，其中 Adapt 224、Reward 32；Data 每帧 768
- condition：两组共用当前帧 Adapt Pilot 的 phase/CFO 和固定 acquisition CIR
- soft-tail：两组共用 Frozen 的递推尾状态，`tail_update_alpha=0.5`
- 状态 embedding：只由 Adapt Pilot 提取 CIR residual、phase/CFO、噪声、置信度
- 更新对象：只训练 Adapter 内部 `state_to_phase_cfo` 的 14 个参数；离线主干、
  `raw_phase/raw_cfo` 及共享 CIR 冻结
- 更新目标：历史多帧 Adapt Pilot 信号重构均值，AdamW，lr=0.005，5 步，
  参数增量范数上限 0.05，proximal 权重 0.01

这里的 AdamW 是累计 replay 明确采用的优化器，不代表配置中的 `sgd` runner 已经
被替换。状态 embedding 只作为已训练映射的输入，参数更新确实发生在神经均衡器
内部；它并非仅更新外部信道估计。

累计 replay 每条轨迹训练一次候选，然后持有参数，在未来窗口测试；可选持续
Reward 检查允许回滚到 Frozen，但不再次训练或重新激活旧候选。它尚未接入
正式持续在线 runner；诊断的 `main_aggregation_allowed=false`，不混入主平均。
单帧排序和旧连续诊断仍使用固定初始尾状态，也不与累计脚本的递推尾状态混合。

## 三组短窗口比较

seed=0/1/2。所有累计长度候选从同一个 checkpoint/零 Adapter 出发，不用 Data
标签选择参数、门限或接受决定。下表仅列累计 5 帧的候选；完整比较还包括累计 1/3 帧。

| 验收方式 | 训练帧 | 独立验收帧 | Data 测试帧 | 5/10/15 dB 最终保留更新的平均改善 |
|---|---|---|---|---|
| 历史 Reward BCE | 0–4 | 无 | 5–7 | 全部拒绝，均为 0 |
| 未来 Reward BCE | 0–4 | 5–6 | 7–9 | 全部拒绝，均为 0 |
| 未来 Reward 信号重构 + 硬错数保护 | 0–4 | 5–6 | 7–9 | +0.0723 / +0.0723 / +0.0434 个百分点 |

两个 BCE 方案不能保留高 SNR 下的小幅有益候选：32 个 Reward 符号经常已全部
判对，BCE 很小。单独增加未来验收帧不能解决这一问题。历史累计 1 帧在 5 dB
还接受过有害候选，因此也不能将“历史 BCE 变好”视为可靠数据收益保证。

信号重构验收测量基础补偿和 Adapter 作用后的 Reward 接收信号与已知卷积参考
之间的 MSE，按 condition 的噪声方差归一化。固定规则为：两个验收帧的平均改善
超过 0.0005、任一帧改善不低于 -1e-5，且每个验收帧的 Reward 硬判错数不增加。
SNR<5 dB 仍冻结。接受决定只读 Pilot；之后才计算未来 Data 指标。

| SNR | Frozen Data BER | 验收后 Data BER | 绝对改善（百分点） | 接受 seed | 正收益 / 负收益 seed |
|---|---:|---:|---:|---:|---|
| 0 dB | 20.7755% | 20.7755% | 0 | 0/3 | 0/3、0/3 |
| 5 dB | 5.4977% | 5.4253% | +0.0723 | 1/3 | 1/3、0/3 |
| 10 dB | 1.0561% | 0.9838% | +0.0723 | 1/3 | 1/3、0/3 |
| 15 dB | 0.1591% | 0.1157% | +0.0434 | 2/3 | 2/3、0/3 |

“原始候选”是训练后尚未验收的结果；“验收后”是在拒绝时回到 Frozen 的结果。
10 dB 的原始候选在 3/3 seed 上有益，平均改善 0.3038 个百分点，但验收只保留
一个。5 dB 的原始候选有一个负收益 seed；0 dB 原始候选全部变差，因低 SNR
冻结规则而没有提交。这仍不是全 seed、全 SNR 稳定改善的证据。

## 独立验证

固定上述学习率、步数、14 个更新参数及验收规则，使用新 seed=3/4/5，轨迹延长到
24 帧。前 5 帧训练、后 2 帧只用 Reward 验收、再后 17 帧只做 Data 评估。该
验证同时检验新轨迹泛化和一次更新后的收益保持，不把验收帧的 Data 混入测试。

| SNR | Frozen Data BER | 验收后 Data BER | 绝对改善（百分点） | 接受 seed | 正收益 / 负收益 seed |
|---|---:|---:|---:|---:|---|
| 0 dB | 18.4385% | 18.4385% | 0 | 0/3 | 0/3、0/3 |
| 5 dB | 4.7130% | 4.7054% | +0.0077 | 1/3 | 1/3、0/3 |
| 10 dB | 0.6230% | 0.6332% | -0.0102 | 1/3 | 0/3、1/3 |
| 15 dB | 0.0945% | 0.0945% | 0 | 1/3 | 0/3、0/3 |

每个 SNR 只有 3 条独立轨迹、51 个 Data 测试帧，不能将逐帧数当成独立 seed 数。
短窗口和长窗口的 seed 不同，不能仅根据两个总体平均推断收益衰减。

进一步分窗检查发现，10 dB 的 seed=4 通过未来 Reward 验收，但随后首个 3 帧
Data 窗口已经略微变差，平均改善为 -0.0434 个百分点；中间 6 帧为 -0.0651
个百分点，后 8 帧持平。它的 17 帧整体 Data BCE 改善为 +0.000527，但硬 BER
多错了 4 个 bit（每条轨迹的测试 Data 总数为 17×768=13056）。这说明连续目标
改善和硬判错数改善可以不同，问题不只是在长时间持有后失效。

同一 SNR 下，seed=3 的候选后续平均有益却被 Reward 拒绝；seed=4 的候选后续
平均有害却被接受。14 个映射参数确实改变了 Data logits，主因仍是 Pilot 评分
对 Data 硬 BER 的判断不可靠，不能用增加帧数、学习率或步数掩盖。

补充持有期 Reward 诊断：对 10 dB、seed=4 记录每个测试帧的冻结/候选 Reward
信号误差和硬错数。它不修改原先接受决定、不读取 Data 作为守门条件，只用于判断
持续复查 Reward 是否足以发现此次错误候选。结果显示，第一个测试帧（index=7）
的 Reward 信号改善已为 -0.0566971，而该帧硬 BER 还未恶化。因此最近两帧的
Reward 复查会在本帧 Data 输出前回滚，可以避免之后累计的 4 个额外错误。

## 持续 Reward 复查与回滚

累计脚本新增可选 `--monitor-reward-window 2`：初始接受仍使用独立未来 Reward
两帧；持有期每帧用当前及前一帧的 Reward 检查同一预设门限与硬错数保护。
接收机整帧缓冲，可以先读当前前缀 Pilot，再决定本帧 Data 使用候选还是 Frozen。
任一监测窗失败时，当前候选永久回滚；之后候选副本仅用于诊断，不自动重新激活。
该阶段没有引入新的候选训练次数或 Data 选择逻辑。

在 seed=3/4/5、同一 24 帧轨迹上的探索性复核中，5/10/15 dB 各有一个初始
接受的候选，但它们都在第一个 Data 测试帧被持续 Reward 规则回滚。最终各 SNR
都与 Frozen 完全相同，10 dB 的负收益被消除，同时 5 dB 的小收益也未保留。

这组结果只能支持“持续保护可止损”，不能支持“在线微调已稳定改善”；测试窗口
中的候选保留帧比例为零，不能将全体回到 Frozen 的安全结果称为在线正收益。
这些轨迹用于发现监测缺口，所以后续另用此前未参与这轮验证的 seed=6/7/8、
同一 24 帧设置做固定规则的独立检查。

固定规则的 seed=6/7/8 验证重复了相同现象：5/15 dB 没有候选通过初始验收；
10 dB 只有 seed=6 通过，随后在第一个测试帧回滚。所有测试帧的候选保留比例
仍为零，最终各 SNR 都与 Frozen 相同。

| SNR | Frozen Data BER | 持续保护后 Data BER | 绝对改善（百分点） | 初始接受 / 回滚 seed |
|---|---:|---:|---:|---|
| 0 dB | 20.5627% | 20.5627% | 0 | 0/3、0/3 |
| 5 dB | 5.0194% | 5.0194% | 0 | 0/3、0/3 |
| 10 dB | 0.7583% | 0.7583% | 0 | 1/3、1/3 |
| 15 dB | 0.1149% | 0.1149% | 0 | 0/3、0/3 |

利用日志中原始候选输出和初始 Pilot 验收决定，可以重建“只验收一次”的事后
对照；这是同一 Frozen soft-tail 下的诊断推演，不作为新的独立实验。10 dB 的
seed=6 若始终持有，会多错 1 个 Data bit；持续保护避免了该错误。即使如此，
当前证据仍不足以声称稳定增益，也不足以声称已有可靠的候选排序器。

## 复现

历史 Reward BCE 比较：

```powershell
.\.venv-gpu\Scripts\python.exe -B scripts/replay_cumulative_state_mapping.py `
  --config configs/online_phase_signal_state_embedding_formal_moderate.json `
  --pretrained pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt `
  --condition-source pilot_cir_phase --snrs 0 5 10 15 --seeds 0 1 2 `
  --frames 8 --train-frame-counts 1 3 5 `
  --output logs/replay_protocol_audit_20261014/cumulative_aligned_3seed_8f.json
```

未来 Reward BCE 比较：将上面 `--frames` 改为 10，加入
`--validation-frames 2`，输出改为 `cumulative_future_reward_3seed_10f.json`。

未来信号重构验收：

```powershell
.\.venv-gpu\Scripts\python.exe -B scripts/replay_cumulative_state_mapping.py `
  --config configs/online_phase_signal_state_embedding_formal_moderate.json `
  --pretrained pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt `
  --condition-source pilot_cir_phase --snrs 0 5 10 15 --seeds 0 1 2 `
  --frames 10 --train-frame-counts 5 --validation-frames 2 `
  --reward-score signal_reconstruction `
  --output logs/replay_protocol_audit_20261014/cumulative_future_signal_reward_3seed_10f.json
```

独立 seed 长窗口验证：将上一条命令的 `--seeds` 改为 `3 4 5`，`--frames`
改为 24，输出改为 `cumulative_future_signal_reward_newseed_24f.json`。

持有期定位：将独立验证命令的 `--snrs` 改为 `10`、`--seeds` 改为 `4`，输出
改为 `cumulative_signal_reward_hold_monitor_s4_24f.json`。新增的测试帧 Reward
字段属于诊断记录，不改变一次验收后持有的算法。

持续复查：在独立 seed 长窗口命令中加入 `--monitor-reward-window 2`，输出改为
`cumulative_future_signal_reward_monitor2_newseed_24f.json`。固定规则的新 seed
验证则把 `--seeds` 改为 `6 7 8`，输出改为
`cumulative_future_signal_reward_monitor2_freshseed_24f.json`。

## 验证与下一步

- 基础协议与相位顺序修复后，全量测试通过：559 passed。
- 新增未来验收/信号评分后，相关测试通过：75 passed。
- 加入持续回滚后，相关测试通过：76 passed；其中本轮协议测试 11 个。测试包含
  在本帧 Data 评估前回滚、候选回滚后不自动重新激活，以及验证帧和训练/测试
  帧严格隔离。
- 本阶段整理后的最新全量测试通过：563 passed，352.21 秒。
- 两个旧 replay 脚本在真实 EME 配置下的 smoke、正式 Frozen compare smoke 和
  累计 replay smoke 均成功。
- checkpoint SHA256 保持不变；Data 标签不用于在线梯度或验收。

独立长窗口未通过，所以不扩大正式 5 seed × 60 帧矩阵、不接入 Bandit/RL。
当前补齐了持续回滚的 replay 验证，但仍需要能长期保留的有益候选。下一步在
修正后的真实 Level B 协议下，固定 checkpoint、Pilot 和更新对象，重新比较
多帧 Adapt BCE 与信号重构/BCE 联合目标，检查两者能否同时改善未来 Data
连续输出和硬 BER。先解决更新方向与验收评分的可迁移性，再扩大正式矩阵。

0 dB 当前受低 SNR 冻结规则保护，尚没有参数微调的保留收益；更换训练目标后
应先检查其原始候选是否可重复有益，再单独验证是否具备解除冻结的条件，不能
把“没有更新所以没有损失”当成该 SNR 的创新收益。
