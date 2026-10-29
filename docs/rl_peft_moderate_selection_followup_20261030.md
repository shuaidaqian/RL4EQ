# Moderate PEFT 固定动作与 PPO 长轨迹复核（2026-10-30）

## 结论

本记录早期的 gap=600 秒结果来自协议边界修正前的运行，只作历史诊断，不能作为符合当前目标的信息隔离证据。最终审计后的 gap=0、`heldout_edge`、5 dB 复核结果见文末；它显示 moderate 更新有小幅正向信号，但 PPO 仍较少选择该动作，收益区间包含零，目标尚未达成。

## 协议边界修正

代码审计发现，旧版 PPO 选动作时会读取 Adapt 与 Reward Pilot 接收波形拼成的 observation，策略置信度也来自包含 Reward Pilot 波形的前向。这违反“Reward Pilot 只负责验收和回滚”的约定。虽然旧日志标注没有使用 Reward/Data 标签，但这不足以满足严格的信息边界。

现已修正为：PPO 当前动作 observation、PEFT 梯度、tail/CIR 状态递推只接收 Adapt Pilot；先选动作，再构造 Reward Pilot 波形与标签用于动作后的验收、奖励和回滚；完整 Data 波形只在在线控制结束后用于事后评估。phase/CFO 条件也显式从 Adapt-only 接收视图计算，CIR/phase 状态更新会先屏蔽非 Adapt 区域。新增回归测试改变 Reward Pilot 波形，并从 Adapt-only 波形计算 phase/CFO 条件，要求策略 observation 和条件 embedding 均保持不变。全量测试 **631 项全部通过**。

本文件中 `gap=600 秒` 的 5 dB、10/15 dB 和四 SNR 结果均来自该边界修正前，只保留作历史记录，不能作为当前协议下的有效实验结论。后面的 gap=0、`heldout_edge` 结果有独立 Adapt-only 复跑记录，且已在显式屏蔽条件输入后再次复核。旧版 gap=600 秒结果与跨 SNR 结果没有在新协议下重跑。

## 历史诊断（协议修正前）：固定动作对照：5 seed × 24 帧

使用前一阶段相同的 Level B、5 dB、gap 600 秒、Pilot 256、seed 1061–1065 和固定 checkpoint。比较冻结模型与一个固定输出 moderate 动作的诊断策略；每个 8 帧窗口都选择 moderate，PEFT 参数仍只由 Adapt Pilot 更新，Reward Pilot 可逐帧拒绝并恢复。策略本身不执行 PPO 梯度更新。这一组只用于区分候选更新与 PPO 选动作的作用。

| Seed | Frozen BER | 固定 moderate BER | BER 改善（百分点） | Reward 接受 / 拒绝 |
|---:|---:|---:|---:|---:|
| 1061 | 30.2029% | 29.4705% | +0.7324 | 17 / 7 |
| 1062 | 36.7188% | 36.3173% | +0.4015 | 19 / 5 |
| 1063 | 18.6795% | 17.6921% | +0.9874 | 11 / 13 |
| 1064 | 27.1918% | 27.8646% | −0.6727 | 15 / 9 |
| 1065 | 41.5907% | 40.4622% | +1.1285 | 18 / 6 |
| **全部 120 帧** | **30.8767%** | **30.3613%** | **+0.5154** | **80 / 40** |

120 个帧级 PEFT 候选中，Reward 接受 80 个、拒绝 40 个；接受的候选相对本帧更新前 Data BER 平均改善约 `0.91` 个百分点，拒绝的候选都已恢复到本帧更新前状态。尽管如此，seed 1064 的最终平均 BER 仍比 Frozen 差 `0.67` 个百分点，说明逐帧 Reward 门控不能保证累计收益。

## 历史诊断（协议修正前）：PPO 长轨迹：5 seed × 60 帧

冻结与 PPO 两种方法在同一运行内复用相同帧，减少单独运行对比带来的随机差异。固定 SNR 5 dB、gap 600 秒、同一 checkpoint 和 seed；PPO 每 8 帧选一次动作。

| Seed | Frozen BER | PPO-PEFT BER | BER 改善（百分点） | moderate 窗口数 |
|---:|---:|---:|---:|---:|
| 1061 | 31.0091% | 31.0091% | 0.0000 | 0/8 |
| 1062 | 35.2886% | 35.2474% | +0.0412 | 2/8 |
| 1063 | 19.2014% | 19.2014% | 0.0000 | 0/8 |
| 1064 | 32.5347% | 33.1554% | −0.6207 | 2/8 |
| 1065 | 40.4622% | 40.3234% | +0.1389 | 4/8 |
| **全部 300 帧** | **31.6992%** | **31.7873%** | **−0.0881** | **8/40** |

PPO 在 300 帧中有 60 帧选择 moderate，PEFT 都实际改变了参数；47 帧通过帧级 Reward，13 帧被拒绝。窗口 PEFT 偏移上限仍为约 `0.05`，Data 波形和 Data 标签在线使用标记均为 `false`。5 条轨迹中有 2 条没有尝试更新，另有 1 条平均退化明显。最后 4 帧落在不完整的第 8 个窗口，因此这 5 个末尾窗口没有窗口级 PPO reward 更新；帧级 Reward 守门仍照常执行。

## 历史诊断（协议修正前）：Moderate 候选的跨 SNR 筛查：10/15 dB，各 5 seed × 24 帧

为判断前述固定 moderate 的正收益是否只出现在 5 dB，继续使用同一固定 checkpoint、seed、channel gap、Pilot 与 Reward 守门，在 10 和 15 dB 运行冻结与固定 moderate 配对对照。两种方法在同一次运行中复用每个 seed 的同一帧。

| SNR | Frozen BER | 固定 moderate BER | 平均改善（百分点） | 正收益 seed 轨迹 | Reward 接受 / 拒绝 |
|---:|---:|---:|---:|---:|---:|
| 10 dB | 29.2676% | 28.7967% | +0.4709 | 5/5 | 88 / 32 |
| 15 dB | 28.5829% | 28.2368% | +0.3461 | 4/5 | 76 / 44 |

15 个 SNR×seed 轨迹（5/10/15 dB，每点 5 seed）有 13 个平均 BER 改善。合并 360 帧的平均改善为 `+0.4442` 个百分点。所有 240 个新增候选帧都产生非零 PEFT 参数变化，Data 信息在线标记均为 `false`。这支持继续验证 moderate 候选，但两个 SNR 点各有一条退化轨迹，不能视为稳定、正式的主实验结果。

机器可读逐配置摘要见 JSON 文件中的 `fixed_moderate_10_15db_screen` 字段。

## 历史诊断（协议修正前）：提高 moderate 探索先验的 PPO 对照：5 seed × 24 帧

在同样的 5 dB、Level B、gap 600 秒条件下，将 PPO 初始动作概率调整为约 `identity 60.4% / moderate 36.6% / rollback 3.0%`（原 moderate 概率约 14%），仍允许 PPO 根据窗口 Reward 学习。结果为：

- moderate 被选中 `7/15` 个窗口（原先验 `4/15`）；共有 56 帧产生 PEFT 更新，其中 32 个候选通过帧级 Reward、24 个被拒绝，另有 1 次窗口级回滚。
- 相对 Frozen 的平均 BER 改善为 `+0.1769` 个百分点，超过原先验的 `+0.0825`，但低于固定 moderate 的 `+0.5154`。
- 逐 seed 改善为 `0.0000 / −0.1031 / +0.5480 / −0.4991 / +0.9386` 个百分点；仍是 2/5 正收益、2/5 负收益、1/5 未触发更新。

提高探索确实让 PPO 多试更新，但没有复制固定 moderate 的收益。随后为诊断配置增加了可选的轨迹总偏移上限：PEFT 相对该轨迹首次更新时的参数变化不能超过 `0.05`。单独重跑固定 moderate 后，5 dB 的平均收益仍为 `+0.5132` 个百分点，4/5 seed 为正；之前最差的 seed 从 `−0.6727` 收窄至 `−0.2116` 个百分点，但没有消除退化。

固定 moderate + 轨迹总偏移 `0.05` 的四 SNR 筛查（各 5 seed × 24 帧，共 480 帧）结果如下：

| SNR | Frozen BER | 固定 moderate BER | 平均改善（百分点） | 正收益 seed 轨迹 |
|---:|---:|---:|---:|---:|
| 0 dB | 39.1200% | 38.5536% | +0.5664 | 5/5 |
| 5 dB | 30.8767% | 30.3635% | +0.5132 | 4/5 |
| 10 dB | 29.2676% | 28.7977% | +0.4698 | 5/5 |
| 15 dB | 28.5829% | 28.2064% | +0.3765 | 4/5 |
| **全部** | **31.9618%** | **31.4803%** | **+0.4815** | **18/20** |

然后把高 moderate 探索先验与同一轨迹总偏移上限一起用于 PPO，并在同样四 SNR、5 seed × 24 帧下复核。PPO 总体 BER 改善为 `+0.1628` 个百分点，14/20 条 seed×SNR 轨迹为正；逐 SNR 改善为 0 dB `+0.1204`（5/5）、5 dB `+0.2431`（2/5）、10 dB `+0.0673`（3/5）、15 dB `+0.2203`（4/5）。60 个动作窗口里 moderate 占 22 个，identity 占 37 个，rollback_identity 占 1 个；moderate 帧级更新通过 Reward 111 次、拒绝 65 次。总偏移上限和 Data 隔离标记均通过。

这些早期结果不能用于声称当前方案已在 acquisition 老化或跨 SNR 条件下获益，因为 PPO 观察越过了 Reward Pilot 信息边界。它们只能提示 moderate PEFT 候选值得在修正后的协议下重新验证。

策略初始权重作为小型复现实验资产一并提交：固定 moderate 策略 [`docs/results/assets/rl_peft_fixed_moderate_policy.pt`](results/assets/rl_peft_fixed_moderate_policy.pt)，高探索 PPO 初始化 [`docs/results/assets/rl_peft_moderate_exploration_policy_init.pt`](results/assets/rl_peft_moderate_exploration_policy_init.pt)。机器可读摘要见 [`docs/results/rl_peft_moderate_selection_followup_20261030.json`](results/rl_peft_moderate_selection_followup_20261030.json)。原始逐帧日志仍保存在被 `.gitignore` 排除的 `logs/`。

## gap=0 的 heldout_edge 配对复核

为了检查更强的信道状态失配，在 acquisition gap 为 0、Level B `heldout_edge`、5 dB 下，用同一 checkpoint、seed 1061–1065、每 seed 24 帧和 Pilot=256，比较高探索 PPO 与从 moderate 起始策略出发的 PPO。两次运行的 Frozen BER 与逐帧 `state_instance` 完全一致，确认对照复用同一组信道状态。

| 策略 | Frozen BER | 在线 BER | 平均改善 | 正收益 seed | moderate 动作帧 | Reward 接受 / 拒绝 |
|---|---:|---:|---:|---:|---:|---:|
| 高探索 PPO | 4.2904% | 4.2383% | +0.0521 pp | 3/5（1 平、1 负） | 56/120 | 28 / 28 |
| moderate 起始 PPO | 4.2904% | 4.1851% | +0.1053 pp | 5/5 | 120/120 | 64 / 56 |

高探索 PPO 的 seed 改善为 `0.0000、−0.0054、+0.2170、+0.0271、+0.0217 pp`；1061 没有尝试 PEFT，1062 略退化。moderate 起始策略的 seed 改善为 `+0.0705、+0.0109、+0.3906、+0.0271、+0.0271 pp`。加载的 moderate 起始策略在 15 个完整窗口中都选择 moderate；它仍按项目实现执行窗口级 PPO 更新，因此这是“策略始终选择 moderate”的观测结果，不是冻结策略权重的独立算法。

moderate 起始策略的 120 帧都产生了非零 PEFT 候选偏移，轨迹偏移不超过 `0.05`；Reward Pilot 接受 64 帧、拒绝 56 帧，另有 10 个窗口级回滚。事后 Data 指标显示 68 帧的 logits 发生变化、44 帧 Data BCE 改善。所有在线行的 `data_waveform_used_online` 和 `data_labels_used_online` 都为 `false`；Data 只在 Pilot 更新和 Reward 验收/窗口回滚完成后用于评估。离线 checkpoint 未改变。

这组小样本说明 moderate 更新候选有正向信号，PPO 的动作选择会漏掉部分可用更新；但收益只有 `0.1053 pp`，且改善幅度受 seed 影响明显。按 5 条 seed 级配对改善计算的近似 95% t 区间为 `−0.0947 到 +0.3052 pp`，包含零，因此目前不能说收益已被统计确认。它也只测了一个 SNR 和一个状态分布。机器可读逐配置结果见 JSON 中的 `gap0_heldout_edge_5db_policy_comparison`。原始逐帧日志位于被 `.gitignore` 排除的两个 `logs/` 目录。

该 5×24 结果来自协议修正后的重跑：`logs/rl_peft_ppo_explore_adapt_only_gap0_heldout_edge_5db_5s24f_seed1061_1065_20261030/` 与 `logs/rl_peft_moderate_init_adapt_only_gap0_heldout_edge_5db_5s24f_seed1061_1065_20261030/`。120 条在线行的 `policy_observation_source` 都是 `adapt_pilot_only`，Reward Pilot 未用于当前动作 observation，Data waveform/labels 标记均为 `false`。

复现命令（CPU）：

```powershell
python compare.py --config configs/rl_peft_phase_signal_diagnostic.json --policy docs/results/assets/rl_peft_moderate_exploration_policy_init.pt --pretrained D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --methods 'Frozen Offline NN' 'RL-Modulated Neural Block Equalizer' --delays 116 --snrs 5 --seed-start 1061 --num-seeds 5 --frames 24 --pilot-total 256 --reward-pilot-total 32 --pilot-layout prefix --state-split heldout_edge --acquisition-to-data-gap-seconds 0 --update-interval 8 --cir-update fixed --online-condition-source pilot_cir_phase --device cpu --output-dir logs/rl_peft_ppo_explore_adapt_only_gap0_heldout_edge_5db_5s24f_seed1061_1065_20261030
python compare.py --config configs/rl_peft_phase_signal_diagnostic.json --policy docs/results/assets/rl_peft_fixed_moderate_policy.pt --pretrained D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --methods 'Frozen Offline NN' 'RL-Modulated Neural Block Equalizer' --delays 116 --snrs 5 --seed-start 1061 --num-seeds 5 --frames 24 --pilot-total 256 --reward-pilot-total 32 --pilot-layout prefix --state-split heldout_edge --acquisition-to-data-gap-seconds 0 --update-interval 8 --cir-update fixed --online-condition-source pilot_cir_phase --device cpu --output-dir logs/rl_peft_moderate_init_adapt_only_gap0_heldout_edge_5db_5s24f_seed1061_1065_20261030
```

## 新 seed 的 60 帧长轨迹复核

为检查 24 帧正向迹象能否持续，在新的 seed 1066–1070 上重复同一 Level B `heldout_edge`、gap=0、5 dB 配置，每 seed 延长到 60 帧。moderate 起始 PPO 与高探索 PPO 使用同一组 Frozen 帧；逐帧 Frozen BER 和 `state_instance` 完全一致。

| 策略 | Frozen BER | 在线 BER | 平均改善 | 正收益 seed | moderate 动作帧 | Reward 接受 / 拒绝 |
|---|---:|---:|---:|---:|---:|---:|
| moderate 起始 PPO | 4.9188% | 4.8503% | +0.0686 pp | 4/5 | 300/300 | 171 / 129 |
| 高探索 PPO | 4.9188% | 4.8928% | +0.0260 pp | 2/5 | 88/300 | 54 / 34 |

moderate 起始 PPO 的五个 seed 改善为 `+0.1128、+0.1497、+0.0195、−0.0087、+0.0694 pp`；高探索 PPO 为 `+0.0217、+0.1454、−0.0260、−0.0043、−0.0065 pp`。moderate 起始策略的 35 个完整窗口中有 17 个窗口级回滚；高探索 PPO 有 5 个。两种策略轨迹偏移都不超过 `0.05`，300 条在线记录都报告 Data 波形和标签未用于在线更新或动作选择。

moderate 起始 PPO 按连续 20 帧段的平均配对改善是 `+0.0482、+0.0638、+0.0938 pp`，看起来平均优势随帧数增加；但正收益 seed 数从 `4/5` 变为 `3/5、3/5`，且五条 seed 级改善的近似 95% t 区间为 `−0.0122 到 +0.1493 pp`，仍包含零。高探索 PPO 三段平均改善为 `+0.0078、+0.0273、+0.0430 pp`，只有 2/5 seed 全程正向。新 seed 的长轨迹支持 moderate PEFT 候选有小幅持续收益、PPO 仍明显少选有效动作，但并未证明稳健或显著收益。

机器可读数据见 JSON 中的 `gap0_heldout_edge_5db_new_seed_60_frame_comparison`；原始逐帧数据分别保存在被忽略的 `logs/rl_peft_moderate_init_gap0_heldout_edge_5db_5s60f_seed1066_1070_20261030/` 和 `logs/rl_peft_ppo_explore_episode_cap005_gap0_heldout_edge_5db_5s60f_seed1066_1070_20261030/`。

该 5×60 对照随后又在更严格的代码路径下复跑：phase/CFO 条件改为从显式屏蔽后的 Adapt-only 视图计算，CIR 与物理状态估计也先屏蔽非 Adapt 区域。有效的最终日志目录为 `logs/rl_peft_moderate_init_explicit_adapt_v2_gap0_heldout_edge_5db_5s60f_seed1066_1070_20261030/` 和 `logs/rl_peft_explore_explicit_adapt_v2_gap0_heldout_edge_5db_5s60f_seed1066_1070_20261030/`。与前一次 Adapt-only 日志逐行核对后，600 条记录的 BER、动作、Reward 接受/拒绝、回滚和 Frozen 信道状态完全一致；300 条 PPO 在线行均为 `adapt_pilot_only`，Reward observation、Data waveform 和 Data labels 越界标记均为 0。固定 moderate 起始 PPO 改善 `0.0686 pp`、4/5 seed 为正；高探索 PPO 改善 `0.0260 pp`、2/5 seed 为正。两组 seed 级区间仍包含零，不能说已经稳定超过离线模型。

复现命令（CPU）：

```powershell
python compare.py --config configs/rl_peft_phase_signal_diagnostic.json --policy docs/results/assets/rl_peft_fixed_moderate_policy.pt --pretrained D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --methods 'Frozen Offline NN' 'RL-Modulated Neural Block Equalizer' --delays 116 --snrs 5 --seed-start 1066 --num-seeds 5 --frames 60 --pilot-total 256 --reward-pilot-total 32 --pilot-layout prefix --state-split heldout_edge --acquisition-to-data-gap-seconds 0 --update-interval 8 --cir-update fixed --online-condition-source pilot_cir_phase --device cpu --output-dir logs/rl_peft_moderate_init_explicit_adapt_v2_gap0_heldout_edge_5db_5s60f_seed1066_1070_20261030
python compare.py --config configs/rl_peft_phase_signal_diagnostic.json --policy docs/results/assets/rl_peft_moderate_exploration_policy_init.pt --pretrained D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --methods 'Frozen Offline NN' 'RL-Modulated Neural Block Equalizer' --delays 116 --snrs 5 --seed-start 1066 --num-seeds 5 --frames 60 --pilot-total 256 --reward-pilot-total 32 --pilot-layout prefix --state-split heldout_edge --acquisition-to-data-gap-seconds 0 --update-interval 8 --cir-update fixed --online-condition-source pilot_cir_phase --device cpu --output-dir logs/rl_peft_explore_explicit_adapt_v2_gap0_heldout_edge_5db_5s60f_seed1066_1070_20261030
```

下一步先完成显式 Adapt-only 条件输入下的 5 dB 重放，然后检查 PPO 是否能利用已有的 CIR/phase 状态信息选择 PEFT 更新。离线 checkpoint、PEFT 候选、Reward 验收和 Data 隔离保持不变；只有策略收益在独立 seed 上可重复，才扩大到多 SNR 和正式矩阵。
