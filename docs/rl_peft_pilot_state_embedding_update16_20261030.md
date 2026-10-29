# Pilot 状态 embedding 与 PPO 更新间隔复核（2026-10-30）

## 结论

Adapt Pilot 的 CIR、phase/CFO、噪声、置信度和重构误差摘要确实进入了 PPO observation，也使少数 PPO loss 数值发生变化；但 5 个 seed、300 帧中，有无 embedding 的动作序列、PEFT 接受/拒绝结果和逐帧 Data BER 完全相同。这个 embedding 在本轮没有带来可测的均衡收益。

两组相对 Frozen Offline NN 都平均改善 `+0.1011` 个百分点，4/5 个 seed 为正；按 seed 计算的 95% t 区间为 `−0.0769` 到 `+0.2791` 个百分点，包含零。这是小样本诊断，不能据此宣称在线收益稳定或显著。

## 固定设置

- Level B `eme_long_memory_v2`、`heldout_edge`，delay 116，5 dB、acquisition gap 0 秒。
- 5 个 seed：1081–1085；每个 seed 连续 60 帧。前缀 Pilot 总长 256，其中 Adapt Pilot 224、Reward Pilot 32。
- 同一离线 checkpoint：`pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`，SHA256 为 `8CD3912D1C957C9689C51730AFF9D27A8F74942186C802A7483E5781AF65AEC4`，本轮未重训或修改。
- 使用高探索 PPO 初始化策略；动作窗口长度 8 帧。状态 embedding 配对试验的 PPO 更新间隔为 16 帧；随后又在相同 seed 和帧轨迹上比较关闭 embedding 时的 update interval=8 与 16。
- 新增状态向量包含 CIR residual、phase、CFO、噪声方差、Pilot 置信度和重构误差。关闭时 observation 为 16 维，开启时为 22 维；开启组 300 帧里出现 40 个不同的状态向量，说明特征并非恒零。

## 结果

| Seed | Frozen BER | 开启 embedding | 关闭 embedding | BER 改善（两组相同） |
|---:|---:|---:|---:|---:|
| 1081 | 4.2730% | 4.2296% | 4.2296% | +0.0434 pp |
| 1082 | 5.3624% | 5.3668% | 5.3668% | −0.0043 pp |
| 1083 | 8.2075% | 7.8624% | 7.8624% | +0.3451 pp |
| 1084 | 5.0000% | 4.8893% | 4.8893% | +0.1107 pp |
| 1085 | 5.0477% | 5.0369% | 5.0369% | +0.0109 pp |
| **平均** | **5.5781%** | **5.4770%** | **5.4770%** | **+0.1011 pp** |

开启和关闭 embedding 时，300 帧的动作完全一致：`identity` 176 帧、`peft_phase_signal_moderate` 124 帧；两组逐帧 BER、Frozen 基线、信道状态和输入哈希均一致。PPO 在每组发生 15 次更新；其中 7 行 policy loss 数值不同，但没有造成一次动作切换。中等 PEFT 候选有 63 帧通过 Reward Pilot 验收、61 帧被拒绝，并发生 9 次窗口回滚。

PEFT 候选在 124 帧实际产生非零参数变化，67 帧的 Data logits 事后评估显示变化；全帧平均 logit 绝对变化为 `0.0257`，平均 Data BCE 改善为 `0.000235`。这说明 PEFT 路径仍能改变一部分数据输出，但 embedding 没有改善候选选择，也没有把当前小收益变得更稳定。

### 更新间隔对照

为隔离策略学习频率的影响，固定关闭 embedding，在同一组 seed、信道轨迹、checkpoint 和初始 PPO 策略上，将更新间隔 8 与 16 帧配对比较：

| 更新间隔 | PPO 更新次数 | identity 帧 | moderate PEFT 帧 | Reward 接受 / 拒绝 | 平均 BER 改善 |
|---:|---:|---:|---:|---:|---:|
| 8 帧 | 35 | 176 | 124 | 63 / 61 | +0.1011 pp |
| 16 帧 | 15 | 176 | 124 | 63 / 61 | +0.1011 pp |

两种间隔的动作、Reward 结果和全部 300 帧 Data BER 都逐帧相同；Frozen BER、信道状态和输入哈希也完全一致。更频繁的策略更新增加了优化次数，却没有改变最终动作或均衡结果，因此目前没有证据支持靠继续缩短 update interval 获得收益。

在线边界检查覆盖了全部 300 条在线记录：动作 observation 来源均为 Adapt Pilot，Reward Pilot 没有进入当前动作 observation，Data 波形与标签在线使用标记均为 `false`。Data BCE、logit 与 BER 只在动作、更新及 Reward 验收完成后用于仿真评估。

机器可读汇总见 [`docs/results/rl_peft_pilot_state_embedding_update16_20261030.json`](results/rl_peft_pilot_state_embedding_update16_20261030.json)。原始逐帧日志保存在 `.gitignore` 排除的两个 `logs/` 目录中。

## 复现命令

以下命令固定同一高探索策略初始化和离线 checkpoint；运行前将 checkpoint 路径调整为本机上 SHA256 匹配的文件。

```powershell
python compare.py --config configs/rl_peft_phase_signal_diagnostic.json --policy docs/results/assets/rl_peft_moderate_exploration_policy_init.pt --pretrained D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --methods 'Frozen Offline NN' 'RL-Modulated Neural Block Equalizer' --delays 116 --snrs 5 --seed-start 1081 --num-seeds 5 --frames 60 --pilot-total 256 --reward-pilot-total 32 --pilot-layout prefix --state-split heldout_edge --acquisition-to-data-gap-seconds 0 --update-interval 16 --cir-update fixed --online-condition-source pilot_cir_phase --device cpu --online-ppo-pilot-state-embedding --output-dir logs/rl_peft_pilot_state_embedding_rngpaired_gap0_heldout_edge_5db_5s60f_update16_seed1081_1085_20261030
python compare.py --config configs/rl_peft_phase_signal_diagnostic.json --policy docs/results/assets/rl_peft_moderate_exploration_policy_init.pt --pretrained D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --methods 'Frozen Offline NN' 'RL-Modulated Neural Block Equalizer' --delays 116 --snrs 5 --seed-start 1081 --num-seeds 5 --frames 60 --pilot-total 256 --reward-pilot-total 32 --pilot-layout prefix --state-split heldout_edge --acquisition-to-data-gap-seconds 0 --update-interval 16 --cir-update fixed --online-condition-source pilot_cir_phase --device cpu --no-online-ppo-pilot-state-embedding --output-dir logs/rl_peft_pilot_no_state_embedding_rngpaired_gap0_heldout_edge_5db_5s60f_update16_seed1081_1085_20261030
python compare.py --config configs/rl_peft_phase_signal_diagnostic.json --policy docs/results/assets/rl_peft_moderate_exploration_policy_init.pt --pretrained D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --methods 'Frozen Offline NN' 'RL-Modulated Neural Block Equalizer' --delays 116 --snrs 5 --seed-start 1081 --num-seeds 5 --frames 60 --pilot-total 256 --reward-pilot-total 32 --pilot-layout prefix --state-split heldout_edge --acquisition-to-data-gap-seconds 0 --update-interval 8 --cir-update fixed --online-condition-source pilot_cir_phase --device cpu --no-online-ppo-pilot-state-embedding --output-dir logs/rl_peft_pilot_no_state_embedding_rngpaired_gap0_heldout_edge_5db_5s60f_update8_seed1081_1085_20261030
```

## 解释与后续

本轮 embedding 的“输入接通”已验证，但“有用的动作排序”没有验证成功。更关键的现象仍是：在线 PEFT 的平均 BER 改善只有约 `0.10 pp`，而 seed 级不确定区间跨过零。因此不扩大 SNR 矩阵，也不把 embedding 写成方法贡献。

后续不再调这组 embedding 或 PPO 更新间隔。应检查策略有没有学到 Pilot reward 对 PEFT 候选的可靠排序；现有证据显示即使 observation 特征和 PPO loss 有变化，最终动作序列仍不变，而且平均 BER 改善幅度小、置信区间跨零。在 Pilot-only 留出轨迹上先证明候选排序和动作选择确实变化并带来一致的 Reward 改善，再扩大 SNR；Data 波形和标签继续只用于最终仿真评估。
