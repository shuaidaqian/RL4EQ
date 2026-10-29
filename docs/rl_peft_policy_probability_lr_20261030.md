# PPO 动作概率与学习率复核（2026-10-30）

## 结论

状态 embedding 没有实质改变 PPO 的动作概率；提高 PPO 学习率则能改变策略概率，并在独立 seed 上实际多选了 moderate PEFT。5 dB 下合并两批共 10 个 seed，高学习率的在线 BER 相对 Frozen Offline NN 平均改善 `+0.168 pp`，10/10 个 seed 为正。但它相对默认学习率 `3e-5` 的额外改善只有 `+0.0098 pp`，配对 95% 区间为 `−0.0050` 到 `+0.0246 pp`，跨过零。因此 `1e-3` 是值得进入四 SNR 验证的候选参数，还不能说它比默认策略更好，也没有证明多 SNR 稳定收益。

本轮没有使用 Data 波形或标签做在线动作选择、PEFT 更新、reward 或回滚。它们只用于每帧在线流程结束后的仿真评估。离线 checkpoint 与 PEFT 候选不变。

## 先看策略概率

为分清“PPO 学不动”与“概率动了但采样动作没变”，现在每条在线记录会写入所有离散动作概率、实际抽中动作的概率和当时的学习率。

在 seed 1083 的 60 帧诊断里，7 个窗口 reward 有 5 个为零、1 个为正、1 个为负。update interval 从 16 降到 8，使 PPO 更新数从 3 次增到 7 次；moderate 概率变化范围从约 `0.00056` 扩大到 `0.00084`，但动作序列和 BER 没变。embedding 开关的最大概率差约 `3.6×10⁻⁷`，接近浮点误差。

仍在同一 seed 上只调学习率：默认 `3e-5` 时 moderate 概率变化范围约 `0.00084`；调到 `3e-4` 后约 `0.0088`；调到 `1e-3` 后约 `0.0443`。这个 seed 的抽样动作和 BER 仍相同，说明更大概率变化并不自动保证选到不同动作。

## 5 dB 的配对结果

先在 seeds 1081–1085 上做学习率配对：

| 策略学习率 | Frozen BER | 在线 BER | 平均改善 | 正收益 seed | moderate PEFT 帧 |
|---:|---:|---:|---:|---:|---:|
| `3e-5` | 5.5781% | 5.4770% | +0.1011 pp | 4/5 | 124/300 |
| `1e-3` | 5.5781% | 5.4701% | +0.1081 pp | 5/5 | 140/300 |

高学习率只在 seed 1082 改变了抽样动作；该 seed 从 `−0.0043 pp` 变为 `+0.0304 pp`。五个 seed 的平均增量仅 `+0.0069 pp`，95% 配对区间为 `−0.0123` 到 `+0.0262 pp`。

随后在独立 seeds 1086–1090 上验证：

| 策略学习率 | Frozen BER | 在线 BER | 平均改善 | 正收益 seed | moderate PEFT 帧 |
|---:|---:|---:|---:|---:|---:|
| `3e-5` | 5.0113% | 4.7956% | +0.2157 pp | 4/5 | 112/300 |
| `1e-3` | 5.0113% | 4.7830% | +0.2283 pp | 5/5 | 148/300 |

高学习率在这批数据里改变了 36/300 帧的动作，Reward 接受/拒绝为 `83/65`，并有 7 次窗口回滚；默认学习率为 `61/51`，5 次窗口回滚。BER 正向 seed 从 4/5 变成 5/5，但平均增量只有 `+0.0126 pp`，95% 配对区间为 `−0.0203` 到 `+0.0454 pp`。

把两批 10 个 seed 合并仅作 5 dB 探索性摘要：默认学习率平均改善 `+0.1584 pp`、8/10 为正；`1e-3` 平均改善 `+0.1682 pp`、10/10 为正。高学习率与默认值的配对增量为 `+0.0098 pp`，95% 区间仍跨零。收益看起来更一致，但差距很小，且目前只在 5 dB 验证。

所有对照的 Frozen BER、输入哈希和信道轨迹按 seed/帧配对；在线 Data 波形/标签边界越界计数为 0。机器可读数据和逐帧日志目录见 [`docs/results/rl_peft_policy_probability_lr_20261030.json`](results/rl_peft_policy_probability_lr_20261030.json)。

## 下一步

保持状态 embedding 关闭、`update_interval=8`，在新的 heldout seed 上用 `1e-3` 策略学习率跑 Level B 主 SNR `0/5/10/15 dB` 的 5 seed × 60 帧矩阵，并与 Frozen Offline NN 同轨迹比较。只有各 SNR 不出现系统性退化、改善幅度在 seed 间可复现，才将这个学习率带入与传统非神经 baseline 的正式比较。当前 5 dB 的正收益不能代替四个 SNR 的主结果。

复现本轮 5 dB 配对的核心命令：

```powershell
python compare.py --config configs/rl_peft_phase_signal_diagnostic.json --policy docs/results/assets/rl_peft_moderate_exploration_policy_init.pt --pretrained D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --methods 'Frozen Offline NN' 'RL-Modulated Neural Block Equalizer' --delays 116 --snrs 5 --seed-start 1086 --num-seeds 5 --frames 60 --pilot-total 256 --reward-pilot-total 32 --pilot-layout prefix --state-split heldout_edge --acquisition-to-data-gap-seconds 0 --update-interval 8 --cir-update fixed --online-condition-source pilot_cir_phase --device cpu --no-online-ppo-pilot-state-embedding --output-dir logs/rl_peft_pilot_default_lr_gap0_heldout_edge_5db_5s60f_update8_seed1086_1090_20261030
python compare.py --config configs/rl_peft_phase_signal_diagnostic.json --policy docs/results/assets/rl_peft_moderate_exploration_policy_init.pt --pretrained D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --methods 'Frozen Offline NN' 'RL-Modulated Neural Block Equalizer' --delays 116 --snrs 5 --seed-start 1086 --num-seeds 5 --frames 60 --pilot-total 256 --reward-pilot-total 32 --pilot-layout prefix --state-split heldout_edge --acquisition-to-data-gap-seconds 0 --update-interval 8 --cir-update fixed --online-condition-source pilot_cir_phase --device cpu --no-online-ppo-pilot-state-embedding --online-ppo-learning-rate 0.001 --output-dir logs/rl_peft_pilot_lr1e3_gap0_heldout_edge_5db_5s60f_update8_seed1086_1090_20261030
```
