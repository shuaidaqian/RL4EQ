# RL-PEFT 高学习率四 SNR 筛查（2026-10-30）

## 结论

固定离线 checkpoint 后，用 PPO 在线学习率 `1e-3` 在 Level B 的 0/5/10/15 dB、5 个 held-out seed、每个 seed 60 帧上做了配对筛查。四个 SNR 的平均 BER 都比 Frozen Offline NN 略低，但差值很小且种子间波动明显；每个 SNR 的 95% 配对 t 区间都跨零。因此目前只能说“没有看到跨 SNR 的平均退化”，不能说在线 PEFT 已稳定、显著地超过离线模型。

这次使用的是 `configs/rl_peft_phase_signal_diagnostic.json`，其中标记 `diagnostic_only=true`、`main_aggregation_allowed=false`。结果是候选参数筛查，不是正式主结果。

## 结果

改善定义为 `Frozen Offline NN BER − 在线 BER`，正数表示在线更好。每个 seed 先对 60 帧求平均，再计算 5 个 seed 的均值和配对 95% t 区间。

| SNR | Frozen BER | 在线 BER | 平均改善 | 正收益 seed | 95% 配对区间 |
|---:|---:|---:|---:|---:|---:|
| 0 dB | 20.4952% | 20.3173% | +0.1780 pp | 4/5 | [−0.4706, +0.8265] pp |
| 5 dB | 5.2036% | 5.0156% | +0.1879 pp | 4/5 | [−0.2485, +0.6244] pp |
| 10 dB | 0.7652% | 0.7305% | +0.0347 pp | 5/5 | [−0.0151, +0.0846] pp |
| 15 dB | 0.1719% | 0.1671% | +0.0048 pp | 4/5 | [−0.0029, +0.0125] pp |

跨四个 SNR 看，正收益主要集中在 0/5 dB；10/15 dB 的绝对差值很小。按 seed 汇总四个 SNR 时，只有 3/5 个 seed 的平均差值为正，说明“每个 SNR 的均值略好”还没有转化成稳定的逐 seed 收益。

## 在线行为与数据边界

- PPO 选了 528/1200 帧的 `peft_phase_signal_moderate`，其余帧执行 identity 或 rollback identity；moderate 候选在 273 帧通过当前帧 Reward Pilot 守门，29 次窗口级回滚。
- 每条在线记录的 `policy_observation_source` 都是 `adapt_pilot_only`；Data 标签使用计数为 0，Data 波形在线使用标记为 `false`。
- 离线 checkpoint SHA256 为 `8CD3912D1C957C9689C51730AFF9D27A8F74942186C802A7483E5781AF65AEC4`，没有重训或修改。策略初始化文件 SHA256 为 `149CBFF70E1561195FFD984E1249EFAE99C39EED6F7DFB0E13FB52F4155678E5`。
- 程序每帧只调用一次 `env.next_frame()`，再把同一个帧对象传给所选方法。日志 `input_hash` 的构造本身包含方法名，所以 Frozen 与在线的哈希必然不同，不能拿哈希字符串是否相等来判断帧是否配对。
- `pre_update_ber_data` 表示在线接收机用自身当前历史状态、在本帧动作前算出的 BER；它不是 Frozen Offline NN 的 BER，二者不相等是预期行为。
- 本比较衡量整条在线接收链相对 Frozen 的变化，不能单独归因于 PEFT 参数更新。后续需要增加“同一 Adapt-only 状态处理、但固定 PEFT 参数”的消融，才可把参数微调的独立贡献分离出来。

## 判断与下一步

`1e-3` 在四 SNR 的平均 BER 上方向一致，但改进较小，置信区间不确定；它还不是已证明的最优学习率。下一步在正式 Level B 主配置 `configs/eme_long_memory_v2.json` 上，固定同一 checkpoint、同一 held-out seed、同一信道轨迹，比较 Proposed、Frozen Offline NN 和配置指定的传统 baseline。该比较回答整条 Proposed 系统是否胜过传统方法；同时补一项固定 PEFT 的 Adapt-only 消融，避免把所有增量都归功于 PEFT。只有这两项结果清楚后，再决定是否扩大主矩阵或改在线策略。

机器可读逐 seed 汇总见 [`docs/results/rl_peft_multisnr_lr1e3_20261030.json`](results/rl_peft_multisnr_lr1e3_20261030.json)，逐帧日志在本地目录 `logs/rl_peft_pilot_l1e3_gap0_heldout_edge_0_5_10_15db_5s60f_seed1091_1095_20261030/`。
