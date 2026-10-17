# Pilot-only 留出 replay 与波形隔离结论（2026-10-17）

## 目的

在固定离线 checkpoint、Level B、`pilot_cir_phase` 和 prefix Pilot 协议下，确认在线 PEFT 更新是否真的只使用 Adapt Pilot，并检查 Adapt Pilot 留出指标能否排序后续 Data 段收益。

固定 checkpoint：

```text
pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt
SHA256 8CD3912D1C957C9689C51730AFF9D27A8F74942186C802A7483E5781AF65AEC4
```

## 修复内容

此前留出 replay 只从 `adapt_mask` 移除了留出 Pilot，但更新帧的 `rx_symbols` 仍保留整帧波形。对于带跨位置注意力的均衡器，这会让留出 Pilot、Reward Pilot 或 Data 波形通过前向路径影响在线梯度。

本次修复包括：

- `_split_adapt_pilot` 返回的更新帧只保留更新 Pilot 的接收波形，其余位置全部清零；
- `apply_adapt_only_peft_update` 内部再次构造 Adapt-only 更新视图，避免其他调用者漏传隔离帧；
- 候选排序的 Adapt、留出 Pilot、Reward 和信道重构分数只在 Pilot-only 视图上计算；
- 多帧 Reward 验收也只在 Adapt + Reward Pilot 视图上计算，跨帧 soft tail 只由 Pilot 可见位置更新；
- Data 波形和 Data 标签只在更新完成后的原始帧上做事后 BER/BCE 与连续输出诊断；
- replay 结果显式记录 `data_waveform_used_online=false` 和 `online_update_uses_data_waveform=false`；
- 子窗口 replay 同样清零非更新 Pilot 波形，并把原始帧仅用于留出/事后评估。

## 验证

针对性测试：

```text
64 passed
```

完整测试：

```text
587 passed
```

新增隔离检查验证：

- 更新 Pilot 与留出 Pilot 互斥；
- 更新帧非更新位置的接收波形全为零；
- 修改 Data 接收波形不会改变在线 PEFT 参数更新；
- 修改 Data 接收波形不会改变 Pilot 候选排序分数。
- 修改 Data 接收波形不会改变多帧 Reward gate 的决策和 Reward 指标。

多帧 Reward gate smoke 已成功生成 3 个候选结果；结果中的内部模型/soft-tail 状态不会写入 JSON，且所有在线 Data 隔离字段均为 `false`。

固定协议 smoke（10 dB、seed=2、1 帧、Pilot=256、留出比例=0.25）确认更新 Pilot 为 168 个、留出 Pilot 为 56 个，PEFT 参数确实发生变化，且所有在线隔离字段为 `false`。

## Pilot-only 排序结果

在 10 dB、seed=2/3/4、每条轨迹 2 帧、候选为 `phase_trend_signal_reconstruction`、`phase_trend_bce` 和 `head_bce` 的 replay 中：

| Pilot-only 排序分数 | 与事后 Data BER 改善 Spearman | 是否通过 0.6 门槛 |
|---|---:|---|
| Adapt BCE | -0.023 | 否 |
| Adapt 留出 Pilot BCE | -0.398 | 否 |
| Reward BCE | 0.398 | 否 |
| Pilot 信号重构 | 0.306 | 否 |

因此当前 Pilot-only 信号仍不能稳定排序候选更新，不能进入正式 `5 seed × 60 帧` 主矩阵，也不能引入 Contextual Bandit。旧 replay 中接近 0.45 甚至更高的 BER 结果也不能作为当前方法性能结论；它们来自错误的 acquisition 相位条件或未隔离的诊断路径。

10 dB 这一组有 6 条 seed/frame 轨迹，每条轨迹评估 3 个候选；表中的 18 个候选结果不是 18 条独立轨迹。更大的主 SNR 复核有 12 条 seed/SNR 轨迹，每条评估 3 个候选，36 个候选行同样不能当成 36 个独立样本。

在主 SNR `0/5/10/15 dB`、seed=2/3/4、每条轨迹 1 帧的复核中，相关性为：Adapt BCE `0.073`、Adapt 留出 Pilot BCE `-0.285`、Reward BCE `0.220`、Pilot 信号重构 `0.242`，均未通过 `0.6` 门槛。0/5/15 dB 的 9 个非 identity 候选行均没有正 Data BER 改善，10 dB 只有 1/9 个候选行改善。

随后把 PEFT 更新从 `lr=1e-4, steps=1` 提高到 `lr=1e-3, steps=5` 做强度诊断。参数变化确实增大，但 12 条主 SNR 轨迹上的排序仍未通过：Adapt BCE `0.464`、留出 Pilot BCE `0.168`、Reward BCE `0.589`、Pilot 重构 `0.129`。按 Reward 选择每条轨迹的最佳候选后，12 条轨迹中 11 条尝试更新，只有 6 条 Data BER 改善；因此更大的更新幅度不能单独解决 Pilot 到 Data 的映射失配。

## 当前结论

本阶段已经修复并验证了“在线更新是否偷看 Data 波形”的实现问题，但尚未解决 Pilot 代理与 Data 目标之间的排序失配。当前不能声称在线微调稳定超过离线模型，也没有修改离线 checkpoint。

此前接近随机的 BER 异常已定位为 `acquisition` 相位条件被沿用到当前帧；不应与本次 `pilot_cir_phase` replay 混为一谈。留出 replay 的整帧波形隔离问题是本次另外发现并修复的实现缺陷。

## 固定调度器入口修复与复核

发现 `compare.py` 的命令行参数默认值为 `bandit`，会覆盖配置文件中已经写明的
`online_scheduler: fixed`。这使主入口在尚未满足 Pilot-only 排序门槛时，可能因为没有匹配的
Bandit 动作而直接 `skip`，无法区分“Reward Pilot 拒绝候选”和“根本没有候选”。现已修复为：

- 未显式传 `--scheduler` 时读取配置文件的 `online_scheduler`，缺省值为 `fixed`；
- 只有显式传 `--scheduler bandit` 才启用安全 Contextual Bandit；
- 结果摘要和每帧记录使用同一个有效调度器值；
- 回归测试覆盖配置默认 `fixed` 与显式 `bandit` 两条路径。

固定 `fixed` 调度器后的强失配诊断（Level B、delay=116、600 s acquisition gap、
`pilot_cir_phase`、Pilot=256、Reward Pilot=32、候选为 phase signal 的保守/中等/激进三档）结果：

| 条件 | Online 相对 Frozen 的结果 |
|---|---:|
| 10 dB，seed=2/3/4，4 帧 | 平均 BER 改善 0.01584；12 个逐帧配对中 7 个改善、3 个持平、2 个变差 |
| 0 dB，seed=2/3/4，2 帧 | 当前配置 `freeze_below_snr_db=5`，6 帧均不更新，Online 与 Frozen 持平 |
| 5 dB，seed=2/3/4，2 帧 | 平均 BER 改善 0.00890；6 帧中 2 个改善、1 个变差、3 个持平 |
| 10 dB，seed=2/3/4，2 帧 | 平均 BER 改善 0.01063；6 帧中 3 个改善、1 个变差、2 个持平 |
| 15 dB，seed=2/3/4，2 帧 | 平均 BER 改善 0.02582；6 帧中 2 个改善、0 个变差、4 个持平 |

12 条 10 dB 轨迹共 36 个候选的 Reward 排名与事后 Data BER 改善 Spearman 为 `0.809`；
但真正经过门控后只有 7/12 帧变好，因此候选排序已有可利用信号，尚未达到稳定在线增益的要求。
上述 Data BER 只在更新完成后作事后评估，所有记录仍为
`data_waveform_used_online=false`、`data_labels_used_online=false`。

本次固定调度器复核日志：

```text
logs/compare_fixed_scheduler_strong_mismatch_smoke_20261017/
logs/compare_fixed_scheduler_strong_mismatch_10db_seed2_4_20261017/
logs/compare_fixed_scheduler_strong_mismatch_main_snrs_seed2_4_f2_20261017/
```

补充的低 SNR 与排序模式诊断：

- 临时解除 `0 dB` 的 `freeze_below_snr_db=5` 后，6 个帧配对中 3 个改善、1 个变差、2 个持平，平均改善 `0.01584`；但 Reward 排名与 Data BER 改善 Spearman 只有 `0.098`。
- 将排序从 `robust_lcb` 改为 `aggregate_bce` 后，0/5/10/15 dB 的候选级 Spearman 约为 `0.537/0.332/0.724/0.603`。
- 加入 Pilot 信号重构一致性约束的 `pilot_consensus` 后，对应相关性约为 `0.060/0.311/0.725/0.689`。

这说明 10/15 dB 的 Pilot BCE 排序已有较明显信号，但 0/5 dB 仍不可靠；不同排序公式没有在所有 SNR 上同时稳定。当前应优先研究低 SNR 下的 Pilot 置信度与候选安全门，而不是继续调大更新幅度或引入 Data 伪标签。

## 两个安全策略的追加诊断

- `online_defer_peft_commit` 将当前帧候选延迟到下一帧 Reward Pilot 再提交。在 10 dB、seed=2/3/4、4 帧中只有 1/12 帧改善、1/12 帧变差，平均收益为负；暂不作为主策略。
- 将 Reward Pilot 从 2 个子窗口增加到 4 个子窗口后，同一条件下有 6/12 帧改善、5 帧持平、1 帧变差，平均改善 `0.01378`。它比延迟提交可靠，也略好于 2 窗口的 7/12 改善、2 帧变差结果，但仍未达到逐帧稳定门槛。

因此当前可保留 4 窗口作为后续候选的安全消融，不把它包装成已经稳定成功的在线方法；正式主矩阵仍需等待跨 SNR、跨 seed 的 Pilot-only 排序和逐帧收益同时通过。

当前仍不能宣称在线方法在所有主 SNR、seed 和帧上稳定超过 Frozen，也不能进入正式
`5 seed x 60 帧` 主矩阵或启用 Bandit 主比较。下一步是单独解除 0 dB 的低 SNR 冻结做诊断，
并比较不同 Pilot-only 排名模式；如果低 SNR 解除冻结仍不稳定，应增加 Pilot-only 状态可靠性门，
而不是使用 Data 伪标签或修改离线 checkpoint。

下一步应继续在固定 checkpoint 和同一 Level B 协议下设计真正独立、可解释的 Pilot-only 候选/状态特征，并先用 replay 证明排序相关性稳定通过，再扩大 seed、SNR 和帧数；不应通过放宽门限或引入 Data 伪标签来制造收益。
