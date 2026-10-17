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

## 当前结论

本阶段已经修复并验证了“在线更新是否偷看 Data 波形”的实现问题，但尚未解决 Pilot 代理与 Data 目标之间的排序失配。当前不能声称在线微调稳定超过离线模型，也没有修改离线 checkpoint。

此前接近随机的 BER 异常已定位为 `acquisition` 相位条件被沿用到当前帧；不应与本次 `pilot_cir_phase` replay 混为一谈。留出 replay 的整帧波形隔离问题是本次另外发现并修复的实现缺陷。

下一步应继续在固定 checkpoint 和同一 Level B 协议下设计真正独立、可解释的 Pilot-only 候选/状态特征，并先用 replay 证明排序相关性稳定通过，再扩大 seed、SNR 和帧数；不应通过放宽门限或引入 Data 伪标签来制造收益。
