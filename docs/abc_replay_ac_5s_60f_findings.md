# A/C 正式统计矩阵结果

日期：2026-09-29

## 运行协议

- 主场景：Level B，`eme_long_memory_v2`
- 最大时延：116 symbols
- impairment：`cfo_phase_tiny`
- checkpoint：`pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`
- SNR：0/5/10/15 dB
- seed：0/1/2/3/4
- 连续帧：每个 seed 60 帧
- Pilot：prefix，总长度 256，其中 Adapt=224、Reward=32
- 调度器：fixed；更新间隔 4
- 候选：A、C；B 暂停
- 比较量：同一 seed、同一帧、同一信道轨迹的 `BER_frozen - BER_candidate`
- Data 标签：不进入在线 observation、状态更新、动作选择、reward 或参数更新

输出文件：

- [逐帧记录](../logs/abc_replay_ac_5s_60f/frame_metrics.jsonl)
- [配对汇总](../logs/abc_replay_ac_5s_60f/summary.json)

## 完整性审计

本轮共 3600 条记录：Frozen、A、C 各 1200 条。A/C 与 Frozen 的配对键均为 1200/1200，差集为 0。B 没有目录、记录或统计条目。

三组方法的在线边界审计均满足 `data_labels_used_online=false`。Frozen 的 CIR/PEFT 更新均为 false；A、C 的 CIR 更新均为 true，PEFT 更新均为 false，RL 均为 false。

## 配对结果

正值表示候选 BER 低于 Frozen。CI 为按 seed 重采样、再按连续 10 帧 block 重采样的 5000 次 bootstrap 95% 区间。

| 候选 | SNR | Frozen BER | Candidate BER | 平均配对差 | 胜出帧数 | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| A | 0 dB | 21.3841% | 21.3841% | +0.0000 pp | 0/300 | [0.0000, 0.0000] pp |
| A | 5 dB | 5.7066% | 5.7066% | +0.0000 pp | 0/300 | [0.0000, 0.0000] pp |
| A | 10 dB | 0.9549% | 0.8720% | +0.0829 pp | 149/300 | [-0.0391, +0.2031] pp |
| A | 15 dB | 0.1641% | 0.1011% | +0.0629 pp | 110/300 | [-0.0026, +0.1641] pp |
| C | 0 dB | 21.3841% | 21.3932% | -0.0091 pp | 36/300 | [-0.0365, +0.0182] pp |
| C | 5 dB | 5.7066% | 5.7122% | -0.0056 pp | 7/300 | [-0.0182, +0.0052] pp |
| C | 10 dB | 0.9549% | 0.8715% | +0.0833 pp | 149/300 | [-0.0365, +0.2031] pp |
| C | 15 dB | 0.1641% | 0.1011% | +0.0629 pp | 110/300 | [-0.0000, +0.1615] pp |

## 当前结论

1. 5 seed × 60 帧确认了 3 seed × 30 帧的方向：A 在 0/5 dB 严格持平，在 10/15 dB 平均优于 Frozen；C 在 0/5 dB 略退化，在 10/15 dB 与 A 基本一致。
2. 当前 replay 仍没有实现“每个 SNR 点都超过 Frozen”。尤其 A 在 0/5 dB 的 600 条配对记录全部持平，低 SNR 缺口不是小样本偶然波动。
3. 10/15 dB 的正向差异已经在更大矩阵中复现，但在正式主门槛中仍需补齐严格的逐 SNR block-bootstrap CI，并结合预先约定的下界大于 0 标准。
4. C 没有显示独立于 A 的收益：它在低 SNR 略退化，在 10/15 dB 与 A 产生完全相同的均值和胜出计数。
5. B 继续暂停进入主比较。下一步应针对低 SNR 的 Pilot 状态估计和门控机制做诊断或调整，不能把当前 A/C 结果表述为全 SNR 在线优势。
