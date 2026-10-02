# A/C 扩大统一 replay 结果

日期：2026-09-28

## 本轮目的

按照当前路线决定，暂停 B，只扩大 A/C 的统一 replay。Frozen Offline NN 保留为配对参考，不作为候选方法。

## 运行协议

- 主场景：Level B，`eme_long_memory_v2`
- 最大时延：116 symbols
- impairment：`cfo_phase_tiny`
- checkpoint：`pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`
- SNR：0/5/10/15 dB
- seed：0/1/2
- 连续帧：每个 seed 30 帧
- Pilot：prefix，总长度 256，其中 Adapt=224、Reward=32
- 调度器：fixed；更新间隔 4
- 比较方式：同一 seed、同一帧、同一信道轨迹的 `BER_frozen - BER_candidate`
- Data 标签：不进入在线 observation、状态更新、动作选择、reward 或参数更新，只用于最终 BER 统计

输出文件：

- [逐帧记录](../logs/abc_replay_ac_3s_30f/frame_metrics.jsonl)
- [配对汇总](../logs/abc_replay_ac_3s_30f/summary.json)

## 记录完整性审计

| 方法 | 记录数 | 配对键 | Data 标签在线使用 | CIR 更新 | PEFT 更新 | RL |
|---|---:|---|---|---|---|---|
| Frozen Offline NN | 360 | 完整 | 全部 false | 全部 false | 全部 false | 全部 false |
| ABC-A Pilot State Gate | 360 | 与 Frozen 完全一致 | 全部 false | 全部 true | 全部 false | 全部 false |
| ABC-C Physics Unfolded | 360 | 与 Frozen 完全一致 | 全部 false | 全部 true | 全部 false | 全部 false |

本轮只生成 Frozen、`candidate_1`（A）和 `candidate_2`（C），没有生成 B 的目录或记录。合计 1080 条记录。

## 配对结果

正值表示候选 BER 低于 Frozen。

| 候选 | SNR | Frozen BER | Candidate BER | 平均配对差 | 候选胜出帧比例 | 95% block-bootstrap CI |
|---|---:|---:|---:|---:|---:|---:|
| A | 0 dB | 21.7593% | 21.7593% | +0.0000 pp | 0/90 | [0.0000, 0.0000] pp |
| A | 5 dB | 6.0127% | 6.0127% | +0.0000 pp | 0/90 | [0.0000, 0.0000] pp |
| A | 10 dB | 1.0388% | 0.9592% | +0.0796 pp | 36/90 | [-0.0709, +0.2185] pp |
| A | 15 dB | 0.2069% | 0.1360% | +0.0709 pp | 37/90 | [+0.0043, +0.1693] pp |
| C | 0 dB | 21.7593% | 21.7477% | +0.0116 pp | 18/90 | [-0.0130, +0.0318] pp |
| C | 5 dB | 6.0127% | 6.0185% | -0.0058 pp | 2/90 | [-0.0159, +0.0029] pp |
| C | 10 dB | 1.0388% | 0.9592% | +0.0796 pp | 36/90 | [-0.0738, +0.2286] pp |
| C | 15 dB | 0.2069% | 0.1360% | +0.0709 pp | 37/90 | [+0.0072, +0.1664] pp |

区间按 seed 重采样，再在每个被抽中的 seed 内按连续 10 帧 block 重采样，5000 次重复。该结果用于本轮筛选，不替代正式 5 seed × 60 帧主矩阵和 3 seed × 200 帧长期稳定性实验。

## 按 seed 的稳定性观察

- A 在 0/5 dB 三个 seed 都严格持平。
- A 在 10 dB 的 seed 均值分别为 +0.1259 pp、-0.0781 pp、+0.1910 pp，方向不一致。
- A 在 15 dB 的三个 seed 均为正，均值分别为 +0.0564 pp、+0.0043 pp、+0.1519 pp。
- C 在 0 dB 只有很小的正向差异，在 5 dB 略退化；其 10/15 dB 结果与 A 基本重合。

## 当前判断

1. A/C 的扩大 replay 没有证明在线方法在每个 SNR 点都超过 Frozen Offline NN。
2. 15 dB 是目前唯一在本轮区间下界大于 0 的配置，说明高 SNR 下 Pilot CIR 状态更新可能有可重复收益。
3. 0/5 dB 的 A 持平和 C 的轻微波动说明当前状态更新在低 SNR 下没有带来可检测的收益；这也是主门槛未满足的直接原因。
4. C 与 A 在 10/15 dB 几乎重合，当前关闭 neural residual 的物理代理没有表现出独立收益。
5. B 继续暂停进入主比较。下一轮若继续 A/C，应优先分析低 SNR 的 Pilot 状态估计误差、更新是否被置信度门控，以及 0/5 dB 是否需要重新校准 Pilot 观测到 CIR/相位状态的映射；在此之前不应宣称在线均衡已全面超过离线阶段。
