# EME 在线均衡阶段性结果

## 研究问题

本阶段验证的问题不是“离线神经网络能否在某个固定信道上降低 BER”，而是：在同一 Level B 极端稀疏长回波 profile 内，acquisition 得到的信道状态逐渐老化后，在线接收机能否只利用当前帧前缀 Adapt Pilot 恢复状态，并在连续帧上优于 Frozen Offline NN。

主配置保持：

- `profile_name=eme_long_memory_v2`；
- `level=B`；
- `max_delay_seconds=0.0116`，符号率等效 `max_delay=116`；
- `strong_path_count=12..24`，`diffuse_energy_ratio=0.20..0.35`；
- `coherence_time_seconds=120`；
- `pilot_layout=prefix`，`pilot_total=128`，其中 Adapt Pilot 为 96 符号、Reward Pilot 为 32 符号；
- residual CFO 和慢相位扰动来自 `cfo_phase_tiny`；
- acquisition 与数据开始之间增加 `acquisition_to_data_gap_seconds=30`，按相干时间计算一次复 tap OU 状态推进。

该 gap 不是快速 Doppler。它表示 acquisition 完成到正式数据开始之间的校准老化，使在线问题具有明确的因果来源：Frozen 方法仍使用 acquisition 条件，在线方法使用当前 Adapt Pilot 恢复状态。

## 当前在线方法

在线链路为：

```text
acquisition CIR
    -> 固定主径 support 的 Pilot-only 稀疏 CIR 更新
    -> 由 Adapt Pilot 产生 PEFT 梯度候选
    -> Reward Pilot 选择候选或 skip
    -> 保留/回滚参数
```

`pilot_sparse_cir_update()` 只使用当前帧接收信号、Adapt Pilot 符号和上一帧 soft tail。它不读取 Data 标签、Reward Pilot 标签、真实 CIR 或真实 CFO。实现先使用 acquisition CFO 先验和当前 Adapt Pilot 的公共相位估计去除慢旋转，再在 acquisition 主径 support 上估计复 tap 增益。Reward Pilot 只承担候选选择和回滚，不参与梯度更新。

神经 PEFT 更新默认使用 `adapter_lora + conditioner_film`。当前正式比较通过 `compare.py --cir-update pilot_sparse` 启用共享的 pilot-only CIR 更新；传统 DFE-RLS 同样使用该模式，保证状态估计信息边界一致。

## 证据一：连续漂移恢复

实验目录：

```text
logs/eme_aged_pilotsparse_30f_2s/
```

配置为 Level B、`delay=116`、`SNR=10 dB`、2 个 seed、30 帧、prefix 128 Pilot、held-out edge 状态、`pilot_sparse` CIR 更新。逐帧 Data BER 汇总如下：

| 方法 | 前 5 帧累计 | 前 10 帧累计 | 前 20 帧累计 | 前 30 帧累计 |
|---|---:|---:|---:|---:|
| Frozen Offline NN | 0.04743 | 0.04715 | 0.04676 | 0.09604 |
| Pilot-Driven Online Adaptation | 0.05011 | 0.04766 | **0.04556** | **0.06174** |
| CFO+DD-Phase DFE-RLS | 0.22969 | 0.21847 | 0.22109 | 0.21804 |

第 30 帧单帧 BER 为：Frozen `0.27344`，Online `0.08594`，传统 DFE-RLS `0.15179`。这说明在线优势主要来自持续跟踪慢 tap 漂移，而不是首帧瞬时增益。在线适配接受次数为 30 帧中的 20 次。

## 证据二：主 SNR 切片

实验目录：

```text
logs/eme_main_aged_15f_1s/
```

该实验覆盖主 SNR `0/5/10/15 dB`，每个配置 1 个 seed、15 帧。累计平均 BER 如下：

| SNR | Frozen Offline NN | Pilot-Driven Online Adaptation | CFO+DD-Phase DFE-RLS |
|---:|---:|---:|---:|
| 0 dB | 0.22336 | 0.29874 | 0.45781 |
| 5 dB | 0.07046 | 0.07522 | 0.22314 |
| 10 dB | 0.02679 | 0.02865 | 0.22098 |
| 15 dB | 0.01488 | 0.01577 | 0.17582 |

该切片用于说明 SNR 依赖性，不作为最终显著性结论。低 SNR 下 32 符号 Reward Pilot 的方差较大，Pilot 梯度更新容易被拒绝或选错；中高 SNR 下应重点报告随帧数变化的恢复曲线。

## 当前结论与未完成项

已经解决的主要问题：

1. acquisition 与数据状态之间具有明确、可审计的信道老化来源；
2. 老化仍由 120 秒相干时间推导，没有引入快速时变；
3. Pilot-only 稀疏 CIR 更新解决了 residual CFO/慢相位对 tap LS 的污染；
4. 在线方法在连续 20/30 帧上超过 Frozen Offline NN，并明显超过传统 DFE-RLS；
5. 在线更新的 Data 标签隔离和 Reward Pilot 回滚边界已经写入逐帧日志。

仍需在论文正式结果前完成：

- 主矩阵至少 60 帧、多个 seed 的重复实验；
- 0/5/10/15 dB 下分别报告前 1、5、10、20、30、60 帧窗口；
- 对 `fixed CIR`、`pilot_sparse CIR`、`decision-directed CIR` 做消融；
- 对 `Frozen NN`、`Pilot-only CIR`、`Pilot PEFT`、`Pilot CIR + PEFT` 做因果拆分；
- 检验 Reward Pilot 分块一致性门控是否能降低低 SNR 误更新；
- RL/Contextual Bandit 只作为动作调度消融，不能替代上述在线恢复主线。

因此当前阶段不能声称所有 SNR、所有首帧都超过 Frozen；可以严谨声称：在校准老化后的 Level B 长回波连续帧场景中，Pilot-driven online recovery 已显示出随时间累积的优势，且已超过公平传统非神经基线。
