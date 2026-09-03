# EME 在线 Reward Pilot 切分与跨帧更新率诊断

## 目的

本阶段检验两个容易混淆的问题：

1. 固定总 prefix Pilot=128 时，把更多符号分给 Reward Pilot，是否能让 Pilot-only Reward 更可靠地判断在线更新；
2. `update_interval` 是否真正控制 PEFT/CIR 的跨帧更新率，而不是只停留在命令行参数层面。

实验不改变 Level B 信道、离线 checkpoint、总 Pilot 数量或 Data 评估方式。在线更新仍只用当前帧 Adapt Pilot 求梯度，Reward Pilot 只用于动作验收和回滚，Data 标签只用于离线统计。

## 固定协议

| 项目 | 设置 |
|---|---|
| 信道 | `eme_long_memory_v2`，Level B，`max_delay=116` |
| 帧长度 | 1024 符号 |
| 长回波 | 12--24 条强径，diffuse energy ratio 0.20--0.35 |
| 状态 | acquisition 后 30 s，coherence time 120 s |
| 扰动 | `cfo_phase_tiny` |
| SNR | 5/10/15 dB（切分诊断），0/5/10/15 dB（更新率主切片） |
| 重复 | 3 seeds，30 连续帧 |
| 方法 | `Pilot-conditioned frozen NN` 与 `Pilot-Driven Online Adaptation` |
| checkpoint | `pretrained/eme_meta_from_offline_32/model_best.pt` |
| CIR | `pilot_sparse`，alpha=0.2 |
| 总 Pilot | 128，prefix layout |

## Reward Pilot 切分结果

下表是每个 SNR、3 seeds、30 帧合并后的 Data BER。两个切分实验都采用每帧尝试 PEFT 的旧行为，用于隔离 Pilot 资源分配因素；因此不能把它们当作最终跨帧更新率协议。

| SNR | 切分 | Frozen NN | Online | Frozen-Online |
|---:|---|---:|---:|---:|
| 5 dB | Adapt 64 / Reward 64 | 21.45% | 22.23% | -0.78 pp |
| 10 dB | Adapt 64 / Reward 64 | 17.47% | 15.55% | +1.92 pp |
| 15 dB | Adapt 64 / Reward 64 | 15.18% | 16.66% | -1.48 pp |
| 5 dB | Adapt 80 / Reward 48 | 19.34% | 24.38% | -5.04 pp |
| 10 dB | Adapt 80 / Reward 48 | 17.11% | 17.24% | -0.13 pp |
| 15 dB | Adapt 80 / Reward 48 | 13.41% | 13.23% | +0.19 pp |

结果没有支持“Reward Pilot 越长，PEFT 更新越可靠”的假设。64/64 只在 10 dB 有正增量，80/48 在 5 dB 反而更差。主要原因不是总 Pilot 不足，而是单帧 Reward Pilot 的局部 BCE 改善不能稳定代表同一帧后续 Data 段的 BER 改善。增加 Reward 符号不能替代更合理的状态表示、更新候选或跨帧评价窗口。

## 跨帧更新率修正

此前 `compare.py` 接收了 `--update-interval`，但 `Pilot-Driven Online Adaptation` 没有实际使用它，导致即使命令指定 `--update-interval 8`，PEFT 仍会每帧尝试更新。这已修正为：

- 第 1 帧执行初始化更新；
- 后续只在第 `1 + k * update_interval` 帧执行 CIR/PEFT 更新；
- 非调度帧仍使用 Adapt Pilot 估计当前可见相位状态并推理；
- 非调度帧不执行 PEFT 梯度更新，也不生成 CIR 候选；
- 每帧写入 `online_update_scheduled`、`online_update_skipped` 和 `online_update_interval`。

2 帧端到端 smoke（`update_interval=2`）的审计结果为：

| 帧 | scheduled | skipped | PEFT/CIR 更新 |
|---:|---:|---:|---|
| 1 | true | false | 按安全门控尝试 |
| 2 | false | true | 不执行 |

## 修正后的 96/32 更新率切片

该矩阵使用 `reward_pilot_total=32`、`adapt_pilot_total=96`、`update_interval=8`，共 720 条逐帧记录。

| SNR | Frozen NN | Online | Frozen-Online | Online 较优帧比例 | PEFT 接受 |
|---:|---:|---:|---:|---:|---:|
| 0 dB | 31.20% | 30.56% | +0.64 pp | 45.6% | 0/90 |
| 5 dB | 22.69% | 21.54% | +1.16 pp | 40.0% | 0/90 |
| 10 dB | 15.22% | 15.83% | -0.60 pp | 40.0% | 4/90 |
| 15 dB | 12.54% | 12.93% | -0.39 pp | 36.7% | 8/90 |

每个 SNR 都准确记录了 12 个调度帧和 78 个跳过帧。该结果说明更新率参数已经真实生效，但当前在线 PEFT 的独立增益仍不稳定。0 dB 的更新按既有规则冻结；5 dB 以上仅少量 Reward 验收成功，15 dB 的物理状态平均置信度仍约为 0.143，尚不足以支持把 sparse CIR tracking 写成性能来源。

## 当前研究判断

可以写入阶段性研究记录：

- Level B 极端稀疏长回波下，当前帧 Pilot 条件化神经块均衡器是有效的在线可见状态恢复机制；
- 在线过程已经具备清晰的信息边界、Reward Pilot 安全验收和跨帧更新率控制；
- 固定总 Pilot 下的 Adapt/Reward 资源切分本身不能解决 Reward 与 Data 的代理失配；
- `update_interval` 的缺失是实现缺口，现已修正并完成端到端审计。

当前不能写入论文主结论：

- 不能声称现有 PEFT 在所有 SNR 或长期窗口稳定超过 Frozen NN；
- 不能声称 64/64 或 80/48 是最优 Pilot 切分；
- 不能声称当前物理状态估计已经可靠恢复 residual CFO 或稀疏 CIR；
- 不能因为 Online 相对传统 baseline 更好，就把 Pilot 条件化收益全部归因于 PEFT 微调。

## 后续路线

下一步应固定 96/32 与 `update_interval=8` 作为当前协议基线，先做 Pilot-only replay 的短窗口候选排序诊断：以多个连续帧的 Reward Pilot 累积改善、一致性和回滚率作为候选评价，而不是只看单帧 BCE。只有该离线诊断对后续 Data BER 的排序相关性稳定为正，才引入低维 Contextual Bandit 来选择更新强度和保持帧数；若仍不成立，应优先重构 phase/CIR 状态特征和安全候选，而不是增加 PPO、学习率或网络规模。

## 可复现实验目录

- `logs/eme_reward_split_64_64_30f_3s/`
- `logs/eme_reward_split_80_48_30f_3s/`
- `logs/eme_update_rate_main_30f_3s/`
- `logs/eme_update_interval_smoke/`
