# 强 CFO/相位失配下的 Pilot-only PEFT 候选排序（2026-10-30）

## 结论

我固定离线 checkpoint，在强 CFO/相位失配下比较了三种 Pilot-only 选法：多尺度更新配 Adapt BCE 排序、多尺度更新配 Adapt 信号重构排序，以及 `phase_trend/head` 不同 PEFT 目标配 Adapt BCE 排序。三种方法的候选分数对事后 Data BER 的排序相关性都只有约 0.40–0.42，低于预设 0.6 门槛；收益主要来自 0 dB，5–15 dB 没有稳定改善。因此这轮仍不能证明在线 PEFT 能稳定超过离线模型。

## 实验怎么做

- Level B `eme_long_memory_v2`，heldout_edge 强失配、gap=0；SNR 0/5/10/15 dB，5 个 seed，每个 seed/SNR 3 帧，共 60 个小样本 trial/排序方法。
- 每帧的 224 个 Adapt Pilot 拆成不重叠的更新子窗（168 符号）和排序子窗（56 符号）；Reward Pilot 32 符号单独负责验收。Reward BCE 必须改善超过 `0.0005`，且硬错数不能增加。
- 离线 checkpoint SHA256 为 `8CD3912D1C957C9689C51730AFF9D27A8F74942186C802A7483E5781AF65AEC4`，未训练或修改。Data 波形和标签没有进入在线更新、排序、Reward 或拒绝决策，只在决策结束后计算 BER/BCE。
- 两种多尺度排序方法使用完全相同的 240 个候选和逐候选 Data 事后指标；只有 Adapt Pilot 排序分数不同。

## 结果

| Pilot-only 方法 | 候选分数与 Data BER 的 Spearman | Reward 接受 | 平均事后 BER 改善 | BER 正收益 trial |
|---|---:|---:|---:|---:|
| 多尺度候选 + Adapt BCE 排序 | 0.412 | 16/60 | +0.119 pp | 13.3% |
| 多尺度候选 + Adapt 信号重构排序 | 0.400 | 22/60 | +0.128 pp | 15.0% |
| BCE/信号/Head 候选 + Adapt BCE 排序 | 0.421 | 21/60 | +0.119 pp | 16.7% |

按 SNR 看，平均 BER 改善（正值代表优于冻结 checkpoint）为：

| SNR | 多尺度 + Adapt BCE | 多尺度 + Adapt 信号重构 | BCE/信号/Head 候选 |
|---:|---:|---:|---:|
| 0 dB | +0.503 pp | +0.503 pp | +0.477 pp |
| 5 dB | -0.035 pp | +0.017 pp | -0.009 pp |
| 10 dB | +0.000 pp | -0.017 pp | -0.000 pp |
| 15 dB | +0.009 pp | +0.009 pp | +0.009 pp |

多尺度信号重构排序的 Data BER 相关性为 0.400，Adapt BCE 排序为 0.412；直接 BCE/信号/Head 候选的 Adapt BCE 排序为 0.421。三者都没有达到排序门槛。合并所有 SNR 的正平均值主要由 0 dB 贡献：5、10、15 dB 的改善接近零，且有些候选虽然通过 Reward Pilot，事后 Data BER 仍持平或轻微变差。

这说明目前卡点不是“参数没更新”：不同候选确实更新了 `phase_trend` 或 head，PEFT 参数 L2 变化随更新幅度增加；而是 **Pilot 子窗分数和 Reward Pilot 仍不能可靠判断哪一种参数变化会改善后续 Data BER**。高 SNR 下很多候选的 Data BER 完全持平，硬 BER 的离散性也会让小样本排序更难。

## 决策

这批结果是诊断筛选，不是正式性能矩阵；我不会据此挑出 Data 指标最好的候选，也不会把它部署为主方法。下一步应先回到信道失配强度：将 CFO/phase noise 从当前 `cfo_phase_light`（约 0.001–0.003）提升到已定义的 `cfo_phase_mid`（约 0.003–0.008），先做 identity-only 与固定 PEFT 的小型配对 replay，确认是否存在 PEFT 能明确解决、冻结模型无法处理的失配区间。若仍无明确差异，就该承认当前 checkpoint 在该设置下没有可转移的在线收益，再决定是否需要改变离线适应目标或论文问题定义；在此之前不做大矩阵。

机器可读结果见 [`docs/results/rl_peft_pilot_multiscale_ranking_strong_gap0_20261030.json`](results/rl_peft_pilot_multiscale_ranking_strong_gap0_20261030.json)。三组逐候选原始日志保存在本机 `logs/`（被 Git 忽略）。
