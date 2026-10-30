# 强 CFO/相位失配下的 Pilot-only PEFT 复核（2026-10-30）

## 当前结论

离线 checkpoint、Level B 信道和 Pilot 总长度保持固定后，三种 Pilot-only 候选排序都无法可靠预测后续 Data BER；在 `cfo_phase_mid` 下做 5 seed × 4 SNR × 8 帧的持续 PEFT 更新，也没有证明稳定超过冻结模型。少数 seed 的收益明显，但 seed 间差异大，四个 SNR 的 95% 配对区间都跨过零。**目前不应进入正式 5 seed × 60 帧矩阵，也没有理由接入 Bandit。**

## 固定条件与数据隔离

- Level B `eme_long_memory_v2`，116 符号最大时延，前缀 Pilot，总长度 256（224 Adapt Pilot + 32 Reward Pilot），acquisition gap=0。
- 两批实验都使用 `heldout_edge`、SNR 0/5/10/15 dB；候选排序批次使用 `cfo_phase_light`（CFO 绝对值和 phase-noise 标准差范围 0.001–0.003），持续 PEFT 批次使用更强的 `cfo_phase_mid`（范围 0.003–0.008）。
- 两批实验使用相同离线 checkpoint，SHA256 为 `8CD3912D1C957C9689C51730AFF9D27A8F74942186C802A7483E5781AF65AEC4`；未重训、未修改。
- Adapt Pilot 负责更新/候选排序，Reward Pilot 负责接受或拒绝；Data 波形和标签仅在所有在线决定完成后用于仿真评估。160 帧 replay 的每条记录均标记 `data_waveform_used_for_update_or_decision=false`、`data_labels_used_for_update_or_decision=false`。

## 先前 replay 的协议修正

最初几组候选排序 replay 虽加载了失配 profile，但辅助入口没有把 `heldout_edge` 传给信道构造器，因此 heldout 的强路径数和弥散能量范围没有生效。那批首轮结果不作为 heldout_edge 证据。状态划分传递已经修正。对本报告涉及的两批旧日志，我按相同配置和 seed 重建了各 20 个信道状态，确认实际路径数为 22–24、弥散能量占比为 0.291–0.341；候选排序批次实际 `|CFO|` 为 0.00124–0.00248、phase-noise 标准差为 0.00109–0.00245，持续 PEFT 批次分别为 0.00360–0.00670 和 0.00322–0.00662，均落在各自配置范围内。为避免之后只能依赖重建，replay 日志现已增加逐条 `channel_state_metadata`；相关 25 项协议与排序测试通过。以下候选排序数字均来自修正后的 replay。

## 修正后的 Pilot-only 候选排序

每种方法为 5 seed × 4 SNR × 3 帧，共 60 个 trial、240 个候选。每个 trial 内比较候选分数排序与事后 Data BER 排序；预先设定的 Spearman 通过门槛为 0.6。

| 候选与 Pilot 排序分数 | Spearman 与 Data BER | Reward 接受 | 平均提交 BER 改善 | 获得正 BER 改善的 trial |
|---|---:|---:|---:|---:|
| 多尺度 `phase_trend` 候选 + Adapt BCE 排序 | 0.282 | 14/60 | 0.165 pp | 16.7% |
| 多尺度 `phase_trend` 候选 + Adapt 信号重构排序 | 0.296 | 22/60 | 0.169 pp | 21.7% |
| BCE/信号/Head 候选 + Adapt BCE 排序 | 0.208 | 24/60 | 0.158 pp | 18.3% |

三种方法都远低于 0.6 门槛。正向平均值主要来自 0 dB；在 10/15 dB，提交后的 Data BER 大多持平或只有极小变化。Reward 接受只能说明留出的 Reward Pilot 达到门槛，不能保证 Data BER 改善。

随后把独立 Adapt 排序视图从 56 个符号扩大到 112 个，其他条件、候选、trial 和实际 Data 输出保持配对不变；两种排序方案各 60 个 trial、240 个候选。候选与 Data 指标在两次运行中逐项一致，只有 Pilot 排序分数改变：

| Pilot 排序分数 | 排序视图 | Spearman 与 Data BER | Spearman 与 Data BCE | Reward 接受 | 正 BER 改善 trial |
|---|---:|---:|---:|---:|---:|
| Adapt BCE | 56 | 0.282 | 0.140 | 23.3% | 16.7% |
| Adapt BCE | 112 | 0.237 | 0.343 | 35.0% | 18.3% |
| 信号重构 | 56 | 0.296 | 0.343 | 36.7% | 21.7% |
| 信号重构 | 112 | 0.415 | 0.467 | 40.0% | 21.7% |

增加排序样本对信号重构分数有帮助，但仍未到 0.6 门槛；对 BCE 排序，Data BER 相关性反而下降。正向收益 trial 比例几乎没变，所以**仅仅把同一帧内用于排序的 Adapt Pilot 加长，不足以解决稳定迁移问题**。扩大窗口的配对实验也记录了逐 episode 的实际信道状态。

## `cfo_phase_mid` 持续 PEFT replay

固定使用 `phase_trend` PEFT 参数，Adapt Pilot 上以信号重构损失做 5 步更新，学习率 0.0025；Reward Pilot 验收，单次及累计参数变化上限均为 0.05。每个 seed/SNR 连续 8 帧，共 160 帧。验收通过的 PEFT 状态跨帧保留。

主统计单位是每个 seed/SNR 轨迹的**最后一帧累计 BER 改善**，正值表示优于同轨迹冻结 checkpoint。每个 SNR 对 5 个 seed 求均值和 95% t 区间：

| SNR | 平均 BER 改善 | 95% 区间 | 正收益 seed |
|---:|---:|---:|---:|
| 0 dB | +1.46 pp | [-3.31, +6.22] pp | 2/5 |
| 5 dB | +1.17 pp | [-1.15, +3.50] pp | 2/5 |
| 10 dB | +1.22 pp | [-0.70, +3.15] pp | 3/5 |
| 15 dB | +0.78 pp | [-1.05, +2.61] pp | 2/5 |

先按 seed 平均四个 SNR 后，平均改善为 +1.16 pp，95% 区间为 [-1.52, +3.84] pp；只有 3/5 个 seed 为正。seed 1092 在所有 SNR 都有较明显收益，其余 seed 多数接近零或有轻微回退，说明整体正均值被单个 seed 拉高。全程 160 次中有 55 次通过 Reward 接受门槛，Adapt 排序门槛通过 102 次；接受率本身没有转化为跨 seed 的稳定 Data BER 改善。

## 跨帧累计 Adapt 更新复核

为检验多帧 Pilot 是否能弥补单帧排序不稳，在 `cfo_phase_mid`、gap=0 下，将第 0 帧起连续 1/3/5 帧的 Adapt Pilot 信号重构损失累计用于更新 `phase_trend` 内的 state-mapping 参数（14 个参数，lr=0.0025，5 步）。三个候选共用第 5、6 帧的独立 Reward Pilot 窗口做验收，再在第 7、8 帧测 Data；因此 1/3/5 帧比较使用相同未来验收与测试帧。共 5 seeds × 4 SNR，每种累计长度 20 条轨迹、40 个 Data 帧。

| Adapt 累计帧数 | Reward 接受轨迹 | 0 dB BER 改善 | 5 dB BER 改善 | 10 dB BER 改善 | 15 dB BER 改善 |
|---:|---:|---:|---:|---:|---:|
| 1 | 16/20 | -0.82 pp（0/5 seed 为正） | +2.16 pp（5/5） | +1.12 pp（5/5） | +0.46 pp（3/5） |
| 3 | 14/20 | -0.36 pp（0/5） | +0.57 pp（2/5） | +1.20 pp（5/5） | +0.48 pp（4/5） |
| 5 | 16/20 | -1.35 pp（0/5） | +0.56 pp（4/5） | +1.13 pp（5/5） | +0.63 pp（4/5） |

每个 SNR、每种累计长度按 5 个 seed 计算的 95% 配对区间都跨过零。单帧 Adapt 在 5/10 dB 的方向较一致，但在 0 dB 变差；累积到 3/5 帧没有带来更大、更稳的收益。更重要的是，在 0 dB 有害候选仍可能通过信号重构 Reward。由此看，当前卡点从“多看几帧”转到 **Reward Pilot 的验收分数能否识别有害 PEFT 更新**；还没有满足四个 SNR 都可靠改善的目标。

## Reward Pilot 验收分数对照

为定位低 SNR 的有害更新，同一单帧 Adapt 信号重构更新分别用未来两帧 Reward Pilot 的信号重构误差、BCE、相对联合分数验收。三组使用相同 checkpoint、seed/SNR 轨迹、相同 Adapt 更新与相同 Data 测试帧（第 3、4 帧）；候选更新前的 Data 输出逐项完全一致，只有 Reward 验收分数不同。

| Reward 验收分数 | 接受轨迹 | 0 dB BER 改善 | 5 dB BER 改善 | 10 dB BER 改善 | 15 dB BER 改善 |
|---|---:|---:|---:|---:|---:|
| 信号重构 | 14/20 | -0.51 pp（1/5 seed 正） | +0.59 pp（2/5） | +0.82 pp（5/5） | +0.40 pp（3/5） |
| BCE | 7/20 | -0.51 pp（1/5） | +0.60 pp（2/5） | +0.16 pp（2/5） | +0.04 pp（1/5） |
| 相对联合分数 | 11/20 | -0.51 pp（1/5） | +0.59 pp（2/5） | +0.66 pp（3/5） | +0.36 pp（2/5） |

所有 SNR 的 seed 级 95% 区间仍跨零。三种门控在 0 dB 恰好接受相同轨迹：其中包含一个约 `-6.25 pp` 的有害 seed，说明换 Reward 分数没有识别出这个问题。单独 BCE 门控明显减少了接受次数，但也把 10/15 dB 的正收益大幅压低；联合分数没有解决 0 dB 风险。**因此，当前失败不能靠把 Reward 阈值收紧或简单混合分数来修复。**

## Adapt 更新目标对照

保持 BCE Reward 门控、checkpoint、信道、seed、Adapt 更新位置和未来测试帧不变，只将 Adapt 更新损失从接收波形信号重构改为已知 Pilot 比特的 BCE：

| Adapt 更新损失 | Reward 接受轨迹 | 0 dB BER 改善 | 5 dB BER 改善 | 10 dB BER 改善 | 15 dB BER 改善 |
|---|---:|---:|---:|---:|---:|
| 信号重构 | 7/20 | -0.51 pp（1/5 seed 正） | +0.60 pp（2/5） | +0.16 pp（2/5） | +0.04 pp（1/5） |
| Adapt BCE | 9/20 | -0.49 pp（1/5） | +0.64 pp（2/5） | -0.72 pp（1/5） | -0.10 pp（0/5） |

每个 SNR 的 95% 区间仍跨零。Pilot BCE 更新没有修复 0 dB；在 10/15 dB 反而更差，其中 10 dB 的候选 Data BER 改善均值在 Reward 决策前已是负值。说明“用 Pilot 的已知比特做梯度”本身也不保证更新方向能泛化到 Data。

## PEFT 参数位置对照

最后保持 Adapt 信号重构目标、Reward BCE 门控、seed/SNR 和未来帧不变，把更新范围从 14 个 state-mapping Adapter 参数缩小到 2 个直接 phase/CFO Adapter 参数。两者使用相同轨迹：

| PEFT 更新范围 | Reward 接受轨迹 | 0 dB BER 改善 | 5 dB BER 改善 | 10 dB BER 改善 | 15 dB BER 改善 |
|---|---:|---:|---:|---:|---:|
| 14 个 state-mapping 参数 | 7/20 | -0.51 pp（1/5 seed 正） | +0.60 pp（2/5） | +0.16 pp（2/5） | +0.04 pp（1/5） |
| 2 个 phase/CFO 参数 | 5/20 | -0.39 pp（0/5） | +0.10 pp（1/5） | +0.08 pp（2/5） | +0.03 pp（1/5） |

所有区间仍跨零。直接 phase/CFO 参数确实发生了更新，但候选 Data 收益整体更小，没能优于 state-mapping 更新。下一轮测试把两种 Adapter 参数放开一起更新，检查表达能力是否不足；如果组合更新仍没有稳定收益，就暂停继续盲目增加可训练参数。

## 对研究路线的影响

已确认的进展是：在线更新作用于真实 `phase_trend` PEFT 参数；通过 Reward 的更新会保留并跨帧影响模型；在线过程不使用 Data；正确的 heldout 强失配已经覆盖。尚未解决的核心问题是：**Pilot 上哪个候选更好，仍不能可靠代表后续 Data 均衡更好；而持续更新的 BER 改善也未在多个 seed 上稳定出现。**

因此暂不扩大正式矩阵、不改变离线训练、不开放 Data 参与更新，也暂不引入 Bandit。下一步固定 Adapt 信号重构目标和 Reward BCE 门控，把 14 个 state-mapping 与 2 个 phase/CFO Adapter 参数一起更新，观察组合 PEFT 更新是否稳定提高 Data BER；若不通过多 seed/SNR 门槛，就停止扩大当前候选范围，回到数据/信道可辨识性问题。

机器可读汇总见 [`docs/results/rl_peft_pilot_corrected_replay_20261030.json`](results/rl_peft_pilot_corrected_replay_20261030.json)。逐帧原始日志保存在被 Git 忽略的 `logs/` 目录。

## 验证记录

为审计 replay 日志，本阶段加入 `channel_state_metadata` 字段，并新增测试核对日志与实际信道状态一致。完整测试套件通过：633 passed；相关协议与排序测试：25 passed。结果 JSON 可解析，`git diff --check` 通过。
