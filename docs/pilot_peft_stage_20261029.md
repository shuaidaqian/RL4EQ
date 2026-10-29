# Pilot-only PEFT 排序复核（2026-10-29）

## 本阶段结论

固定离线 checkpoint 后，Adapt Pilot 可以驱动 PEFT 参数变化，某些候选也会明显改变 Data logits；但当前 Pilot-only Reward 还不能稳定判断这些变化会不会改善 Data BER。跨 seed 留出排序没有达到预设门槛，因此本阶段不启动正式 `5 seed × 60 帧` 主矩阵，也不能声称在线微调已经稳定优于冻结离线模型。

本轮工作属于 Pilot/PEFT 诊断，不替换仓库规定的 Level B 离散安全动作 PPO 主路线。`gap=600 s` 是 acquisition 老化压力诊断，不纳入主实验平均。

## 固定条件和数据隔离

- 信道：Level B `eme_long_memory_v2`，最大时延 116 符号；前缀 Pilot 总长 256（Adapt 224、Reward 32）。
- SNR：5、10、15 dB；acquisition gap：600 秒。
- 所有回放复用同一离线 checkpoint：`pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`，SHA256 为 `8CD3912D1C957C9689C51730AFF9D27A8F74942186C802A7483E5781AF65AEC4`。没有重新训练或改写 checkpoint。
- Adapt 更新和状态特征只看 Adapt Pilot；Reward 验收前向只看 Pilot 波形；Reward 的验收/回滚完成后才读取完整帧计算 Data 指标。Data 波形和标签均未用于在线更新、状态更新、Reward 决策或策略选择。

代码审计发现，累计 replay 先前没有在所有前向路径屏蔽 Data 波形，且前缀 Pilot 未覆盖的尾状态会被模型输出覆盖。本轮修复了这些路径，并新增“污染 Data 波形后 Pilot 更新与验收输出仍不变”的回归测试。事后 Data 指标仍然使用 Data 标签，这是仿真评估用途，不能作为在线能力证据。

## 实验结果

| 诊断 | 结果 | 判断 |
|---|---|---|
| Adapt 目标与累计长度 | BCE、信号重构、联合目标；1/3/5 帧历史，共 81 个候选、27 个配对排序组 | 没有稳定赢家 |
| Pilot Reward 对 Data BCE 的配对方向一致率 | 27/62 = 43.5%（并列不计） | 低于随机排序水平 |
| Pilot Reward 对 Data BER 的配对方向一致率 | 15/31 = 48.4%（并列不计） | 接近随机，不能可靠选候选 |
| 不同 Reward 评分作用于同一候选 | 候选平均 Data BCE 改善 0.00176，但平均 Data BER 改善为 −0.0164 个百分点；Data logit 平均绝对变化约 0.0997 | 参数更新确实影响 Data 输出，问题在 Pilot 指标与 Data BER 的迁移/排序 |
| 聚合信号重构 Reward | 平均提交 BER 改善 +0.0125 个百分点，但仅 29.6% 试验有正 BER 改善；约 42.9% 被接受候选事后 BER 仍变差 | 相对较好，但不稳定 |
| 逐帧严格信号重构门控 | 正 BER 改善试验 11.1%，平均提交 BER 改善 −0.0068 个百分点 | 更严格门控没有解决问题 |
| 当前 PEFT 候选留一 seed OOF | 36 个试验、216 个候选；Pilot Reward 与 Data BER Spearman 0.112，去除 identity 后 0.118 | 远低于 0.6 通过门槛 |
| Pilot 特征可靠性模型留 seed OOF | 与 Data BER Spearman −0.040；与 Data BCE Spearman 0.340 | 能解释部分连续损失变化，但不能可靠预测 BER 获益 |
| OOF 选择策略的正 BER 收益比例 | 19.4% 试验 | 未显示跨 seed 可用性 |

完整的精简机器可读汇总见 [`pilot_peft_stage_20261029.json`](results/pilot_peft_stage_20261029.json)。原始逐帧 GPU 日志留在本机 `logs/`（该目录由 `.gitignore` 排除）；版本库不包含这些大型原始文件。

## 代码和验证

- `scripts/replay_cumulative_state_mapping.py`：分离 Adapt、Pilot 验收和事后 Data 前向；屏蔽 Data 波形，避免未观测尾状态被更新。
- `scripts/evaluate_pilot_reliability_oof.py`：拒绝任何声明在线使用 Data 标签或 Data 波形的输入日志。
- 增加协议隔离、OOF 隔离和汇总脚本测试。
- 定向测试：30 项通过。全量 `pytest` 结果在提交前记录。

当前 worktree 没有 `.venv-gpu`，本轮使用系统 Anaconda Python 执行测试；GPU replay 日志是本轮已有的运行结果，没有在此环境重新跑 GPU 实验。

## 下一步

1. 暂停扩大 seed、帧数和 SNR 矩阵，也不接入 bandit。
2. 保留能确实改变 Data 连续输出的相位趋势 Adapter/head 候选；剔除当前步长下几乎不改变 Data logits 的普通 Adapter/attention LoRA 候选。
3. 只在 Pilot 内重做候选筛选：用互不重叠的 Adapt 子窗口生成候选、另一段 Adapt Pilot 作筛选，再由独立 Reward Pilot 验收；全过程继续屏蔽 Data 波形和标签。
4. 先在留出 seed 上达到预设 Spearman ≥ 0.6，且各 seed 的 Data BER 收益没有系统性退化，再进入正式矩阵。达不到时，应调整 PEFT 调制位置或 Pilot 可观测性，而不是继续扩大样本掩盖排序失败。

## 连续帧参数持久化复核（2026-10-29）

为确认被 Reward Pilot 接受的相位趋势 Adapter 是否会在后续帧保留并继续产生收益，固定 checkpoint 和 `gap=600 s`，运行 3 seeds × 3 SNR × 12 帧连续 replay。每条 seed/SNR 轨迹都从同一冻结模型开始；Reward 接受后把候选参数带入下一帧，拒绝则回到上一帧参数。Adapt、独立排序窗和 Reward Pilot 仍按原协议隔离，Data 指标只在该帧所有在线决定结束后计算。

精简机器可读结果见 [`pilot_persistent_peft_gap600_3s3snr12f_20261029.json`](results/pilot_persistent_peft_gap600_3s3snr12f_20261029.json)；原始逐帧 replay 保存在被 `.gitignore` 排除的本机 `logs/` 目录。

固定参数为 Level B `eme_long_memory_v2`、delay `116`、prefix Pilot `256`（Reward `32`）、SNR `5/10/15 dB`、seed `1051/1052/1053`、每轨迹 `12` 帧；更新对象为 `phase_trend`，目标为 Pilot 信号重构，学习率 `0.0025`、每帧 `5` 步、单次参数增量上限 `0.05`。checkpoint SHA256 仍为 `8CD3912D1C957C9689C51730AFF9D27A8F74942186C802A7483E5781AF65AEC4`，未重新训练或修改。

| 指标 | 结果 | 解读 |
|---|---:|---|
| 逐帧 Data BER 平均改善 | `+0.3581 pp` | 以当前累计 Adapter 状态对比冻结模型，在每帧 Data 上测量后再跨帧平均；不是把 12 帧拼接后计算的单一 BER |
| seed×SNR 轨迹均值为正 | `7/9` | 5 dB 为 `3/3`，10 dB 为 `2/3`，15 dB 为 `2/3`；仍有两条轨迹平均退化 |
| 最后一帧 BER 改善为正 | `6/9` | 另有一条为零、两条为负，无法证明每条轨迹都稳定改善 |
| Reward 接受 / 独立排序通过 | `68/108` / `90/108` | 在线门控有实际更新，但更新接受并不保证每条轨迹最终受益 |
| 连续输出变化 | Data BCE 平均改善 `0.0581`；logit 平均绝对变化 `0.846`；软输出变化 `0.0210`；方向翻转比例 `2.19%` | PEFT 确实影响 Data 输出，不是梯度路径失效 |
| 前 6 帧 / 后 6 帧平均 BER 改善 | `+0.3159 pp` / `+0.4003 pp` | 后半段略高，但逐帧收益上下波动，不能说优势稳定随帧数扩大 |

仅在 68 个已接受更新上做事后排序检查时，Reward Pilot 信号重构改善与该次更新实际 Data BER 改善的 Spearman `ρ=0.591`。按 seed 分别为 `0.696/0.488/0.563`，按 SNR 分别为 `0.627/0.486/0.581`。这是重复帧上的探索性相关，不是留出 seed OOF 结果；低于预设的跨 seed 稳定性要求，不能据此接入 bandit 或宣称 Pilot reward 已可靠预测 Data 收益。

**阶段结论：**参数持久化后总体平均收益为正，且更新会明显改变 Data 连续输出；但收益幅度小、不同 seed/SNR 仍有退化，逐帧收益不单调，Reward 排序也没有跨 seed 稳定。因而在线 PEFT 仍未达到“稳定明确地超过冻结离线模型”的门槛。本矩阵是 gap 老化诊断，不替代 gap=0 主场景，也不计入正式主平均。

下一步继续在 Pilot-only replay 中改进多帧排序可靠性：固定 checkpoint、信道、pilot 总长和 seed 划分，比较当前单帧 Reward 与跨帧累计 Reward/状态一致性分数；只有留出 seed 的排序达到既定门槛、且各 SNR 不出现系统性 BER 退化，才进入正式矩阵。仍不使用 Data 波形或标签参与在线更新、动作选择或门控。

## 多帧 Reward 一致性门控复核（2026-10-30）

在相同 checkpoint、seed、SNR 和 12 帧信道轨迹上，将单帧 Reward 门控与最近 3 帧 Reward Pilot 一致性门控对照。多帧候选必须满足：最近至多 3 帧的平均 Reward BCE 和信号重构分数都改善；至少 `2/3` 的窗口信号误差改善且 Reward Pilot 硬错数不增加；当前帧硬错数也不能增加。历史视图只保留对应 Pilot 的波形/已知 Pilot 符号，Data 波形、Data 标签、真 CIR 和真实 tail 在 Pilot 视图中均被清零或移除。

| 门控 | 平均逐帧 BER 改善 | 接受更新 | 正收益 seed×SNR 轨迹 | 5 dB | 10 dB | 15 dB |
|---|---:|---:|---:|---:|---:|---:|
| 单帧 Reward | `+0.3581 pp` | `68/108` | `7/9` | `+0.3798 pp` | `+0.4087 pp` | `+0.2857 pp` |
| 最近 3 帧一致性 | `+0.1013 pp` | `44/108` | `6/9` | `+0.0253 pp` | `+0.3798 pp` | `−0.1013 pp` |

多帧门控更谨慎，但没有减少 Data BER 退化帧（单帧 `38/108`，多帧 `42/108`），也没有让收益跨 seed 更稳定；15 dB 的平均收益转为负值。由此淘汰“最近 3 帧均值 + 多数一致性”作为当前主门控，后续不再按这组 seed 调阈值。精简对照结果见 [`pilot_reward_history3_comparison_20261030.json`](results/pilot_reward_history3_comparison_20261030.json)，完整逐帧日志仍在本机 `logs/`。

此外，Pilot 视图现在会把不可见位置的 `bits`、`tx_symbols` 清零，并清除仿真真 CIR 与真实 tail；这让数据隔离从“调用方承诺不读”变成输入对象本身不含这些值。新增回归测试覆盖该约束和多帧门控汇总。

下一步保留已较好的单帧 Reward 守门，改测直接作用于均衡器内部 CIR PEFT 的 `channel_residual` 参数组，继续固定离线 checkpoint；这样可以判断瓶颈是否来自相位趋势参数组表达能力不足，而不是继续调松多帧门槛。结果仍先作为 Pilot-only 诊断，未跨 seed 复现前不进入正式统计矩阵。

## CIR 残差 PEFT 与累计偏移上限（2026-10-30）

按上述计划，使用同一 Pilot 更新、单帧 Reward 门控，改为只训练 `channel_residual`。随后增加以冻结离线参数为中心、总 L2 偏移上限 `0.05` 的信任区域，对照原本仅限制“每帧单步增量不超过 `0.05`”的设置。两组均为 3 seeds × 3 SNR × 12 帧，固定 checkpoint 未变。

| 参数组 / 限制 | 平均逐帧 BER 改善 | Data BCE 平均改善 | 接受更新 | 有意义正收益轨迹（>0.01 pp） | 最大最终参数偏移 |
|---|---:|---:|---:|---:|---:|
| `phase_trend`，单帧门控（参照） | `+0.3581 pp` | `+0.0581` | `68/108` | `7/9` | `0.0885` |
| `channel_residual`，仅单步限幅 | `−0.0181 pp` | `−0.0067` | `38/108` | `3/9` | `0.1777` |
| `channel_residual`，总偏移限幅 `0.05` | `+0.0024 pp` | `−0.0025` | `40/108` | `4/9` | `0.0500` |

总偏移限制确实将参数偏移压在 `0.05` 内，也减少了 Data BER 退化帧（无总限幅 `38/108`，加总限幅 `29/108`），但平均收益仍接近零且 Data BCE 仍变差；15 dB 两个 CIR 残差版本的平均 BER 改善都为负。`channel_residual` 因此不保留为主更新对象。详细汇总见 [`pilot_persistent_channel_residual_gap600_3s3snr12f_20261030.json`](results/pilot_persistent_channel_residual_gap600_3s3snr12f_20261030.json) 和 [`pilot_persistent_channel_residual_trust_region_20261030.json`](results/pilot_persistent_channel_residual_trust_region_20261030.json)。

至此，直接 Pilot 梯度 PEFT 的两个核心候选（`phase_trend` 与 `channel_residual`）和多帧 Reward 门控都还没有达到跨 seed、跨 SNR 稳定收益门槛。继续扩大这些消融只会增加样本，不能解决候选更新排序不稳。下一步回到仓库规定的离散安全动作路线：固定 PEFT 参数组，由 PPO 用 Pilot 状态选择是否更新及更新强度，Reward Pilot 用窗口级结果验收；先做小门槛比较，未跨配置稳定改善前不启动正式大矩阵。

复核 OOF 诊断的命令：

```powershell
python scripts/evaluate_pilot_reliability_oof.py `
  --input logs/pilot_rank_current_peft_gap600_3s3snr4f_seed60_62_20261029.json `
  --output logs/pilot_rank_current_peft_gap600_3s3snr4f_seed60_62_oof_20261029.json
```
