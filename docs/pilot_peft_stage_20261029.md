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

复核 OOF 诊断的命令：

```powershell
python scripts/evaluate_pilot_reliability_oof.py `
  --input logs/pilot_rank_current_peft_gap600_3s3snr4f_seed60_62_20261029.json `
  --output logs/pilot_rank_current_peft_gap600_3s3snr4f_seed60_62_oof_20261029.json
```
