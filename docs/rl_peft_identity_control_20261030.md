# RL-PEFT Identity-only 因果对照（2026-10-30）

## 最重要的发现

我补跑了一个“在线 Pilot 状态流程照常，但永远选 identity、不更新 PEFT 参数”的对照。它在四个 SNR、全部 5 个 seed 上与 `Frozen Offline NN` 的 BER **逐项完全相同**。因此，这套配置下在线方法相对离线模型的那一点收益，确实来自 PEFT 参数更新，不是另一个状态处理步骤带来的假收益。

但收益仍很小、不够稳定：四个 SNR 的配对区间都跨零。这证明 PEFT 路径有效地改变了均衡器输出，却还没有证明它能稳定改善数据段 BER。

## 配对结果

完整方法与 identity-only 对照使用同一 checkpoint、同一配置、相同 seed/frame 和信道状态元数据。两组都只让 Adapt Pilot 进入在线 observation；Data 波形和标签只用于动作完成后的仿真评估。

| SNR | Frozen / identity-only BER | 完整 PEFT BER | PEFT 平均增益 | 正收益 seed | 95% 配对区间 |
|---:|---:|---:|---:|---:|---:|
| 0 dB | 20.4952% | 20.3173% | +0.1780 pp | 4/5 | [−0.4706, +0.8265] pp |
| 5 dB | 5.2031% | 5.0156% | +0.1875 pp | 4/5 | [−0.2492, +0.6242] pp |
| 10 dB | 0.7652% | 0.7305% | +0.0347 pp | 5/5 | [−0.0151, +0.0846] pp |
| 15 dB | 0.1719% | 0.1671% | +0.0048 pp | 4/5 | [−0.0029, +0.0125] pp |

identity-only 对照共 1200 帧，1200 帧都选择 identity，PEFT 参数变化范数全部为 0。它和完整方法按 1200 个 seed/SNR/frame 键配对，方法内帧标识和信道状态元数据均一致。

## 参数确实改了输出，但 Pilot 改善没有稳定传到 Data

完整方法在 1200 帧里提议 PEFT 更新 528 次；Reward Pilot 当帧守门接受 273 次、拒绝 255 次，窗口回滚 29 次。被接受的更新在 Data 段上平均产生 `0.153` 的 logits 变化和 `0.00477` 的软输出变化。用 Data 做的事后评估显示，其中 63.4% 的更新改善 Data BCE；但硬 BER 只有 119 次变好、60 次变坏，另外 94 次不变。

迁移效果还随 SNR 改变：被接受更新的 Data BCE 正收益比例从 0 dB 的 87.4%，降到 5 dB 的 59.0%、10 dB 的 51.7%，在 15 dB 只有 46.3%。这些 Data 指标仅用于分析，没有参与策略 observation、PEFT 更新、Reward 或回滚。

所以现在卡点更具体了：不是“参数路径没有梯度”或“Adapter 完全不影响数据输出”，而是 **当前 PEFT 更新候选在更强状态失配之外的收益偏小，而且 Reward Pilot 排序到 Data 收益的能力随 SNR 变弱**。

## 下一步

按研究方向先把状态失配加大：固定同一离线 checkpoint 和 PEFT 参数设置，在 `cfo_phase_light` 下继续做完整 PEFT 与 identity-only 的配对 replay。若较强失配下仍没有可复现增益，再调整 Pilot-only 候选更新目标或 Reward 排序；不动离线训练，也不让 Data 进入在线更新。

机器可读结果见 [`docs/results/rl_peft_identity_control_20261030.json`](results/rl_peft_identity_control_20261030.json)，逐帧日志分别保存在 `logs/rl_peft_main_l1e3_gap0_heldout_edge_0_5_10_15db_5s60f_seed1091_1095_20261030/` 和 `logs/rl_peft_identity_control_gap0_heldout_edge_0_5_10_15db_5s60f_seed1091_1095_20261030/`。
