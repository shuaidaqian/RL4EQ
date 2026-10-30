# RL-PEFT Level B 正式方法比较（2026-10-30）

## 结论先说

在 Level B `heldout_edge` 信道上，离线神经均衡器整体明显好于本轮配置的传统均衡器；加上在线 Pilot 驱动 PPO/PEFT 后，平均 BER 在四个 SNR 都略有下降。但相对冻结离线网络的增量只有 **0.005–0.188 个百分点**，5 个 seed 的配对区间都跨零，所以“在线参数微调能稳定进一步提升”仍未被证明。

这轮能确认的是：在线链路确实执行了 Adapter 参数更新，且 Data 波形和标签没有参与更新；这轮单独看还不能把增益归因于 PEFT。后续 identity-only 对照已经补上这个缺口，结果见[因果对照记录](rl_peft_identity_control_20261030.md)。另一个问题是传统 baseline 中 SC-FDE 的实现与当前线性信道不匹配，不能用它的近随机 BER 来宣称 Proposed 的优势。

## 实验与完整性

使用固定离线 checkpoint（SHA256 `8CD3912D…AEC4`），Level B、延迟 116、`cfo_phase_tiny`、`heldout_edge`、0/5/10/15 dB；每个 SNR 5 个 seed（1091–1095），每个 seed 60 帧。Pilot 总长 256，其中 Adapt 224、Reward 32。策略学习率 `1e-3`、窗口 8 帧；状态 embedding 关闭，CIR 更新固定。

结果标记为正式 Level B 主配置，配置允许主结果聚合。日志有 6000 行，五种方法各 1200 行，1200 个信道帧都包含五种方法的完整记录，没有重复记录。在线方法的 1200 条记录全部标记 `data_labels_used_online=false`、`data_waveform_used_online=false`，动作 observation 全部来自 Adapt Pilot。

## BER 结果

改善值按“对照 BER − Proposed BER”计算，正值代表 Proposed 更好；区间用 5 个 seed 的配对 t 区间计算。

| SNR | Frozen 离线 NN | 在线 Proposed | 相对 Frozen 改善（95% 区间） | 本轮较好的 Pilot 传统 baseline | 相对它的改善 |
|---:|---:|---:|---:|---:|---:|
| 0 dB | 20.495% | 20.317% | +0.178 pp [−0.471, +0.826] | DFE-RLS 43.247% | +22.929 pp，5/5 seed 更好 |
| 5 dB | 5.203% | 5.016% | +0.188 pp [−0.249, +0.624] | DFE-RLS 24.368% | +19.352 pp，5/5 seed 更好 |
| 10 dB | 0.765% | 0.730% | +0.035 pp [−0.015, +0.085] | LMMSE-FIR 9.934% | +9.204 pp，5/5 seed 更好 |
| 15 dB | 0.172% | 0.167% | +0.005 pp [−0.003, +0.012] | DFE-RLS 6.597% | +6.430 pp，5/5 seed 更好 |

相对 Frozen 的逐 seed 正收益分别为 4/5、4/5、5/5、4/5。优势没有在所有 SNR 上随在线帧数持续扩大：10 dB 的后 20 帧改善增加，但 0 dB 后 20 帧优势变小，5/15 dB 也有起伏。整体目标“在线微调稳定明显超过离线”尚未达到。

## 传统 baseline 的限制

- `SC-FDE-MMSE` 在四个 SNR 都约为 49.76%–49.93% BER，接近随机判决。代码直接把整帧做 FFT 并按循环信道均衡；当前环境使用线性卷积，帧结构没有循环前缀。因此本轮将 SC-FDE 视为不匹配的诊断项，不作为有效 baseline 优势证据。
- 两种 Pilot 相位补偿 baseline（`CFO+DD-Phase LMMSE-FIR`、`CFO+DD-Phase DFE-RLS`）有实际信号处理，不是随机输出；但实现将前向 FIR 限制为 41 taps，DFE 反馈仅 5 taps，而信道最大延迟为 116。需再验证它们的滤波长度和参数是否足以代表长记忆信道，不能只凭这次结果写成强对比结论。
- 传统 baseline 结果说明当前神经离线模型在这个压力信道上有明显系统优势；这不是在线 PEFT 的独立收益证据。

## 在线动作做了什么

1200 帧中，策略选择 moderate PEFT 528 帧、identity 608 帧、rollback identity 64 帧。moderate 更新有 273 帧通过当前帧 Reward Pilot 守门，255 帧被拒绝；窗口级回滚 29 次。也就是说，参数更新路径确实工作，但最终数据段收益很小，说明下一个关键问题是验证“PEFT 更新相对相同 Adapt-only 接收状态的无更新版本”到底有没有因果增益。

## 下一步

下一步是在更强状态失配 `cfo_phase_light` 下复跑完整 PEFT 与 identity-only 配对，以检验收益能否随失配增大而变得更明显；同时继续校准长记忆传统 baseline 的滤波器容量。

机器可读汇总见 [`docs/results/rl_peft_main_comparison_l1e3_20261030.json`](results/rl_peft_main_comparison_l1e3_20261030.json)，逐帧日志位于 `logs/rl_peft_main_l1e3_gap0_heldout_edge_0_5_10_15db_5s60f_seed1091_1095_20261030/`。
