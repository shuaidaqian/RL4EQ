# PPO 调制中等步长 PEFT 的小样本复核（2026-10-30）

## 结论

本轮确认了 PPO 能实际触发 Pilot 驱动的 PEFT 更新，帧级 Reward 拒绝和窗口累计参数限幅也在工作。但在线 BER 没有稳定超过冻结离线 checkpoint：5 个 seed 中只有 1 个平均改善、2 个退化、2 个因始终选择 identity 而与冻结模型相同。因此这只是链路与动作候选诊断，不是在线方法达标，也不进入正式主结果。

## 固定协议

- Level B `eme_long_memory_v2`，最大时延 116，前缀 Pilot 总长 256（Adapt 224、Reward 32），SNR 5 dB，acquisition gap 600 秒。
- 5 个固定 seed：1061–1065；每个 seed 连续 24 帧，PPO 每 8 帧选择一次窗口动作，共 15 个决策窗口。
- 离线 checkpoint 固定为 `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`，SHA256：`8CD3912D1C957C9689C51730AFF9D27A8F74942186C802A7483E5781AF65AEC4`；未重新训练或改写。
- 动作表为 `identity`、`peft_phase_signal_moderate`、`rollback_identity`。中等步长候选只更新 `phase_trend` PEFT，使用 Adapt Pilot 信号重构目标，学习率 `0.0025`、5 步；Reward Pilot 按帧验收或恢复，窗口 reward 非正时恢复窗口开始时的 PEFT 参数。
- PEFT 参数相对窗口起点的累计 L2 偏移最多为 `0.05`。Data 波形和标签只在当帧在线决定结束后用于仿真评估；逐帧日志中 `data_waveform_used_online` 与 `data_labels_used_online` 均为 `false`。
- 该配置标记为 `diagnostic_only`，gap 600 秒是 acquisition 老化压力诊断，不代表 gap=0 主场景。

## 结果

冻结模型与 PPO 方法使用相同的 seed、帧号和由 seed 决定的信道轨迹，逐帧比较：

| Seed | Frozen BER | PPO-PEFT BER | BER 改善（百分点） | moderate 窗口数 |
|---:|---:|---:|---:|---:|
| 1061 | 30.2029% | 30.2029% | 0.0000 | 0/3 |
| 1062 | 36.7188% | 36.8218% | −0.1031 | 1/3 |
| 1063 | 18.6795% | 18.6795% | 0.0000 | 0/3 |
| 1064 | 27.1918% | 27.6150% | −0.4232 | 1/3 |
| 1065 | 41.5907% | 40.6521% | +0.9386 | 2/3 |
| **全部 120 帧** | **30.8767%** | **30.7943%** | **+0.0825** | **4/15** |

总体微小正值主要由 seed 1065 拉动，不能作为稳定收益。两条完全未选择 PEFT 的轨迹与 Frozen BER 完全相同；其余三条里有两条退化。32 个 moderate 帧都产生了非零参数变化，其中 22 帧通过 Reward、10 帧被拒绝；窗口级回滚为 0 次。最大窗口累计偏移是 `0.050000004`，与上限的微小差异来自浮点舍入。

更新确实改变了 Data 连续输出：moderate 帧的平均 Data logit 绝对变化为 `0.1307`，平均 Data BCE 改善为 `0.0200`。但连续损失改善没有变成跨 seed 稳定的硬 BER 改善，说明当前瓶颈仍在 Pilot Reward 与 Data BER 的对应关系，以及 PPO 在每条轨迹只有 3 次窗口决策时的动作选择稳定性。

机器可读摘要：[`docs/results/rl_peft_moderate_replay_20261030.json`](results/rl_peft_moderate_replay_20261030.json)。原始日志保存在本机被 `.gitignore` 排除的 `logs/` 目录。

## 复现命令

```powershell
python compare.py --config configs/rl_peft_phase_signal_diagnostic.json --pretrained 'D:\Research\RL4EQ\pretrained\eme_bce_all_32_20260905_pilot256\model_best.pt' --method-group proposed --delays 116 --snrs 5 --num-seeds 5 --seed-start 1061 --frames 24 --pilot-total 256 --reward-pilot-total 32 --acquisition-to-data-gap-seconds 600 --pilot-layout prefix --update-interval 8 --output-dir logs/rl_peft_moderate_gap600_5db_5s24f_seed1061_1065_20261030 --device cpu
python compare.py --config configs/rl_peft_phase_signal_diagnostic.json --pretrained 'D:\Research\RL4EQ\pretrained\eme_bce_all_32_20260905_pilot256\model_best.pt' --methods 'Frozen Offline NN' --delays 116 --snrs 5 --num-seeds 5 --seed-start 1061 --frames 24 --pilot-total 256 --reward-pilot-total 32 --acquisition-to-data-gap-seconds 600 --pilot-layout prefix --output-dir logs/rl_peft_frozen_gap600_5db_5s24f_seed1061_1065_20261030 --device cpu
```

本机使用 PyTorch `2.11.0+cpu`，未检测到 CUDA。验证命令与结果在本阶段提交记录中注明。

## 下一步

先做一个小型配对消融，区分“moderate PEFT 候选是否有用”和“PPO 是否会挑对更新”：在相同固定轨迹上比较冻结模型、固定使用 moderate（仍由 Adapt Pilot 更新并由 Reward Pilot 拒绝/回滚）和 PPO 选动作三种诊断。保持 checkpoint、信道、Pilot、5 dB 和 seed 划分不变，先看逐 seed Data BER 与 Reward 接受率；若固定 PEFT 也不能跨 seed 改善，就不扩大 PPO 矩阵，而回到 Pilot-only 排序目标。三种策略仍严格屏蔽 Data 波形与标签的在线使用。
