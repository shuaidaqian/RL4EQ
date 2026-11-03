# 第六轮只读诊断：验证 Pilot 守门与 Data 变化

## 目的与边界

本诊断只读取[第六轮已归档评估轨迹](../detection_subspace_long_v6_adapt_history8_v1_20261103/frame_metrics.jsonl.gz)，按 B 臂评估记录比较最终保留状态的验证 Pilot BCE 变化与同帧 Data BCE/BER 变化。它没有重新运行接收机、执行更新或选择动作；Data 标签只用于动作后离线诊断。帧按 SNR/seed 会话分组，逐帧相关性在同一会话内计算，不把 32 帧当作 32 个独立样本。

复现命令：

```powershell
python scripts/diagnose_subspace_gate_alignment.py `
  --trajectory docs/results/detection_subspace_long_v6_adapt_history8_v1_20261103/frame_metrics.jsonl.gz `
  --output docs/results/detection_subspace_long_v6_gate_alignment_v1_20261103/analysis.json
```

机器可读结果 [`analysis.json`](analysis.json) 记录输入轨迹 SHA256、seed 会话级汇总和逐会话描述量。输入仍是重复开发 seed，不是确认集。

## 观察结果

| SNR | seed 会话内 Spearman 均值 | B 候选接受数 | 接受行最终保留 Data BCE 平均改善 |
|---:|---:|---:|---:|
| 0 dB | `+0.108` | 17 | `+4.14e-5` |
| 5 dB | `-0.136` | 17 | `-7.27e-6` |
| 10 dB | `-0.337` | 9 | `-1.13e-5` |
| 15 dB | `-0.596` | 1 | `-5.25e-5` |

这里“Data BCE 改善”定义为 Frozen Data BCE 减去最终 Online Data BCE，正值表示改善。会话内 Spearman 以最终保留 Pilot BCE 改善与该 Data BCE 改善为变量。0 dB 只有弱正关系；其余三个点为负。5/10 dB 接受候选后的平均 Data BCE 变化为负。15 dB 只有一个接受候选，均值和相关性没有稳定解释力。BER 变化由少量符号翻转量化，接受行的 BER 均值不能替代连续 BCE 诊断。

## 结论与下一步

现有证据支持“验证 Pilot 守门与同帧 Data 改善之间的校准不足”是当前 R2 瓶颈之一；它不证明改变验证统计量必然改善持续 Data BER，也不重新判定第六轮核心目标通过。更新日志没有保存 Adapt/Data 梯度向量，因此本归档不能重算更新梯度方向余弦；那需要单独的只读离线梯度采样，Data 梯度只作为事后参考。

下一项 R2 工作应保持信道和资源不变，先在独立的开发诊断数据上比较合法验证 Pilot 连续损失与动作后 Data 连续损失的排序/校准，再据此只改一个预先固定的守门统计量，并匹配离线内循环与在线回放。不得用 Data 标签在线控制，不得扩展 seed 来追逐正结果。R3 仍未启动。
