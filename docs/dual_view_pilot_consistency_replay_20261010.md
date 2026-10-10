# 双视图 Adapt Pilot 一致性 replay

日期：2026-10-10

## 目的

当前在线微调的主要问题是：Adapt Pilot 上的损失下降不一定代表 Data 段会变好。
本轮把同一帧的 Adapt Pilot 拆成前、后两个子段，各自独立预览
`phase_trend + Pilot 信号重构` 的参数更新，只有两个参数更新方向的余弦相似度达到阈值时
才保留完整 Pilot 更新。这个过程只使用 Adapt Pilot 和已知 Pilot 符号；Data 标签只用于
事后计算 BER/BCE。

## 固定设置

- checkpoint：`pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`
- 信道：Level B，`delay=116`
- Pilot：prefix，总长度 256
- acquisition 到普通帧间隔：120 s（状态老化诊断）
- SNR：5、10 dB
- seed：0、1、2
- 每个 seed：2 帧
- 参数更新：`phase_trend`，Pilot 复数信号重构，AdamW，5e-3，2 步
- 一致性阈值：参数更新方向余弦 `>= 0`

## 结果

逐帧结果：[`logs/dual_view_analytic_s01_gap120_3s_2snr_2f_20261010.json`](../logs/dual_view_analytic_s01_gap120_3s_2snr_2f_20261010.json)

| SNR | 一致帧比例 | 无门控平均 Data BER 改善 | 一致性门控平均 Data BER 改善 | 无门控正收益帧比例 | 门控正收益帧比例 |
|---:|---:|---:|---:|---:|---:|
| 5 dB | 50.0% | 0.0000 pp | -0.0651 pp | 16.7% | 0.0% |
| 10 dB | 66.7% | 0.0000 pp | 0.0000 pp | 0.0% | 0.0% |

结论：两个子段的参数方向一致性不能可靠预测 Data BER。原因是 Adapter 只有两个相位
参数，噪声和相位估计误差会让方向余弦在接近 0 和 ±1 之间跳变；一次方向相反的更新
有时反而能改善 Data，因此简单的“方向一致才更新”会错过有效更新。

## 同轮解析相位写入探针

脚本还记录了一个诊断性对照：从 Adapt Pilot 估计公共相位/CFO，直接写入
`phase_trend` Adapter 参数，并用不同平滑系数限制步长。即使使用 0.1 平滑，5/10 dB
平均 Data BER 改善分别为 `-10.42/-14.32 pp`；多个样本的 Pilot 估计被裁剪到
`±0.5 rad` 或 `±0.0012 cycles/symbol`。这说明当前 acquisition CIR 与 Pilot 参考波形
之间的误差足以让解析估计失真，不能直接把物理状态估计当作参数微调方向。

## 当前决策

1. 不把双视图方向门控接入正式在线主流程。
2. 不把解析相位/CFO 直接写参数接入主流程。
3. 保留 replay 脚本作为负结果和复现实验：
   [`scripts/replay_dual_view_pilot_consistency.py`](../scripts/replay_dual_view_pilot_consistency.py)
4. 下一步转向 Reward Pilot 的物理一致性验收：候选参数更新必须同时满足 Reward Pilot
   的硬判决不退化、复数信号重构误差不退化和跨帧最坏退化约束；通过验收后才保留
   Adapter 参数。这个方案仍然是参数微调，Reward Pilot 只做验收/回滚，不参与梯度。

## 复现命令

```powershell
.\.venv-gpu\Scripts\python.exe scripts/replay_dual_view_pilot_consistency.py `
  --config configs/online_peft_phase_trend_head.json `
  --pretrained pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt `
  --output logs/dual_view_analytic_s01_gap120_3s_2snr_2f_20261010.json `
  --snrs 5 10 --seeds 0 1 2 --frames 2 --delay 116 --pilot-total 256 `
  --gap-seconds 120 --lr 5e-3 --steps 2 --cosine-threshold 0 `
  --analytic-smoothing 0.1
```

