# Adapt损失梯度方向只读筛查（2026-11-05）

## 目的

在正式改变在线更新器前，只读比较现有v11 B checkpoint上几种Adapt Pilot损失的系数梯度方向，判断难样本挖掘是否可能修复Pilot/Data迁移。只用了v11隔离的selection开发会话（每SNR两个seed）；现有Pilot更新规则重放后，再分别计算候选Adapt梯度和离线Data BCE梯度。Data梯度只作离线机制参照，没有进入任何在线更新或门控。

复现：

```powershell
& 'D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe' scripts/diagnose_adapt_loss_alignment.py `
  --manifest configs/audits/detection_subspace_long_v11_smooth_ber_outer_v1.json `
  --feature-cache logs/detection_subspace_training_v11_smooth_ber_outer_v1_20261104/offline_features.pt `
  --checkpoint logs/detection_subspace_training_v11_smooth_ber_outer_v1_20261104/B_post_adapt.pt `
  --output docs/results/detection_subspace_adapt_loss_alignment_v1_20261105/analysis.json `
  --device cuda
```

## 结果

下表是每个seed内逐帧余弦均值，再对两个seed等权平均。梯度范数是同样的会话/seed聚合。

| Adapt损失 | 指标 | 0 dB | 5 dB | 10 dB | 15 dB |
|---|---|---:|---:|---:|---:|
| BCE，T=2（v11线上目标） | Adapt/Data梯度余弦 | 0.843 | 0.832 | 0.664 | 0.834 |
| BCE，T=1 | Adapt/Data梯度余弦 | 0.817 | 0.813 | 0.405 | 0.483* |
| 最高损失25% | Adapt/Data梯度余弦 | -0.390 | 0.789 | 0.405 | 0.483* |
| 最高损失50% | Adapt/Data梯度余弦 | 0.556 | 0.810 | 0.405 | 0.483* |
| focal，gamma=2 | Adapt/Data梯度余弦 | -0.584 | -0.534 | 0.216 | 无有效方向** |
| BCE，T=2（v11线上目标） | Adapt梯度范数 | 5.31e-3 | 5.99e-3 | 5.07e-4 | 7.56e-8 |

`*` 15 dB仅一个seed具有非零Data梯度；另一个seed的Data梯度为零，故方向不可定义。`**` 15 dB两个seed都没有可用的非零focal梯度方向。

## 判断

难样本聚焦没有呈现全SNR优势：top-25%在0 dB与Data梯度反向，focal在0/5 dB反向；top-50%在0/5/10 dB也低于当前T=2 BCE。当前T=2 BCE在可计算的四点上方向对齐最好，但15 dB Adapt梯度范数只有约`7.6e-8`，并且选择会话仍不足以证明梯度方向能转成BER收益。因此不把hard mining或focal作为在线机制，也不据两个seed作统计推断。

此诊断不改变研究判断：v11当前更新仍未稳定优于Frozen。结果只排除了一个直观候选方向，后续重点转向离线外循环是否明确优化“同一表示下Online相对Frozen的增益”。
