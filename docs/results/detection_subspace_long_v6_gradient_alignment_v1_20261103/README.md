# 第六轮只读诊断：在线系数梯度方向

## 目的与方法

本诊断使用第六轮已经生成的评估特征缓存、B 臂 checkpoint 和逐帧轨迹，重建每个动作前的 8 维系数状态。固定卷积基与来源接收机，不执行优化或状态变更；分别计算最近最多 8 帧 Adapt Pilot BCE、当前帧验证 Pilot BCE 和当前帧离线 Data BCE 对在线系数的梯度。Data 标签只用于事后参照，不能进入更新器或在线动作。

对齐度按每个 SNR/seed 的 32 帧先求会话均值，再对 3 个 seed 会话求均值。帧不视为独立样本。特征缓存、压缩轨迹和 B checkpoint 的 SHA256 均写入 [`analysis.json`](analysis.json)，每帧还核对接收波形和 lookahead 波形指纹。

复现命令：

```powershell
python scripts/diagnose_subspace_gradient_alignment.py `
  --feature-cache logs/detection_subspace_training_v6_adapt_history8_v1_20261103/evaluation_features.pt `
  --trajectory docs/results/detection_subspace_long_v6_adapt_history8_v1_20261103/frame_metrics.jsonl.gz `
  --checkpoint logs/detection_subspace_training_v6_adapt_history8_v1_20261103/B_post_adapt.pt `
  --output docs/results/detection_subspace_long_v6_gradient_alignment_v1_20261103/analysis.json
```

## 结果

| SNR | Adapt 梯度范数 | Data 梯度范数 | Adapt/Data 余弦 | 验证/Data 余弦 |
|---:|---:|---:|---:|---:|
| 0 dB | `2.59e-3` | `2.58e-3` | `+0.408` | `+0.254` |
| 5 dB | `1.64e-3` | `2.11e-3` | `+0.236` | `+0.056` |
| 10 dB | `6.52e-4` | `1.69e-3` | `+0.224` | `-0.012` |
| 15 dB | `3.51e-6` | `1.54e-3` | `+0.169` | `+0.440` |

每项都是 3 个重复开发会话的等权均值，没有置信区间或多重比较校正。15 dB 的验证/Data 余弦不能掩盖 Adapt 梯度范数仅约 Data 梯度的 `0.23%`；该点的更新信号接近消失。0–10 dB Adapt/Data 方向有弱到中等正对齐，但验证/Data 方向在 5 dB 很弱、10 dB近零。结合前一份守门诊断中 5/10 dB 接受后的 Data BCE 均值为退化方向，可见问题同时包含在线更新信号质量与验证守门校准，不能只调高守门阈值解决。

## R2 判断

本结果是重复开发数据上的机制诊断，不是独立确认，不支持进入 R3。下一次开发迭代应预先固定一个温度/尺度处理单因素，首先检查四个 SNR 的 Adapt 梯度有效尺度和 Adapt/Data 对齐是否共同改善；验证 Pilot 守门保持不变，避免把更新目标与守门同时改动。已有固定温度 0.5 的第四轮开发结果为失败，因此新检查应先用只读梯度对照判断是否存在可行的全局尺度，再决定是否训练；不得引入按 SNR 的规则或改变信道配置。
