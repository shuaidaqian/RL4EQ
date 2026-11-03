# 第六轮只读诊断：Adapt 全局温度梯度筛查

## 范围

本诊断重用第六轮已归档的 evaluation 特征、B checkpoint 和逐帧系数轨迹。只对最近最多8帧合法 Adapt Pilot BCE 的 logits 应用统一温度 `0.25/0.5/1/2/4`，分别重算 Adapt 梯度；验证Pilot梯度和 Data 事后梯度不变。Data 标签仅用于离线梯度方向参照，没有进入更新或动作选择。所有输入及SHA256记录在 [`analysis.json`](analysis.json)。

本批是重复开发数据上的只读机制诊断，不是新的信道样本，也不是在线收益或确认结果。信道配置、Pilot、接收窗口、物理跟踪、子空间结构与会话参数均未改变。

复现命令：

```powershell
& 'D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe' scripts/diagnose_subspace_gradient_alignment.py `
  --feature-cache logs/detection_subspace_training_v6_adapt_history8_v1_20261103/evaluation_features.pt `
  --trajectory docs/results/detection_subspace_long_v6_adapt_history8_v1_20261103/frame_metrics.jsonl.gz `
  --checkpoint logs/detection_subspace_training_v6_adapt_history8_v1_20261103/B_post_adapt.pt `
  --adapt-temperatures 0.25,0.5,1,2,4 `
  --output docs/results/detection_subspace_long_v6_temperature_sweep_v1_20261103/analysis.json --device cuda
```

## 结果

Adapt/Data 梯度余弦，按每个 seed 会话先平均、再对三个重复开发会话等权平均：

| Adapt温度 | 0 dB | 5 dB | 10 dB | 15 dB | 四点最差 |
|---:|---:|---:|---:|---:|---:|
| 0.25 | -0.350 | -0.239 | +0.115 | +0.327 | -0.350 |
| 0.5 | -0.249 | -0.206 | +0.150 | +0.291 | -0.249 |
| 1 | +0.408 | +0.236 | +0.224 | +0.169 | +0.169 |
| 2 | +0.450 | +0.319 | +0.290 | +0.258 | +0.258 |
| 4 | +0.451 | +0.324 | +0.302 | +0.219 | +0.219 |

温度2在四个主要SNR上的梯度方向均较温度1改善，并且其最差点高于温度4。15 dB Adapt 梯度范数从约 `3.51e-6` 增至 `2.63e-5`；这说明全局尺度值得进入一次预先固定的匹配训练，不证明在线 BER 会改善。温度0.5的梯度表现较差，与既有训练失败方向一致。

所有值来自已观察的三个开发seed；没有置信区间、没有多重比较校正，不能推断独立泛化或正式效果。
