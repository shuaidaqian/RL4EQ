# 第七轮：统一 Adapt 温度的匹配训练与开发评估

## 目标与边界

本轮接续 v6 的只读温度梯度筛查，仅将最近最多8帧合法 Adapt Pilot BCE 的全局 logit 温度设为2.0。信道与来源 checkpoint、Pilot/接收窗口、8维 `temporal_conv_v1`、物理状态跟踪、验证 Pilot 守门、初始系数回退、0.5锚点收缩、A/B训练预算及开发 seed 均沿用 v6。温度对四个SNR统一，不使用SNR条件规则。`channel_changes=false`、`heldout_confirmation=false`。

清单 SHA256 为 `7ad25dead444c063e828a87f40c112595a9d0c9eca30d51d968821ebf12657e3`，现有信道配置 SHA256 为 `535c1ce48a3ed194c0efc90b4d55afad371179d4d1b8415a3ba57ed40da6d149`，来源 checkpoint SHA256 为 `11e58a5d94801ae613ec8f7a55679ed2304e0864535c7b0468b316827e2bec51`。几何审计选择8维，离线普通监督可达性通过。A/B各64步外层训练，均选中第64步；最终评估为3个重复开发 seed、四SNR、每会话32帧，共768条记录。

## 结果

B臂在线增益定义为 Frozen BER 减 Online BER，正值表示改善：

| SNR | Frozen BER | B Online BER | 配对增益及95% seed区间 | 接受 / 回退 | 有害保留 |
|---:|---:|---:|---:|---:|---:|
| 0 dB | 0.17309571 | 0.17306858 | `+2.71e-5` `[-8.96e-5, +1.44e-4]` | 30 / 13 | 0 |
| 5 dB | 0.03346083 | 0.03344727 | `+1.36e-5` `[-4.48e-5, +7.19e-5]` | 35 / 12 | 0 |
| 10 dB | 0.00339084 | 0.00339084 | `0` `[0, 0]` | 1 / 0 | 0 |
| 15 dB | 0.00055610 | 0.00055610 | `0` `[0, 0]` | 1 / 0 | 0 |

这三个seed区间只是开发性配对描述；15 dB错误数不足，零变化不代表等效或非劣。A臂在线增益在四点也没有可靠改善，B-A交互分别约 `+1.36e-5/+2.71e-5/0/0`，均不能确立适配训练的稳定额外收益。在线候选更新约48–53 ms/帧，为缓存特征后的子空间增量，不是端到端部署延迟。

只读复盘显示B臂 Adapt/Data梯度余弦为 `0.462/0.321/0.319/0.259`，四点均高于v6基线的 `0.408/0.236/0.224/0.169`；15 dB Adapt梯度范数由约 `3.51e-6` 增至 `2.52e-5`。然而验证/Data梯度余弦仍只有约 `0.276/0.074/0.024/0.447`，且10/15 dB候选各只接受一次。验证Pilot与同帧Data BCE改善的会话内Spearman均值为 `0.195/0.053/-0.558/-0.596`，每点仅三个重复开发会话。温度解决了部分Adapt梯度尺度/方向问题，没有解决守门校准、有效更新频率和 BER 增益问题。

因此本轮**没有完成核心目标，也不支持进入独立确认**。Data只用于离线评分和梯度事后诊断，从未用于更新、选择、验收或回滚。当前不能把小幅点估计写成在线微调有效。

## 复现与产物

几何检查：

```powershell
& 'D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe' online_train.py --method subspace_audit --audit-manifest configs/audits/detection_subspace_long_v7_adapt_temperature2_v1.json --output-dir logs/detection_subspace_inspection_v7_adapt_temperature2_v1_20261104 --device cuda
```

匹配训练与评估：

```powershell
& 'D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe' pretrain.py --stage detection_subspace --subspace-manifest configs/audits/detection_subspace_long_v7_adapt_temperature2_v1.json --inspection-dir logs/detection_subspace_inspection_v7_adapt_temperature2_v1_20261104 --save-dir logs/detection_subspace_training_v7_adapt_temperature2_v1_20261104 --device cuda
```

本目录保留清单、几何审计、机器汇总、A/B checkpoint、只读守门/梯度诊断和完整压缩轨迹。未压缩轨迹 SHA256 为 `45fdbd00d8642e3b4ddef9b2762f165dc031a250b946021d2f3b7de1b43591ab`；gzip解压字节须与此一致。GPU训练与评估耗时约529.19秒；所有结果仍属开发证据。
