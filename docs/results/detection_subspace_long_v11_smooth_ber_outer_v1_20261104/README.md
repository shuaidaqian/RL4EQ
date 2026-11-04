# 第十一轮：用平滑误码边界目标训练上下文更新器

## 单因素与隔离

本轮保留 v10 的在线上下文更新器、Adapt 历史、验证Pilot门控、投影、四SNR、长回波信道、Pilot预算、接收窗口、物理跟踪、seed与A/B训练预算。唯一改动是把离线外层当前/下一帧Data目标和checkpoint选择目标从BCE改为固定温度 `T=1.0` 的平滑误码代理：

`mean(sigmoid(-signed_logit / T))`

A/B两臂统一使用该目标训练与选择；在线更新仍仅读取Adapt Pilot梯度历史，验证Pilot只执行原硬门控。Data标签仅用于离线外循环、checkpoint选择和动作后的独立评分，不进入在线接口。SNR没有独立温度或其他分支参数。

清单SHA256：`67060019ad546620383a5a87e428a827e89c527f4ed0178f2e45c8192658a462`；信道配置和来源checkpoint SHA256沿用此前固定值 `535c1ce48a3ed194c0efc90b4d55afad371179d4d1b8415a3ba57ed40da6d149` 与 `11e58a5d94801ae613ec8f7a55679ed2304e0864535c7b0468b316827e2bec51`。本轮使用3个重复开发seed，不是独立确认。

## 开发结果

几何门槛与8维监督作用烟测通过。A/B各64次外层更新，评估覆盖4 SNR × 3开发seed × 每会话32帧 × 2臂，共768条记录。

| SNR | A监督臂 BER 增益及95%区间 | B平滑误码训练臂 BER 增益及95%区间 | B−A交互 | B守门接受数 |
|---:|---:|---:|---:|---:|
| 0 dB | `+1.36e-5` `[-4.48e-5,+7.19e-5]` | `+4.07e-5` `[+4.07e-5,+4.07e-5]` | `+2.71e-5` | 45 |
| 5 dB | `约0` `[-1.01e-4,+1.01e-4]` | `-4.07e-5` `[-1.42e-4,+6.04e-5]` | `-4.07e-5` | 43 |
| 10 dB | `0` `[0,0]` | `0` `[0,0]` | `0` | 3 |
| 15 dB | `0` `[0,0]` | `0` `[0,0]` | `0` | 0 |

0 dB的B增益虽在三个开发seed上同向，但量化后仅相当于每个32帧会话约减少1个Data错误；区间窄是因为三个会话的离散误差变化恰好相同，不能据此宣称稳健或推广。5 dB的B臂一个会话无变化、另两个会话合计增加约3个错误，点估计为退化且区间跨零。10/15 dB没有BER变化，15 dB没有被接受的候选。B的checkpoint平滑误码选择损失为 `0.0736802`，仍高于A的 `0.0736188`。

## 机制结论

Adapt/Data系数梯度余弦按0/5/10/15 dB约为 `0.629/0.497/0.339/0.233`，明显高于v10的 `0.371/0.094/0.166/0.086`。这说明外层目标确实改变了上下文更新器偏好的方向。B臂Frozen BCE减Online BCE约为 `+6.22e-5/+2.34e-5/+9.83e-7/0`；然而连续BCE改善未稳定转成硬BER改善。验证/Data梯度余弦在5/10 dB约 `0.125/0.030`，留出守门信号仍然偏弱。

本轮提供“训练目标影响梯度方向”的机制证据，以及一个很小的0 dB开发点增益；它同时暴露5 dB退化、选择损失劣于A、其余SNR无变化。所有统计都来自重复开发seed，不是独立确认；不能进入R3，不能宣称目标完成。下一步应先解决动作后验证Pilot风险估计与Data泛化不一致，并在冻结正式应用主区间/退化界限后设计新的独立会话确认；不得用本轮Data结果反向调在线门控。

## 复现

```powershell
& 'D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe' -m pytest -q -p no:cacheprovider tests/test_detection_subspace.py
& 'D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe' online_train.py --method subspace_audit --audit-manifest configs/audits/detection_subspace_long_v11_smooth_ber_outer_v1.json --output-dir logs/detection_subspace_inspection_v11_smooth_ber_outer_v1_20261104 --device cuda
& 'D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe' pretrain.py --stage detection_subspace --subspace-manifest configs/audits/detection_subspace_long_v11_smooth_ber_outer_v1.json --inspection-dir logs/detection_subspace_inspection_v11_smooth_ber_outer_v1_20261104 --save-dir logs/detection_subspace_training_v11_smooth_ber_outer_v1_20261104 --device cuda
```

归档包括清单、检查结果、A/B checkpoint、摘要、压缩逐帧轨迹、梯度对齐与门控复盘；原始特征缓存不纳入版本库。CUDA设备为GTX 1650。定向测试 **27 passed**；全量 `python -m pytest -q -p no:cacheprovider` **928 passed**（370.41秒）。
