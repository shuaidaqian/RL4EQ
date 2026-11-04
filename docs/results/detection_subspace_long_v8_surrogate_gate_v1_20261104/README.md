# 第八轮：硬门控拒绝分支直通梯度开发评估

## 单因素与实现

本轮固定v7的 `eme_long_memory_v2`、来源checkpoint、8维 `temporal_conv_v1`、统一Adapt温度2、最近8帧Adapt、Pilot/接收窗口、物理状态跟踪、验证硬门控、初始值回退/锚点收缩、四个SNR、A/B预算及开发seed。唯一新增的是**离线训练反向**对硬拒绝/回退分支使用直通梯度：前向精确执行原线上门控，拒绝时输出与部署相同的初始/回退系数；外层梯度近似穿过未门控候选。线上 `create_graph=false` 不使用代理梯度，部署前向规则不变。

清单 SHA256：`98385788a574ac04f407c370fc166dcc7d2b18ac54bf003af1355c2c0053de35`；信道配置 SHA256 `535c1ce48a3ed194c0efc90b4d55afad371179d4d1b8415a3ba57ed40da6d149`，来源checkpoint SHA256 `11e58a5d94801ae613ec8f7a55679ed2304e0864535c7b0468b316827e2bec51`。审计仍选择8维，监督可达性检查通过。A/B各64步外层训练，选中step64；评估沿用3个已观察开发seed、四SNR、每seed32帧，共768条方法记录。不是独立确认。

## 在线 Data BER

增益定义为同臂 Frozen BER 减 Online BER，正值代表改善：

| SNR | A普通监督增益及95%区间 | B直通训练增益及95%区间 | B接受数 | B有害保留数 | B−A交互 |
|---:|---:|---:|---:|---:|---:|
| 0 dB | `+1.36e-5` `[-1.41e-4,+1.68e-4]` | `-1.36e-5` `[-7.19e-5,+4.48e-5]` | 29 | 1 | `-2.71e-5` |
| 5 dB | `-1.36e-5` `[-7.19e-5,+4.48e-5]` | `+1.36e-5` `[-4.48e-5,+7.19e-5]` | 28 | 0 | `+2.71e-5` |
| 10 dB | 0 `[0,0]` | 0 `[0,0]` | 5 | 0 | 0 |
| 15 dB | 0 `[0,0]` | 0 `[0,0]` | 1 | 0 | 0 |

各区间只有三个重复开发seed，不能作确认推断；15 dB错误不足，零变化不代表等效或非劣。B在0 dB点估计退化、5 dB点估计改善，且区间跨零；10/15 dB没有可判断增量。v8未显示直通外层梯度能把局部梯度改善转成可靠Data检测收益，因此**不把此代理确立为方法，不进入R3**。

v8 B臂Adapt/Data梯度余弦为约 `0.478/0.331/0.260/0.204`；相较v7 `0.462/0.321/0.319/0.259`，0/5 dB略升而10/15 dB下降。5/10 dB验证/Data余弦仅约 `0.043/-0.029`。训练选择损失B为 `0.1058933`，略差于A的 `0.1058667`。所有Data量均为动作之后的开发诊断。

## 阶段判断

已排除两种局部修补：逐帧Pilot阈值/置信界门控没有可靠校准；仅恢复硬拒绝分支外层梯度也没有产生增量。隔离Data副本仍说明8维子空间存在可降低Data BCE、部分帧可改善BER的方向，但Adapt与Data目标如何映射仍未解决。下一轮应把更新方向本身作为离线学习对象，例如在低维系数空间学习Pilot梯度到参数增量的受约束预条件/传输映射；在线仍只消费合法Adapt Pilot并实际更新神经系数，验证Pilot保持独立硬门控，Data仅用于离线外层训练和动作后评分。

## 复现

```powershell
& 'D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe' online_train.py --method subspace_audit --audit-manifest configs/audits/detection_subspace_long_v8_surrogate_gate_v1.json --output-dir logs/detection_subspace_inspection_v8_surrogate_gate_v1_20261104 --device cuda
& 'D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe' pretrain.py --stage detection_subspace --subspace-manifest configs/audits/detection_subspace_long_v8_surrogate_gate_v1.json --inspection-dir logs/detection_subspace_inspection_v8_surrogate_gate_v1_20261104 --save-dir logs/detection_subspace_training_v8_surrogate_gate_v1_20261104 --device cuda
```

压缩轨迹 SHA256：`eeff1bb7758f73813a3065afd1eb52a33bbbfc3a05954ee8f3f1504558a36f18`；解压原始轨迹 SHA256：`a17fd020d20faf80381a8befc0e593c299849a0183d18beda6b0c63474449a51`。摘要、清单、A/B checkpoint、几何检查、完整轨迹与v8门控/梯度只读复盘共同保留在归档和相邻诊断目录。
