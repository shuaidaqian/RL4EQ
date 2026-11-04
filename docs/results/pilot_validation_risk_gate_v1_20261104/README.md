# 第十二轮：离线训练的 Pilot-only 动作风险门控

## 设计与信息边界

本轮固定 v11 的 8 维上下文更新器、Adam更新规则、投影、锚点回退、独立验证Pilot硬门控、信道/同步/物理状态、Pilot预算、接收窗口与四SNR。离线训练一个 11 维岭回归器，估计候选更新造成的 Data BCE 改善；特征仅由更新前后验证Pilot的逐符号BCE变化、硬BER变化、logit幅度/符号变化、候选增量范数和Adapt梯度范数组成。在线策略仍先通过原有验证Pilot硬门控，再要求风险模型预测的 Data BCE 改善大于0。

Data 仅用于离线训练风险回归器的目标、选择会话的只读校准报告和评估会话的事后评分。在线门控接口只接收上述 Pilot 特征，不接收 Data、真实CIR、相位或CFO。风险目标定义为“当前Data BCE减去同次候选动作的Data BCE”，正值代表改善。所有候选动作（不只原硬门控接受的动作）进入训练，排除已触发离线初始值回退的帧；训练、选择与评估继续使用基础清单中互斥的数据角色。

固定风险门控清单 SHA256：`3c84465fe0db137f586723ee4104795834eebf414197b7ecb6ae8ee9941e92cb`；基础 v11 manifest SHA256：`67060019ad546620383a5a87e428a827e89c527f4ed0178f2e45c8192658a462`；v11 B checkpoint SHA256：`7f2b20c9fe9396c6d692d349acb6bf45760984905ef72f7730a84cc90d3d54ac`。岭系数固定为0.1，决策阈值固定为预测 Data BCE 改善 `>0`，没有按SNR分支，也没有用评估标签调阈值。

## 离线校准

风险模型用 131 个训练候选拟合，训练样本中99个实际 Data BCE 改善；独立选择角色包含66个候选，49个改善。阈值0下，选择角色报告6个误收（预测有益但事后 Data BCE 未改善）、8个误拒（预测无益但事后 Data BCE 改善），RMSE约 `2.93e-5`、预测与目标 Pearson 相关约 `0.465`。训练集相关约 `0.804`，选择集下降，显示泛化校准明显弱于拟合表现。选择指标只作报告，没有据其更改固定阈值。

## 评估角色结果

在同一 v11 B checkpoint 的独立开发评估会话中，逐帧分别重放原硬门控与“硬门控+风险门控”，每个策略保有自己的在线系数历史。此评估角色仍是已用于开发的会话，不是独立确认。

| SNR | 原硬门控接受/有害数 | 风险门控接受/拒绝/有害数 | 两策略在线BER变化 |
|---:|---:|---:|---:|
| 0 dB | 45 / 3 | 41 / 4 / 3 | 相同，仍约 `+1.36e-5` |
| 5 dB | 43 / 6 | 38 / 7 / 5 | 相同，仍约 `-1.36e-5` |
| 10 dB | 3 / 3 | 2 / 1 / 2 | 相同，0 |
| 15 dB | 0 / 0 | 0 / 0 / 0 | 相同，0 |

0 dB风险门控拒绝的4个候选事后均有正Data BCE改善，而已接受动作中的有害数没有下降。5 dB少接受5个动作，只少1个有害动作，并误拒6个有益候选；BER没有改变。10 dB从3个有害接受动作降到2个，但剩余动作仍全部有害；BER同样不变。15 dB没有候选通过原硬门控。Data BCE标签只用于这些离线事后计数。

## 阶段判断

Pilot-only回归器在选择角色上有一定排序信息，但固定正向预测门控没有形成跨SNR一致的风险收益权衡，也没有降低在线BER。它在0 dB拒绝了有益动作，在5 dB略微降低有害接受率同时误拒不少有益动作，在10 dB仍保留有害动作。因此**不把该风险门控设为默认在线策略，核心目标仍未完成，也不进入R3**。不能用本次评估Data重新拟合、改阈值或调各SNR的策略。

本轮说明“再加一道只依赖验证Pilot的拒绝条件”不足以解决验证/Data失配。下一步应先离线分析回归器在未被原硬门控接受的候选上的反事实排序，判断是否值得研究由Pilot-only风险模型替代原硬门控；该分支必须单独预注册开发门槛，仍只允许Pilot信息在线决策。独立确认前还需冻结应用主要区间、允许退化界限、统计方法与样本量。

## 复现与验证

```powershell
& 'D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe' -m pytest -q -p no:cacheprovider tests/test_detection_subspace.py
& 'D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe' scripts/train_pilot_risk_gate.py --manifest configs/audits/pilot_validation_risk_gate_v1.json --base-training-dir logs/detection_subspace_training_v11_smooth_ber_outer_v1_20261104 --output-dir logs/pilot_validation_risk_gate_final_v1_20261104 --device cuda
```

归档保留清单、回归模型、训练/选择/评估逐动作诊断和输入指纹；模型与全部诊断记录见同目录。特征缓存未提交。定向测试 **30 passed**；全量 `python -m pytest -q -p no:cacheprovider` **931 passed**（376.54秒）。
