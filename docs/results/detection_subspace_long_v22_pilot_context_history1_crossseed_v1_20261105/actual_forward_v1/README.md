# v22：单帧更新跨seed复测：真实模型前向修正评估

> 此归档取代父目录原README中的BER表。原表以 `basis @ coefficients` 线性近似评分，与非线性上下文Adapter的在线更新/验证路径不一致，现已撤回；请勿引用旧表作为实验结论。

本次仅重用原清单、评估特征缓存和A/B checkpoint，以 `DetectionSubspace.logits(observation, coefficients)` 重算Frozen、更新前、候选与保留系数。未重训、未改信道或资源、未改更新控制。Data标签只用于事后评分。该修正确认的是评分路径，不构成在线增益或独立确认。

| SNR | A Frozen BER | A Online BER | A增益 | B Frozen BER | B Online BER | B增益（95%区间） | B−A交互 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 dB | 0.15894912 | 0.15900337 | -5.43e-05 | 0.15894912 | 0.15898981 | -4.07e-05 [-0.000142, +6.04e-05] | +1.36e-05 |
| 5 dB | 0.026448568 | 0.026448568 | -1.94e-11 | 0.026448568 | 0.026448568 | -1.94e-11 [-1.03e-10, +6.41e-11] | +0 |
| 10 dB | 0.0018581815 | 0.0018581815 | +0 | 0.0018581815 | 0.0018581815 | +0 [+0, +0] | +0 |
| 15 dB | 8.1380211e-05 | 8.1380211e-05 | +0 | 8.1380211e-05 | 8.1380211e-05 | +0 [+0, +0] | +0 |

## 解释

v22在新开发seed上真实前向结果与旧BER表大体一致：0 dB的A、B均退化，B−A只有很小正交互；5/10/15 dB几乎没有变化。跨seed在线收益没有成立。

所有四个SNR均保留报告；开发数据不能用于正式确认。在线优于同源Frozen的可归因、可重复证据尚未建立。

## 可复现性与输入指纹

- 评估记录数：`768`；子空间维数：`2`。
- 输入清单、特征缓存、A/B checkpoint的SHA-256见同目录`summary.json`；压缩逐帧指标也带SHA-256。
- 设备：NVIDIA GeForce GTX 1650，CUDA；重算脚本：[`scripts/recompute_subspace_evaluation.py`](../../../../scripts/recompute_subspace_evaluation.py)。
- `core_confirmed=false`、`confirmation_ready=false`。
