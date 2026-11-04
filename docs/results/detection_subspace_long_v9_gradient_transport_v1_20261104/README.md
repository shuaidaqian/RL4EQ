# 第九轮：离线学习 Adapt 梯度传输映射

## 研究问题与唯一改动

本轮继续在原有 116 符号长回波、0/5/10/15 dB、既有 Pilot/接收窗口和物理跟踪下研究“离线训练适应信道，在线微调均衡”。相较 v8，新增一个 8 维对称正定梯度传输矩阵，将在线合法 Adapt Pilot 产生的 Adam 系数增量左乘该矩阵。矩阵从单位阵初始化，更新后仍经过原有逐步/累计半径投影及独立验证 Pilot 硬守门。在线只改 8 个神经系数；Data 和隐藏真值不进入在线接口。

清单 SHA256：`40f11bc5dab9b242c9e4825aef372592bfcaf2c1db569c5cd51612acfcdd3e41`。信道配置 SHA256：`535c1ce48a3ed194c0efc90b4d55afad371179d4d1b8415a3ba57ed40da6d149`；来源 checkpoint SHA256：`11e58a5d94801ae613ec8f7a55679ed2304e0864535c7b0468b316827e2bec51`。数据角色与开发 seed 沿用清单，未作独立确认。

## 实现及固定预算检查

8 维矩阵由有界下三角因子构造 `P=L L^T`：初始 `P=I`，非对角因子经 `0.25*tanh` 限幅，对角因子经 `exp(0.5*tanh)` 限幅。测试覆盖单位阵时更新与旧实现严格一致、矩阵正定/有界/条件数可控、非单位矩阵改变更新方向，以及离线展开外层对映射存在有限非零梯度。原 checkpoint 不含该字段时仍默认关闭映射并沿用原 state dict。

几何检查在全部指定帧上通过 2/4/8 维候选，按预定最大合格维度规则选 8 维；独立监督作用烟测可改变子空间基、系数和 Data logits。随后 A/B 各执行 64 步、相同初始化和数据预算；A 保持单位映射，B 用 Adapt 内循环与当前/下一帧 Data 外循环学习映射。完整评估为 3 个重复开发 seed × 4 SNR × 32 帧 × 2 臂，共 768 条记录。

## 在线 BER 结果

增益定义为 Frozen BER 减 Online BER，正数为改善。区间以每个 SNR 三个开发 seed 为单位，不能作独立确认推断。

| SNR | A 监督臂增益（95%区间） | B 梯度传输臂增益（95%区间） | B−A 交互 | B 接受数 |
|---:|---:|---:|---:|---:|
| 0 dB | `+1.36e-5` `[-1.41e-4,+1.68e-4]` | `0` `[0,0]` | `-1.36e-5` | 17 |
| 5 dB | `-1.36e-5` `[-7.19e-5,+4.48e-5]` | `0` `[0,0]` | `+1.36e-5` | 17 |
| 10 dB | `0` `[0,0]` | `0` `[0,0]` | `0` | 6 |
| 15 dB | `0` `[0,0]` | `0` `[0,0]` | `0` | 1 |

15 dB 错误数不足，零变化不代表等效或非劣。B 仅在 0/5 dB 各有 17 帧更新被接受，最终 768 条配对评估没有任何 B 臂 Data BER 改变。B checkpoint 的传输矩阵偏离单位阵 Frobenius 范数约 `0.0572`，特征值范围约 `[0.9471,1.0002]`，条件数 `1.0561`；矩阵确实学习到有限变化，但没有转化为 BER 收益。B 选择损失 `0.1059295`，略高于 A 的 `0.1058667`。平均在线子空间更新耗时约 39 ms；不含接收、acquisition 与来源状态跟踪，不能视为端到端延迟。

## 机制复盘

- Adapt/Data 系数梯度余弦按 0/5/10/15 dB 分别约 `0.396/0.135/0.144/0.089`；Adapt/验证余弦约 `0.236/0.159/0.084/0.277`。验证/Data 余弦在 5/10 dB 约 `-0.008/-0.089`，说明留出 Pilot 对 Data 改善方向的指示很弱。
- 原验证门控接受候选的动作后 Data BCE 有害比例在 0/5/10/15 dB 约 `0.35/0.53/0.83/1.00`。区块置信下界门控降低接受次数，但没有跨 SNR 的可靠可用性；此分析只读动作后 Data 评分，未用于控制。
- 独立 Data-only 诊断在 0/5/10/15 dB 的 96 帧上均降低 Data BCE，但只有约 `39.6%/38.5%/25.0%/13.5%` 帧改善硬 BER。它证明低维表示仍有局部 Data 可达方向，不证明合法 Pilot 能找到这些方向；Data 标签仅用于隔离离线诊断。

因此，本轮不支持将“学习一个低维固定预条件矩阵”确立为有效在线方法。当前主要瓶颈仍是 Pilot 更新/留出门控与 Data 泛化方向不一致，且高 SNR 可用错误稀少、Adapt 梯度极弱。核心目标未完成，不进入 R3。

## 复现与归档

```powershell
& 'D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe' -m pytest -q -p no:cacheprovider tests/test_detection_subspace.py
& 'D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe' online_train.py --method subspace_audit --audit-manifest configs/audits/detection_subspace_long_v9_gradient_transport_v1.json --output-dir logs/detection_subspace_inspection_v9_gradient_transport_v1_20261104 --device cuda
& 'D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe' pretrain.py --stage detection_subspace --subspace-manifest configs/audits/detection_subspace_long_v9_gradient_transport_v1.json --inspection-dir logs/detection_subspace_inspection_v9_gradient_transport_v1_20261104 --save-dir logs/detection_subspace_training_v9_gradient_transport_v1_20261104 --device cuda
```

本归档保留固定清单、几何检查、A/B checkpoint、总结、压缩逐帧轨迹与梯度对齐/验证门控/Data 可达性诊断。原始特征缓存不纳入版本库；摘要和诊断记录各自保留了输入/缓存指纹。CUDA 环境为 GTX 1650。定向子空间测试 **22 passed**；全量 `python -m pytest -q -p no:cacheprovider` **923 passed**（367.53 秒）。全量运行时仅出现本次新增测试中的 tensor 转 float 警告，已修正并重跑定向测试确认 **22 passed** 无警告。
