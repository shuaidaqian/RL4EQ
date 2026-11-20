# 衰落分布与帧末观测边界的确定性数学核验

本目录不是接收机性能实验。没有生成随机轨迹，没有新增或复用研究 seed，没有使用 P8 中途性能来选择方法。理论依据见[推导说明](../../continual_v2_fading_genie_bound_20261120.md)。

## 已核验内容

- `deterministic_quadrature.json`：三个说明性 PDP、四 SNR、七个时间，共 84 个积分。核验初始 AWGN、单 tap 长期瑞利闭式与下界非下降。
- `finite_frame_boundary_examples.json`：两个显式假设的 117 tap 功率模板、四 SNR、七个时间，共 56 个整体积分和对应完整列/截断列分量。在 1024 符号、Data 索引 256–1023、记忆 116 的当前帧观测下，652 个 Data 比特卷积列完整，116 个被帧末截断；整体下界等于两个分量按比特数加权。
- 18 项定向测试覆盖复噪声功率的因子 2、无记忆 AWGN、不可观测末比特的 0.5、CIR 相位不变性、信号/噪声共同缩放不变性、帧末计数、衰落积分和非法物理输入。

第二个功率模板将既有离线平均功率向量归一化为总功率 1。它是说明性假设，不是从某个 P8 会话读取的真实初始 PDP，也不能代表所有项目会话。

## 数值示例的解释范围

在归一化离线平均模板的 15 dB 示例中，计入帧末截断后的整体额外信息 BER 下界由数学初始时刻约 `1.18e-7` 变为一个相干时间后的约 `4.81e-5`。这些数值不等于合法接收机 BER，不作为 P6/P8 的真实下界报告，也不足以证明目标不可达。均匀模板给出的量级不同，进一步说明 PDP 与帧末边界必须明确，不能只凭总功率或任意 Perfect-CSI 参考判断问题难度。

当前 P8 只保存完整 CIR 范数等评分标量，没有逐帧 tap 能量，因此可以在完整结果后算更宽松的全能量下界，无法还原精确的逐比特帧尾下界。若后续判断确实需要更紧的边界，应在新的独立协议中明确保存相应评分标量；不会重放旧 seed 或回写为 P8 已观察到的结果。

## 复核

以下命令只做确定性数学核验，不运行接收会话：

```powershell
.venv-gpu/Scripts/python.exe scripts/verify_fading_genie_bound.py
.venv-gpu/Scripts/python.exe scripts/verify_finite_frame_bound.py
.venv-gpu/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_physical_ber_bounds.py
```

评分函数位于 `evaluation/physical_ber_bounds.py`，没有接入当前接收机或 P8 比较入口。
