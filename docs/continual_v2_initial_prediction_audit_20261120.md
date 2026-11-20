# 持续接收机首帧状态预测审计

审计时间：2026-11-20。适用对象为当前 P6 开发实现，不是 BER 性能结论。

## 核查结果

当前生成器在 acquisition 发送结束后已完成一次 CIR 演化，而递归接收机在第一个 Data 帧前没有做状态预测。`acquisition_to_data_gap_seconds=0` 只表示没有额外空档，并不取消发送函数在帧末执行的演化。

调用链为：

1. `CommunicationEnvironment.reset_episode` 调用 `channel.transmit` 发送 acquisition。
2. `ExtremeDelayChannel.transmit` 用当前 CIR 生成 acquisition，保存 `last_cir_used`，随后调用 `_evolve_taps`。因此零额外空档下，首个 Data 波形使用的是演化一步后的 CIR。
3. `estimate_acquisition` 从 acquisition 波形估计对应的 CIR；其协方差尚未包含之后的过程创新。
4. `ContinualReceiver.frames_seen` 初始化为 0，`step` 仅在其为真时执行 `predict_cir_state`。首个 Data Pilot 因而沿用 acquisition 原始先验；第二个 Data Pilot 才开始预测。

定向检查使用合成的三维复先验、`rho=0.8` 和屏蔽隐藏字段后的接收视图，替换 Pilot 更新为记录先验的恒等操作，并隔离解码输出。检查确认首帧先验严格等于 acquisition 原值，第二帧先验满足一次 AR(1) 均值和协方差预测。检查不使用研究会话，不比较 BER。

## 当前参数的量级

冻结离线统计文件指纹与 P6 清单一致，离线 seeds 233001–233010 与在线开发 seeds 234001–234010 不重叠。`rho=0.9999149638055597`，功率向量维数为 117、功率和约为 1.067862671。

- 一步均值缩放相对差为 `1-rho ≈ 8.50e-5`。
- 一步 AR(1) 创新协方差迹为 `(1-rho²) sum(D) ≈ 1.8161e-4`。
- 当前随机游走的一步创新协方差迹为 `117 × 1e-6 = 1.17e-4`。

这些是模型计算量级，不能代替接收性能实验。实际遗漏的协方差变化还包括 AR(1) 的 `(rho²-1)P` 项，不能把创新迹直接当作完整协方差误差。

## 对研究证据的处理

当前 P6 保留既有实现和预登记运行；它仍可比较两种既定接收实现，但不能称作从 acquisition 到首帧均与生成器严格匹配的状态空间滤波。两条递归臂都存在首帧未预测的问题，也不代表这种影响会在配对差中抵消。尤其需要谨慎解释以首帧为参照的自身改善和增益增长。

本轮结束后的 BER、成本报告须明确此初始化限制，以及本轮未记录后验协方差轨迹和合法观测残差的既有限制。不得根据中途结果更换方法、删除会话或把本轮改写为修正后的实验。

后续修正入口：先明确 acquisition 估计所对应的状态时刻，令首个 Data Pilot 前也执行相应的过程预测；为零空档和非零空档分别定义转移间隔，避免额外空档被漏计或重复计数。采用定向时序测试核验预测次数与间隔，再用新的登记开发 seed 验证 BER 轨迹。修正不能靠读取当前会话真实 CIR；当前 P6 运行中不修改其算法源文件，以保留必要时校验续跑的条件。

## 核查源文件指纹

| 文件 | SHA-256 |
|---|---|
| `training/continual_receiver.py` | `4fcc2dee9a70c944c9ed9382d2218f4e9d2dc6e093127cad558b624337074b81` |
| `env/comm_env.py` | `2357ca25c6ab052516de27f9d378982fa8600aff7501f0920f38c97bd5e956d6` |
| `env/extreme_delay_channel.py` | `b65fb8b0191dec2561bf06bac9c04c3c52e6d7fb7d357951fca82004f564bc56` |
