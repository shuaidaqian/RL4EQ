# GPU-only 接收路径计时（开发资源测量）

- 目的：估算拟议 1024 帧四 SNR 筛查的目标接收路径成本；不做 CPU 对照。
- 本次只计时，不计算或记录 BER、NMSE 或其他质量指标；计时数据不属于正式筛查，也不作为研究效果证据。
- 设备、代码/checkpoint 指纹、seed、warm-up 与计时帧数、逐 SNR/逐臂延迟见 `gpu_timing_ar1_only.json`。
- 计时接收状态：recursive Pilot-only 与 recursive soft Data 使用已冻结的 AR(1) 动态参数；acquisition Frozen 仅为时间对照臂。
- 每 SNR 使用 10 帧，其中前 2 帧作为 warm-up，后 8 帧计时；臂为 acquisition Frozen、recursive Pilot-only、recursive soft Data。
- 单步接收延迟使用 CUDA synchronize 后计量；环境帧生成和 acquisition/session 初始化单列记录。
- 单 seed / 8 计时帧仅为粗略资源估算。线性外推到 1024 帧不包括长会话状态变化、IO、准备与审计开销，不可视为正式长跑实测。
- 全程使用新登记的 timing-only seed；不纳入后续筛查 BER 分析。
