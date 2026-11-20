# P8a：schema 3 真实入口核验

首次实验 `p8a_long_diagnostics_smoke_dev_1x8`，seed 236001，源码提交 `eab6c6e`，在首帧接收完成后的事后评分阶段失败。评分误将加载器默认挂载的恒等 `OnlinePhaseTrendAdapter` 一律拒绝，因而没有写出性能记录（row_count=0）。[失败主结果](smoke_1x8.json)及 seed 登记原样保留，不复用该 seed、不覆盖为修复后结果。

实现已将这个经过审计的确定性旋转纳入实际解码预处理评分。新增恒等与非零相位趋势适配器测试，全量 268 passed。修正不改变接收算法或在线状态更新。

后续独立入口核验使用新 seed 236002、CPU、四 SNR、8 帧、Frozen/当前 Pilot/修复 AR(1) 三臂。见[新协议](../../protocols/continual_v2_long_diagnostic_smoke_v2_dev_20261120.json)。进程正常退出，全部 96 行、32 个配对帧、12 个 session-arm 组完成并审计通过，seed registry 已记 completed，仍为 development。

schema 3 的状态、同步、波形及原有合法观测字段全部为有限标量，108 行 Adapt 观测、噪声来源、计数、波形哈希与同帧物理评分配对通过。配置、checkpoint、冻结动态参数和运行源码指纹匹配。接收步骤均值：Frozen 16.52 ms、当前 Pilot 23.17 ms、AR(1) 24.27 ms；用于长程资源规划，包含短样本启动效应，不含事后评分及落盘，不能作为端到端实时测量。未按 BER 调整接收参数或选择方法；一个短开发会话不证明总体性能、增长或饱和。

- [修复后主结果](smoke_v2_1x8.json)
- [完整逐帧结果](smoke_v2_1x8.jsonl)
- [审计及逐会话预登记窗口汇总](smoke_v2_1x8_audit.json)

源码提交 `01c94ff`。仅重建审计、不运行接收或复用 seed：

```powershell
python scripts/audit_long_diagnostics.py --result docs/results/continual_v2_long_diagnostic_smoke_dev_20261120/smoke_v2_1x8.json
```

后续长程诊断另用 seeds 236101–236102，独立预登记；不会把本次短会话并入长程或确认数据。

完成后分析链路验证：使用本轮完整96行生成[全部预登记窗口报告](smoke_v2_1x8_report.md)和[逐会话配对汇总](smoke_v2_1x8_paired_summary.json)。这只是8帧入口核验，不是长程研究结论。另按原运行全部指纹保存[source_snapshot.json](source_snapshot.json)；对当前工作树与归档字节两种审计方式，逐会话所有数值完全一致。归档不执行其中代码，未来接收源码升级后可显式使用 `--source-snapshot` 复核，不需要重跑旧seed。
