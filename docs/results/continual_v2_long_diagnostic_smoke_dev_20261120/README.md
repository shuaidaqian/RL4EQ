# P8a：schema 3 真实入口核验

首次实验 `p8a_long_diagnostics_smoke_dev_1x8`，seed 236001，源码提交 `eab6c6e`，在首帧接收完成后的事后评分阶段失败。评分误将加载器默认挂载的恒等 `OnlinePhaseTrendAdapter` 一律拒绝，因而没有写出性能记录（row_count=0）。[失败主结果](smoke_1x8.json)及 seed 登记原样保留，不复用该 seed、不覆盖为修复后结果。

实现已将这个经过审计的确定性旋转纳入实际解码预处理评分。新增恒等与非零相位趋势适配器测试，全量 268 passed。修正不改变接收算法或在线状态更新。

后续独立入口核验使用新 seed 236002、CPU、四 SNR、8 帧、Frozen/当前 Pilot/修复 AR(1) 三臂。见[新协议](../../protocols/continual_v2_long_diagnostic_smoke_v2_dev_20261120.json)。尚未将本页视作完成报告；完整性和资源核验将在全部 96 行完成后归档，短会话不作增长、饱和或性能确认。
