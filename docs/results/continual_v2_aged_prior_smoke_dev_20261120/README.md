# P9a：匹配老化先验的真实入口核验

新seed237001，四SNR、四臂、每会话8帧，共128行、32个配对帧，原进程正常退出。完整schema3审计通过；四SNR首帧的错误数、实际CIR评分、soft评分、先验/后验迹及合法残差等保存字段均精确一致。逐位logits和CIR的一致性已由接收路径单元测试覆盖。整合后全量测试为324 passed in 31.62s。

证据：[完整审计](smoke_1x8_audit.json)、[首帧匹配](first_frame_matched_audit.json)、[全部窗口](smoke_1x8_report.md)、[源码字节](source_snapshot.json)。原始JSONL本地保留，不重复运行该seed。

旧current_pilot_static与匹配老化current_pilot_aged_ar1分别标记；后续历史信息增量主要比较后者和recursive_pilot_ar1，并保留acquisition Frozen整体增益。各臂自己的同步与soft-tail轨迹可能不同，因此该差值仍是完整递归接收策略的增量，不直接拆为纯CIR因果效应。

本次8帧仅用于真实入口、首帧匹配和保存链路核验，不据短会话BER选择方法、不作增长、饱和或神经贡献证明。四SNR完整目标及独立确认均未完成。
