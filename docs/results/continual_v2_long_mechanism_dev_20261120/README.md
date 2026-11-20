# P8：长程状态、同步及物理难度机制开发诊断

[冻结协议](../../protocols/continual_v2_long_mechanism_dev_20261120.json)：新 seeds 236101–236102，四 SNR，每会话 11,719 帧（1200.0256 s），三臂 acquisition Frozen、当前 Pilot、修复 AR(1)，共 281,256 行。设备 CPU；不复用旧开发或确认 seed，不修改信道、Pilot、SNR、checkpoint、递归参数或合法信息边界。神经权重冻结，解码器为 `known_adapt_real_physics`。

P8a 全部 96 行通过 schema 3 的真实入口验证后才冻结本实验。每帧完整处理并持续保存状态与 tail，事后评分不回流。所有预定会话完成后才分析性能，不依中途 BER 调参、选方法或停止。实验身份为 development；不作增长、饱和、保持或神经协同的独立确认。

主要问题是长程 BER 变化是否伴随 CIR 误差、合法残差、预测不确定性、同步残差、真实信道功率及频谱形状变化。伴随关系不等于机制因果；物理难度指标也不是最优 BER 下界。四 SNR 研究目标不因此缩小。

启动命令如下；一旦登记启动，不得重新运行该命令覆盖或重复旧 seed：

```powershell
$env:PYTHONUTF8=1
.venv-gpu/Scripts/python.exe scripts/compare_continual_variants.py --manifest docs/protocols/continual_v2_long_mechanism_dev_20261120.json --seeds 236101 236102 --frames 11719 --experiment p8_long_mechanism_dev_2x11719 --output docs/results/continual_v2_long_mechanism_dev_20261120/mechanism_2x11719.json --device cpu --stream-rows --long-diagnostics --checkpoint-interval 128
```

运行期间仅检查进程及 JSONL 完整性，避免并行读取正在原子替换的主 JSON。长程 JSONL 本地保留、Git 忽略，完成后登记指纹及全量审计；不会将短会话或旧格式记录补写成这次诊断。尚未形成完整结果或机制结论。

启动源码提交 `7d12a3a`。2026-11-20 18:31 +08:00，原工具会话 `78511` 仍在运行；已核验 0 dB seed 236101 的前 3456 个配对帧，共 10,368 行，索引连续、108 行 Adapt、768 Data 比特和配对波形哈希通过。本检查点不分析中途 BER 或机制评分，不改变协议。

2026-11-20 18:44 +08:00，首个 0 dB seed 236101 会话三臂完整结束；11,719 帧、35,157 行审计通过，原进程继续第二个会话。连同第二会话前896帧的固定前缀完整性见[first_session_integrity_checkpoint.json](first_session_integrity_checkpoint.json)，没有生成中途性能汇总。

独立源码审查发现，本轮 current_pilot 使用未计物理老化的 acquisition 原始先验。因此其与递归 AR(1) 的长期差值不能完全归因于历史观测信息积累。匹配动态的老化先验单帧对照已在隔离分支提交 `5dd01dc`，未整合进本轮；本轮保留原臂，后续新对照必须新登记，不回写本结果。
