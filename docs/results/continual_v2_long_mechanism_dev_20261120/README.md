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

2026-11-20 19:03 +08:00：0 dB 两个会话全部完成；按固定字节前缀审计 82,218 行、27,406 个配对帧，包含两个完整会话与首个 5 dB 会话前3968帧。schema3全部字段、数值有限性、计数、配对波形/物理评分及同会话噪声一致性通过，见[0 dB完整性检查点](zero_db_integrity_checkpoint.json)。原进程继续运行，未生成中途性能汇总。

## 运行源码字节归档与完成后分析

[source_snapshot.json](source_snapshot.json)保存13个源码文件、配置、协议及冻结动态文件，共16个原始 UTF-8 字节内容。全部在运行期间按预登记指纹核验，未执行或替换归档内代码；快照 SHA-256 为 `bb4f7562358ec5e15d038f32f897328066869838d69e5a351c6041f1f1e45b5f`。部分运行文件的 CRLF 与 Git 对象的 LF 不同，快照保留实际字节，可在未来源码升级后显式复核。没有包含 checkpoint 权重，分析时仍需按原指纹核验权重。

完整结果后才可执行以下离线命令。它不运行研究会话，不改 seed registry；未全部完成时拒绝生成性能报告：

```powershell
python scripts/report_long_mechanism.py --result docs/results/continual_v2_long_mechanism_dev_20261120/mechanism_2x11719.json --source-snapshot docs/results/continual_v2_long_mechanism_dev_20261120/source_snapshot.json
```

报告保留全部预登记窗口和逐会话配对差，分别列自身 BER、Frozen BER、增益及相邻窗变化，不作饱和或通过判定。分析代码指纹与运行代码指纹分开保存。当前仅用已完成P8a验证该分析链路，未对本轮生成性能报告。

2026-11-20 19:46 +08:00：0/5 dB各两会话、10 dB首会话完整，共5/8个会话。固定字节前缀177,705行、59,235配对帧审计通过，见[首个10 dB完整性检查点](first_ten_db_integrity_checkpoint.json)。原进程78511继续第二个10 dB会话；全部13项运行源码仍匹配启动指纹。当前不分析中途性能、不读取运行中的主JSON，接收算法和预登记保持不变。
