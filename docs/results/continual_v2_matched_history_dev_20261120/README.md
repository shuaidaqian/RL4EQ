# P9：匹配物理动态的历史信息增量开发比较

已预登记新seeds237101–237106，四SNR、1024连续帧、三臂，共73,728行。全部24个SNR-session、72个session-arm组、24,576个配对帧完成；完整schema3、原始源码字节、acquisition、行顺序、比特数、合法估噪和同波形哈希审计通过，重复/缺失/哈希错配均为0。主结果SHA-256 `3f190f272a17d4952be993ae8c41ca71b32f6257616cd2581396365c6168a74a`，JSONL SHA-256 `39c1d04c339bd197ae4dabdc1e8b5e38c2a18a22ce2206e6131d75dc305e247c`。原始JSONL本地保留且Git忽略。

acquisition Frozen保留整体增益锚点；current_pilot_aged_ar1每帧从原acquisition按累计间隔预测，仅更新当前合法Adapt；recursive_pilot_ar1使用相同冻结rho和D，但累计历史普通帧Pilot。各臂相同合法估噪、解码和soft-tail规则，自己的同步与tail轨迹仍可能改变。配对差解释为完整递归策略的历史信息增量，不直接称纯CIR中介因果效应。

递归臂相对老化acquisition单帧先验的全会话BER增量，在每个SNR都六个会话同向；未校正、会话级t区间如下：

| SNR | 全会话 BER 增量均值 | 未校正95% t区间 | 同向会话 |
|---:|---:|---:|---:|
| 0 dB | 0.02668 | [0.02413, 0.02923] | 6/6 |
| 5 dB | 0.01139 | [0.00895, 0.01382] | 6/6 |
| 10 dB | 0.001081 | [0.000488, 0.001673] | 6/6 |
| 15 dB | 0.0000507 | [0.0000223, 0.0000790] | 6/6 |

这支持本开发范围内完整递归策略相对匹配物理时间先验的正增量，但不能把它拆成纯CIR中介效应。Online自身BER必须独立考察：从首帧[0,1)到末窗[256,1024)的首减末变化，在0/5/10/15 dB分别为 `+0.01505 [0.00231,0.02778]`、`+0.00464 [-0.00134,0.01062]`、`−0.000330 [-0.00186,0.00120]`、`−0.0000379 [-0.000635,0.000559]`；6个会话中首窗更差的数量为5/6、5/6、2/6、1/6。5 dB及更高SNR的区间跨零不能证明自身改善；10/15 dB点估计为Online回升。Frozen BER在同一时期明显变差，因此Frozen优势扩大不能替代自身改善。

本轮观察104.8576秒，短于P8的约1200秒，不能重新验证或消除长程末段回升，也不缩短最终目标。未冻结实质差、饱和容差或总体确认功效；不以不显著证明保持。

[预登记协议](../../protocols/continual_v2_matched_history_dev_20261120.json)固定首帧起的六个窗口。全部会话结束后才分析；没有依据中途BER改方法、窗口或帧数。神经权重冻结，当前物理解码分支不证明离线神经贡献。

证据：[全量审计](matched_6x1024_audit.json)、[预登记窗口报告](matched_6x1024_report.md)、[配对变化](matched_6x1024_paired_summary.json)、[完整流核验](completed_stream_integrity.json)、[会话级区间](session_level_paired_intervals.json)、[源码快照](source_snapshot.json)。区间在完成后按6个会话计算，未做四SNR/窗口多重校正，不是预登记确认检验；相关帧不作为独立样本。原始JSONL与运行时源码快照保留在本地，可依据指纹复核。P9仍属development，不构成四SNR目标确认。

启动命令（登记后不得重复执行或覆盖）：

```powershell
.venv-gpu/Scripts/python.exe scripts/compare_continual_variants.py --manifest docs/protocols/continual_v2_matched_history_dev_20261120.json --seeds 237101 237102 237103 237104 237105 237106 --frames 1024 --experiment p9_matched_history_dev_6x1024 --output docs/results/continual_v2_matched_history_dev_20261120/matched_6x1024.json --device cpu --stream-rows --long-diagnostics --checkpoint-interval 128
```
