# P9：匹配物理动态的历史信息增量开发比较

已预登记新seeds237101–237106，四SNR、1024连续帧、三臂，共73,728行。当前尚未形成完整结果。原始JSONL本地保留且Git忽略，完成后登记指纹与完整审计。

acquisition Frozen保留整体增益锚点；current_pilot_aged_ar1每帧从原acquisition按累计间隔预测，仅更新当前合法Adapt；recursive_pilot_ar1使用相同冻结rho和D，但累计历史普通帧Pilot。各臂相同合法估噪、解码和soft-tail规则，自己的同步与tail轨迹仍可能改变。配对差解释为完整递归策略的历史信息增量，不直接称纯CIR中介因果效应。

本轮观察104.8576秒，检验匹配对照下的历史信息价值及其跨会话差异。P8已揭示1200秒末段BER回升，本轮较短时域不重新验证或消除该问题，也不缩短最终目标。未冻结实质差、饱和容差或总体确认功效，不以不显著证明保持，不以Frozen退化替代Online自身改善。

[预登记协议](../../protocols/continual_v2_matched_history_dev_20261120.json)固定首帧起的六个连续窗口。全部会话结束后再分析，不依中途BER改方法、窗口、帧数或停止。P9a真实入口与首帧匹配全部通过后才启动。神经权重冻结，当前物理解码分支不证明离线神经贡献。

启动命令（登记后不得重复执行或覆盖）：

```powershell
.venv-gpu/Scripts/python.exe scripts/compare_continual_variants.py --manifest docs/protocols/continual_v2_matched_history_dev_20261120.json --seeds 237101 237102 237103 237104 237105 237106 --frames 1024 --experiment p9_matched_history_dev_6x1024 --output docs/results/continual_v2_matched_history_dev_20261120/matched_6x1024.json --device cpu --stream-rows --long-diagnostics --checkpoint-interval 128
```
