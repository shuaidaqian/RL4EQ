# P11：匹配老化先验下的长程历史增量开发比较

预登记新seed237301–237306，四SNR、每会话11,719帧、三臂，共843,768条逐帧记录，观察约1,200秒。当前尚未完成。

本轮把P9已验证的当前Pilot老化对照延长到P8原时域，保留acquisition Frozen锚点，并与递归Pilot比较。递归算法、8次物理求解、checkpoint、合法噪声估计、channel/Pilot和同步方法固定；只比较完整接收策略中是否带入历史Pilot后验。各臂soft-tail及其后续输入依各自输出自然演化。

所有预登记窗口均从acquisition后第一帧Data开始：`[0,1)`, `[1,4)`, `[4,16)`, `[16,64)`, `[64,256)`, `[256,1024)`, `[1024,2048)`, `[2048,4096)`, `[4096,8192)`, `[8192,11719)`。全部会话结束及完整审计通过前，不生成BER报告，不依中途BER、诊断或运行速度选方法或停实验。原始JSONL本地保留，不进入Git。

P8只有每SNR两个会话，且current_pilot未使用老化先验；P9有六个会话但只覆盖约105秒。P11补这两项之间的证据缺口，但六会话仍属开发集，不是最终独立确认。它不预先建立增长实质阈值、饱和容差或保持判据，不显著不能说等价，Frozen退化也不能代替Online自身改善。

启动命令（已预登记，不得重跑覆盖）：

```powershell
.venv-gpu/Scripts/python.exe scripts/compare_continual_variants.py --manifest docs/protocols/continual_v2_matched_history_long_dev_20261120.json --seeds 237301 237302 237303 237304 237305 237306 --frames 11719 --experiment p11_matched_history_long_dev_6x11719 --output docs/results/continual_v2_matched_history_long_dev_20261120/matched_long_6x11719.json --device cuda --stream-rows --long-diagnostics --checkpoint-interval 128
```
