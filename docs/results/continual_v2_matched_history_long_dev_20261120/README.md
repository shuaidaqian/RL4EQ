# P11：匹配老化先验下的长程历史增量开发比较

实验现已完成并审计：预登记新seed237301–237306，四SNR、每会话11,719帧、三臂，共843,768条逐帧记录、281,256个配对帧，观察约1,200秒。24个SNR-session均完整；重复、缺失、配对哈希错配和非法字段均为0，源码快照验证通过。

本轮把P9已验证的当前Pilot老化对照延长到P8原时域，保留acquisition Frozen锚点，并与递归Pilot比较。递归算法、8次物理求解、checkpoint、合法噪声估计、channel/Pilot和同步方法固定；只比较完整接收策略中是否带入历史Pilot后验。各臂soft-tail及其后续输入依各自输出自然演化。

所有预登记窗口均从acquisition后第一帧Data开始：`[0,1)`, `[1,4)`, `[4,16)`, `[16,64)`, `[64,256)`, `[256,1024)`, `[1024,2048)`, `[2048,4096)`, `[4096,8192)`, `[8192,11719)`。全部会话结束及完整审计通过前，不生成BER报告，不依中途BER、诊断或运行速度选方法或停实验。原始JSONL本地保留，不进入Git。

P8只有每SNR两个会话，且current_pilot未使用老化先验；P9有六个会话但只覆盖约105秒。P11补足了匹配老化先验下的长时域开发证据，但六会话仍属开发集，不是最终独立确认。全会话平均BER（Frozen / 匹配老化当前Pilot / 递归Pilot）分别为：0 dB `0.27288 / 0.16956 / 0.12337`，5 dB `0.20847 / 0.05961 / 0.02958`，10 dB `0.18238 / 0.01065 / 0.00204`，15 dB `0.17567 / 0.001221 / 0.000378`。递归Pilot相对匹配当前Pilot的窗口优势总体扩大，但窗口并非严格单调；Frozen后段明显恶化，故Frozen相对优势扩大不能替代Online自身BER改善。10 dB Online末窗BER高于首窗，15 dB首帧为零而后续出现错误；本结果没有证明四SNR的Online自身持续改善，也没有建立增长实质阈值、饱和容差、等价或保持判定。神经权重冻结且使用物理解码分支，不证明离线神经闭环训练贡献。

启动命令（已预登记，不得重跑覆盖）：

```powershell
.venv-gpu/Scripts/python.exe scripts/compare_continual_variants.py --manifest docs/protocols/continual_v2_matched_history_long_dev_20261120.json --seeds 237301 237302 237303 237304 237305 237306 --frames 11719 --experiment p11_matched_history_long_dev_6x11719 --output docs/results/continual_v2_matched_history_long_dev_20261120/matched_long_6x11719.json --device cuda --stream-rows --long-diagnostics --checkpoint-interval 128
```


完成证据：全量报告、逐会话配对窗口汇总、schema 3 审计、完整性检查点和源码快照分别见 `matched_long_6x11719_report.md`、`matched_long_6x11719_paired_summary.json`、`matched_long_6x11719_audit.json`、`integrity_checkpoint.json` 和 `source_snapshot.json`。原始逐帧 JSONL 按约定保留在本地且不纳入 Git。
