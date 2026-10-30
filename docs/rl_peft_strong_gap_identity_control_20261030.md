# 更强 CFO/相位失配下的 RL-PEFT 对照（2026-10-30）

## 结论

强失配压力下，在线 PEFT 的平均 BER 在四个 SNR 都略低于 identity-only 对照，但 **5 个 seed 的配对 95% 区间在四个 SNR 上仍全部跨零**。因此，本轮没有证明在线微调能稳定提升 BER；它也没有验证“失配越强，在线增益就越大”。

本轮只测试参数微调路线：离线 checkpoint 固定，Adapt Pilot 更新内部 `phase_trend` 参数，更新目标为 Pilot 信号重构；Reward Pilot 做逐帧验收和窗口回滚，Data 波形与标签只在在线操作全部结束后用于仿真评估。

## 配置与配对检查

- Level B `eme_long_memory_v2`，delay 116，gap=0，prefix Pilot 总长 256（Adapt 224、Reward 32）；四个 SNR 为 0/5/10/15 dB，每个 5 个 seed、每个 seed/SNR 60 帧。
- 强失配由 `heldout_edge` 覆盖为 CFO 绝对值 0.001–0.003 周期/符号、phase noise 标准差 0.001–0.003；实际样本 CFO 范围为 -0.0017 到 0.0025，phase-noise 标准差为 0.0011 到 0.0024。
- 两组共 1200 帧，SNR/seed/frame 键、输入哈希和信道状态元数据逐项完全一致。identity-only 组 1200 帧均执行 identity，PEFT 与调制参数变化范数均为 0。
- 两组均没有在线使用 Data 波形或标签；策略 observation 全部来自 Adapt Pilot。离线 checkpoint SHA256 固定为 `8CD3912D1C957C9689C51730AFF9D27A8F74942186C802A7483E5781AF65AEC4`。
- 这是 diagnostic-only pressure test，不纳入主平均。

## BER 对照

增益按 `identity-only BER − 完整 PEFT BER` 计算，正值代表 PEFT 较好；置信区间以 5 个 seed 的配对差值计算 t 区间。

| SNR | identity-only BER | 完整 PEFT BER | 平均增益 | 正收益 seed | 95% 配对区间 |
|---:|---:|---:|---:|---:|---:|
| 0 dB | 21.1489% | 21.1376% | +0.0113 pp | 3/5 | [-0.4059, +0.4285] pp |
| 5 dB | 4.9657% | 4.8390% | +0.1267 pp | 4/5 | [-0.0898, +0.3432] pp |
| 10 dB | 0.7083% | 0.6897% | +0.0187 pp | 3/5 | [-0.0392, +0.0766] pp |
| 15 dB | 0.1628% | 0.1554% | +0.0074 pp | 4/5 | [-0.0030, +0.0177] pp |

5 dB 的平均增益最大，为 0.1267 个百分点，4/5 个 seed 为正，但区间仍包含零。其余 SNR 的平均收益约 0.007–0.019 个百分点，也都不确定。与此前 tiny 失配对照相比，这一轮没有显示各 SNR 的增益一致变大。

## 参数更新是否真的生效

完整策略在 1200 帧中提议 PEFT 更新 492 次；Reward Pilot 当帧接受 255 次、拒绝 237 次，窗口层再回滚 29 次。最终保留下来的更新为 247 次。以 Data 作事后分析，这 247 次更新中 Data BCE 有 68.8% 改善，平均 logits 变化为 0.151、平均 soft output 变化为 0.0041；单帧硬 BER 变好 105 次、变坏 39 次、持平 103 次。

这说明 PEFT 参数路径确实改变了 Data 连续输出，且大多数被保留更新改善 Data BCE；但大量帧的硬 BER 不变，跨帧累计后整体 BER 收益仍小且不显著。Data 指标没有参与在线 observation、目标、Reward、动作选择或回滚。

## 下一步

暂不扩大正式 seed/frame 矩阵，也不重训离线模型。下一步应在同一强失配设置上做小规模 Pilot-only 候选 replay：固定 `phase_trend` 参数组，用互不重叠的 Adapt 子窗分别生成候选和筛选候选，Reward Pilot 独立验收；比较保守/当前/较强的更新强度，并在留出 seed 上看 Pilot 分数能否正确排序候选。只有排序跨 seed 可靠、且最终 BER 改善稳定，才值得扩大统计矩阵。全程不读取 Data 波形或标签做在线决策。

完整机器可读结果见 [`docs/results/rl_peft_strong_gap_identity_control_20261030.json`](results/rl_peft_strong_gap_identity_control_20261030.json)。原始逐帧日志保存在本机 `logs/`（该目录被 Git 忽略），路径为：

- `logs/rl_peft_light_strong_gap0_heldout_edge_0_5_10_15db_5s60f_seed1091_1095_20261030/`
- `logs/rl_peft_light_strong_identity_control_gap0_heldout_edge_0_5_10_15db_5s60f_seed1091_1095_20261030/`
