# EME 长记忆 v2：同步先验与在线状态阶段性结果

## 1. 本轮修正的研究问题

本轮没有修改 `eme_long_memory_v2` 的信道参数。修正对象是接收机中的两个可审计问题：

1. 长回波 CIR 参考波形发生相消时，低幅度 Pilot 的相位在噪声下不可靠，不能与高幅度 Pilot 等权拟合；
2. residual CFO 是同步后的慢状态，在一个 episode 内应近似恒定，不能把每帧 Pilot 拟合噪声直接当成 CFO 漂移。

## 2. 实现约束

`baseline/synchronization_compensation.py` 现在在传入 CIR 参考波形时，使用参考波形幅度的上半区样本进行相位直线拟合。参考幅度近似恒定时不会改变有效 Pilot 数；参考波形存在深衰落或相消时，低幅度样本不会主导拟合。

神经接收机和传统接收机均可使用 acquisition frame 估计出的 CFO 搜索结果。该结果来自接收端已知的 acquisition 发送序列，不是真实 impairment 参数。每个普通帧仍使用当前帧 Adapt Pilot 更新 phase0 截距；CFO 作为慢状态先验传入 phase feature。Offline NN、Pilot 在线适配和 RL 消融使用同一先验。

在线适配器读取 `Frame.receiver_view()` 中的 `rx_symbols`、`adapt_symbols`、`adapt_mask` 和 `model_region_ids`。它不从隐藏 Data 区域读取发送符号，即使内部损失只选取 Adapt Pilot，也不依赖完整 `tx_symbols`。

## 3. 小规模验证

使用 `eme_long_memory_v2`、`delay=116`、`SNR=10 dB`、1 个 seed、4 帧的诊断结果如下：

| 方法 | 平均 Data BER |
|---|---:|
| CFO+DD-Phase LMMSE-FIR | 0.1122 |
| CFO+DD-Phase DFE-RLS | 0.1127 |
| Offline NN only | 0.2305 |
| Pilot-Driven Online Adaptation | 0.2305 |

该 impaired checkpoint 尚未通过主目标。它证明了接口和状态估计链路可运行，但不能证明神经网络超过传统方法。当前结果应作为失败诊断，而不是论文正结果。

在同一 episode 的独立诊断中，acquisition CFO 估计为 `-0.0004000`，真实信道 episode CFO 为 `-0.0003959`。不使用 acquisition CFO 先验时，局部 Pilot 拟合的 CFO 特征可在连续帧明显漂移；使用先验后，各 block 的 CFO 特征保持在 acquisition 估计值，phase0 则随跨帧相位状态变化。

## 4. clean 长记忆 sanity

`eme_long_memory_v2_clean` 使用相同的 116 符号长记忆、固定 support、跨帧 soft tail 和 120 s 相干时间，但关闭 residual CFO/相位扰动。使用已有 clean 基座、2 个 seed、每个 seed 4 帧的结果为：

| SNR | LMMSE-FIR | DFE-RLS | Offline NN | Pilot Online |
|---:|---:|---:|---:|---:|
| 0 dB | 0.2938 | 0.2979 | 0.2383 | 0.2676 |
| 5 dB | 0.1749 | 0.1915 | 0.0890 | 0.0893 |
| 10 dB | 0.0875 | 0.0898 | 0.0618 | 0.0617 |
| 15 dB | 0.0508 | 0.0511 | 0.0511 | 0.0509 |

这组结果支持“长记忆下非因果神经块均衡器可超过有限记忆 FIR/DFE”的 sanity 结论，但在线相对 Offline 的增益很小，且低 SNR 更新可能有害。因此不能把 clean 结果直接写成 residual CFO 场景的主结论。

## 5. Traditional-only 阶段性冻结

`logs/eme_long_memory_v2_traditional_calibration_2026-08-28_fresh/summary.json` 保存了当前代码的 impaired v2 traditional-only 阶段性校准：5 个 seed、每个 seed 20 帧、4 个主 SNR、10 个传统方法，共 4000 行逐帧记录。摘要 SHA-256 为 `1B3CB7C19D806C1D974C4CDBB047565633057ECD2DAF6141F1E3A9611F128A48`。

其中公平的 `CFO+DD-Phase` 结果为：

| SNR | LMMSE-FIR | DFE-RLS |
|---:|---:|---:|
| 0 dB | 0.4279 | 0.4467 |
| 5 dB | 0.3004 | 0.3042 |
| 10 dB | 0.1882 | 0.1554 |
| 15 dB | 0.1303 | 0.1311 |

该文件用于当前代码阶段的 profile 冻结，不满足正式论文统计的 60/100 帧要求。曾尝试的 60 帧运行发生并发缓存写入，已停止且不引用；正式统计必须在单一进程、独立输出目录中重跑。

## 6. 当前论文判断

当前可辩护的命题仍然是：在公平的前缀 Pilot、有限记忆传统接收机和跨帧 ISI 约束下，在线 Pilot 驱动适配有机会改善固定离线模型；不能声称已经超过所有传统均衡器。`SC-FDE-MMSE` 利用循环结构和整块处理，作为强诊断参考单独报告；若它优于 proposed，应保留该结果并明确接收机边界。

下一步应优先增加 impaired 场景的离线训练覆盖，并报告在线方法相对 Offline NN 的 paired 前后帧变化；不应继续通过放大 CFO、相位噪声或随意修改长回波 profile 制造优势。
