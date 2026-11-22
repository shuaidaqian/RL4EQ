# 递归 Pilot 后段 BER 回升的只读归因（2026-11-22）

## 状态

- 只读描述性分析，使用已完成的 P8/P9/P11 逐帧流；**不生成新 seed、不改实验代码、不占 GPU、不构成确认或因果结论**。
- 原始数据：[P11](../../results/continual_v2_matched_history_long_dev_20261120/matched_long_6x11719.jsonl)、[P9](../../results/continual_v2_matched_history_dev_20261120/matched_6x1024.jsonl)、[P8](../../results/continual_v2_long_mechanism_dev_20261120/mechanism_2x11719.jsonl)。P11/P8 原始流按约定仅在本地保留。
- 汇总：[`late_rebound_attribution.json`](late_rebound_attribution.json)；脚本：[`scripts/analyze_late_rebound.py`](../../../scripts/analyze_late_rebound.py)。
- 窗口沿用 P11 预登记：`[0,1) … [8192,11719)`；中段取 `[1024,4096)`，晚段取 `[4096,11719)`。真实 CIR/ESNR 只做事后诊断，不进入任何接收路径。

## 方法

按 `(SNR, seed, arm, window)` 聚合错误数、比特数和接收/真值诊断均值；对每个 session 计算晚段减中段的 BER 变化与真实有效 ESNR 变化，再在 6 个开发 session 上做描述性相关。P8/P9 只作交叉参照（P9 仅覆盖前 1,024 帧，无法覆盖晚段）。

## 主要发现

1. **晚段 BER 变化与真实信道轨迹的晚段 ESNR 变化同向**。P11 的 6 个 seed 分成两组：
   - 237301–237304：真实有效 ESNR 晚段下降 0.69–0.80 dB，递归 Pilot 的 BER 全部回升；
   - 237305–237306：真实有效 ESNR 晚段上升 0.56–0.73 dB，递归 Pilot 的 BER 反而下降。
2. **递归 Pilot 的 ΔBER 与 ΔESNR 强负相关**（P11，6 session，描述性）：0 dB `r=−0.93`，5 dB `r=−0.95`，10 dB `r=−0.98`；与 ΔCIR 范数 `r≈−0.95`、与 Δ对齐 NMSE `r≈0.89–0.97`、与 Δ相位 RMS `r≈0.86–0.92`、与 Δ高置信错误率 `r≈0.89–0.92` 同向。15 dB 计数少、变化小，`r=−0.31` 不稳定。
3. **当前 Pilot 臂不是这样**：P11 中即使信道 ESNR 改善的两个 seed，当前 Pilot 的 BER 仍略升；P8 的两个 seed 晚段 ESNR 均上升（+0.78/+0.84 dB），递归 Pilot 在 0 dB 的 ΔBER 为 −0.0206/−0.0080，而当前 Pilot 为 +0.0895/+0.0418。递归臂跟随信道变好/变差，当前 Pilot 臂单向退化。
4. 逐帧诊断显示晚段没有递归滤波器发散迹象：相位 RMS、Pilot 创新、协方差迹在晚段基本平稳，真实 CIR 范数与估计范数同步变化；`data_bce/brier` 与高置信错误率随信道变难而上升。

## 解读与限制

- 这表明 P8/P11 观察到的“Online 自身后段 BER 回升”**主要混杂于有限长度 AR(1) 轨迹在晚段的真实信道变化**（4/6 条轨迹变差），而不是递归估计器本身的失稳；递归估计器在 2/6 条变好的轨迹上表现为 BER 下降。
- 这只是描述性归因：n=6（P11）且呈 4/2 两组，相关系数近似组间对比；P8 只有 2 个 session；真实 ESNR 为事后量；不据此宣称因果或总体效应。
- 10 dB/15 dB 的晚窗绝对错误数仍达数万/数千，不是少量帧驱动；但 15 dB 的 session 内变化小、相关不稳定。
- 同一 seed 的真实信道轨迹跨 SNR 相同，因此四 SNR 是配对设计；信道轨迹差异来自 seed。

## 对研究协议的含义（提议，未冻结）

1. 四 SNR “自身改善/饱和保持”不能只看单组轨迹的绝对 BER 曲线；否则轨迹晚段变差会被误读为方法失效。
2. 确认协议应在**开发阶段**预先规定：使用多条独立信道轨迹并平衡其晚段走向，或在分析中按真实有效 ESNR 分层报告（真值仅事后使用）；不得在确认集上看到曲线后挑选轨迹或窗口。
3. 递归 Pilot 仍应作为主线；当前 Pilot 臂的晚段退化不能作为在线方法的主要对照解释来源。

## 复现

```powershell
.venv-gpu\Scripts\python.exe scripts/analyze_late_rebound.py
```
