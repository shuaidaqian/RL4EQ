# 贝叶斯递归 CIR 估计：3 seed × 16 帧开发诊断（2026-11-06）

状态：**开发机制诊断，不是正式确认。** Data 标签只用于事后 BER；真实 CIR 只用于 NMSE/oracle 诊断。估计器只使用 acquisition 已知符号、Adapt Pilot 已知符号、接收波形与模型 soft 判决。

## 目的

离线饱和诊断显示 CIR 估计误差是 5/10 dB 的主要剩余误差；简单 Data LS / Pilot ridge 因欠定、相位残差和错误传播失败。本诊断检验一个有原则的贝叶斯递归估计器：

- acquisition 帧构造归一化 CIR 先验均值与协方差；
- 每帧先做 Adapt Pilot 后验（已知符号），再做 Data 后验（模型 soft 判决）；
- Data 行噪声 `R_t = sigma^2 + sum_l |h_l|^2 Var_{t-l}`，错误符号自动降权；
- 后验协方差跨帧持久并加过程噪声 `Q=1e-4`；协方差收缩对应"收益随帧增长后饱和"。

## 协议

- 强离线 checkpoint：`D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`。
- 配置：`configs/eme_long_memory_v2_online_dd_tail_v1.json`（`dd_soft` tail、`alpha=1.0`）。
- 开发 seed：95001、95002、95003；每 seed/SNR 16 连续帧。
- Frozen：acquisition CIR；Bayes：每帧 Pilot→Data 后验，用于下一帧。
- 两者共用同一 acquisition 先验、同一 dd_soft tail 规则与同一波形。Data 标签不进入估计。

## 结果

| SNR | Frozen BER | Bayes BER | 配对增益（Frozen−Bayes） | 3 seed 增益 | NMSE Frozen→Bayes |
|---:|---:|---:|---:|---|---:|
| 0 dB | 0.214817 | 0.185086 | **+0.029731 ± 0.0532** | 0.00065 / 0.03703 / 0.05151 | 0.1357→0.0727 |
| 5 dB | 0.052626 | 0.045654 | **+0.006972 ± 0.00518** | 0.00496 / 0.00610 / 0.00985 | 0.0478→0.0206 |
| 10 dB | 0.008219 | 0.006375 | **+0.001845 ± 0.00134** | 0.00155 / 0.00138 / 0.00260 | 0.0169→0.0076 |
| 15 dB | 0.001465 | 0.001438 | +0.000027 ± 0.000095 | 0 / 0.00008 / 0 | 0.0069→0.0040 |

区间为 3 seed 的描述性 95% 区间（t 临界值 4.303，n=3），不是确认区间。**四个 SNR 均为 3/3 seed 同向为正或非负。**

逐帧平均增益（Frozen−Bayes）：

- 5 dB：`0, +0.0017, +0.0061, +0.0100, +0.0104, +0.0082, +0.0004, +0.0091, +0.0087, +0.0117, +0.0126, +0.0030, +0.0082, +0.0069, +0.0091, +0.0052`；前 5 帧增长，之后在约 `0.008–0.012` 饱和。
- 10 dB：`0, 0, +0.0039, -0.0009, +0.0013, +0.0013, +0.0022, +0.0009, +0.0026, +0.0009, +0.0030, +0.0048, +0.0035, +0.0017, +0.0009, +0.0035`；约第 3 帧起转正并小幅波动上升。
- 0 dB：逐帧受 session 方差主导（有 `-0.0195`、`+0.2435` 等大幅波动），但 3 seed 平均为正；不能按单帧解读。
- 15 dB：接近零，错误数太少。

## 判断

1. **这是本轮第一个可重复的在线物理状态增益**：四个 SNR 全部 3/3 seed 同向；5 dB 点估计 `+0.0070`、10 dB `+0.0018`，NMSE 降低约 55%–57%。
2. **5/10 dB 的增益已达到因果 oracle 真实 CIR 的水平**：3×16 因果 oracle 为 5 dB `-0.00705`、10 dB `-0.00109`，Bayes 为 `+0.00697`、`+0.00185`。说明在 5/10 dB，Bayes 估计器基本吃到了 CIR 的可修复头room。
3. **5/10 dB 的逐帧轨迹出现"先增长、后饱和"**，与用户的时间增长假设方向一致；但这是物理状态的增长，不是神经参数的成长。
4. **归因边界**：本结果的 Frozen 使用 acquisition CIR，Bayes 使用在线后验；因此它是**物理状态增益**，不是神经参数微调增益。正式核心主张要求 Frozen 与 Online 共享同一物理跟踪，再由神经参数提供额外增量。
5. **下一步必须做共享 Bayes CIR 的四臂**：
   - Frozen：Bayes CIR + 冻结网络；
   - Online：Bayes CIR + 低维神经状态/head 修正；
   - 只有 Online 相对共享 Bayes Frozen 仍有可重复正增量，才能宣称神经微调有效。
6. 3 seed × 16 帧仍属开发筛查；0 dB 的 session 方差很大，需要 30–35 个 session；5 dB 需 10–15、10 dB 20–25、15 dB 10–15（128 帧/session）才能做确认级检验。

## 复现

```powershell
python scripts/diagnose_bayesian_cir.py --config configs/eme_long_memory_v2_online_dd_tail_v1.json --checkpoint D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --seeds 95001 95002 95003 --frames 16 --process-noise 1e-4 --ridge 1e-2 --device cuda --output docs/results/bayesian_cir_diag_v1_20261106/bayes_cir_q1e4_r1e2.json
```

配对差与逐帧汇总见 `summary.json`。定向测试 `tests/test_bayesian_cir.py` 4 项通过。
