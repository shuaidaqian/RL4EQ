# Bayes CIR 扩样本：30 session × 128 帧（2026-11-12）

状态：**开发扩展筛查，不是正式确认。** 新 seed 170031–170060（30 个独立 session），
不与已用开发/确认 seed 重叠。目标：判断配对增益的波动是偶发噪声还是结构性非单调。

## 协议

- 配置：`configs/eme_long_memory_v2_online_dd_tail_v1.json`；`dd_soft` tail `alpha=1.0`。
- checkpoint：`D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`。
- seed：170031–170060；每 seed/SNR 128 帧；SNR 0/5/10/15 dB。
- Bayes：`Q=1e-4`、`ridge=1e-2`；Frozen 为同源 acquisition CIR。
- 统计单位是 **session（seed）**；窗口为 16 帧，共 8 窗。
- Data 标签只事后 BER；真实 CIR 只事后 NMSE。

## 主结果（30 seed，均值 ± 95% CI，t=2.045）

| SNR | Frozen BER | Bayes BER | 配对增益 | 负向 seed | NMSE Frozen → Bayes |
|---:|---:|---:|---:|---:|---:|
| 0 dB | 0.2114 ± 0.0078 | 0.1812 ± 0.0070 | **+0.03025 ± 0.00494** | 1/30 | 0.1402 → 0.0414 |
| 5 dB | 0.05538 ± 0.00511 | 0.04800 ± 0.00452 | **+0.00738 ± 0.00165** | 1/30 | 0.0556 → 0.0204 |
| 10 dB | 0.008689 ± 0.001725 | 0.006276 ± 0.001319 | **+0.00241 ± 0.00058** | 0/30 | 0.0257 → 0.0073 |
| 15 dB | 0.001535 ± 0.000466 | 0.001100 ± 0.000292 | **+0.000435 ± 0.000235** | 0/30 | 0.0158 → 0.0040 |

**四 SNR 的配对增益 95% CI 都不跨零**；负向 seed 最多 1/30。

## 窗口斜率（每 seed 8 个窗口，再跨 seed 统计）

| SNR | 窗口斜率 | 正斜率 seed | 末窗 − 首窗 | 正差值 seed |
|---:|---:|---:|---:|---:|
| 0 dB | +0.001895 ± 0.002238 | 22/30 | +0.01496 ± 0.01941 | 18/30 |
| 5 dB | **+0.000765 ± 0.000520** | 27/30 | +0.00433 ± 0.00467 | 24/30 |
| 10 dB | **+0.000437 ± 0.000154** | 29/30 | **+0.00320 ± 0.00123** | 27/30 |
| 15 dB | **+0.000111 ± 0.000071** | 25/30 | **+0.000876 ± 0.000573** | 25/30 |

## 结论

1. **增益不是偶发**：四 SNR 的配对增益 CI 都不跨零，30 个独立 session 里负向最多 1 个。
2. **5/10/15 dB 的增长成立**：窗口斜率 CI 都不跨零；10/15 dB 的末窗 − 首窗也显著。
3. **0 dB 有增益，但“增长”不成立**：增益本身 `+0.03025 ± 0.00494` 显著；
   但窗口斜率 `+0.001895 ± 0.002238` 跨零，末窗 − 首窗也跨零，只有 22/30 seed 斜率为正。
   0 dB 的形状跨 seed 不稳定，应报告为“增益存在、形状不稳定”，不能写成增长。
4. 这与之前的 10×128 文件 `0db_criteria_robustness_v1_20261110` 一致：
   0 dB 前三个窗口波动、晚窗升高，不是稳健的早期收敛或后段饱和。

## 对“多跑几次取均值”的回答

- 扩样本让**均值更稳**：增益 CI 收窄、负向 seed 降到 1/30；
- 但**不会把结构性非单调变成单调**：0 dB 的斜率 CI 仍跨零；
- 因此正确报告方式是“session 级均值 + CI + 斜率的 CI”，而不是要求逐窗口单调。

## 证据边界

- 新 seed 170031–170060 是开发 seed，**不是正式确认**；确认集 100001–100010 未重测。
- 统计单位是 session；不能用 bit 级二项分布算 CI（0 dB 设计效应约 90）。
- 本诊断不改变在线只读约束，不引入 Data 标签回流。
- 0 dB 的形状结论只适用于当前配置与 checkpoint。

## 复现

```powershell
$env:PYTHONUTF8=1
python scripts/diagnose_bayesian_cir.py `
  --config configs/eme_long_memory_v2_online_dd_tail_v1.json `
  --checkpoint D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt `
  --seeds 170031 170032 170033 170034 170035 170036 170037 170038 170039 170040 `
          170041 170042 170043 170044 170045 170046 170047 170048 170049 170050 `
          170051 170052 170053 170054 170055 170056 170057 170058 170059 170060 `
  --frames 128 --process-noise 1e-4 --ridge 1e-2 --device cuda `
  --output docs/results/bayesian_cir_expanded_v1_20261112/confirmation.json

python scripts/summarize_bayesian_expanded.py `
  --input docs/results/bayesian_cir_expanded_v1_20261112/confirmation.json `
  --window 16 --output docs/results/bayesian_cir_expanded_v1_20261112/summary.json
```

## 指纹

| 文件 | SHA256 |
| --- | --- |
| `model_best.pt` | `8cd3912d1c957c9689c51730aff9d27a8f74942186c802a7483e5781af65aec4` |
| `confirmation.json` | `263f402be1ff104792f95086d271df90622b0194c21ab9b08b684178960bacb4` |
| `summary.json` | `58f8da5365f7080adfd786952ff06d23000ce712dbdb7fa3412ce1b0700bc9f0` |
