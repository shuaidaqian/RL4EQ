# 传统基线扩样本公平对照（2026-11-10）

状态：**开发扩展筛查，不是正式确认。** 新 seed 170001–170005；不与已用开发/确认 seed 重叠。
所有方法共享同一 episode、同一接收波形、同一 Pilot 预算和同一 acquisition。

## 目的

把此前的 3 seed × 16 帧开发对照扩到 **5 seed × 64 帧**，确认 Frozen/Bayes
相对传统方法的优势不是小样本偶然。

## 协议

- 配置：`configs/eme_long_memory_v2_online_dd_tail_v1.json`；`dd_soft` tail `alpha=1.0`。
- checkpoint：`D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`。
- seed：170001–170005；每 seed/SNR 64 帧；SNR 0/5/10/15 dB。
- 方法：
  - `frozen`：acquisition CIR + 冻结神经模型 + dd_soft tail；
  - `bayes`：同一先验 → Pilot/Data soft 后验 CIR + 冻结神经模型 + dd_soft tail；
  - `CFO+DD-Phase LMMSE-FIR` / `CFO+DD-Phase DFE-RLS` / `SC-FDE-MMSE`：代码默认传统实现。
- 传统方法使用同一 acquisition CIR 和合法 Pilot/判决反馈，不读 Reward/Data 标签。
- `SC-FDE-MMSE` 无相位补偿，仅作弱对照。

## 结果（Data BER，均值 ± 95% CI，5 seed，t=2.776）

| SNR | Frozen | Bayes | LMMSE-FIR | DFE-RLS | SC-FDE-MMSE |
|---:|---:|---:|---:|---:|---:|
| 0 dB | 0.1979 ± 0.0157 | **0.1745 ± 0.0164** | 0.4293 ± 0.0107 | 0.4344 ± 0.0156 | 0.4973 ± 0.0028 |
| 5 dB | 0.0460 ± 0.0084 | **0.0400 ± 0.0072** | 0.1389 ± 0.0380 | 0.1505 ± 0.0381 | 0.4982 ± 0.0031 |
| 10 dB | 0.00584 ± 0.00224 | **0.00428 ± 0.00179** | 0.0455 ± 0.0147 | 0.0428 ± 0.0125 | 0.4976 ± 0.0032 |
| 15 dB | 0.000960 ± 0.000526 | **0.000846 ± 0.000515** | 0.02369 ± 0.01118 | 0.01978 ± 0.00870 | 0.4974 ± 0.0036 |

## 结论

- **Frozen/Bayes 在四个 SNR 都显著优于三个传统方法**，BER 低 1–2 个数量级；
  CI 与所有传统方法均不重叠。
- Bayes 相对 Frozen 在四个 SNR 的点估计都更低；0 dB `0.1745` vs `0.1979`，
  5 dB `0.0400` vs `0.0460`，10 dB `0.00428` vs `0.00584`，15 dB `0.000846` vs `0.000960`。
- 与 2026-11-07 的 3 seed × 16 帧开发对照方向一致，扩样本后差距没有缩小。
- `SC-FDE-MMSE` 接近 0.5，符合“无相位补偿”的预期，只作弱对照。

## 证据边界

- 新 seed 170001–170005 是开发 seed，**不是正式确认**；正式确认需预注册、冻结判据并一次性运行。
- 传统方法使用代码默认实现，未加载 clean 分支未包含的 frozen calibration；`SC-FDE-MMSE` 无相位补偿。
- 本对照不改变在线只读约束，不引入 Data 标签回流。
- 成本沿用 11-07 profile：Frozen/Bayes 约 416/444 ms/帧，传统方法按各自实现。

## 复现

```powershell
$env:PYTHONUTF8=1
python scripts/compare_traditional_baselines.py `
  --config configs/eme_long_memory_v2_online_dd_tail_v1.json `
  --checkpoint D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt `
  --seeds 170001 170002 170003 170004 170005 --frames 64 --device cuda `
  --output docs/results/traditional_comparison_confirm_v1_20261110/comparison.json
```

## 指纹

| 文件 | SHA256 |
| --- | --- |
| `model_best.pt` | `8cd3912d1c957c9689c51730aff9d27a8f74942186c802a7483e5781af65aec4` |
| `comparison.json` | `2e55d193548d609a1f78d3d0a542d974f42d2b278f652cea8b6350ee38c2a088` |
