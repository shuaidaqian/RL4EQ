# 传统基线扩样本公平对照（2026-11-10，2026-11-19 修正 CFO 先验）

状态：**开发扩展筛查，不是正式确认。** 新 seed 170001–170005；不与已用开发/确认 seed 重叠。
所有方法共享同一 episode、同一接收波形、同一 Pilot 预算和同一 acquisition。

> **2026-11-19 修正**：旧版传统 DD-Phase 跟踪器的 CFO 状态从 0 开始，与神经两臂直接使用 acquisition CFO 不对称，也不符合“共享 acquisition CIR/CFO”的表述。现以同一 acquisition CFO 初始化，再按当前帧 Pilot 平滑跟踪；同 seed 重跑。旧表见 git 历史。

## 目的

把此前的 3 seed × 16 帧开发对照扩到 **5 seed × 64 帧**，确认 Frozen/Bayes
相对传统方法的优势不是小样本偶然。

## 协议

- 配置：`configs/eme_long_memory_v2_online_dd_tail_v1.json`；`dd_soft` tail `alpha=1.0`。
- checkpoint：`D:/Research/02_研究点二_神经均衡/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`。
- seed：170001–170005；每 seed/SNR 64 帧；SNR 0/5/10/15 dB。
- 方法：
  - `frozen`：acquisition CIR + 冻结神经模型 + dd_soft tail；
  - `bayes`：同一先验 → Pilot/Data soft 后验 CIR + 冻结神经模型 + dd_soft tail；
  - `CFO+DD-Phase LMMSE-FIR` / `CFO+DD-Phase DFE-RLS` / `SC-FDE-MMSE`：代码默认传统实现，DD-Phase 跟踪器以同一 acquisition CFO 初始化。
- 传统方法使用同一 acquisition CIR/CFO 和合法 Pilot/判决反馈，不读 Reward/Data 标签。
- `SC-FDE-MMSE` 无相位补偿，仅作弱对照。

## 结果（Data BER，均值 ± 95% CI，5 seed，t=2.776）

| SNR | Frozen | Bayes | LMMSE-FIR | DFE-RLS | SC-FDE-MMSE |
|---:|---:|---:|---:|---:|---:|
| 0 dB | 0.197868 ± 0.015726 | **0.174459 ± 0.016411** | 0.439465 ± 0.027417 | 0.444047 ± 0.012094 | 0.497339 ± 0.002821 |
| 5 dB | 0.046016 ± 0.008386 | **0.040019 ± 0.007235** | 0.150720 ± 0.061685 | 0.156213 ± 0.039930 | 0.498246 ± 0.003078 |
| 10 dB | 0.005835 ± 0.002243 | **0.004281 ± 0.001791** | 0.043734 ± 0.015161 | 0.040120 ± 0.012855 | 0.497591 ± 0.003193 |
| 15 dB | 0.000960 ± 0.000526 | **0.000846 ± 0.000515** | 0.022331 ± 0.011314 | 0.017737 ± 0.008178 | 0.497428 ± 0.003565 |

## 结论

- **Frozen/Bayes 在四个 SNR 都显著优于三个传统方法**：BER 低约 2.5×（0 dB）到 26×（15 dB）；CI 与所有传统方法均不重叠。
- Bayes 相对 Frozen 在四个 SNR 的点估计都更低；0 dB `0.1745` vs `0.1979`，
  5 dB `0.0400` vs `0.0460`，10 dB `0.00428` vs `0.00584`，15 dB `0.000846` vs `0.000960`（与修正前一致，神经两臂未改动）。
- 与 2026-11-07 的 3 seed × 16 帧开发对照方向一致，扩样本后差距没有缩小。
- `SC-FDE-MMSE` 接近 0.5，符合“无相位补偿”的预期，只作弱对照。

## CFO 先验修正的影响

- 修正后传统基线在 10/15 dB 变好（LMMSE `0.0455→0.0437`、`0.0237→0.0223`；DFE `0.0428→0.0401`、`0.0198→0.0177`），在 0/5 dB 略差（LMMSE `0.4293→0.4395`、`0.1389→0.1507`；DFE `0.4344→0.4440`、`0.1505→0.1562`）。
- 方向与 3×16 开发对照不完全相同（那里各 SNR 都不劣于旧版），说明 acquisition CFO 先验在低 SNR 会引入估计误差，但幅度不足以改变任何结论。
- 两张表都以“共享 acquisition CFO”为准；旧实现数字保留在 git 历史。

## 证据边界

- 新 seed 170001–170005 是开发 seed，**不是正式确认**；正式确认需预注册、冻结判据并一次性运行。
- 传统方法使用代码默认实现，DD-Phase 以 acquisition CFO 初始化，未加载 clean 分支未包含的 frozen calibration；`SC-FDE-MMSE` 无相位补偿。
- 本对照不改变在线只读约束，不引入 Data 标签回流。
- 成本（本次运行）：Frozen/Bayes ≈450/485 ms/帧，LMMSE ≈286 ms，DFE-RLS ≈839 ms，SC-FDE ≈1 ms；均不实时。

## 复现

```powershell
$env:PYTHONUTF8=1
python scripts/compare_traditional_baselines.py `
  --config configs/eme_long_memory_v2_online_dd_tail_v1.json `
  --checkpoint D:/Research/02_研究点二_神经均衡/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt `
  --seeds 170001 170002 170003 170004 170005 --frames 64 --device cuda `
  --output docs/results/traditional_comparison_confirm_v1_20261110/comparison.json
```

## 指纹

| 文件 | SHA256 |
| --- | --- |
| `model_best.pt` | `8cd3912d1c957c9689c51730aff9d27a8f74942186c802a7483e5781af65aec4` |
| `comparison.json` | `5489a51eca80cd2b332ac62242134f8c97d92a8a2e01364fa471a4d713d396d4` |
