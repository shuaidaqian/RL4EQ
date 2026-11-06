# Data 位置软边缘化 ELBO 方向诊断（2026-11-06）

状态：**开发机制诊断，不是正式确认。** Data 标签只用于事后 Data BCE/BER 与 oracle 方向；ELBO 本身只读接收波形、模型 soft 输出与估计 CIR。

## 目的

Adapt Pilot BCE 在 10/15 dB 已接近零，无法提供可迁移到 Data 的梯度。本诊断检验：Data 位置软边缘化模型似然（ELBO）是否提供一个与 Data 判决改善同向的更新方向，尤其是在高 SNR。

ELBO 定义见[training/data_aided.py](../../../training/data_aided.py)：

```text
L = E_q[||y - H s||^2] / sigma^2
  = ||y - H mu||^2 / sigma^2 + sum_l |h_l|^2 Var_{t-l} / sigma^2
mu = tanh(logit/2),  Var = 1 - mu^2
```

`H` 使用 Pilot 驱动估计 CIR，不使用真实 CIR；不读取 Data 发送符号或 Data 比特。

## 协议

- 强离线 checkpoint：`D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`。
- 配置：`configs/eme_long_memory_v2_online_dd_tail_v1.json`（`dd_soft` tail、`alpha=1.0`）。
- 开发 seed：90001、90002；每 seed/SNR 6 连续帧；Frozen 模型。
- 三个 head 梯度方向：Adapt BCE、Data BCE（oracle，不可部署）、ELBO。
- 一次固定半径 head 步进：`w ← w − r · g/||g||`，`r=0.01/0.05`；动作后重新计算 Data BCE/BER。
- 统计均为逐帧平均；2 seed × 6 帧，仅作机制方向筛查。

## 结果

| SNR | cos(ELBO, Data) | cos(Adapt, Data) | r | ELBO 步 ΔBCE | ELBO 步 ΔBER | Oracle 步 ΔBER | Adapt 步 ΔBER |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 dB | −0.118 | +0.770 | 0.01 | +1.04e-4 | −1.09e-4 | −3.26e-4 | −1.09e-4 |
| 0 dB | −0.118 | +0.770 | 0.05 | +5.38e-4 | +1.09e-4 | −9.77e-4 | +1.09e-4 |
| 5 dB | +0.148 | +0.458 | 0.01 | +4.7e-5 | −1.09e-4 | −1.09e-4 | −1.09e-4 |
| 5 dB | +0.148 | +0.458 | 0.05 | +2.49e-4 | +1.09e-4 | 0 | −1.09e-4 |
| 10 dB | +0.137 | −0.047 | 0.01 | −6e-6 | 0 | 0 | 0 |
| 10 dB | +0.137 | −0.047 | 0.05 | −3.0e-5 | +1.09e-4 | +1.09e-4 | 0 |
| 15 dB | +0.244 | −0.008 | 0.01 | −1e-6 | 0 | 0 | 0 |
| 15 dB | +0.244 | −0.008 | 0.05 | −6e-6 | 0 | 0 | 0 |

每帧 ELBO 梯度范数均值约 `0.123/0.081/0.021/0.0009`（0/5/10/15 dB）。

## 判断

1. **ELBO 在高 SNR 确实携带 Data 相关方向**：5/10/15 dB 的 `cos(ELBO, Data)` 约 `+0.15/+0.14/+0.24`，而 Adapt/Data 在 10/15 dB 约 `−0.05/−0.01`。这支持"Pilot 梯度消失后，Data 侧仍有可用信息"。
2. **但 ELBO 单独不够强**：15 dB 的 ELBO 梯度范数只有约 `9e-4`；一次固定半径步进没有稳定 BER 收益；0 dB 的 ELBO 方向与 Data 负相关。
3. **ELBO 更适合作为辅助项或 CIR 精化的目标**：它提供的不是可直接替代 Pilot 的全局 head 方向，而是 Data 位置的物理残差信息。离线饱和诊断显示 CIR 估计误差在 5/10 dB 占约 20%–33% 相对 BER，远大于 head 的逐帧可修空间。
4. **不能据此宣称在线增益**：这是 2 seed × 6 帧的机制筛查；没有跨 session 重复性，也没有随帧增长证据。

## 下一步

- 用 Data-aided CIR 精化直接作用于 CIR 估计误差；对照 Frozen / Pilot-only CIR / Data-aided CIR / Oracle CIR。
- 若 CIR 精化闭合真实 CIR 差距，再把 ELBO 作为状态更新的辅助目标；若不能，检查置信阈值、错误传播和 CIR 可辨识性。
- 仍保持同波形、同 Pilot、同物理跟踪；Data 标签只做事后评分。

## 复现

```powershell
python scripts/diagnose_data_aided_elbo.py --config configs/eme_long_memory_v2_online_dd_tail_v1.json --checkpoint D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --seeds 90001 90002 --frames 6 --radius 0.01 --device cuda --output docs/results/data_aided_elbo_diag_v1_20261106/elbo_diag_r001.json
python scripts/diagnose_data_aided_elbo.py --config configs/eme_long_memory_v2_online_dd_tail_v1.json --checkpoint D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --seeds 90001 90002 --frames 6 --radius 0.05 --device cuda --output docs/results/data_aided_elbo_diag_v1_20261106/elbo_diag_r005.json
```
