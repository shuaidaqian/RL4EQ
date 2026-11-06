# 数据辅助 CIR 精化四模式诊断（2026-11-06）

状态：**开发机制诊断，不是正式确认。** Data 标签只用于事后 BER；真实 CIR 只用于隔离 oracle 与 NMSE，不回流任何在线状态。`data_cir` 只读接收波形与模型 soft 判决，不读 Data 发送符号或 Data 比特。

## 目的

离线饱和诊断显示 CIR 估计误差是 5/10 dB 的主要剩余误差（相对 BER 约 20%–33%）。本诊断检验：用 Data 位置的模型 soft 判决做加权 LS 精化 CIR，能否把 Frozen 的 CIR 误差降下来并转化为 Data BER 改善；同时与 Pilot-only sparse、真实 CIR oracle 对照。

## 协议

- 强离线 checkpoint：`D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`。
- 配置：`configs/eme_long_memory_v2_online_dd_tail_v1.json`（`dd_soft` tail、`alpha=1.0`）。
- 开发 seed：92001、92002、92003；每 seed/SNR 16 连续帧；Frozen 模型，不更新任何神经参数。
- 四种物理状态：
  - `frozen`：始终使用 acquisition CIR；
  - `pilot_sparse`：`pilot_sparse_cir_update`，Adapt Pilot + 相位补偿 + top-24 稀疏 tap；
  - `data_cir`：`data_aided_cir_update`，Data soft 判决 + Pilot 相位线去旋转 + 加权正规方程；
  - `oracle_cir`：该帧真实 CIR（仅诊断）。
- CIR 更新只用于下一帧，保持因果；同一 `(seed, SNR)` 的波形只生成一次，四模式严格配对。
- `data_cir` 参数：`alpha=0.5`、置信阈值 `0.5`、ridge `1e-3`、`max_paths` 对照为 24。

## 结果

3 seed × 16 帧；ΔBER 为相对同 seed Frozen 的配对差，括号内为 3 seed 的描述性 95% 区间（t 临界值 4.303，n=3，不能当作确认区间）。

| SNR | Frozen BER | pilot_sparse ΔBER | data_cir ΔBER | oracle_cir ΔBER |
|---:|---:|---:|---:|---:|
| 0 dB | 0.213569 | +0.042996 ± 0.0675 | **+0.152832 ± 0.1230** | −0.034261 ± 0.0683 |
| 5 dB | 0.036919 | +0.041504 ± 0.1290 | +0.000353 ± 0.00367 | −0.007053 ± 0.00455 |
| 10 dB | 0.003852 | +0.001356 ± 0.00332 | −0.000570 ± 0.000572 | −0.001085 ± 0.00122 |
| 15 dB | 0.000461 | +0.000081 ± 0.000165 | −0.000027 ± 0.000252 | +0.000054 ± 0.000095 |

CIR NMSE（去掉第 0 帧，均值）：

| SNR | Frozen | pilot_sparse | data_cir | oracle_cir |
|---:|---:|---:|---:|---:|
| 0 dB | 0.1214 | 0.3432 | 0.6107 | 0.00014 |
| 5 dB | 0.0435 | 0.1577 | 0.0315 | 0.00014 |
| 10 dB | 0.0157 | 0.0297 | 0.0069 | 0.00014 |
| 15 dB | 0.0066 | 0.0156 | 0.0025 | 0.00014 |

逐帧轨迹显示：

- 0 dB：`data_cir` 从第 4 帧起逐步发散（第 6 帧均值 0.540，第 15 帧 0.580）；`pilot_sparse` 也明显退化。数据符号错误率约 21%，soft 判决不足以支撑 LS。
- 5 dB：`data_cir` 基本持平；`pilot_sparse` 在第 11 帧后发散（0.157–0.176）。
- 10 dB：`data_cir` 的 CIR NMSE 从 `0.0157` 降到 `0.0069`，但 BER 只有 `−5.7e-4` 且区间跨零；NMSE 改善没有等比例转化为判决改善。
- 15 dB：所有模式接近持平，错误数太少。

## 判断

1. **CIR 头room 是真实的**：真实 CIR 在 0/5/10 dB 都优于 Frozen（0 dB `−0.0343`、5 dB `−0.00705`、10 dB `−0.00109`），与离线饱和诊断一致。
2. **一次性 Data 加权 LS 不能可靠吃到这个头room**：它依赖模型 soft 判决，而 soft 判决与当前 CIR 误差相关；低 SNR 下错误传播使 CIR 发散，高 SNR 下 NMSE 改善不足以转化为 BER。
3. **Pilot-only sparse 也不稳定**：相位补偿 + top-24 稀疏 tap 在 5 dB 后发散；不能用它作为共享物理跟踪的最终版本。
4. **不能据此进入 N-DACC 的简单 Kalman 版本**：如果观测 `r_t = h_ls,t − h_pilot,t` 本身有偏，Kalman 只会平滑一个坏观测。
5. **下一步优先级**：
   - 先做**同帧 ELBO head 适配**：离线饱和显示同帧 oracle head 在 5/10 dB 显著（`−2.34e-3`/`−7.81e-4`），ELBO 与 Data 梯度在 5/10/15 dB 正相关（`+0.15/+0.14/+0.24`）；检验合法 ELBO 能否吃到其中一部分。
   - 若 ELBO 不足，再做**学习式 CIR 估计器**（离线外层用真实 CIR/Data 标签训练去偏与置信度），而不是直接 Kalman 平滑 LS。
   - 保留 `frozen`、`pilot_sparse`、`data_cir`、`oracle_cir` 作为后续四臂的物理侧对照。

## 复现

```powershell
python scripts/diagnose_data_aided_cir.py --config configs/eme_long_memory_v2_online_dd_tail_v1.json --checkpoint D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --seeds 92001 92002 92003 --frames 16 --modes frozen pilot_sparse data_cir oracle_cir --device cuda --output docs/results/data_aided_cir_diag_v1_20261106/cir_modes_a05_t05.json
```

均值与配对差汇总见 `summary.json`。
