# EME 测量约束信道冻结记录

## 结论与适用范围

正式 Level B 配置冻结为 `B-core`，profile 名为 `eme_measurement_v1`。冻结发生在任何 Proposed、神经网络或 RL 结果产生之前；选择证据仅来自两种传统方法和信道统计。正式主平均只包含 `B-core`，SNR 固定为 0/5/10/15 dB，Pilot 固定为 128 bit 前缀。

`B-sanity` 只用于 clean sanity check，不进入 Level B 主平均；`B-anomaly` 只用于异常散射体诊断；`B-upper` 定义为 Level C pressure profile，也不进入 Level B 主平均。

## 证据来源与数字化边界

- 受版本控制的正式冻结证据：`data/eme/calibration/eme_measurement_channel_calibration_2026-08-26/summary.json`。
- 该文件从本地校准产物 `logs/eme_measurement_channel_calibration_2026-08-26/summary.json` 机械复制；冻结时两者均为 18061 byte 且逐字节一致。正式证据不依赖被 `.gitignore` 排除的 `logs/` 目录。
- 正式冻结证据 SHA-256：`db174fa0d988ae24c2e2109ed6385f8baf56db5701328d573afcd603e0bd3961`。
- 候选定义：`configs/eme_measurement_channel_candidates.json`。
- 外部物理约束：Evans 1965，DOI `10.6028/jres.069d.195`，Fig. 8；仓库内数字化数据为 `data/eme/evans_1965_fig8_envelope.csv`，来源清单为 `data/eme/reference_manifest.json`。
- Fig. 8 只被人工数字化为稀疏的归一化回波上下包络锚点。11.6 ms 端点是论文正文给出的物理雷达深度支撑边界，不是 Fig. 8 的额外定量观测点；3.6 cm 上包络在端点保持最后观测值 -22.5 dB，68 cm 下包络在 -40 dB 处按右删失处理。
- 这些包络约束不构成 1.296 GHz 的精确功率时延分布（PDP）。`strong_path_count` 和 `diffuse_energy_ratio` 的所有范围都是 configured candidates，不是 EME 直接测量值。
- 仿真采用 2000 symbol/s 的等效 1 sample/symbol 表示，因此 `sample_rate_hz = symbol_rate_hz = 2000`。11.6 ms 支撑映射到 `D = 24` 个离散时延 sample；这不是对射频前端采样过程的复现。

## 校准命令与规模

中等校准的可复现命令为：

```powershell
.\.venv-gpu\Scripts\python.exe scripts/calibrate_eme_measurement_channel.py --config configs/eme_measurement_channel_candidates.json --seeds 0 1 2 3 4 --frames 100 --snrs 0 5 10 15 --output-dir logs/eme_measurement_channel_calibration_2026-08-26
```

校准覆盖 4 个候选、5 个 seed、每个 seed 100 frame、4 个 SNR，以及 2 种传统方法。每个候选/方法/SNR 的 BER 分母为 192000 个 Data bit。传统方法为：

- `CFO+DD-Phase LMMSE-FIR`
- `CFO+DD-Phase DFE-RLS`

两种方法只使用 acquisition/Adapt Pilot、接收信号、基于 Pilot 的 CFO 初始化和传统 DD 相位/自适应规则。Data 标签只用于最终 BER 仿真评估，不用于适配。该校准没有运行或读取任何 Proposed、神经网络或 RL 结果。

## 四项候选

| 候选 | 强径数范围 | 弥散能量比范围 | 相干时间/s | 异常散射体 | impairment | CFO 限制 | 校准角色 | 冻结后用途 |
|---|---:|---:|---:|---|---|---:|---|---|
| B-sanity | [3, 5] | [0.03, 0.08] | 240 | 否 | clean | 0.0012 | sanity | Level B sanity，不进主平均 |
| B-core | [3, 7] | [0.05, 0.15] | 120 | 否 | cfo_phase_tiny | 0.0012 | main | Level B 正式主平均 |
| B-anomaly | [3, 7] | [0.05, 0.15] | 120 | 是 | cfo_phase_tiny | 0.0012 | 校准时 main 候选 | 仅 diagnostic，不进主平均 |
| B-upper | [4, 7] | [0.10, 0.20] | 60 | 是 | cfo_phase_tiny | 0.0012 | 校准时 main 候选 | Level C pressure，不进主平均 |

候选参数的语义统一为 `configured_modeling_candidates_not_direct_eme_measurements`。

## 固定选择顺序

冻结前已经固定且未按 Proposed 结果调整的选择顺序为：

1. `physical_delay`：必须保持 11.6 ms 物理支撑和 `D=24`。
2. `support_stability`：强径支撑在在线帧间不得发生非预期跳变。
3. `cross_frame_error`：跨帧卷积历史必须连续。
4. `frame_lag_correlation`：实测 lag magnitude 应接近由帧时长和相干时间确定的目标 rho。
5. `envelope_rmse`：在前述硬约束满足后比较数字化包络 RMSE。
6. `traditional_ber`：最后检查传统方法难度是否适合主研究场景，不以任何 Proposed 结果反向挑选信道。

## 完整信道统计

`rho target = exp(-0.256 / coherence_time_seconds)`；`rho 误差`为 `|lag magnitude - rho target|`。`support changes` 统计 5 seeds × 100 frames，`cross error` 为跨帧 impulse 检查误差。

| 候选 | D | support changes | taps90 | diffuse | rho target | lag magnitude | rho 误差 | RMSE/dB | cross error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B-sanity | 24 | 0 | 2.572 | 0.056032300 | 0.998933902 | 0.999152554 | 0.000218652 | 4.489477 | 0 |
| B-core | 24 | 0 | 4.250 | 0.162078047 | 0.997868941 | 0.997859273 | 0.000009667 | 1.982971 | 0 |
| B-anomaly | 24 | 0 | 3.410 | 0.080311142 | 0.997868941 | 0.998626820 | 0.000757880 | 2.568524 | 0 |
| B-upper | 24 | 0 | 4.166 | 0.115823733 | 0.995742423 | 0.997014985 | 0.001272562 | 2.574158 | 0 |

## 完整传统 BER

以下数值由受版本控制的 `summary.json` 程序化读取后转录；每行均为 192000 个 Data bit。

| 候选 | SNR/dB | CFO+DD-Phase LMMSE-FIR | CFO+DD-Phase DFE-RLS |
|---|---:|---:|---:|
| B-sanity | 0 | 0.30364583333333334 | 0.3103385416666667 |
| B-sanity | 5 | 0.11977083333333334 | 0.127515625 |
| B-sanity | 10 | 0.0211875 | 0.02246875 |
| B-sanity | 15 | 0.0033229166666666667 | 0.004203125 |
| B-core | 0 | 0.3505520833333333 | 0.36172916666666666 |
| B-core | 5 | 0.18785416666666666 | 0.19925520833333332 |
| B-core | 10 | 0.06636458333333334 | 0.07319270833333333 |
| B-core | 15 | 0.026505208333333332 | 0.031989583333333335 |
| B-anomaly | 0 | 0.3097552083333333 | 0.3158697916666667 |
| B-anomaly | 5 | 0.17841666666666667 | 0.18847916666666667 |
| B-anomaly | 10 | 0.053234375 | 0.060296875 |
| B-anomaly | 15 | 0.025427083333333333 | 0.031802083333333335 |
| B-upper | 0 | 0.337375 | 0.33959375 |
| B-upper | 5 | 0.20039583333333333 | 0.21040104166666668 |
| B-upper | 10 | 0.121265625 | 0.10784895833333333 |
| B-upper | 15 | 0.064640625 | 0.07459895833333334 |

## 冻结 B-core 的理由

四个候选都满足 `D=24`、支撑变化为 0、跨帧误差为 0 的硬约束。`B-core` 的 lag magnitude 与 rho target 误差仅 `9.6672387e-6`，并具有最低的包络 RMSE（1.982971 dB）。它在 tiny residual CFO 与慢相位扰动下形成比 clean sanity 更有区分度、但不依赖异常散射体强制出现的中等难度传统 BER，因此适合作为唯一 Level B 主配置。

`B-sanity` 的用途是验证 clean 链路基本正确，不能稀释正式主平均。`B-anomaly` 不因一次 Tycho 异常观测就被强制写入每次正式场景；单次观测不足以确定异常散射体的发生概率、幅度和稳定性，因此只保留诊断角色。`B-upper` 具有更短相干时间、更高 configured 弥散范围和异常散射体，是 Level C 压力测试，不参与 Level B 主平均。

## 已知限制与入口状态

- 校准规模是 5 seeds × 100 frames 的中等 Monte Carlo，不代表实测总体置信区间。
- 信道统计可访问仿真 true CIR 以评估模型是否满足配置约束；在线接收机 observation、reward、动作选择和适配不得使用该上界。
- Data 标签只用于离线监督或仿真 BER 评估，本次传统校准的适配过程未使用 Data 标签。
- 校准只覆盖两种带合理 Pilot CFO/相位补偿的传统 baseline，没有使用神经网络、RL 或 Proposed 结果。
- 异常散射体、强径数和弥散能量比是模型候选，不是从 EME 数据直接估计出的概率模型。
- 数字化包络稀疏且带人工读图误差；11.6 ms 端点包含支撑延伸/删失语义，不能被解释成精确观测 PDP。
- `continual_ppo_eme_measurement_v1.json` 保留了 `model`、`main_delays`、`main_snrs`、`rho`、Pilot 和 impairment 等现有入口可读取字段，同时新增物理字段。当前 `pretrain.py`、`online_train.py` 和 `compare.py` 尚未完成 `eme_measurement_v1` 物理字段的端到端透传，不能仅凭该配置声称训练入口已经运行 EME 测量约束信道。
