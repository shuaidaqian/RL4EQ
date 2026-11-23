# P5 四 SNR 独立确认（运行中）

状态：**running**，2026-11-23 10:42 启动。协议与判据见 [`docs/protocols/p5_confirmation_20261123.json`](../../protocols/p5_confirmation_20261123.json) 和 [P5 框架草案](../../protocols/p5_confirmation_framework_draft_20261123.md)。

## 设计

- 三臂：`frozen_acquisition`、`current_pilot_aged_ar1`、`recursive_pilot_ar1`；同一 checkpoint、同波形。
- 每 SNR 11,719 帧；四个 SNR 用互不重叠的新 seed（250001–250700 中选出 527 个）：
  - 0 dB：269，5 dB：69，10 dB：73，15 dB：116。
- seed 按保持窗 `[8192,11719)` 相对平台窗的真实 ESNR 漂移匹配零均值对称目标；只使用信道生成器，不读取 BER。
- 运行实现：三臂 batch 前向 + CUDA Graph（`--accel graph`）、`--minimal-diagnostics`（schema 4）、checkpoint 间隔 512 帧。
- 启动前证据：三臂 1,024 帧等价性（BER 逐帧一致、logits 差 ≤3.2e-6）、11,719 帧三臂预检（35,157 行完整、波形 0 错配）、1,024 帧中断续跑验证、全量 359 tests passed。

## 输出

| SNR | 实验名 | 主 JSON | 原始流（本地，不进入 Git） |
|---:|---|---|---|
| 0 | `p5_confirmation_snr0_20261123` | `snr0.json` | `snr0.jsonl` |
| 5 | `p5_confirmation_snr5_20261123` | `snr5.json` | `snr5.jsonl` |
| 10 | `p5_confirmation_snr10_20261123` | `snr10.json` | `snr10.jsonl` |
| 15 | `p5_confirmation_snr15_20261123` | `snr15.json` | `snr15.jsonl` |

启动与续跑：`scripts/run_p5_confirmation_20261123.ps1`；完成后四个实验的 registry 状态由 `scripts/finalize_registry.py` 置为 completed。

## 预登记分析

- 主族：四 SNR `adv_current = (B−O)_plateau − (B−O)_early`，Holm，双侧 α=0.05。
- 次族：0/5 dB `own_gain = O_plateau − O_early` 超过 0.005 绝对最小效应，单侧 Holm。
- 第三族：保持 `O_hold − O_plateau` 的 TOST（0/5 dB 0.002 绝对；10/15 dB 10% 相对），Holm。
- 描述性：`adv_frozen`、10/15 dB 自身改善、真实 ESNR 漂移；10/15 dB 自身改善仍受开发门槛限制，不宣称确认。
- 工具：`scripts/audit_p5_confirmation.py`（完整性）与 `scripts/analyze_p5_confirmation.py`（统计）。两者均流式读取，不在内存中保留全部逐帧数据。

## 中断与恢复

- 任一实验中断：保留主 JSON/JSONL 与 registry；使用相同参数加 `--resume` 续跑，不覆盖、不换 seed。
- 若主 JSON 的 `row_count` 小于 JSONL 行数（写入间隙被杀），将 JSONL 截断到 `row_count` 行后再续跑；丢失的帧会在重放前缀时重算。
- 不自动补 seed、不基于中途 BER 停止或延长。

## 报告

全部四个实验 `completed` 且审计通过后，运行审计与预登记分析，生成会话级结果、精确 p 值/区间与最终报告；在此之前不发布部分确认结论。
