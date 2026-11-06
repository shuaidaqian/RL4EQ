# 物理 Bayes CIR 在线适配：确认协议 v1（冻结）

状态：**冻结于 2026-11-06，确认 seed 100001–100010。** 开发 seed（90001–99010）全部排除；确认集不得用于调参、选 checkpoint 或改判据。

## 1. 主张

**离线神经均衡器 + 合法在线贝叶斯物理状态（CIR）适配**，相对同源 acquisition-CIR Frozen 网络，获得可归因、可重复的 Data BER 增益，并随会话帧数增加总体扩大直到饱和。

- Frozen：同源 checkpoint + acquisition CIR + 同一 dd_soft tail 规则；神经权重冻结。
- Bayes：同一 checkpoint + 贝叶斯递归 CIR 后验（acquisition 先验 → Pilot 后验 → Data soft 后验）；神经权重冻结。
- 在线只读已接收波形、Adapt Pilot 已知符号、模型 soft 输出与 Pilot 驱动估计；不读 Data 标签或真实 CIR/相位/CFO。

## 2. 假设

- **H1（主效应）**：`mean_s Δ_s > 0`，其中 `Δ_s = mean_sessions(BER_frozen,s − BER_bayes,s)`，四个主要 SNR 等权。单侧 session 级配对检验，α=0.05。
- **H2（安全护栏）**：每个主要 SNR 的 `Δ_s` 单侧 95% 下界 ≥ 0；点估计不得为负。允许退化量为 0。
- **H3（增长）**：每个 SNR 的 16 帧窗口 `gain_w = BER_frozen,w − BER_bayes,w`；会话级斜率与末窗减首窗为正。逐 SNR 报告，不用四 SNR 平均掩盖 0 dB 的饱和/波动。
- 多重比较：H1/H2 的四 SNR 用 Holm 控制；H3 逐 SNR 描述性报告并给出等权平均。

## 3. 设计

- 配置：`configs/eme_long_memory_v2_online_dd_tail_v1.json`（Level B，116 符号记忆，prefix Pilot=256：Adapt=224/Reward=32，0/5/10/15 dB，dd_soft tail `alpha=1.0`）。
- Checkpoint：`D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`，SHA256 `8cd3912d1c957c9689c51730aff9d27a8f74942186c802a7483e5781af65aec4`。
- Bayes 超参数（冻结）：acquisition ridge `1e-2`、过程噪声 `Q=1e-4`、Data 行噪声尺度 `1.0`（`R_t = sigma^2 + sum_l |h_l|^2 Var_{t-l}`）；不按真实 SNR 调度。
- 会话：10 个独立 session（seed 100001–100010），每 session 128 连续帧；Frozen/Bayes 同波形、同 tail 规则、同 checkpoint。
- 统计单位：独立 session；帧与 bit 不冒充独立样本。

## 4. 停止规则

- 若 H1 单侧检验不显著，或任一 SNR 的 H2 下界 <0：报告确认失败，不在同一确认集重测。
- 若数值失败（NaN/OOM）且与方法无关：可用同一 seed 重跑并记录；不得在看到 BER 后换 seed。
- 若需改配置、超参数、checkpoint 或判据：视为新方法，必须用新的确认 seed 重新走确认。
- 确认集只运行一次；开发 seed 不得并入。

## 5. 报告

- 四 SNR 的逐 session BER、配对差、绝对/相对改善、单侧置信界、Holm 校正。
- 16 帧窗口 `gain_w`、会话级斜率、末窗减首窗、稳态；0 dB 单独报告饱和/波动。
- CIR NMSE（Frozen/Bayes/真实 CIR oracle，仅诊断）、后验协方差迹/条件数、Data 有效行数。
- 计算量、每帧延迟、内存；Frozen/Bayes 差异。
- 标签隔离证据与失败/异常会话。
- 神经参数微调子主张不在本确认内；另行协议。

## 6. 运行

```powershell
python scripts/diagnose_bayesian_cir.py --config configs/eme_long_memory_v2_online_dd_tail_v1.json --checkpoint D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --seeds 100001 100002 100003 100004 100005 100006 100007 100008 100009 100010 --frames 128 --process-noise 1e-4 --ridge 1e-2 --device cuda --output docs/results/bayesian_cir_confirmation_v1_20261106/confirmation.json
```

## 7. 开发证据（仅用于设计，不并入确认）

- 3×16（95001–95003）：四 SNR 3/3 seed 同向为正；5/10 dB `+0.00697/+0.00185`。
- 5×32（97001–97005）：5/10 dB `+0.00561±0.00223` / `+0.00141±0.00084`，区间不跨零。
- 10×64（98001–98010）：四 SNR `+0.0351/+0.00575/+0.00155/+0.00013`，区间全部不跨零。
- 10×128（99001–99010）：四 SNR `+0.02227/+0.00615/+0.00220/+0.00049`，区间全部不跨零；0 dB 先增长后饱和，5/10/15 dB 持续增长。

这些 seed 全部属于开发集，确认集从 100001 开始。
