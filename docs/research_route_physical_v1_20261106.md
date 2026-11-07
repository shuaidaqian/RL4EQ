# 研究路线（物理版 v1）：离线神经均衡器 + 在线贝叶斯 CIR 适配

## 0. 地位

- 本分支 `physical-bayes-cir-v1` 是**物理算法快照**，本文件是当前唯一事实源。
- 历史 PEFT/RL/子空间实验保留在 `codex/online-peft` 与 `docs/route-v4-20261106` 分支；本快照不保留旧实验文档与脚本。
- 机器可读状态见[research_route_v4_20261106.json](protocols/research_route_v4_20261106.json)；确认协议见[confirmation_physical_v1_20261106.md](protocols/confirmation_physical_v1_20261106.md)。

## 1. 核心主张

**离线学习神经块均衡器；在线用合法观测做贝叶斯物理状态（CIR）适配；相对同源 acquisition-CIR Frozen 网络，获得可归因、可重复的 Data BER 增益；并证明该增益随同一会话传输帧数增加而总体扩大，直到饱和。**

- Frozen：同源 checkpoint + acquisition CIR + 同一 dd_soft tail 规则；神经权重冻结。
- Bayes：同一 checkpoint + 贝叶斯递归 CIR 后验（acquisition 先验 → Pilot 后验 → Data soft 后验）；神经权重冻结。
- 本快照**不宣称**"在线神经参数微调"；该子主张在开发中未成立，另行处理。

## 2. 方法

### 2.1 共享物理状态

- `dd_soft` soft-tail：用当前帧最后 `max_delay` 个 Data 位置的模型 soft 判决递推下一帧 tail，与离线课程训练的 `online_soft` 对齐；只读接收波形与模型输出，不读 Data 标签。
- acquisition CIR：用已知 acquisition 帧和 CFO 网格估计；Frozen 与 Bayes 的起点相同。
- 相位/CFO：由 Adapt Pilot 与当前 CIR/tail 估计，供神经 conditioner 与 CIR 更新使用。

### 2.2 贝叶斯递归 CIR

[train/bayesian_cir.py](../training/bayesian_cir.py)：

1. acquisition 帧构造归一化 CIR 先验均值与协方差 `Sigma_acq`。
2. 每帧 Adapt Pilot 已知符号做 Pilot 后验（已知观测，噪声为接收噪声）。
3. Data 位置用模型 soft 判决 `tanh(logit/2)` 做后验；行噪声
   `R_t = sigma^2 + sum_l |h_l|^2 Var_{t-l}`，
   低 SNR 的错误符号自动降权，不按真实 SNR 调度。
4. 后验协方差跨帧持久并加过程噪声 `Q`；协方差收缩对应"随帧增长后饱和"。
5. 更新后的 CIR 只用于下一帧，保持因果。

### 2.3 信息边界

- 在线只读：已接收波形、Adapt Pilot 已知符号、模型 soft 输出、Pilot 驱动的 CIR/CFO/相位/tail 估计。
- 禁止在线读取：Data 标签、真实 CIR/相位/CFO、Reward Pilot 回流内循环、未来帧标签。
- 真实 CIR 只用于事后 NMSE/oracle 诊断；Data 标签只用于事后 BER。

## 3. 独立确认 v1（2026-11-06）

独立 seed 100001–100010，10 session × 128 帧；开发 seed 90001–99010 全部排除；配置、超参数（`Q=1e-4`、ridge `1e-2`）、checkpoint 在运行前冻结。

| SNR | Frozen BER | Bayes BER | ΔBER | 单侧 95% 下界 | Holm p | 负向 seed |
|---:|---:|---:|---:|---:|---:|---:|
| 0 dB | 0.2081 | 0.1713 | **+0.03680** | +0.03029 | 1e-5 | 0/10 |
| 5 dB | 0.0479 | 0.0399 | **+0.00803** | +0.00627 | 2e-5 | 0/10 |
| 10 dB | 0.00623 | 0.00426 | **+0.00197** | +0.00146 | 6e-5 | 0/10 |
| 15 dB | 0.000985 | 0.000724 | **+0.000260** | +0.000149 | 0.001 | 0/10 |

- **H1（四 SNR 等权主效应）通过**：`+0.011767`，单侧下界 `+0.010188`，p=`1.26e-7`。
- **H2（逐 SNR 安全护栏）通过**：四 SNR 下界均 >0，Holm 校正 p 均 <0.05，10/10 seed 无负向。
- **H3（增长）部分通过**：5/10/15 dB 的 16 帧窗口斜率与末窗减首窗为正；**0 dB 斜率 `-0.001238/bin`，按预注册判据不通过**（增益大且稳定，但第 2 窗后早期饱和/波动）。
- NMSE：`0.132→0.038 / 0.052→0.017 / 0.023→0.0065 / 0.014→0.0035`。
- 完整结果与逐 seed/窗口见[确认归档](results/bayesian_cir_confirmation_v1_20261106/README.md)。

## 4. 边界与未完成项

- 0 dB 的 H3 不通过；2-seed Data 后验/相位变体筛查显示 0 dB 的增益是早期收敛、后段饱和+晚窗为正，不是增长；5/10/15 dB 才支持增长。若仍要求 0 dB 增长，需要新的机制或新的预注册判据。
- 同帧 EM/block_phase 的 BER 使用同帧更新后的 CIR，与已确认的跨帧因果 Bayes 不同，需单独标注或改为跨帧。
- 成本 profile：模型前向 ≈400 ms、tail 独立前向 ≈400 ms、Pilot/Data 后验合计 ≈30 ms；tail 复用与独立前向逐位一致，把总耗时从 ≈830 ms 降到 ≈435 ms（4.2× 预算）。模型前向是剩余瓶颈；当前不实时。
- 传统基线公平比较已完成 3×16 开发：Frozen/Bayes 四 SNR 显著优于 CFO+DD-Phase LMMSE/DFE-RLS；SC-FDE 无相位补偿，仅作弱对照；确认级对照待扩样本。
- 神经参数微调子主张未成立，不在本快照的确认范围内。
- 确认集只运行一次；不得在看到结果后调参或换 seed 重测。

## 5. 开发证据（设计依据，不并入确认）

| 规模 | seed | 0 dB | 5 dB | 10 dB | 15 dB |
|---|---:|---:|---:|---:|---:|
| 3×16 | 95001–95003 | 3/3 同向 | +0.00697 | +0.00185 | 中性 |
| 5×32 | 97001–97005 | +0.0318 | +0.00561 | +0.00141 | 中性 |
| 10×64 | 98001–98010 | +0.0351 | +0.00575 | +0.00155 | +0.00013 |
| 10×128 | 99001–99010 | +0.0223 | +0.00615 | +0.00220 | +0.00049 |

## 6. 复现

```powershell
$env:PYTHONUTF8=1
python -m pytest -q -p no:cacheprovider

# 物理确认（独立 seed；本分支已冻结配置）
python scripts/diagnose_bayesian_cir.py --config configs/eme_long_memory_v2_online_dd_tail_v1.json --checkpoint D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --seeds 100001 100002 100003 100004 100005 100006 100007 100008 100009 100010 --frames 128 --process-noise 1e-4 --ridge 1e-2 --device cuda --output docs/results/bayesian_cir_confirmation_v1_20261106/confirmation.json

# tail 协议诊断 / 离线饱和诊断
python scripts/diagnose_tail_protocol_alignment.py --config configs/eme_long_memory_v2_online_dd_tail_v1.json --checkpoint D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --seeds 90001 90002 90003 --frames 6 --device cuda --output docs/results/tail_alignment_v1_20261106/tail_protocol_alignment.json
```

## 7. 传统基线公平比较（2026-11-07）

[对照归档](results/traditional_comparison_v1_20261107/README.md)（3 seed × 16 帧，seed 130001–130003）：

- 同一 episode、同一接收波形、同一 Pilot 预算（256：Adapt 224 / Reward 32）与同一 acquisition CIR/CFO；传统方法不读 Reward/Data 标签、不使用神经网络/RL。
- 四 SNR 的 Data BER：

| SNR | Frozen | Bayes | CFO+DD-Phase LMMSE-FIR | CFO+DD-Phase DFE-RLS | SC-FDE-MMSE |
|---:|---:|---:|---:|---:|---:|
| 0 dB | 0.214572 | **0.181424** | 0.436551 | 0.406359 | 0.490641 |
| 5 dB | 0.052544 | **0.049344** | 0.149495 | 0.170546 | 0.485677 |
| 10 dB | 0.008138 | **0.007270** | 0.041314 | 0.035862 | 0.484185 |
| 15 dB | 0.001519 | **0.001383** | 0.016710 | 0.017442 | 0.482829 |

- Bayes−Frozen：0/5/10/15 dB `−0.0331/−0.0032/−0.00087/−0.00014`。
- 逐 (seed, frame) 配对差（传统−Bayes / 传统−Frozen）全部为正；48 对/SNR。
- 成本（当前 GPU，逐帧均值）：Frozen ≈795 ms、Bayes ≈829 ms、LMMSE ≈268 ms、DFE-RLS ≈761 ms、SC-FDE ≈2 ms；102.4 ms 帧预算下不实时。
- 强传统为 CFO+DD-Phase LMMSE/DFE；`SC-FDE-MMSE` 未做 CFO/相位补偿，接近随机，仅作弱对照。
- 公平性检查：同帧 rx hash 一致、Frozen/传统共享同一 acquisition CIR、传统 `uses_neural_network=False`、`uses_reward_or_data_labels=False`。
- 开发对照；传统方法使用代码默认实现，未加载 clean 分支未包含的 frozen calibration 资产。

## 8. 0 dB Data 后验变体（2026-11-07）

[变体筛查归档](results/diagnose_0db_data_posterior_v1_20261107/README.md)（2 seed × 16 帧，seed 140001–140002；同帧重解码，非跨帧因果）：

| SNR | Frozen | soft | oracle | EM2 | EM3 | block_phase | EM+block |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 dB | 0.186808 | 0.168457 | 0.168050 | **0.162557** | 0.164144 | 0.166585 | 0.169312 |
| 5 dB | 0.052327 | **0.047526** | 0.047648 | 0.048462 | 0.049276 | 0.047892 | 0.048584 |
| 10 dB | 0.006673 | **0.005941** | 0.006063 | 0.006144 | 0.006144 | 0.006022 | 0.006185 |
| 15 dB | 0.001017 | 0.001221 | 0.001180 | 0.001221 | 0.001180 | 0.001221 | 0.001180 |

- 0 dB 的 2-seed 增益：soft `+0.0184`、oracle `+0.0188`、EM2 `+0.0243`（seed `0.0395/0.0090`，异质）、EM3 `+0.0227`、block_phase `+0.0202`（`0.0151/0.0254`，同向）、EM+block `+0.0175`。
- **0 dB 的 H3 斜率仍为负**：soft `−0.0152`、oracle `−0.0109`、EM2 `−0.0110`、block `−0.0135`；但第四窗已转正（`+0.010 ~ +0.025`）。
- **0 dB 的形状是“早期收敛 → 第三窗负 → 第四窗正”**，不是增长；**0 dB 应写成早期收敛/后段饱和且晚窗为正，5/10/15 dB 才是增长**。
- 5/10 dB 的 soft 最好；EM2/EM3 略差；15 dB 两个 seed 的 Bayes 都略差于 Frozen，属于 session 噪声。
- 误差 lag-1 约 0；EM 的收益来自“更新 CIR 后重解码”，不是误差相关模型。
- 成本：soft/block ≈1.7 s/帧、EM2 ≈2.1 s、EM3 ≈2.8 s；102.4 ms 帧预算下不实时。
- 同帧重解码与已确认的跨帧因果不同；正式使用需明确时序或改为跨帧。

## 9. 成本 profile（2026-11-07）

[profile 归档](results/profile_bayes_cost_v1_20261107/README.md)（1 seed × 4 SNR × 8 帧，当前 GPU）：

| SNR | forward | pilot | data | tail_fwd | total_fwd | total_reuse | 预算比 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 dB | 413 ms | 8.6 ms | 19.3 ms | 402 ms | 842 ms | 441 ms | 4.3× |
| 5 dB | 400 ms | 9.6 ms | 21.2 ms | 395 ms | 826 ms | 431 ms | 4.2× |
| 10 dB | 401 ms | 9.7 ms | 20.4 ms | 400 ms | 831 ms | 431 ms | 4.2× |
| 15 dB | 402 ms | 10.4 ms | 21.7 ms | 408 ms | 841 ms | 434 ms | 4.2× |

- tail 复用与独立前向**逐位一致**（`tail_match=True`），总耗时 ~830→~435 ms；零精度 2×。
- 模型前向 ~400 ms 是剩余瓶颈；Pilot/Data 后验合计 ~30 ms；102.4 ms 下约 4.2×。
- 优先级：tail 复用（立即）→ 模型前向（warm-start/层数/batch/蒸馏）→ 后验。

## 10. 文档

- [确认协议](protocols/confirmation_physical_v1_20261106.md)
- [确认判据草案](protocols/confirmation_criteria_v4_draft_20261106.md)
- [机器可读状态](protocols/research_route_v4_20261106.json)
- [贝叶斯 CIR 开发归档](results/bayesian_cir_diag_v1_20261106/README.md)
- [独立确认归档](results/bayesian_cir_confirmation_v1_20261106/README.md)
- [传统基线公平对照](results/traditional_comparison_v1_20261107/README.md)
- [0 dB Data 后验变体](results/diagnose_0db_data_posterior_v1_20261107/README.md)
- [成本 profile](results/profile_bayes_cost_v1_20261107/README.md)
- [tail 协议对齐](results/tail_alignment_v1_20261106/README.md)
- [功效与错误构成](results/power_error_composition_20261106/README.md)
- [离线饱和/oracle](results/offline_saturation_20261106/README.md)
- [方法调研](results/method_survey_20261106/README.md)
- [离线训练与基线](offline_equalizer_training_and_baseline_20261105.md)
