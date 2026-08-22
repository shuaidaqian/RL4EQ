# 迭代记录：修复后 cfo_phase_tiny 20-frame 小矩阵

时间戳：2026-08-13 00:23:36  
分支：`codex/continual-ppo-unfolded-equalizer`  
工作目录：`D:\Research\RL4EQ`

## 1. 本轮目标

上一轮已经修复 CFO/phase traditional compensation 的逐帧 CFO 过估计问题。修复前的 `20-frame cfo_phase_tiny` 矩阵不能继续作为公平结论，因为补偿版传统 baseline 被错误斜率估计人为打残。

本轮目标：

```text
使用修复后的 traditional compensation，
重跑 cfo_phase_tiny 3 delays × 4 SNR × 1 seed × 20 frames 小矩阵，
替代上一轮修复前结果。
```

## 2. 执行命令

使用上一轮已经训练好的 500-step checkpoint，不重新训练：

```powershell
.\.venv-gpu\Scripts\python.exe compare.py --config configs/continual_ppo_cfo_phase_tiny.json --method-group all --pretrained pretrained/cfo_phase_tiny_phasevector_500steps_2026-08-12/model_best.pt --delays 20 30 40 --snrs 0 5 10 15 --num-seeds 1 --frames 20 --pilot-total 128 --pilot-layout prefix --impairment-profile cfo_phase_tiny --output-dir logs/compare_cfo_phase_tiny_phasevector_500steps_compfix_20frames_2026-08-13
```

结果：

```text
saved logs/compare_cfo_phase_tiny_phasevector_500steps_compfix_20frames_2026-08-13
```

耗时约 9 分 20 秒。

输出：

```text
logs/compare_cfo_phase_tiny_phasevector_500steps_compfix_20frames_2026-08-13/frame_metrics.jsonl
logs/compare_cfo_phase_tiny_phasevector_500steps_compfix_20frames_2026-08-13/summary.json
```

共 4080 行，`impairment_profile` 全部为 `cfo_phase_tiny`。

## 3. 修复后方法级均值

| 方法 | mean BER_data | 样本数 |
|---|---:|---:|
| LMMSE-FIR | 0.283767 | 240 |
| CFO-Corrected LMMSE-FIR | 0.283767 | 240 |
| RLS Linear | 0.284071 | 240 |
| CFO-Corrected DFE-RLS | 0.289562 | 240 |
| LMS | 0.314355 | 240 |
| CFO+DD-Phase LMMSE-FIR | 0.322949 | 240 |
| CFO+DD-Phase DFE-RLS | 0.327452 | 240 |
| NLMS | 0.352246 | 240 |
| RL-Modulated Neural Block Equalizer | 0.368056 | 240 |
| Offline NN only | 0.368392 | 240 |
| NN + Fixed Modulation | 0.368392 | 240 |
| NN + Discrete PEFT Scheduler | 0.368392 | 240 |
| NN + Rule Modulation | 0.376584 | 240 |
| Fixed CG-BPSK-DD Block Detector | 0.522732 | 240 |
| SC-FDE-MMSE | 0.524425 | 240 |
| Perfect-CSI Block | 0.537565 | 240 |
| DFE-RLS | 0.538303 | 240 |

补偿器诊断：

```text
CFO-Corrected LMMSE-FIR max_abs_cfo = 0.001000
CFO-Corrected DFE-RLS   max_abs_cfo = 0.001000
CFO+DD-Phase LMMSE-FIR  max_abs_cfo = 0.000938
CFO+DD-Phase DFE-RLS    max_abs_cfo = 0.000987
```

说明修复后的补偿器不再出现 `0.008–0.014` 的不合理 CFO 估计。

## 4. 修复前后差异

| 方法 | 修复前 mean BER | 修复后 mean BER | 变化 |
|---|---:|---:|---:|
| LMMSE-FIR | 0.283767 | 0.283767 | +0.000000 |
| RLS Linear | 0.284071 | 0.284071 | +0.000000 |
| CFO-Corrected LMMSE-FIR | 0.461708 | 0.283767 | -0.177941 |
| CFO-Corrected DFE-RLS | 0.466439 | 0.289562 | -0.176877 |
| CFO+DD-Phase LMMSE-FIR | 0.477593 | 0.322949 | -0.154644 |
| CFO+DD-Phase DFE-RLS | 0.471376 | 0.327452 | -0.143924 |
| RL-Modulated Neural Block Equalizer | 0.368056 | 0.368056 | +0.000000 |
| Offline NN only | 0.368392 | 0.368392 | +0.000000 |

解释：

- 修复只影响 traditional compensation，不影响 Proposed。
- 修复后补偿版传统 baseline 明显变强。
- 修复前“Proposed 超过补偿 DFE”的结论不能作为最终公平结论。

## 5. Proposed 与传统 baseline 差距

`RL-Modulated Neural Block Equalizer` 相对关键 baseline：

```text
RL - LMMSE-FIR:                 +0.084288
RL - RLS Linear:                +0.083984
RL - CFO-Corrected LMMSE-FIR:   +0.084288
RL - CFO-Corrected DFE-RLS:     +0.078494
RL - CFO+DD-Phase LMMSE-FIR:    +0.045106
RL - CFO+DD-Phase DFE-RLS:      +0.040603
RL - DFE-RLS:                   -0.170247
RL - SC-FDE-MMSE:               -0.156369
```

当前结论：

```text
Proposed 超过未补偿 DFE-RLS 和 SC-FDE-MMSE；
但仍弱于 LMMSE-FIR / RLS Linear / 修复后的 CFO-Corrected LMMSE-FIR / CFO-Corrected DFE-RLS。
```

因此项目第一目标尚未达成。

## 6. 逐配置观察

Proposed 并非所有配置都弱于传统。在部分 delay/SNR 子配置中，Offline NN / RL 已经超过 LMMSE/RLS：

```text
d30/s10:
  Offline NN:   0.311458
  RL-Modulated: 0.311458
  LMMSE-FIR:    0.333203
  RLS Linear:   0.333203

d30/s15:
  Offline NN:   0.296615
  RL-Modulated: 0.296615
  LMMSE-FIR:    0.326432
  RLS Linear:   0.326693

d40/s5:
  RL-Modulated: 0.332552
  Offline NN:   0.334115
  LMMSE-FIR:    0.359766
  RLS Linear:   0.360156

d40/s15:
  Offline NN:   0.238021
  RL-Modulated: 0.238021
  LMMSE-FIR:    0.332422
  RLS Linear:   0.332031
```

但在 d20 系列、d30/s0、d30/s5、d40/s0、d40/s10 等配置中，传统 LMMSE/RLS 仍更强。

这说明：

```text
神经接收机已经在某些 Level B impaired 子区间有竞争力；
但主目标要求逐配置明显超过传统 baseline，当前还不成立。
```

## 7. 在线帧趋势

前 5 帧 vs 后 5 帧：

| 方法 | first5 mean BER | last5 mean BER |
|---|---:|---:|
| LMMSE-FIR | 0.172917 | 0.358507 |
| RLS Linear | 0.172092 | 0.359028 |
| CFO-Corrected DFE-RLS | 0.178385 | 0.358377 |
| Offline NN only | 0.311719 | 0.449783 |
| NN + Rule Modulation | 0.299609 | 0.482118 |
| RL-Modulated Neural Block Equalizer | 0.311068 | 0.449783 |

观察：

- 所有方法后 5 帧都比前 5 帧更差，说明该 episode 轨迹后段更难。
- Proposed 没有体现“在线帧数越多优势越明显”。
- `RL-Modulated` 仍几乎等同 `Offline NN only`。

这继续支持之前判断：

```text
不要扩大 PPO；
先增强神经块接收机本体。
```

## 8. 验证

命令：

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests/test_traditional_baselines.py tests/test_evaluation_contract.py::test_compare_cli_writes_real_level_b_metrics tests/test_evaluation_contract.py::test_docs_share_single_research_contract tests/test_evaluation_contract.py::test_default_entrypoints_use_prefix_pilot_and_correct_main_snrs -q -p no:cacheprovider
```

结果：

```text
16 passed in 21.71s
```

## 9. 结论

修复后 20-frame 矩阵替代上一轮修复前矩阵，成为当前 `cfo_phase_tiny` 小样本判断依据。

当前最重要结论：

```text
1. 修复后的传统补偿 baseline 更公平、更强。
2. Proposed 在部分主配置已超过 LMMSE/RLS，但不是逐配置超过。
3. PPO 没有带来在线收益，RL-Modulated 基本等同 Offline NN。
4. 当前论文第一目标尚未达成。
5. 下一步应增强神经本体，尤其显式 phase/CFO correction branch，而不是扩大 PPO。
```

## 10. 下一步建议

建议下一轮实现：

```text
Phase/CFO correction branch for UnfoldedEqualizer
```

最小设计：

- 输入：`condition.latent_residual` 的 phase vector、CIR support/noise、region ids。
- 输出：每帧分段 phase correction，例如 4 或 8 个 segment 的相位偏置，再插值到整帧。
- 在均衡主干前对 `rx_iq` 做可学习相位校正。
- 离线训练通过最终 BER/BCE 反向传播学习。
- 在线 RL 仍只调制安全 Adapter/FiLM/LoRA/head，不直接输出相位轨迹。

目标：

```text
先让 Offline NN only 在 d20/d30/d40 × 0/5/10/15 上接近或超过 LMMSE/RLS；
再判断 PPO 是否能在此基础上进一步降低 BER。
```

