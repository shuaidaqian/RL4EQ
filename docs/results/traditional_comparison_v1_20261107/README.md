# 传统基线公平比较 v1（2026-11-07，2026-11-19 修正 CFO 先验）

状态：**开发对照，不是正式确认。** 所有方法共享同一 episode、同一接收波形、同一 Pilot 预算（256：Adapt 224 / Reward 32）与同一 acquisition CIR/CFO；传统方法不使用神经网络/RL，不读取 Reward/Data 标签。

> **2026-11-19 修正**：旧版 `TraditionalPhaseState` 的 CFO 状态从 0 开始，DD-Phase 跟踪器首帧只补偿一半估计值，而神经两臂直接使用 acquisition CFO，与“共享 acquisition CIR/CFO”的表述不符。现以同一 acquisition CFO 初始化传统 DD-Phase 状态，再按当前帧 Pilot 平滑跟踪；同 seed 重跑。旧表见 git 历史。

## 协议

- 配置：`configs/eme_long_memory_v2_online_dd_tail_v1.json`；强离线 checkpoint SHA256 `8cd3912d1c957c9689c51730aff9d27a8f74942186c802a7483e5781af65aec4`。
- 开发 seed：130001–130003；每 seed/SNR 16 帧；0/5/10/15 dB。
- Frozen：acquisition 先验 CIR + dd_soft tail + 冻结神经模型。
- Bayes：同一先验 → Pilot/Data 后验 CIR + dd_soft tail + 冻结神经模型。
- 传统：`CFO+DD-Phase LMMSE-FIR`、`CFO+DD-Phase DFE-RLS`、`SC-FDE-MMSE`；同一 acquisition CIR、合法 Adapt Pilot/判决反馈，DD-Phase 状态以同一 acquisition CFO 初始化。
- 公平性检查（脚本自动记录）：同帧 rx hash 一致；Frozen 与传统共享同一 acquisition CIR；传统 CFO 先验为 acquisition（逐行记录 `acquisition_cfo_cycles_per_symbol`）；传统 `uses_neural_network=False`、`uses_reward_or_data_labels=False`。

## 结果

| SNR | Frozen | Bayes | CFO+DD-Phase LMMSE-FIR | CFO+DD-Phase DFE-RLS | SC-FDE-MMSE |
|---:|---:|---:|---:|---:|---:|
| 0 dB | 0.214572 | **0.181424** | 0.412842 | 0.406114 | 0.490641 |
| 5 dB | 0.052544 | **0.049344** | 0.135634 | 0.165392 | 0.485677 |
| 10 dB | 0.008138 | **0.007270** | 0.032986 | 0.028781 | 0.484185 |
| 15 dB | 0.001519 | **0.001383** | 0.011475 | 0.008328 | 0.482829 |

Bayes 相对 Frozen 的 Data BER 改善：0 dB `−0.033148`、5 dB `−0.003200`、10 dB `−0.000868`、15 dB `−0.000136`（与修正前一致，神经两臂未改动）。

逐 (seed, frame) 配对差（传统 − Bayes / 传统 − Frozen；正值表示神经/物理更好）：

| SNR | LMMSE-FIR − Bayes / − Frozen | DFE-RLS − Bayes / − Frozen | SC-FDE − Bayes / − Frozen |
|---:|---:|---:|---:|
| 0 dB | +0.231418 / +0.198269 | +0.224691 / +0.191542 | +0.309218 / +0.276069 |
| 5 dB | +0.086290 / +0.083089 | +0.116048 / +0.112847 | +0.436334 / +0.433133 |
| 10 dB | +0.025716 / +0.024848 | +0.021512 / +0.020643 | +0.476915 / +0.476047 |
| 15 dB | +0.010091 / +0.009956 | +0.006944 / +0.006809 | +0.481445 / +0.481310 |

每个 SNR 48 对配对；全为正。

## 成本（当前 GPU，逐帧均值）

| 方法 | 平均耗时 | 迭代数 | 备注 |
|---|---:|---:|---|
| Frozen | ≈422 ms | 12（4 展开 + 8 warm-start） | 含 dd_soft tail 前向 |
| Bayes | ≈460 ms | 12 | 比 Frozen 多约 38 ms（Pilot/Data 后验） |
| CFO+DD-Phase LMMSE-FIR | ≈279 ms | 41 | |
| CFO+DD-Phase DFE-RLS | ≈791 ms | 224 | 与神经模型同量级 |
| SC-FDE-MMSE | ≈1.3 ms | 1024 | 未做 CFO/相位补偿，BER 接近随机 |

102.4 ms 帧预算下，当前神经/物理分支**不实时**；传统 LMMSE 更快但 BER 差约一个数量级。

## 判断

1. **Frozen/Bayes 在四个 SNR 都显著优于三种传统基线**；同波形、同 Pilot、同 acquisition 下，Bayes−Frozen 的额外物理增益也成立。
2. **强传统基线是 CFO+DD-Phase LMMSE-FIR / DFE-RLS**；两者在 5/10/15 dB 的 BER 约为神经的 2–15 倍。`SC-FDE-MMSE` 在 `run_traditional_equalizer` 中未做 CFO/相位补偿，接近随机，只能作为弱对照。
3. **CFO 先验的影响**：3×16 上修正后传统基线全部不劣于旧版，10/15 dB 改善最明显（LMMSE `0.0413→0.0330`、`0.0167→0.0115`；DFE `0.0359→0.0288`、`0.0174→0.0083`）；0 dB 基本不变。5×64 扩展对照中 0/5 dB 方向相反（见该归档），但四 SNR 的结论都不变。
4. **成本是新的瓶颈**：神经/物理分支约 422–460 ms/帧，DFE-RLS 约 791 ms/帧，LMMSE 约 279 ms/帧；当前 GPU 上均不实时。若用于实时链路，需要模型剪枝/蒸馏、批量推理或 CIR 后验的低秩/递归加速。
5. 本对照是 3 seed × 16 帧开发；传统方法使用 `run_traditional_equalizer` 默认实现，DD-Phase 状态以 acquisition CFO 初始化，未加载 clean 分支未包含的 frozen calibration 资产。正式报告可恢复校准资产并在确认 seed 上做同波形对照（不改变确认主假设）。

## 公平性记录

```text
same_episode_for_all_methods = true
traditional_uses_neural_network = false
traditional_uses_reward_or_data_labels = false
acquisition_cir_shared_by_frozen_and_traditional = true
traditional_phase_cfo_prior = acquisition
pilot_total = 256, pilot_layout = prefix
```

## 复现

```powershell
python scripts/compare_traditional_baselines.py --config configs/eme_long_memory_v2_online_dd_tail_v1.json --checkpoint D:/Research/02_研究点二_神经均衡/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --seeds 130001 130002 130003 --frames 16 --device cuda --output docs/results/traditional_comparison_v1_20261107/comparison.json
```

## 边界

- 3 seed × 16 帧开发对照；不是正式确认。
- 传统方法使用代码默认实现；未加载额外 frozen calibration。
- `SC-FDE-MMSE` 无 CFO/相位补偿，不能代表强传统基线。
- 成本为当前 GPU 的逐帧均值，未做优化；不把开发对照写成确认。
- 2026-11-19 的 CFO 先验修正使用同一开发 seed 重跑，属于实现修正而非新假设；旧实现数字保留在 git 历史。
