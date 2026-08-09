# 迭代记录：Reward/Data 相关性、动作空间、PEFT 与 Level B 难度诊断

时间戳：2026-08-02 20:46:31 +08:00

## 1. 本轮目标

本轮执行四个诊断任务：

1. 检查 Reward Pilot loss 改善是否能预测 Data BER 改善。
2. 检查 RL 低维 modulation 动作空间是否存在有效动作。
3. 比较“低维 modulation”与“真实 PEFT 参数更新”。
4. 重新校准 Level B 难度区间，判断传统 baseline 是否真的足够困难。

执行原则：

- 不启动正式 pilot sweep。
- 不启动 10 seeds × 1000 frames 正式矩阵。
- Data 标签只用于离线诊断分析，不进入在线 observation、reward、动作选择或 PPO update。
- 诊断中的 Data 使用不是 Data Oracle baseline，不作为正式方法。

## 2. 新增代码与测试

### 2.1 新增诊断模块

新增：

```text
evaluation/research_diagnostics.py
scripts/diagnose_research_assumptions.py
tests/test_research_diagnostics.py
```

诊断模块提供：

- `summarize_reward_data_correlation()`
  - 统计 Reward Pilot loss 改善与 Data BER 改善的 Spearman 相关性。
- `structured_modulation_candidates()`
  - 构造围绕 identity 的低维 modulation 候选。
- `evaluate_modulation_candidates()`
  - 在同一帧上扫描 modulation 候选，输出 Reward/Data paired 改善。
- `apply_adapt_only_peft_update()`
  - 只用 Adapt Pilot loss 对真实 PEFT 参数做梯度更新。
- `run_level_b_difficulty_scan()`
  - 只扫描传统非神经、非 RL baseline 的 Level B 难度。

### 2.2 防御性修复

测试暴露：

- `copy.deepcopy(model)` 后，参数上的 `_peft_group` 标记可能丢失。
- 这会导致 PEFT 诊断中 `model.trainable_parameters()` 为空。

修复：

- 在 `apply_adapt_only_peft_update()` 内部增加 `_retag_unfolded_peft_groups()`。
- 当 PEFT 组没有可训练参数时，按参数名恢复：
  - conditioner → `conditioner_film`
  - head → `head`
  - adapter → `adapter`
  - attention LoRA → `attention_lora`
  - FFN LoRA → `ffn_lora`

### 2.3 CLI import 修复

测试暴露：

- 直接运行 `scripts/diagnose_research_assumptions.py` 时，Python import path 只有 `scripts/`，找不到 `agent/`、`env/` 等仓库包。

修复：

- 脚本启动时把仓库根目录加入 `sys.path`。

### 2.4 配对 seed 修复

初版 probe 后发现：

- 神经诊断与传统难度扫描使用了不同 env seed offset。
- 这会导致 Offline NN 与传统 baseline 不是同一批信道轨迹，不能直接比较。

修复：

- `run_level_b_difficulty_scan()` 增加 `seed_offset` 参数。
- CLI 中神经诊断和传统扫描统一使用 `seed_offset=90000`。

## 3. 测试结果

新增诊断测试：

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests\test_research_diagnostics.py -q -p no:cacheprovider
```

结果：

```text
5 passed in 11.79s
```

测试覆盖：

- Reward/Data paired improvement 汇总字段。
- modulation 动作扫描字段。
- PEFT 更新只依赖 Adapt Pilot，不依赖 Reward/Data 标签。
- Level B 传统 baseline 扫描不混入神经网络或 RL。
- CLI smoke 能写出 `research_diagnostics.json`。

相关回归测试：

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests\test_research_diagnostics.py tests\test_rl_modulated_equalizer.py tests\test_traditional_baselines.py -q -p no:cacheprovider
```

结果：

```text
18 passed in 57.60s
```

最终全量测试：

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

结果：

```text
109 passed in 245.79s (0:04:05)
```

## 4. Probe 1：宽一点的单 seed / 单帧扫描

命令：

```powershell
.\.venv-gpu\Scripts\python.exe scripts\diagnose_research_assumptions.py --config configs\continual_ppo.json --pretrained pretrained\level_b_probe_from_a_1500_p128_offset\model_best.pt --delays 20 40 --snrs 0 5 10 --pilot-totals 64 96 128 --pilot-layouts prefix two_block --seeds 0 --frames 1 --peft-groups head adapter_lora --peft-steps 1 --peft-lr 1e-4 --device cuda --output-dir logs\research_diagnostics_2026-08-02_probe_paired
```

输出：

```text
logs\research_diagnostics_2026-08-02_probe_paired\research_diagnostics.json
```

关键结果：

```text
Offline NN mean BER_data = 0.053568
Reward/Data Spearman = -0.033193
num_pairs = 360
gate_pass = False
```

低维 modulation 动作：

```text
action_samples = 360
fraction_reward_improved = 0.411111
fraction_data_improved = 0.172222
fraction_both_improved = 0.055556
best_mean_data_action = adapter_gate_1.2
```

PEFT：

```text
adapter_lora:
  mean_reward_loss_improvement = 0.006668
  mean_data_ber_improvement = 0.000542
  fraction_reward_improved = 0.861111
  fraction_data_improved = 0.222222

head:
  mean_reward_loss_improvement = 0.000188
  mean_data_ber_improvement = 0.000145
  fraction_reward_improved = 0.500000
  fraction_data_improved = 0.055556
```

解释：

- 单 seed / 单帧下，Reward/Data 相关性基本为 0，甚至略负。
- 低维动作中确实存在局部有效动作，但“Reward 改善”和“Data BER 改善”高度不一致。
- adapter_lora PEFT 对 Reward Pilot 改善很明显，但 Data BER 改善比例仍低。

## 5. Probe 2：focused 多 seed / 多帧扫描

为了降低单 seed 偶然性，又跑了 focused probe：

```powershell
.\.venv-gpu\Scripts\python.exe scripts\diagnose_research_assumptions.py --config configs\continual_ppo.json --pretrained pretrained\level_b_probe_from_a_1500_p128_offset\model_best.pt --delays 20 40 --snrs 5 10 --pilot-totals 64 128 --pilot-layouts two_block --seeds 0 1 2 --frames 3 --peft-groups head adapter_lora --peft-steps 1 --peft-lr 1e-4 --device cuda --output-dir logs\research_diagnostics_2026-08-02_focused
```

输出：

```text
logs\research_diagnostics_2026-08-02_focused\research_diagnostics.json
```

覆盖：

```text
2 delays × 2 SNR × 2 pilot_total × 1 layout × 3 seeds × 3 frames
= 72 paired frames
```

### 5.1 Reward Pilot loss 与 Data BER 相关性

结果：

```text
Spearman = 0.214686
num_pairs = 720
gate_threshold = 0.6
gate_pass = False
```

按主配置拆分：

| delay | SNR | pilot_total | layout | Spearman | pairs | pass |
|---:|---:|---:|---|---:|---:|---|
| 20 | 5 | 64 | two_block | 0.0361 | 90 | False |
| 20 | 5 | 128 | two_block | 0.1977 | 90 | False |
| 20 | 10 | 64 | two_block | 0.0633 | 90 | False |
| 20 | 10 | 128 | two_block | 0.5620 | 90 | False |
| 40 | 5 | 64 | two_block | 0.0725 | 90 | False |
| 40 | 5 | 128 | two_block | 0.1976 | 90 | False |
| 40 | 10 | 64 | two_block | 0.2788 | 90 | False |
| 40 | 10 | 128 | two_block | 0.3299 | 90 | False |

判断：

- 当前 Reward Pilot loss 改善不能稳定预测 Data BER 改善。
- 最接近门槛的是 `delay=20, SNR=10, pilot_total=128, two_block`，Spearman=0.562，但仍低于 0.6。
- 这解释了为什么 PPO 再训练也不稳定：reward 信号本身对真正目标不够可靠。

### 5.2 RL 低维 modulation 动作空间有效性

整体结果：

```text
action_samples = 720
fraction_reward_improved = 0.438889
fraction_data_improved = 0.205556
fraction_both_improved = 0.105556
```

动作按平均 Data BER 改善排序：

| action | mean_data_ber_improvement | mean_reward_loss_improvement |
|---|---:|---:|
| adapter_gate_1.2 | 0.001839 | 0.002451 |
| lora_scale_1.2 | 0.001023 | -0.000515 |
| head_temperature_0.8 | 0.000992 | -0.013420 |
| head_bias_minus_0.2 | 0.000765 | -0.003654 |
| film_minus_0.1 | 0.000279 | -0.000050 |
| film_plus_0.1 | -0.000186 | -0.000058 |
| head_temperature_1.2 | -0.001230 | -0.002533 |
| lora_scale_0.8 | -0.002191 | -0.003244 |
| adapter_gate_0.8 | -0.002759 | -0.004464 |
| head_bias_plus_0.2 | -0.021608 | -0.020528 |

判断：

- 动作空间不是完全无效：`adapter_gate_1.2` 平均能带来小幅 Data BER 改善。
- 但动作有效性很弱：
  - 只有约 20.6% 动作样本改善 Data BER。
  - 只有约 10.6% 同时改善 Reward 和 Data。
- 多个 Data 改善动作对应 Reward 反而变差，例如 `lora_scale_1.2`、`head_temperature_0.8`。
- 这说明当前 PPO 即使能探索到动作，也不容易被 Reward Pilot 正确强化。

### 5.3 真实 PEFT vs 低维 modulation

focused probe 结果：

```text
adapter_lora:
  mean_reward_loss_improvement = 0.000807
  mean_data_ber_improvement = -0.000041
  fraction_reward_improved = 0.680556
  fraction_data_improved = 0.097222

head:
  mean_reward_loss_improvement = 0.000154
  mean_data_ber_improvement = 0.000129
  fraction_reward_improved = 0.638889
  fraction_data_improved = 0.125000
```

判断：

- 真实 PEFT 更新能稳定改善 Reward Pilot loss。
- 但 1-step Adapt-only PEFT 并没有稳定改善 Data BER。
- adapter_lora 甚至出现平均 Data BER 轻微退化。
- 因此“直接做真实 PEFT”不是自动解决方案；核心问题仍是 Adapt/Reward 对 Data 的代表性和更新策略。

### 5.4 Level B 难度重新校准

focused probe 中 Offline NN 平均：

```text
Offline NN mean BER_data = 0.022755
```

逐配置 Offline NN vs 最强传统 baseline：

| delay | SNR | pilot_total | layout | Offline NN | 最强传统 | 传统 BER | NN 是否更低 |
|---:|---:|---:|---|---:|---|---:|---|
| 20 | 5 | 64 | two_block | 0.028770 | SC-FDE-MMSE | 0.034474 | True |
| 20 | 5 | 128 | two_block | 0.033565 | SC-FDE-MMSE | 0.036458 | True |
| 20 | 10 | 64 | two_block | 0.004216 | SC-FDE-MMSE | 0.008681 | True |
| 20 | 10 | 128 | two_block | 0.006366 | SC-FDE-MMSE | 0.009549 | True |
| 40 | 5 | 64 | two_block | 0.041419 | SC-FDE-MMSE | 0.051835 | True |
| 40 | 5 | 128 | two_block | 0.042245 | SC-FDE-MMSE | 0.053530 | True |
| 40 | 10 | 64 | two_block | 0.012153 | SC-FDE-MMSE | 0.019593 | True |
| 40 | 10 | 128 | two_block | 0.013310 | SC-FDE-MMSE | 0.021991 | True |

传统 baseline 整体均值：

```text
DFE-RLS overall mean = 0.219608
SC-FDE-MMSE overall mean = 0.029514
LMMSE-FIR overall mean = 0.195297
RLS Linear overall mean = 0.196620
```

判断：

- 在 focused 多 seed / 多帧下，传统 baseline 里真正强的是 `SC-FDE-MMSE`，不是 `DFE-RLS`。
- 当前 DFE-RLS 对这些配置并不强，之前某次 compare 中 DFE-RLS 很强可能受具体 seed / frame 轨迹影响，需要更大 paired matrix 才能定论。
- Offline NN 在这 8 个 focused 配置上均超过最强传统 baseline，但不是所有配置都满足 `BER_data < 0.01`：
  - 满足 <0.01：delay=20, SNR=10 的两个 pilot 配置。
  - 不满足 <0.01：delay=20, SNR=5；delay=40, SNR=5/10。

## 6. 对四个问题的直接回答

### 6.1 Reward Pilot loss 能预测 Data BER 改善吗？

当前证据：不能稳定预测。

focused probe：

```text
Spearman = 0.214686 < 0.6
```

结论：

- PPO 用当前 Reward Pilot loss 做 reward，不太可能稳定学出超越固定策略的在线策略。
- 如果不改 reward 设计或 Pilot 布局，单纯增加 PPO 训练帧数风险很高。

### 6.2 RL 动作空间是否有有效动作？

当前证据：有弱有效动作，但信号很稀疏、不稳定。

最有效动作：

```text
adapter_gate_1.2
mean_data_ber_improvement = 0.001839
fraction_data_improved = 0.541667
```

但整体：

```text
fraction_data_improved = 0.205556
fraction_both_improved = 0.105556
```

结论：

- 低维 modulation 动作空间不是完全错的。
- 但多数动作不能改善 Data BER。
- 当前 reward 无法可靠挑出 Data 改善动作。

### 6.3 真实 PEFT 是否优于低维 modulation？

当前证据：1-step Adapt-only PEFT 并不优于低维 modulation。

focused probe：

```text
adapter_lora mean_data_ber_improvement = -0.000041
head mean_data_ber_improvement = 0.000129
```

结论：

- 直接把 modulation 换成真实 PEFT 更新不一定解决问题。
- 如果后续让 RL 控制真实 PEFT，需要增加：
  - 多步/多 lr 动作；
  - Reward Pilot 留出早停；
  - last_good rollback；
  - 更强的防过拟合约束；
  - 可能还需要更合理的 Pilot 分布。

### 6.4 Level B 难度应该怎么调整？

当前证据：

- 不是所有 Level B 都对传统算法困难。
- 在 focused probe 中，SC-FDE-MMSE 是最强传统 baseline。
- Offline NN 已经能超过 SC-FDE-MMSE，但尚未在所有配置达到 <0.01。

推荐下一步 Level B 主工作区间：

1. 第一主攻区间：

```text
delay = 20
SNR = 10
pilot_total = 64 / 128
layout = two_block
```

理由：

- Offline NN 已经 <0.01。
- Offline NN 已经超过 SC-FDE-MMSE。
- Reward/Data Spearman 在 `pilot_total=128` 时接近 0.6，但还没过。
- 适合作为“先让 RL 进一步降低 BER”的开发区间。

2. 第二攻关区间：

```text
delay = 40
SNR = 10
pilot_total = 64 / 128
layout = two_block
```

理由：

- Offline NN 超过 SC-FDE-MMSE。
- 但 BER 约 0.012–0.013，略高于 0.01。
- 需要继续离线训练或改 Pilot/reward 后再作为主配置。

3. 暂不作为主门槛：

```text
SNR = 5
```

理由：

- NN 超过传统，但 BER 仍在 0.028–0.042。
- 这适合作为低 SNR 扩展或压力区间，不适合作为当前 PPO <0.01 门槛的第一开发目标。

## 7. 下一阶段建议

不要直接继续 PPO 大训练。应先解决 reward 与 Data BER 不一致。

优先顺序：

1. 改 Pilot 布局，提升 Reward Pilot 对 Data 的代表性。
   - 重点测试 `multi_block`，因为当前 focused 只跑了 `two_block`。
   - Reward Pilot 应更均匀覆盖整帧，而不是只贴在 Adapt 后面。

2. 修改 reward。
   - 不能用 Data 标签。
   - 可以考虑组合：
     - Reward Pilot BCE 改善；
     - Reward Pilot BER 改善；
     - confidence 校准；
     - update 后 logits margin 分布；
     - 参数扰动惩罚；
     - 跨连续帧 reward EMA。

3. 缩小 PPO 开发区间。
   - 先只做：

```text
delay=20, SNR=10, pilot_total=128, two_block/multi_block
```

   - 目标不是马上跑正式矩阵，而是先让 PPO 在该区间稳定超过 Offline NN only。

4. 再考虑真实 PEFT 动作。
   - 当前 1-step PEFT 不够。
   - 若做真实 PEFT，应做为动作空间：

```text
skip
head lr=1e-4 steps=1
head lr=3e-4 steps=1
adapter_lora lr=1e-4 steps=1
adapter_lora lr=1e-4 steps=2
rollback_last_good
```

   - 但前提是 reward 相关性先提升到接近或超过 0.6。

## 8. 本轮结论

本轮最关键的结论不是“RL 不行”，而是：

```text
当前 reward 信号不可靠，导致 PPO 无法稳定利用已有动作空间。
```

同时，Level B 难度校准给出一个更积极的信号：

```text
当前 Offline NN 在 focused 多 seed / 多帧配置下已经超过最强传统 baseline SC-FDE-MMSE。
```

因此下一阶段研究路线应调整为：

```text
先固定 delay=20, SNR=10, pilot_total=128 的可实现主工作区间
-> 优化 Pilot 布局与 reward 相关性
-> 再训练 PPO，让它超过 Offline NN only
-> 最后扩展到 delay=40 和更低 SNR
```
