# 迭代记录：multi_block、Reward surrogate 与离散安全动作诊断

时间戳：2026-08-02 21:19:41 +08:00

## 1. 本轮目标

根据上一轮结论继续推进：

1. 对比 `two_block` 与 `multi_block`，验证分布式 Pilot 是否提升 Reward/Data 相关性。
2. 如果 Reward/Data Spearman ≥ 0.6，则启动小规模 PPO。
3. 如果不过门槛，则先做 composite reward 诊断，不盲目训练 PPO。
4. 如果 action-level reward 有效，再验证 Reward Pilot 逐帧选择离散动作是否能真实降低 Data BER。

本轮仍遵守：

- Data 标签只用于离线诊断评估。
- 在线 reward / observation / action 选择不使用 Data 标签。
- 不启动正式矩阵。
- 不把诊断上界写成正式结果。

## 2. 新增/修改代码

修改：

```text
evaluation/research_diagnostics.py
scripts/diagnose_research_assumptions.py
tests/test_research_diagnostics.py
```

新增诊断能力：

1. `evaluate_modulation_candidates()` 增加字段：

```text
reward_ber_before
reward_ber_after
reward_ber_improvement
reward_margin_before
reward_margin_after
reward_margin_improvement
action_delta_norm
```

2. 新增 `summarize_reward_surrogates()`：

比较以下 reward surrogate 与 Data BER 改善的 Spearman：

```text
reward_loss_delta
reward_ber_delta
reward_margin_delta
loss_plus_ber
loss_plus_margin
ber_plus_margin
```

3. 新增 action-level surrogate 汇总：

按 `action_name` 聚合后再计算 reward surrogate 与 Data BER 改善的 Spearman。

4. 新增 `summarize_reward_selected_actions()`：

模拟“每帧用 Reward Pilot 选择离散动作”，Data 只用于诊断评估。

安全规则：

- 候选包含 `identity`。
- 如果 surrogate 没有正改善，则回退 `identity`。
- tie-break 使用：

```text
surrogate_score
reward_loss_improvement
-action_delta_norm
```

避免 Reward BER 全为 0 时误选大扰动动作。

## 3. Probe 1：two_block vs multi_block focused gate

命令：

```powershell
.\.venv-gpu\Scripts\python.exe scripts\diagnose_research_assumptions.py --config configs\continual_ppo.json --pretrained pretrained\level_b_probe_from_a_1500_p128_offset\model_best.pt --delays 20 --snrs 10 --pilot-totals 128 --pilot-layouts two_block multi_block --seeds 0 1 2 3 4 --frames 5 --peft-groups head adapter_lora --peft-steps 1 --peft-lr 1e-4 --device cuda --output-dir logs\research_diagnostics_2026-08-02_multiblock_gate
```

结果文件：

```text
logs\research_diagnostics_2026-08-02_multiblock_gate\research_diagnostics.json
```

关键结果：

```text
overall Spearman = 0.264560
num_pairs = 500
gate_pass = False
```

按 layout：

```text
two_block Spearman = 0.311573
multi_block Spearman = 0.221364
```

判断：

- `multi_block` 没有提升 Reward/Data 相关性。
- 在当前实现和 checkpoint 下，`two_block` 反而更好。
- 因此不启动 PPO。

Offline NN 与传统 baseline：

```text
Offline NN mean BER_data = 0.014375

two_block:
  Offline NN = 0.014271
  SC-FDE-MMSE = 0.017917

multi_block:
  Offline NN = 0.014479
  SC-FDE-MMSE = 0.018437
```

判断：

- Offline NN 仍超过最强传统 baseline `SC-FDE-MMSE`。
- 但 Offline NN 没有达到 `<0.01`，因此当前 checkpoint 还不满足最终绝对 BER 门槛。

## 4. Probe 2：Composite Reward Surrogate

命令：

```powershell
.\.venv-gpu\Scripts\python.exe scripts\diagnose_research_assumptions.py --config configs\continual_ppo.json --pretrained pretrained\level_b_probe_from_a_1500_p128_offset\model_best.pt --delays 20 --snrs 10 --pilot-totals 128 --pilot-layouts two_block multi_block --seeds 0 1 2 3 4 --frames 5 --peft-groups head adapter_lora --peft-steps 1 --peft-lr 1e-4 --device cuda --output-dir logs\research_diagnostics_2026-08-02_reward_surrogates_actionlevel
```

逐样本 surrogate 结果：

| surrogate | Spearman | pass |
|---|---:|---|
| reward_loss_delta | 0.264560 | False |
| reward_ber_delta | 0.219874 | False |
| reward_margin_delta | 0.086873 | False |
| loss_plus_ber | 0.274845 | False |
| loss_plus_margin | 0.215175 | False |
| ber_plus_margin | 0.169146 | False |

最好的逐样本 surrogate：

```text
loss_plus_ber
Spearman = 0.274845
```

判断：

- 逐帧/逐样本 reward surrogate 都没有达到 0.6。
- 简单 composite reward 不能解决 PPO 信号问题。

## 5. Action-level surrogate 发现

按 `action_name` 聚合后，出现不同结论：

| surrogate | action-level Spearman | num_actions | pass |
|---|---:|---:|---|
| reward_loss_delta | 0.490909 | 10 | False |
| reward_ber_delta | 0.716172 | 10 | True |
| reward_margin_delta | 0.321212 | 10 | False |
| loss_plus_ber | 0.503030 | 10 | False |
| loss_plus_margin | 0.393939 | 10 | False |
| ber_plus_margin | 0.321212 | 10 | False |

关键判断：

```text
逐帧 reward 不可靠；
但动作均值层面，Reward BER delta 能区分好动作和坏动作。
```

这说明当前连续 PPO 的问题是：

- 动作空间太连续；
- 每帧 reward 太噪；
- PPO 在逐帧层面很难稳定学到“哪类动作整体更好”。

更合理的后续方向：

```text
连续 modulation PPO
-> 收缩为少量离散安全动作
-> 用窗口级 / action-level 聚合 reward
-> 再做离散 PPO 或 bandit/PPO hybrid
```

## 6. Probe 3：Reward-selected 离散动作选择

为了验证“用 Reward Pilot 逐帧选择离散动作”是否足够，加入：

```text
summarize_reward_selected_actions(..., surrogate_name="reward_ber_delta")
```

第一次实现排除了 identity，导致 Reward BER 并列时误选扰动动作，结果退化：

```text
mean_data_ber_improvement = -0.001979
action_counts:
  adapter_gate_0.8 = 43
```

根因：

- Reward BER 常常没有变化。
- 排除 identity 后，零改善 tie 会选择第一个非 identity 候选。
- 这会把“没有证据应更新”的帧错误更新。

修复后：

- 候选包含 identity。
- surrogate ≤ 0 时保持 identity。
- tie-break 偏向低扰动。

命令：

```powershell
.\.venv-gpu\Scripts\python.exe scripts\diagnose_research_assumptions.py --config configs\continual_ppo.json --pretrained pretrained\level_b_probe_from_a_1500_p128_offset\model_best.pt --delays 20 --snrs 10 --pilot-totals 128 --pilot-layouts two_block multi_block --seeds 0 1 2 3 4 --frames 5 --peft-groups head adapter_lora --peft-steps 1 --peft-lr 1e-4 --device cuda --output-dir logs\research_diagnostics_2026-08-02_reward_selected_safe
```

结果：

```text
selected_frames = 50
mean_data_ber_improvement = 0.000208
fraction_data_improved = 0.04
action_counts:
  identity = 45
  head_bias_minus_0.2 = 3
  adapter_gate_1.2 = 1
  lora_scale_0.8 = 1
```

判断：

- 安全选择不再明显伤害 Data。
- 但改善极小，远小于固定 `adapter_gate_1.2` 的潜在平均改善。
- 逐帧 Reward-selected 动作选择仍不足以支撑正式 PPO。

## 7. 当前最重要结论

### 7.1 不启动 PPO

当前不满足 PPO 开发门槛：

```text
逐样本 Reward/Data Spearman < 0.6
逐帧 Reward-selected 离散动作平均改善很小
Offline NN 仍未达到 BER_data < 0.01
```

因此，本轮不启动：

- 小规模 Continual PPO。
- 正式矩阵。
- pilot sweep 正式实验。

### 7.2 研究假设进一步收敛

当前 evidence 支持：

```text
神经均衡器已经能超过传统 baseline；
但 RL 贡献还没有建立。
```

更准确的问题是：

```text
如何让 RL 稳定地选择少量安全动作，使 BER_data 进一步低于 Offline NN only。
```

### 7.3 当前最有希望的动作

固定动作诊断中：

```text
adapter_gate_1.2:
  mean_data_ber_improvement = 0.001406
  fraction_reward_improved = 0.76
  fraction_data_improved = 0.42
```

但逐帧 Reward-selected 只选中它 1/50 次，说明当前 selection signal 过于保守或离散 Reward BER 太粗。

## 8. 下一步建议

下一步不要做连续 PPO，应做“离散安全动作 PPO / Bandit-PPO hybrid”的最小实现。

推荐动作集：

```text
identity
adapter_gate_1.2
lora_scale_1.2
film_minus_0.1
head_temperature_0.8
head_bias_minus_0.2
rollback_identity
```

推荐 reward：

```text
窗口级 reward，而不是单帧 reward

reward =
  mean_reward_ber_delta_over_window
  + 0.1 * mean_reward_loss_delta_over_window
  - 0.001 * mean_action_delta_norm
```

推荐窗口：

```text
8 或 16 frames
```

原因：

- action-level `reward_ber_delta` 已过 0.6。
- 单帧 reward 没过。
- 所以 RL 应学习窗口级动作偏好，不应逐帧连续探索。

同时需要并行推进 offline checkpoint：

```text
当前 Offline NN mean BER_data = 0.014375
目标 < 0.01
```

如果 offline 不低于 0.01，即使 RL 带来 0.001–0.002 改善，也很难稳定达成最终 BER 目标。

建议下一阶段门槛：

1. 先继续定向离线训练，让：

```text
delay=20, SNR=10, pilot_total=128
two_block/multi_block mean BER_data < 0.01
```

2. 然后实现离散安全动作 PPO。

3. PPO 小实验只要求先超过 Offline NN only：

```text
RL BER_data < Offline NN only
```

4. 再恢复最终目标：

```text
RL BER_data < 0.01
且超过传统 baseline
```

## 9. 测试结果

相关回归测试：

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests\test_research_diagnostics.py tests\test_rl_modulated_equalizer.py tests\test_traditional_baselines.py -q -p no:cacheprovider
```

结果：

```text
19 passed in 47.95s
```

最终全量测试：

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

结果：

```text
110 passed in 212.02s (0:03:32)
```
