# 迭代记录：Offline NN 门槛与窗口级离散 PEFT-RL 推进

时间戳：2026-08-03 00:49:59 +08:00

## 1. 本轮目标

本轮按新的执行顺序推进：

1. 先把 `Offline NN only` 打到稳定 `BER_data < 0.01`。
2. 再把 RL 从逐帧连续 modulation 改为“离散安全动作 + 窗口级 reward”。
3. 继续推进到 `RL-Modulated Neural Block Equalizer < Offline NN only`。

执行中坚持门槛式推进：没有真实评估数据时不进入下一阶段，不用 Data Oracle，不用 Data 标签参与在线 observation、reward、动作选择或 PEFT 更新。

## 2. 代码改动

### 2.1 离线训练分布修正

发现 `training/curriculum.py` 中 `estimated_cir_level_b` 原先只训练固定配置：

```text
delay=40, SNR=10, pilot_total=128, layout=two_block
```

这与“Level B 主配置 20/30/40 × 10/15/20”不一致。已修改为：

- `CurriculumPhase` 增加 `delay_grid`、`snr_grid`。
- `sample_delay_snr()` 按样本序号循环覆盖主配置网格。
- mixed-delay batch 中 CIR 和 tail 右侧补零到 `model.config.max_delay`，避免 20/30/40 delay 混 batch 时 `torch.stack` 失败。

新增测试：

- `test_level_b_curriculum_phase_cycles_main_delay_snr_grid`
- `test_level_b_curriculum_step_loss_supports_mixed_delay_batch`

### 2.2 RL 改为离散安全动作 + 窗口级 reward

新增：

- `agent/discrete_safe_policy.py`
- `training/windowed_discrete_ppo.py`
- `tests/test_windowed_discrete_ppo.py`

当前动作表：

```text
identity
adapter_gate_1.2
lora_scale_1.2
film_minus_0.1
head_temperature_0.8
head_bias_minus_0.2
peft_head_light
peft_adapter_light
rollback_identity
```

核心逻辑：

- 同一动作作用于一个短窗口。
- reward 使用窗口内 Reward Pilot BER/loss 改善聚合。
- Data BER 只用于仿真评估。
- `peft_head_light` / `peft_adapter_light` 使用 Adapt Pilot 做 1 步真实 PEFT 更新。
- PEFT 更新后用 Reward Pilot guard 判断是否保留，否则回滚参数。

### 2.3 入口切换

`online_train.py` 已从旧连续调制 runner 切换到：

```text
training/windowed_discrete_ppo.py
```

`compare.py` 中 `RL-Modulated Neural Block Equalizer` 也切换到窗口级离散 PPO state / rollout / reward / PPO update 逻辑。

未显式传入 `--policy` 时，`compare.py` 不再自动加载旧 `logs/online/policy.pt`，避免旧连续 policy checkpoint 被误读。

### 2.4 文档同步

已更新：

- `AGENTS.md`
- `README.md`
- `开发框架.md`
- `RL信道均衡研究分析.md`

当前文档明确：逐帧连续 modulation 不再是主路线；当前主路线是窗口级离散安全动作 + PEFT-RL。

## 3. 离线训练结果

### 3.1 初始 checkpoint

使用：

```text
pretrained/level_b_probe_from_a_1500_p128_offset/model_best.pt
```

外部 9 配置评估：

```text
logs/offline_gate_current_2026-08-02
```

总体：

```text
Offline NN only overall BER_data = 0.016840
SC-FDE-MMSE overall BER_data   = 0.019309
DFE-RLS overall BER_data       = 0.001331
```

结论：

- Offline NN 已超过 SC-FDE-MMSE 总体均值。
- Offline NN 未达到 `<0.01`。
- DFE-RLS 仍非常强。

### 3.2 混合 Level B grid 训练

分段训练原因：

- `1000 steps / batch=4` 和 `1000 steps / batch=2` 都表现为长时间无可观测输出。
- 改为每段 `200 steps / batch=2`，每段保存 checkpoint 并外部评估。

训练序列：

```text
level_b_grid_200_from_1500_p128
level_b_grid_400_from_1500_p128
level_b_grid_600_from_1500_p128
level_b_grid_800_from_1500_p128
```

外部 9 配置总体 BER：

| checkpoint | Offline NN overall BER_data |
|---|---:|
| 初始 1500 offset | 0.016840 |
| grid200 | 0.015837 |
| grid400 | 0.013773 |
| grid600 | 0.012712 |
| grid800 | 0.012269 |

结论：

- 混合网格训练稳定改善。
- 但低 SNR 配置仍卡住，尤其 `delay=40, SNR=10`。

### 3.3 低 SNR 聚焦训练

新增实验配置：

- `configs/continual_ppo_focus_snr10.json`
- `configs/continual_ppo_focus_d40_snr10.json`

继续训练：

```text
level_b_focus_snr10_1000_from_1500_p128
level_b_focus_snr10_1200_from_1500_p128
level_b_focus_d40_snr10_1400_from_1500_p128
level_b_focus_d40_snr10_1600_from_1500_p128
```

最佳当前 checkpoint：

```text
pretrained/level_b_focus_d40_snr10_1600_from_1500_p128/model_best.pt
```

对应外部评估：

```text
logs/offline_gate_focus_d40_snr10_1600_2026-08-02
```

9 配置结果：

| delay | SNR | Offline NN mean BER_data | 是否 < 0.01 |
|---:|---:|---:|---|
| 20 | 10 | 0.013542 | 否 |
| 20 | 15 | 0.005382 | 是 |
| 20 | 20 | 0.003993 | 是 |
| 30 | 10 | 0.010764 | 否 |
| 30 | 15 | 0.005556 | 是 |
| 30 | 20 | 0.003472 | 是 |
| 40 | 10 | 0.018403 | 否 |
| 40 | 15 | 0.008507 | 是 |
| 40 | 20 | 0.005556 | 是 |

总体：

```text
Offline NN only overall BER_data = 0.008353
SC-FDE-MMSE overall BER_data     = 0.019309
DFE-RLS overall BER_data         = 0.001331
```

结论：

- 当前 Offline NN 总体已经低于 `0.01`。
- 但严格逐配置门槛仍未通过，失败配置主要是 `SNR=10`。
- `delay=40, SNR=10` 是最硬瓶颈。
- 如果按“每个主配置都 <0.01”的严格门槛，Offline NN 阶段仍未完成。

## 4. RL 实验结果

### 4.1 纯离散 modulation + Reward BER guard

小样本：

```text
logs/rl_vs_offline_probe_windowed_berguard_2026-08-02
```

结果：

```text
Offline NN only overall BER_data                  = 0.009512
RL-Modulated Neural Block Equalizer overall BER_data = 0.009524
```

结论：

- 几乎等于 Offline NN。
- 但未低于 Offline NN。
- Reward BER guard 太严格时，绝大多数动作回退 identity。

### 4.2 加入真实 PEFT 动作

新增：

```text
peft_head_light
peft_adapter_light
```

先尝试 BER-only guard：

```text
logs/rl_vs_offline_probe_peft_2026-08-02
```

结果：

```text
Offline NN only overall BER_data = 0.009512
RL overall BER_data             = 0.009524
PEFT applied frames             = 0
```

PEFT 全部被 BER guard 拒绝。

再尝试 PEFT loss-guard：

```text
logs/rl_vs_offline_probe_peft_lossguard_2026-08-02
```

结果：

```text
Offline NN only overall BER_data = 0.009512
RL overall BER_data             = 0.009573
PEFT applied frames             = 33
```

结论：

- PEFT 被执行后，Reward loss 改善仍不能稳定预测 Data BER 改善。
- 小帧数短窗口下，PEFT 会带来轻微退化。

### 4.3 长 episode 下的 Continual PPO 信号

最难配置单独跑：

```text
logs/rl_vs_offline_d40s10_64f_peft_2026-08-02
delay=40, SNR=10, seeds=3, frames=64
```

结果：

```text
Offline NN only BER_data = 0.055094
RL BER_data             = 0.052856
```

分段结果：

| frame 段 | Offline NN | RL |
|---|---:|---:|
| 1–16 | 0.021050 | 0.021050 |
| 17–32 | 0.048123 | 0.046658 |
| 33–48 | 0.070475 | 0.066406 |
| 49–64 | 0.080729 | 0.077311 |

结论：

- 长 episode 漂移/状态累积后，Offline NN 明显变差。
- PEFT-RL 在后半段有稳定小幅补偿。
- 但绝对 BER 远高于 `<0.01`，说明接收机状态维护或长 episode 鲁棒性仍有问题。

### 4.4 9 配置 64-frame 小矩阵

单 seed：

```text
logs/rl_vs_offline_9cfg_64f_peft_2026-08-02
```

结果：

```text
Offline NN only overall BER_data = 0.012049
RL overall BER_data             = 0.011316
```

三 seed 确认：

```text
logs/rl_vs_offline_9cfg_64f_3seed_peft_2026-08-02
```

总体：

```text
Offline NN only overall BER_data = 0.039216
RL overall BER_data             = 0.038167
```

逐配置 delta：

| delay | SNR | RL - Offline |
|---:|---:|---:|
| 20 | 10 | -0.000475 |
| 20 | 15 | -0.000081 |
| 20 | 20 | -0.002007 |
| 30 | 10 | -0.002048 |
| 30 | 15 | +0.000800 |
| 30 | 20 | -0.001289 |
| 40 | 10 | -0.002238 |
| 40 | 15 | -0.000624 |
| 40 | 20 | -0.001478 |

动作统计：

```text
accepted_fraction = 0.1863
applied_actions:
  identity          1406
  peft_adapter_light 175
  peft_head_light    118
  lora_scale_1.2      11
  head_bias_minus_0.2 14
  head_temperature_0.8 3
  adapter_gate_1.2     1
```

结论：

- 3-seed 长 episode 小矩阵中，RL 总体低于 Offline NN。
- 8/9 个配置 RL 低于 Offline NN。
- `30/15` 轻微退化。
- 绝对 BER 仍显著高于 `<0.01`，因此不能作为最终论文结论。

## 5. 当前判断

### 5.1 已经成立的点

1. Offline NN 总体可以低于 `0.01`，当前最佳总体为 `0.008353`。
2. Offline NN 明显超过 SC-FDE-MMSE 总体均值。
3. 窗口级离散 PEFT-RL 在长 episode 中出现了 `RL < Offline NN only` 的真实信号。
4. RL 的有效动作主要来自 `peft_adapter_light` 和 `peft_head_light`，不是纯 modulation。

### 5.2 尚未成立的点

1. Offline NN 没有在每个主配置都低于 `0.01`。
2. RL 没有在每个主配置都低于 Offline NN；`30/15` 有轻微退化。
3. 长 episode 下绝对 BER 明显升高，说明状态维护、tail 更新或信道漂移适配还不稳。
4. DFE-RLS 仍明显强于当前 Offline NN / RL，当前 proposed 尚未超过最强传统 baseline。

### 5.3 对研究方向的反思

当前结果不说明创新点不可行，但说明原始假设仍需要收窄：

```text
“极端长时延必然击败传统均衡器”
```

这个假设过强。当前 Level B 虽然有长时延，但仍是干净线性 BPSK 问题，DFE-RLS 在当前 pilot 和信道设置下很强。

更准确的创新点应改为：

```text
在极端稀疏长回波、整帧非因果神经均衡器已经具备较强离线先验的条件下，
Continual PPO 通过 Reward Pilot 反馈选择少量安全 PEFT 动作，
在长 episode 漂移/状态累积时进一步降低 BER。
```

这条创新点已有初步数据支持，但还不能写成最终结论。

## 6. 下一阶段建议

优先级按实际瓶颈排序：

1. 修复长 episode 下 Offline NN BER 从短评估 `0.008353` 升到 64-frame `0.039216` 的问题。
   - 检查神经 tail 更新是否误差累积。
   - 对比用真实 tail、soft tail、hard decision tail 的差异。
   - 检查 episode 内 CIR 是否需要在线重新估计，而不是固定 acquisition CIR。

2. 把 `delay=40, SNR=10` 单配置继续作为主要攻关点。
   - 当前短评估仍约 `0.0184`。
   - 单纯加训练步数有改善但变慢，需要结构性分析。

3. 强化 PEFT action 空间。
   - 当前有效动作主要是 adapter/head 轻更新。
   - 可加入 `peft_adapter_head_light`、`peft_head_3step`、`peft_adapter_3step`，但必须继续使用 Reward Pilot guard。

4. reward 不宜只看 loss。
   - Reward loss-only guard 会误接受动作。
   - Reward BER-only guard 太保守。
   - 下一步应考虑窗口级复合指标：Reward BER 优先，loss 作为 tie-break，同时加入跨窗口 EMA。

5. 暂不跑正式矩阵。
   - 当前只通过了方向性小矩阵。
   - 未达到“每配置 BER < 0.01”。
   - 未超过 DFE-RLS。

## 7. 验证记录

全量测试：

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

结果：

```text
117 passed in 245.08s (0:04:05)
```

缓存清理：

- `.venv-gpu` 内部依赖缓存不清理。
- 仓库根目录 `__pycache__` 删除动作被当前工具安全策略拦截，未绕过执行。

## 8. 当前最重要结论

本轮没有完成最终论文目标，但完成了关键路线转折：

```text
纯 modulation RL 基本不可用；
真实 PEFT 动作是必要的；
窗口级 Continual PPO 在长 episode 下已经出现 RL < Offline NN only 的方向性信号；
但绝对 BER 和传统 baseline 对比仍未达标。
```

因此下一阶段不应再回到纯 modulation，也不应立即扩大正式矩阵。应集中解决长 episode 状态累积、CIR 更新和 PEFT action/reward 设计。
