# eme_slow_drift_v1 Gate 2 诊断记录

## 1. 当前结论

`eme_slow_drift_v1` 已完成 traditional-only 冻结，但 Gate 2 尚未通过：

```text
目标：Offline NN 在冻结 profile 上超过最强 traditional baseline。
当前结果：未通过。
主要阻碍：现有 Offline NN 可以单帧过拟合，但不能跨 Level B slow-drift 信道泛化；简单增加训练步数或学习率无效。
```

该结论不改变冻结 profile。后续不得根据 NN/RL 结果重调 `cfo_abs`、`phase_noise_std` 或 `rho`。

## 2. 已运行实验

### 2.1 基础 100-step Offline 训练

```text
config = configs/continual_ppo_eme_slow_drift_v1.json
checkpoint = pretrained/eme_slow_drift_v1_offline100_lr1e4_2026-08-24/model_best.pt
compare = logs/compare_eme_slow_drift_v1_offline100_main_d20_snr10_f4
```

`delay=20, SNR=10, frames=4`：

```text
best traditional: CFO+DD-Phase LMMSE-FIR, mean BER_data = 0.086589
Offline NN only: mean BER_data = 0.419922
```

训练 loss 基本停在 `~1.04`，Offline NN validation 为 `0.531132`。

### 2.2 phase_correction_initial_scale = 1.0

```text
config = configs/continual_ppo_eme_slow_drift_v1_phase_scale1.json
checkpoint = pretrained/eme_slow_drift_v1_phase_scale1_100steps_2026-08-24/model_best.pt
compare = logs/compare_eme_slow_drift_v1_phase_scale1_100_main_d20_snr10_f4
```

`delay=20, SNR=10, frames=4`：

```text
best traditional: CFO+DD-Phase LMMSE-FIR, mean BER_data = 0.086589
Offline NN only: mean BER_data = 0.464193
```

内部 validation 改善到 `0.240057`，但正式 compare 不改善，说明只打开 phase scale 不足以解决泛化。

### 2.3 B-only 续训

```text
checkpoint = pretrained/eme_slow_drift_v1_phase_scale1_bplus300_2026-08-24/model_best.pt
compare = logs/compare_eme_slow_drift_v1_phase_scale1_bplus300_main_d20_snr10_f4
```

`delay=20, SNR=10, frames=4`：

```text
best traditional: CFO+DD-Phase LMMSE-FIR, mean BER_data = 0.086589
Offline NN only: mean BER_data = 0.454427
```

B-only loss 没有继续下降，`first10 = 0.9821`、`last10 = 1.0102`。

### 2.4 更高学习率 B-only 续训

```text
config = configs/continual_ppo_eme_slow_drift_v1_phase_scale1_lr1e3.json
checkpoint = pretrained/eme_slow_drift_v1_phase_scale1_lr1e3_bplus300_2026-08-24/model_best.pt
compare = logs/compare_eme_slow_drift_v1_phase_scale1_lr1e3_bplus300_main_d20_snr10_f4
```

`delay=20, SNR=10, frames=4`：

```text
best traditional: CFO+DD-Phase LMMSE-FIR, mean BER_data = 0.086589
Offline NN only: mean BER_data = 0.402344
NN + Rule Modulation: mean BER_data = 0.393229
```

学习率升到 `1e-3` 后略改善 compare，但 validation 退化到 `0.345407`，B loss 仍停在 `~1.006`，因此不是单纯学习率太低。

### 2.5 单配置 focused 训练

```text
config = configs/continual_ppo_eme_slow_drift_v1_focus_d20_snr10_phase_scale1.json
checkpoint = pretrained/eme_slow_drift_v1_focus_d20_snr10_phase_scale1_b300_2026-08-24/model_best.pt
compare = logs/compare_eme_slow_drift_v1_focus_d20_snr10_phase_scale1_b300_main_f8
```

`delay=20, SNR=10, frames=8`：

```text
best traditional: CFO+DD-Phase LMMSE-FIR, mean BER_data = 0.271810
Offline NN only: mean BER_data = 0.453776
```

focused validation 约 `0.482585`，未证明单配置可超过传统。

### 2.6 单帧过拟合诊断

固定一个 `delay=20, SNR=10` 帧、固定 acquisition CIR，以 `lr=1e-3` 训练同一帧：

```text
step 0:   loss = 1.018407, BER_data = 0.351563
step 50:  loss = 0.129609, BER_data = 0.033854
step 100: loss = 0.059545, BER_data = 0.015625
step 200: loss = 0.034922, BER_data = 0.013021
```

这排除了标签映射、基本反向传播和模型容量完全失效的问题。模型能记住一个帧，但不能泛化到 Level B slow-drift 分布。

### 2.7 CIR/tail 上界诊断

同一 checkpoint 下，将 compare 的条件替换为 true CIR 或 true tail：

```text
acquisition CIR + soft tail: mean BER_data = 0.402344
acquisition CIR + true tail: mean BER_data = 0.402344
true CIR + soft tail:        mean BER_data = 0.448568
true CIR + true tail:        mean BER_data = 0.448568
```

因此当前失败不是 compare 中 soft tail 传递或 acquisition CIR 误差单独造成的。

### 2.8 analytic logit skip 试验

新增默认关闭的模型字段：

```text
analytic_logit_skip_scale
```

最小单元测试证明：identity channel 下打开 analytic skip 且神经 head 置零时，模型可直接恢复 BPSK 判决。

但在 EME slow-drift focused 训练中，`analytic_logit_skip_scale = 0.25` 使 loss 偏高：

```text
probe20 validation = 0.486654
compare Offline NN only = 0.484701
```

说明简单 matched-filter proposal 在 residual CFO/phase 场景下不是可靠 warm-start；传统方法的优势来自 pilot-based CFO/DD phase compensation 后再做 FIR/DFE，而不是未补偿 proposal。

## 3. 排除的解释

```text
不是冻结 profile 太难：traditional-only 已给出可工作的 CFO+DD-Phase baseline。
不是训练入口不可运行：pretrain/compare/online smoke 均通过。
不是标签或反向传播完全错误：单帧可过拟合到 BER_data 约 0.013。
不是 soft tail 或 true CIR 单独造成：true tail/true CIR 上界没有显著改善。
不是单纯训练步数不足：B-only 续训 300 step 未降低 B loss。
不是单纯学习率太低：lr=1e-3 仍未打开泛化。
不是简单 analytic proposal 可解决：matched-filter logit skip 在失同步场景下会放大错误。
```

## 4. 下一步技术路线

Gate 2 的下一步不应继续盲目扩大 PPO 或训练步数，而应先重建 Offline NN 的物理 warm-start：

```text
1. 在 Proposed NN 前端加入 pilot-based phase/CFO compensation feature 或 differentiable compensation branch。
2. 该 branch 只能使用 acquisition/adapt pilot、接收信号、profile-level CFO budget，不读 true signed CFO 或 data label。
3. warm-start 应接近 CFO+DD-Phase LMMSE-FIR 的输入质量，而不是当前未补偿 matched-filter proposal。
4. Offline NN 先在 delay=20, SNR=10 focused 配置超过最强 traditional，再扩展 12 配置。
5. 只有 Offline NN 达到 Gate 2 后，再继续 Rule/Online RL。
```

当前建议的下一项实现不是 RL，而是：

```text
PilotPhaseCompensatedNeuralEqualizer:
  receiver_view + acquisition/adapt pilot -> phase/CFO estimate
  rx_symbols -> compensated rx_iq
  compensated rx_iq + neural block equalizer -> logits
```

公平性边界与 traditional baseline 相同：只用 pilot-based estimate 和 profile-level CFO limit，不使用真实 impairment state。
