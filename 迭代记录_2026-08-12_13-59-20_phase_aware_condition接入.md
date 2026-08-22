# 迭代记录：phase-aware condition 接入

时间戳：2026-08-12 13:59:20

## 1. 本轮目标

上一轮结论是：`phase_tiny` 500-step Offline NN 已经超过部分传统方法，但仍未稳定超过 DFE-RLS。瓶颈不在 PPO，而在神经接收机本体没有显式利用 Adapt Pilot 中可观测的相位残差信息。

本轮目标：

```text
Adapt Pilot phase residual / CFO residual
-> CIRCondition.latent_residual[0:2]
-> curriculum training
-> compare / online inference
```

关键要求：

- 训练和推理都接入，避免分布不一致；
- phase residual 只使用接收端可见信息；
- 不读取 Reward/Data 标签；
- 不读取真实 CFO/相位。

## 2. 已实现内容

### 2.1 phase residual helper

在 `baseline/traditional_equalizers.py` 新增：

```python
estimate_phase_residual_features(receiver_view, cir, soft_tail)
```

它复用传统补偿边界：

```text
rx_symbols
Adapt Pilot symbols
Adapt Pilot mask
CIR
soft_tail
```

输出：

```text
phase_residual
cfo_residual
```

不使用：

- Reward Pilot 标签；
- Data 标签；
- 真实 impairment 参数；
- 神经网络；
- RL。

### 2.2 condition_from_cir phase latent

`agent/cir_estimator.py` 中的接口已扩展为：

```python
condition_from_cir(
    cir,
    snr_db,
    phase_residual=0.0,
    cfo_residual=0.0,
)
```

映射规则：

```text
CIRCondition.latent_residual[:, 0] = phase_residual
CIRCondition.latent_residual[:, 1] = cfo_residual
```

### 2.3 curriculum training 接入

`training/curriculum.py` 已在训练 batch 中为每个 frame 计算：

```python
phase_residual, cfo_residual = estimate_phase_residual_features(
    frame.receiver_view(),
    cir,
    frame.tail_symbols,
)
```

并写入：

```text
condition.latent_residual[0:2]
```

验证路径 `_validate_offline_nn()` 也同步接入，避免训练/验证不一致。

### 2.4 compare / online 接入

`compare.py` 中 proposed 神经方法和 `RL-Modulated Neural Block Equalizer` 已接入 phase-aware condition。

`training/windowed_discrete_ppo.py` 和 `training/rl_modulated_online.py` 也已接入。

`training/continual_ppo.py` 是旧兼容 runner，不直接使用神经 equalizer condition，因此本轮不改。

## 3. 测试

新增/覆盖测试：

```text
tests/test_traditional_baselines.py::test_phase_residual_features_are_label_free_and_nonzero_under_rotation
tests/test_receiver_architecture.py::test_condition_from_cir_can_encode_phase_residual_features
```

相关测试：

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest `
  tests/test_traditional_baselines.py `
  tests/test_receiver_architecture.py `
  tests/test_evaluation_contract.py::test_compare_cli_writes_real_level_b_metrics `
  -q -p no:cacheprovider
```

结果：

```text
21 passed in 26.47s
```

全量测试：

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

结果：

```text
154 passed in 290.61s (0:04:50)
```

## 4. phase-aware smoke

### 4.1 2-step smoke

命令：

```powershell
.\.venv-gpu\Scripts\python.exe pretrain.py `
  --config configs/continual_ppo_phase_tiny.json `
  --stage all `
  --steps 2 `
  --batch-size 1 `
  --amp `
  --save-dir pretrained/phase_tiny_phaseaware_smoke_2026-08-12
```

结果：

```text
saved pretrained/phase_tiny_phaseaware_smoke_2026-08-12
```

compare smoke：

```powershell
.\.venv-gpu\Scripts\python.exe compare.py `
  --config configs/continual_ppo_phase_tiny.json `
  --method-group proposed `
  --pretrained pretrained/phase_tiny_phaseaware_smoke_2026-08-12/model_best.pt `
  --delays 20 `
  --snrs 10 `
  --num-seeds 1 `
  --frames 1 `
  --pilot-total 128 `
  --pilot-layout prefix `
  --impairment-profile phase_tiny `
  --resume `
  --output-dir logs/compare_phaseaware_smoke_2026-08-12
```

结果：

```text
saved logs/compare_phaseaware_smoke_2026-08-12
```

说明训练与推理链路可运行。

### 4.2 100-step 初筛

命令：

```powershell
.\.venv-gpu\Scripts\python.exe pretrain.py `
  --config configs/continual_ppo_phase_tiny.json `
  --stage all `
  --steps 100 `
  --batch-size 2 `
  --amp `
  --save-dir pretrained/phase_tiny_phaseaware_100steps_2026-08-12
```

结果：

```text
saved pretrained/phase_tiny_phaseaware_100steps_2026-08-12
```

compare：

```powershell
.\.venv-gpu\Scripts\python.exe compare.py `
  --config configs/continual_ppo_phase_tiny.json `
  --method-group all `
  --pretrained pretrained/phase_tiny_phaseaware_100steps_2026-08-12/model_best.pt `
  --delays 20 `
  --snrs 10 15 `
  --num-seeds 1 `
  --frames 4 `
  --pilot-total 128 `
  --pilot-layout prefix `
  --impairment-profile phase_tiny `
  --resume `
  --output-dir logs/compare_phaseaware_100steps_2026-08-12
```

## 5. 初筛结果

与上一轮旧 100-step 对比：

```text
old100:
SNR 10:
  Offline NN only = 0.0306
  Rule Modulation = 0.0332
SNR 15:
  Offline NN only = 0.0143
  Rule Modulation = 0.0124

phaseaware100:
SNR 10:
  Offline NN only = 0.0332
  Rule Modulation = 0.0306
SNR 15:
  Offline NN only = 0.0124
  Rule Modulation = 0.0156
```

判断：

- phase-aware condition 接入后链路稳定；
- 100-step 初筛结果是混合的：SNR 15 Offline 稍好，SNR 10 Offline 稍差；
- 没有立刻解决 DFE-RLS 差距；
- 需要更长训练或进一步模型结构增强才能判断它是否真正有效。

## 6. 当前结论

本轮完成的是“相位残差进入神经接收机条件”的工程闭环，不是性能突破。

当前判断：

```text
phase-aware condition 是必要接口；
但仅加入 2 个 latent 标量不足以立刻超过 DFE-RLS；
下一步要么跑 phase-aware 500-step 对齐上一轮训练长度，
要么增强模型对 phase residual 的使用方式。
```

## 7. 下一步建议

建议下一步二选一，推荐先做 A：

### A. 跑 phase-aware 500-step 对齐实验

目的：和上一轮 `phase_tiny_500steps` 公平对比。

```powershell
.\.venv-gpu\Scripts\python.exe pretrain.py `
  --config configs/continual_ppo_phase_tiny.json `
  --stage all `
  --steps 500 `
  --batch-size 4 `
  --amp `
  --save-dir pretrained/phase_tiny_phaseaware_500steps_2026-08-12
```

然后跑同一 compare：

```powershell
.\.venv-gpu\Scripts\python.exe compare.py `
  --config configs/continual_ppo_phase_tiny.json `
  --method-group all `
  --pretrained pretrained/phase_tiny_phaseaware_500steps_2026-08-12/model_best.pt `
  --delays 20 30 40 `
  --snrs 0 5 10 15 `
  --num-seeds 1 `
  --frames 8 `
  --pilot-total 128 `
  --pilot-layout prefix `
  --impairment-profile phase_tiny `
  --resume `
  --output-dir logs/compare_phaseaware_500steps_2026-08-12
```

### B. 若 A 仍不能超过 DFE-RLS，增强 conditioner

可能方向：

- 不只给 latent 两个标量，而是给每个 Pilot block 的 phase slope / residual energy；
- 在 unfolded iterations 中加入相位校正分支；
- 引入跨帧 phase state；
- 让 Rule modulation 的有效策略成为可学习监督先验；
- 增加 unfolded iterations 或显式 phase compensation layer。

当前不建议直接扩大 PPO。PPO 要等 Offline NN 本体接近或超过 DFE-RLS 后再上。
