# 迭代记录：phase correction branch 接入与 100-step 初筛

时间戳：2026-08-13 23:06:08

## 1. 本轮目标

当前主线仍是：先增强 `RL-Modulated Neural Block Equalizer` 的离线神经接收机本体；只有当 Offline NN 在 Level B 主配置上接近或超过传统非神经 baseline 后，再扩大 Continual PPO 开发实验。

本轮针对 `cfo_phase_tiny` 场景接入显式 phase/CFO correction branch，验证它是否能帮助神经块均衡器处理 residual CFO 与慢相位扰动。

## 2. 已完成代码改动

修改文件：

- `agent/unfolded_equalizer.py`
- `configs/continual_ppo_cfo_phase_tiny.json`
- `tests/test_receiver_architecture.py`

主要改动：

- 在 `UnfoldedConfig` 新增：
  - `enable_phase_correction_branch`
  - `phase_correction_segments`
- 默认关闭 phase branch，保证旧 checkpoint 的 `state_dict` 结构不被破坏。
- 在 `UnfoldedEqualizer` 中新增 `_apply_phase_correction(...)`。
- 当 branch 打开时，模型从 `condition.latent_residual` 中读取分段 phase/CFO 特征，对整帧接收 I/Q 做前置相位校正。
- 在 `configs/continual_ppo_cfo_phase_tiny.json` 中打开 phase branch：
  - `enable_phase_correction_branch=true`
  - `phase_correction_segments=4`

新增测试：

- `test_unfolded_equalizer_phase_correction_branch_uses_phase_vector`
- `test_unfolded_equalizer_default_config_keeps_legacy_state_dict_shape`

## 3. 已完成验证

架构与入口相关测试命令：

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests/test_receiver_architecture.py tests/test_meta_adaptation.py::test_pretrain_smoke_writes_strict_loadable_checkpoint tests/test_evaluation_contract.py::test_compare_cli_writes_real_level_b_metrics -q -p no:cacheprovider
```

结果：

```text
15 passed in 76.45s
```

2-step smoke 预训练：

```powershell
.\.venv-gpu\Scripts\python.exe pretrain.py --config configs/continual_ppo_cfo_phase_tiny.json --stage all --steps 2 --batch-size 1 --amp --save-dir pretrained/cfo_phase_tiny_phasebranch_smoke_2026-08-13
```

结果：

```text
saved pretrained/cfo_phase_tiny_phasebranch_smoke_2026-08-13
```

2-step compare smoke：

```powershell
.\.venv-gpu\Scripts\python.exe compare.py --config configs/continual_ppo_cfo_phase_tiny.json --method-group proposed --pretrained pretrained/cfo_phase_tiny_phasebranch_smoke_2026-08-13/model_best.pt --delays 20 --snrs 10 --num-seeds 1 --frames 1 --pilot-total 128 --pilot-layout prefix --impairment-profile cfo_phase_tiny --output-dir logs/compare_cfo_phase_tiny_phasebranch_smoke_2026-08-13
```

结果：

```text
saved logs/compare_cfo_phase_tiny_phasebranch_smoke_2026-08-13
```

100-step 初筛预训练：

```powershell
.\.venv-gpu\Scripts\python.exe pretrain.py --config configs/continual_ppo_cfo_phase_tiny.json --stage all --steps 100 --batch-size 2 --amp --save-dir pretrained/cfo_phase_tiny_phasebranch_100steps_2026-08-13
```

结果：

```text
saved pretrained/cfo_phase_tiny_phasebranch_100steps_2026-08-13
```

100-step 初筛 compare：

```powershell
.\.venv-gpu\Scripts\python.exe compare.py --config configs/continual_ppo_cfo_phase_tiny.json --method-group all --pretrained pretrained/cfo_phase_tiny_phasebranch_100steps_2026-08-13/model_best.pt --delays 20 30 40 --snrs 0 5 10 15 --num-seeds 1 --frames 5 --pilot-total 128 --pilot-layout prefix --impairment-profile cfo_phase_tiny --output-dir logs/compare_cfo_phase_tiny_phasebranch_100steps_2026-08-13
```

结果：

```text
saved logs/compare_cfo_phase_tiny_phasebranch_100steps_2026-08-13
```

## 4. 100-step 初筛结果

`logs/compare_cfo_phase_tiny_phasebranch_100steps_2026-08-13/frame_metrics.jsonl` 的方法均值：

| 方法 | mean BER_data |
|---|---:|
| RLS Linear | 0.172092 |
| LMMSE-FIR | 0.172917 |
| CFO-Corrected LMMSE-FIR | 0.172917 |
| CFO-Corrected DFE-RLS | 0.178385 |
| LMS | 0.232595 |
| CFO+DD-Phase DFE-RLS | 0.236328 |
| CFO+DD-Phase LMMSE-FIR | 0.239931 |
| NLMS | 0.278993 |
| RL-Modulated Neural Block Equalizer | 0.474913 |
| Offline NN only | 0.474957 |
| NN + Rule Modulation | 0.490712 |
| DFE-RLS | 0.548264 |
| SC-FDE-MMSE | 0.554601 |

预训练指标文件：

```text
pretrained/cfo_phase_tiny_phasebranch_100steps_2026-08-13/pretrain_metrics.json
```

关键摘要：

- `offline_nn_validation.mean_ber_data = 0.1380208432674408`
- `offline_nn_validation.gate_pass = false`
- `validation.mean_ber_data = 0.7005208333333334`
- Level B 阶段 loss 仍在较高区间，没有形成稳定收敛。

## 5. 分析

本轮只能说明 phase branch 的代码链路可运行，不能说明它已经提升接收机效果。

主要原因：

1. 100-step 训练步数太少，不足以评价新结构最终性能。
2. 当前实现会把 phase vector 直接作为强制校正作用到整帧输入上。
3. 在长 ISI 场景中，基于前缀 Pilot 得到的 phase/CFO 估计本身会混入信道记忆误差；如果估计偏差较大，强制校正会改变 Offline NN 已经适应的输入分布。
4. 因此 phase branch 可能从一开始就把接收 I/Q 旋到错误方向，导致训练早期更难。

和旧的 500-step 非 phase branch checkpoint 相比，当前 100-step phase branch 明显更弱，但二者训练步数不同，不能直接作为公平对比。

## 6. 当前结论

- phase correction branch 接口与 smoke 链路已打通。
- 当前“强制使用 phase vector 做整帧校正”的实现风险较大。
- Continual PPO 仍不应扩大：当前 Proposed 与 Offline NN only 几乎相同，说明 RL 还没有给出有效增益；在 Offline NN 本体弱于 LMMSE/RLS 时扩大 PPO 不会解决核心问题。

## 7. 下一步

下一步应把 phase correction branch 改为更保守的 gated 形式：

- 新增可学习 `phase_correction_scale`。
- 初始值设为 0 或很小，使 branch 初始近似 identity。
- 训练过程由模型自己学习是否、以及多大程度使用 phase/CFO 校正。
- 先用单元测试验证：
  - 默认关闭 branch 时 checkpoint 兼容性不变。
  - 打开 branch 且 initial scale=0 时输出近似 identity。
  - 打开 branch 且 initial scale=1 时能纠正已知旋转。
  - scale 是可训练参数。
- 通过后再做 2-step smoke、100-step 初筛和更长预训练。

