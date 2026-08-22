# 迭代记录：gated phase branch 与 Perfect-CSI 诊断

时间戳：2026-08-13 23:33:08

## 1. 本轮执行目标

继续当前路线：在不扩大 PPO 的前提下，先增强离线神经块均衡器本体。上一轮 `phase correction branch` 虽然能运行，但 100-step 初筛明显弱于传统 LMMSE/RLS，因此本轮重点是：

1. 把 phase correction branch 从“强制相位校正”改成更保守的 gated 形式。
2. 验证新分支不会破坏旧 checkpoint 默认结构。
3. 跑 smoke 和 100-step 初筛。
4. 解释为什么 `Perfect-CSI Block` 在 `cfo_phase_tiny` 下也很差，避免误判诊断结果。

## 2. 代码改动

修改文件：

- `agent/unfolded_equalizer.py`
- `tests/test_receiver_architecture.py`
- `configs/continual_ppo_cfo_phase_tiny.json`

新增配置项：

```python
phase_correction_initial_scale: float = 0.0
```

实现策略：

- `enable_phase_correction_branch=false` 时：
  - 不注册 `phase_correction` 子模块；
  - 不注册 `phase_correction_scale` 参数；
  - 保持旧 checkpoint `state_dict` 结构不变。
- `enable_phase_correction_branch=true` 时：
  - 注册 `phase_correction` residual branch；
  - 注册可学习标量 `phase_correction_scale`；
  - 最终作用到接收 I/Q 的校正相位为：

```text
phase = phase_correction_scale * (phase_vector_base + learned_residual)
```

- `phase_correction_scale` 初始为 0，使模型初始近似 identity，不再一开始强制使用可能有误差的 phase/CFO 估计。
- `phase_correction` 与 `phase_correction_scale` 纳入 `conditioner_film` PEFT 组，保证在线阶段可作为安全低维调制/微调目标，但默认构造后仍遵守全模型冻结契约。

配置更新：

```json
"enable_phase_correction_branch": true,
"phase_correction_segments": 4,
"phase_correction_initial_scale": 0.0
```

## 3. TDD 验证

新增/更新测试：

- `test_unfolded_equalizer_phase_correction_branch_uses_phase_vector`
  - 当 `phase_correction_initial_scale=1.0` 时，应能纠正已知 phase/CFO 旋转。
- `test_unfolded_equalizer_phase_correction_scale_zero_keeps_identity`
  - 当 `phase_correction_initial_scale=0.0` 时，输入输出应近似一致。
  - 构造后默认冻结。
  - 调用 `set_trainable_groups({"conditioner_film"})` 后，`phase_correction_scale` 可训练。
- `test_unfolded_equalizer_default_config_keeps_legacy_state_dict_shape`
  - 默认配置不包含 phase branch 参数。

RED 阶段结果：

```text
2 failed
TypeError: UnfoldedConfig.__init__() got an unexpected keyword argument 'phase_correction_initial_scale'
```

GREEN 阶段 targeted test：

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests/test_receiver_architecture.py::test_unfolded_equalizer_phase_correction_branch_uses_phase_vector tests/test_receiver_architecture.py::test_unfolded_equalizer_phase_correction_scale_zero_keeps_identity tests/test_receiver_architecture.py::test_unfolded_equalizer_default_config_keeps_legacy_state_dict_shape -q -p no:cacheprovider
```

结果：

```text
3 passed in 3.10s
```

架构与入口合约测试：

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests/test_receiver_architecture.py tests/test_meta_adaptation.py::test_pretrain_smoke_writes_strict_loadable_checkpoint tests/test_evaluation_contract.py::test_compare_cli_writes_real_level_b_metrics -q -p no:cacheprovider
```

结果：

```text
16 passed in 61.03s
```

全量测试：

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

结果：

```text
162 passed in 351.70s (0:05:51)
```

## 4. Smoke 与 100-step 初筛

2-step gated 预训练 smoke：

```powershell
.\.venv-gpu\Scripts\python.exe pretrain.py --config configs/continual_ppo_cfo_phase_tiny.json --stage all --steps 2 --batch-size 1 --amp --save-dir pretrained/cfo_phase_tiny_phasebranch_gated_smoke_2026-08-13
```

结果：

```text
saved pretrained/cfo_phase_tiny_phasebranch_gated_smoke_2026-08-13
```

2-step gated compare smoke：

```powershell
.\.venv-gpu\Scripts\python.exe compare.py --config configs/continual_ppo_cfo_phase_tiny.json --method-group proposed --pretrained pretrained/cfo_phase_tiny_phasebranch_gated_smoke_2026-08-13/model_best.pt --delays 20 --snrs 10 --num-seeds 1 --frames 1 --pilot-total 128 --pilot-layout prefix --impairment-profile cfo_phase_tiny --output-dir logs/compare_cfo_phase_tiny_phasebranch_gated_smoke_2026-08-13
```

结果：

```text
saved logs/compare_cfo_phase_tiny_phasebranch_gated_smoke_2026-08-13
```

100-step gated 预训练：

```powershell
.\.venv-gpu\Scripts\python.exe pretrain.py --config configs/continual_ppo_cfo_phase_tiny.json --stage all --steps 100 --batch-size 2 --amp --save-dir pretrained/cfo_phase_tiny_phasebranch_gated_100steps_2026-08-13
```

结果：

```text
saved pretrained/cfo_phase_tiny_phasebranch_gated_100steps_2026-08-13
```

100-step gated compare：

```powershell
.\.venv-gpu\Scripts\python.exe compare.py --config configs/continual_ppo_cfo_phase_tiny.json --method-group all --pretrained pretrained/cfo_phase_tiny_phasebranch_gated_100steps_2026-08-13/model_best.pt --delays 20 30 40 --snrs 0 5 10 15 --num-seeds 1 --frames 5 --pilot-total 128 --pilot-layout prefix --impairment-profile cfo_phase_tiny --output-dir logs/compare_cfo_phase_tiny_phasebranch_gated_100steps_2026-08-13
```

结果：

```text
saved logs/compare_cfo_phase_tiny_phasebranch_gated_100steps_2026-08-13
```

## 5. 100-step gated 结果

`logs/compare_cfo_phase_tiny_phasebranch_gated_100steps_2026-08-13/frame_metrics.jsonl` 的方法均值：

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
| NN + Rule Modulation | 0.448047 |
| Offline NN only | 0.458854 |
| NN + Fixed Modulation | 0.458854 |
| NN + Discrete PEFT Scheduler | 0.458854 |
| RL-Modulated Neural Block Equalizer | 0.459028 |
| DFE-RLS | 0.548264 |
| Fixed CG-BPSK-DD Block Detector | 0.552821 |
| SC-FDE-MMSE | 0.554601 |
| Perfect-CSI Block | 0.576997 |

与非 gated 100-step 对比：

| 接收机 | 非 gated 100-step | gated 100-step |
|---|---:|---:|
| Offline NN only | 0.474957 | 0.458854 |
| RL-Modulated Neural Block Equalizer | 0.474913 | 0.459028 |

结论：

- gated 版本比强制校正版本略好，说明“避免初始过校正”方向合理。
- 但 100-step gated 仍明显弱于 LMMSE/RLS，当前不能作为有效 Proposed 结果。
- PPO 仍几乎没有带来增益：`RL-Modulated Neural Block Equalizer` 与 `Offline NN only` 基本相同。

## 6. phase scale 学习情况

100-step gated best checkpoint 中：

```text
phase_correction_scale = -0.0054
```

说明模型没有主动学到显著使用显式 phase correction。当前更像是仍在依赖原始接收 I/Q 和 CIR 条件，而不是有效利用 phase branch。

## 7. 关于 Perfect-CSI Block 很差的解释

本轮额外新增并验证了无噪声长延迟一致性测试：

```text
test_perfect_cir_refine_recovers_noiseless_sparse_long_delay_bpsk
```

该测试构造：

- BPSK；
- 长延迟稀疏 CIR；
- 非零跨帧 tail；
- 无噪声；
- 使用 `LinearChannelOperator.forward()` 生成接收信号；
- 使用 `perfect_csi_bpsk_refine_detect()` 恢复。

结果：

```text
1 passed
```

因此，`Perfect-CSI Block` 在 `cfo_phase_tiny` 矩阵中 BER 很差，并不说明块算子在 clean 长延迟场景下错误。更合理解释是：

- `Perfect-CSI Block` 只知道真实 CIR；
- 它没有补偿 residual CFO 和慢相位扰动；
- 在 impaired 场景下，“只知道 CIR”并不是完整物理上界；
- 因此它只能作为诊断参考，不能作为主 baseline，也不能用来判断 proposed 是否已经具备真实上界性能。

这与当前 AGENTS.md 契约一致：`Perfect-CSI Block` 和 `Fixed CG-BPSK-DD Block Detector` 只作为诊断参考，不纳入主成功门槛。

## 8. 当前研究判断

当前没有证据表明“RL 已经能改善 Data BER”。主要事实是：

1. Offline NN 本体仍弱于传统 LMMSE/RLS。
2. PPO/调制动作基本退化到 Offline NN only。
3. phase branch 接入后工程链路可跑，但 100-step 下没有学成有效补偿。
4. 传统 LMMSE/RLS 在当前 `cfo_phase_tiny` + prefix pilot=128 设置下仍然很强。

因此，下一阶段不能直接扩大 PPO。第一性原理上，RL 只能在一个已经具备竞争力的神经接收机上做在线动作选择；如果底座接收机弱于传统方法，PPO 只是在弱模型附近做动作搜索，很难稳定超过传统 baseline。

## 9. 下一步建议

按优先级继续：

1. 先做更长的 gated phase branch 预训练，而不是 100-step 初筛。
   - 建议先跑 500 或 1000 steps。
   - 对比同样步数的非 branch checkpoint。
2. 检查 Offline NN 的训练损失是否真正下降，以及验证集是否和 compare 使用同一 Level B/prefix/0-15 dB/impaired 设置。
3. 如果 gated phase branch 仍不提升：
   - 保留 gated branch 代码但默认关闭；
   - 转向增强 unfolded detector 本体，例如更多 unfolded iterations、更强 CIR/support/noise 显式输入、更稳定的 LMMSE 初始化。
4. 只有当 Offline NN 在主配置上接近或超过 LMMSE/RLS 后，再启动 Continual PPO。
5. PPO 阶段继续坚持窗口级 Reward Pilot reward，不使用 Data 标签作为在线 reward。

