# EME 长记忆 v2：物理 warm-start 与在线均衡阶段性结果

## 1. 本轮修正的根因

上一版 impaired checkpoint 在连续帧上出现“第 1、3 帧较好，第 2、4 帧突然崩溃”的交替现象。问题不是 EME 信道状态跨帧变化过快，也不是训练标签泄漏，而是 phase conditioner 的维度契约不一致：

- `estimate_phase_residual_vector(..., blocks=4)` 只生成 4 个 phase block，即 16 个统计量；
- EME 模型配置却使用 `phase_correction_segments=8`；
- 其余 4 个 segment 从 96 维 latent 的零填充区域读取，参与 phase/CFO 中位数聚合；
- 该错误会在不同 phase wrap 状态下产生不一致的整帧旋转。

现在 EME 主配置统一使用 `phase_correction_segments=4`，并在 `validate_model_dimensions()` 中拒绝不一致的模型配置。默认 legacy 模型仍保持原有配置和 checkpoint 兼容行为。

## 2. 物理 warm-start 结构

原模型的随机神经 head 在训练尚未收敛时会污染长记忆物理迭代结果。当前 EME 模型新增：

```text
已校正的接收帧
  -> acquisition CIR + soft tail
  -> 2 次可见信息约束的 LMMSE/CG 物理 warm-start
  -> 小增益神经残差 head
  -> 整帧输出
```

物理 warm-start 只使用接收端可见的 acquisition CIR、当前帧 Adapt Pilot 导出的 phase/CFO 特征、接收信号和历史 soft tail，不读取真实 CIR、真实 impairment 参数或 Data 标签。`neural_residual_scale=0.1` 用于限制训练早期神经残差对物理判决的破坏；只有显式开启 EME 配置的模型使用该路径，旧默认模型的 `physics_warm_start_iterations=0` 保持不变。

8 次 CG 只作为诊断参考。主配置采用 2 次，避免把高复杂度求解器隐藏在在线均衡结果中。

## 3. 阶段性结果

### 3.1 同一 seed 的修复前后对照

配置为 `eme_long_memory_v2`、`delay=116`、`SNR=10 dB`、`seed=0`、4 帧、前缀 Pilot。

修复 phase block 契约且使用物理 warm-start 后：

| 方法 | 平均 Data BER |
|---|---:|
| CFO+DD-Phase LMMSE-FIR | 0.1122 |
| Offline NN only | 0.0103 |
| Pilot-Driven Online Adaptation | 0.0103 |

Offline NN 的逐帧 BER 为 `0.0112/0.0179/0.0033/0.0089`，不再出现约 `0.8` 的交替崩溃。对应原始逐帧文件为：

```text
logs/eme_long_memory_v2_phase_consistent_compare/frame_metrics.jsonl
```

### 3.2 2 步训练 smoke

采用主配置的 2 次物理 warm-start、神经残差增益 0.1，训练 2 步后：

| 方法 | 平均 Data BER |
|---|---:|
| CFO+DD-Phase LMMSE-FIR | 0.1122 |
| Offline NN only | 0.0572 |
| Pilot-Driven Online Adaptation | 0.0592 |

训练损失从 `0.6947` 降到 `0.3008`。checkpoint 和结果为：

```text
pretrained/eme_long_memory_v2_warmstart_smoke2/model_best.pt
logs/eme_long_memory_v2_warmstart_smoke2_compare/summary.json
```

### 3.3 Level B 小矩阵

使用 3 个独立 seed、4 个主 SNR、每个配置 8 帧、`pilot_total=128`：

| SNR | CFO+DD LMMSE-FIR | CFO+DD DFE-RLS | Offline NN | Pilot Online |
|---:|---:|---:|---:|---:|
| 0 dB | 0.4105 | 0.4283 | 0.1861 | 0.2033 |
| 5 dB | 0.3280 | 0.3721 | 0.0850 | 0.0853 |
| 10 dB | 0.2997 | 0.1818 | 0.0492 | 0.0497 |
| 15 dB | 0.1960 | 0.1489 | 0.0378 | 0.0382 |

原始结果为：

```text
logs/eme_long_memory_v2_warmstart_main3x8/summary.json
logs/eme_long_memory_v2_warmstart_main3x8/frame_metrics.jsonl
```

这组结果说明：在当前 Level B 长记忆、残余 CFO/慢相位和前缀 Pilot 约束下，物理引导神经块均衡器在小矩阵中逐 SNR 超过两个公平传统 baseline。它还不能作为正式论文统计，原因是每个配置只有 3 个 seed、8 帧，而且主模型的训练仍是 smoke 规模。

### 3.4 Level B 5 seed × 30 帧主矩阵

在同一主配置下，使用 5 个独立 seed、每个 seed 30 帧、4 个主 SNR、单进程运行，共生成 2400 条逐帧记录：

| SNR | CFO+DD LMMSE-FIR | CFO+DD DFE-RLS | Offline NN | Pilot Online |
|---:|---:|---:|---:|---:|
| 0 dB | 0.4388 | 0.4447 | 0.2040 | 0.2056 |
| 5 dB | 0.3184 | 0.3170 | 0.0874 | 0.0816 |
| 10 dB | 0.1858 | 0.1715 | 0.0442 | 0.0443 |
| 15 dB | 0.1255 | 0.1325 | 0.0330 | 0.0331 |

原始文件为：

```text
logs/eme_long_memory_v2_warmstart_main5x30/summary.json
logs/eme_long_memory_v2_warmstart_main5x30/frame_metrics.jsonl
```

按前 5 帧与后 5 帧比较，Pilot Online 的 Data BER 为：

| SNR | 前 5 帧 | 后 5 帧 |
|---:|---:|---:|
| 0 dB | 0.1985 | 0.2211 |
| 5 dB | 0.0750 | 0.0819 |
| 10 dB | 0.0395 | 0.0463 |
| 15 dB | 0.0289 | 0.0354 |

600 个在线更新中有 361 个被 Adapt Pilot 更新和 Reward Pilot guard 接受，接受率为 `60.17%`。这证明在线链路可以在 30 帧递推中稳定运行，并且在四个主 SNR 上保持相对传统方法的明显优势；但它没有证明在线更新在每个 SNR 上都进一步超过 Offline NN。0 dB 的后段退化和 5 dB 的改善必须在正式论文中分别报告。

### 3.5 Level B 5 seed × 60 帧延长矩阵

将同一单进程矩阵扩展到每个 seed 60 帧后，共有 4800 条唯一逐帧记录：

| SNR | CFO+DD LMMSE-FIR | CFO+DD DFE-RLS | Offline NN | Pilot Online |
|---:|---:|---:|---:|---:|
| 0 dB | 0.4529 | 0.4473 | 0.2059 | 0.2105 |
| 5 dB | 0.3168 | 0.3038 | 0.0888 | 0.0910 |
| 10 dB | 0.1855 | 0.1817 | 0.0473 | 0.0473 |
| 15 dB | 0.1250 | 0.1215 | 0.0355 | 0.0355 |

Pilot Online 的前 5 帧/后 5 帧 Data BER 为：

| SNR | 前 5 帧 | 后 5 帧 |
|---:|---:|---:|
| 0 dB | 0.1985 | 0.2044 |
| 5 dB | 0.0750 | 0.0884 |
| 10 dB | 0.0395 | 0.0524 |
| 15 dB | 0.0289 | 0.0398 |

1200 个在线更新中有 722 个被接受，接受率仍为 `60.17%`；所有 `(method, SNR, seed, frame)` key 唯一。60 帧结果确认 proposed 在四个主 SNR 上持续明显优于公平传统 baseline，但在线相对 Offline NN 没有稳定的额外增益，甚至在 0/5 dB 有小幅代价。当前证据支持“在线 Pilot 适配在长时间递推中保持性能并受 guard 约束”，不支持“在线更新在所有条件下持续提升 BER”。

## 4. 在线性的证据边界

Pilot-Driven Online Adaptation 的每帧更新仍遵循：

```text
当前帧 Adapt Pilot
  -> Adapt BCE 更新受限 PEFT 参数
  -> Reward Pilot guard/reward
  -> 变差时回滚 PEFT，保留当前 soft tail
```

小矩阵中 0 dB 的 Pilot Online 比 Offline NN 高约 `0.0172` BER，说明低 SNR 下 Pilot 更新会受噪声影响；5/10/15 dB 的两者差异很小。当前最稳妥的结论是“在线方法保持了对传统方法的优势并具备安全 guard”，而不是声称在线更新已经稳定带来额外增益。

在线更新的审计字段包括：`adaptation_accepted`、`adapt_loss_before/after`、`reward_pilot_loss_before/after`、`parameter_delta_norm`、`tail_update_alpha` 和 `data_labels_used_online`。

## 5. 验证命令

相关回归测试：

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_eme_long_memory_profile.py tests/test_receiver_architecture.py
```

完整测试和正式统计仍需在提交前重新运行。正式论文结果还需要单进程、至少 5 个 channel seed、每个配置至少 30/60 帧，并报告前 5 帧、后 5 帧、Reward Pilot 与 Data BER 的配对统计。

## 6. 当前未完成项

1. 对 5 seed × 60 帧结果计算并保存 seed/连续帧 block bootstrap 置信区间。
2. 单列报告 0 dB 的在线更新退化、5 dB 的在线改善和 guard 接受率，不能用总体平均掩盖。
3. 在不使用 Data 标签的前提下比较固定 Pilot 更新、规则调度和离散 RL 调度；RL 只作为在线更新率/校正强度调度器，不替代 Pilot 驱动参数更新。
4. 若 60 帧统计中在线相对 Offline 仍无全 SNR 稳定增益，应将论文创新表述为“Pilot 驱动安全在线适配框架在长记忆模型失配下保持稳健”，而不是捏造显著在线增益。
