# EME 神经均衡条件边界诊断

## 目的

早期版本把“严格冻结 acquisition 条件”单独注册为方法，导致方法数量膨胀，也不能代表第二研究点的主问题。该方法现已从代码和主实验协议中删除。

本阶段将神经接收机的条件来源显式拆分，并在每帧记录信息边界。

## 方法定义

| 方法 | CIR 条件 | 相位条件 | 参数更新 | 用途 |
|---|---|---|---|---|
| `Frozen Offline NN` | acquisition CIR | acquisition 阶段固定条件 | 否 | 离线 checkpoint 冻结对照 |
| `Pilot CIR only` | 当前帧 Pilot 稀疏 CIR | 当前帧 Adapt Pilot | 否 | 识别物理状态恢复的贡献 |
| `Pilot-Driven Online Adaptation` | 当前帧 Pilot 稀疏 CIR | 当前帧 Adapt Pilot | 是，受 Reward Pilot 保护 | 在线均衡主线 |

`Frozen Offline NN` 的网络权重和 acquisition 条件都来自离线接收机；它不读取后续帧 Pilot，也不更新 CIR 或 PEFT 参数。Online 才使用当前帧 Pilot 做状态恢复和参数微调，所有在线方法仍不得读取 Data 标签。

## 审计字段

每帧结果现在包含：

- `condition_source`：`acquisition`、`pilot_phase` 或 `pilot_cir_phase`；
- `pilot_phase_used`：是否读取当前帧 Pilot 相位统计；
- `cir_update_applied`：当前帧是否允许 Pilot 驱动的 CIR 更新；
- `peft_update_applied`：当前帧是否接受 PEFT 参数更新；
- `data_labels_used_online`：在线路径固定为 `false`。

这些字段与 BER 同步写入 `frame_metrics.jsonl`，用于审计而不是作为性能指标。

## 诊断上界修正

新增 `Perfect-CSI + Pilot Phase`，在真实 CIR 诊断路径上使用当前 Adapt Pilot 估计并补偿公共 phase/CFO。原始 `Perfect-CSI Block` 保留为未补偿诊断项，但两者都不进入主 baseline 和主成功门槛。原因是已知 CIR 不能自动消除接收波形上的残余 CFO 与相位旋转。

## Smoke 记录

使用 `eme_long_memory_v2`、Level B、`delay=116`、15 dB、prefix Pilot=128、`cfo_phase_tiny`、1 个 seed、2 帧和已有 checkpoint 进行了协议 smoke。结果如下，仅用于验证边界，不作为论文统计：

| 方法 | 2 帧平均 BER | 当前帧 Pilot 相位 | CIR 更新 | PEFT 更新 |
|---|---:|---:|---:|---:|
| `Frozen Offline NN` | 72.71% | 否 | 否 | 否 |
| `Pilot CIR only` | 6.75% | 是 | 是 | 否 |
| `Pilot-Driven Online Adaptation` | 6.53% | 是 | 是 | 第 2 帧接受 |

该切片说明当前 checkpoint 对相位条件高度敏感；两帧不足以证明 CIR 更新或 PEFT 的稳定增益。正式论文结论必须使用统一的 60/200 帧、多 seed 主矩阵，不能引用此 smoke 代替统计结果。

## 当前结论

本阶段完成的是实验因果边界修正，不是重新训练模型。当前主实验只包含传统均衡器、`Frozen Offline NN` 和完整 `Pilot-Driven Online Adaptation`；`Pilot CIR only` 仅在需要拆分物理状态贡献时作为专项消融，并同时报告 0/5/10/15 dB 的逐配置 BER、后 5 帧 BER、Reward Pilot 指标和在线审计字段。

## 2026-09-03 安全门控复核

上述四个神经条件与两种传统 baseline 已在 Level B、`delay=116`、prefix Pilot=128、0/5/10/15 dB、3 seed、60 帧协议下完成统一矩阵。完整结果存于 `logs/eme_guarded_unified_main_60f_3s/`，解释与主表存于 `docs/eme_guarded_online_state_recovery_results.md`。

复核的核心事实是：当前帧 Pilot 物理条件对神经恢复很重要，不能把这部分收益误报成 PEFT 增益。当前主问题是比较 Frozen 的离线固定条件与 Online 的 Pilot 驱动恢复/微调；若需要单独归因 PEFT，使用 `--online-condition-source acquisition` 做参数微调因果消融。
