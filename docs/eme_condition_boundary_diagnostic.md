# EME 神经均衡条件边界诊断

## 目的

此前 `Offline NN only` 虽然将 `condition_update_mode` 设为 `fixed`，但公共推理路径仍会用当前帧 `Adapt Pilot` 计算相位特征。这样得到的结果无法回答在线收益来自哪里：当前 Pilot 的物理状态恢复、稀疏 CIR 更新，还是神经参数微调。

本阶段将神经接收机的条件来源显式拆分，并在每帧记录信息边界。

## 方法定义

| 方法 | CIR 条件 | 相位条件 | 参数更新 | 用途 |
|---|---|---|---|---|
| `Offline NN only` | acquisition CIR | acquisition 阶段冻结的相位条件 | 否 | 严格离线冻结基线 |
| `Pilot-conditioned frozen NN` | acquisition CIR | 当前帧 Adapt Pilot | 否 | 识别当前 Pilot 相位条件的贡献 |
| `Pilot CIR only` | 当前帧 Pilot 稀疏 CIR | 当前帧 Adapt Pilot | 否 | 识别物理状态恢复的贡献 |
| `Pilot-Driven Online Adaptation` | 当前帧 Pilot 稀疏 CIR | 当前帧 Adapt Pilot | 是，受 Reward Pilot 保护 | 在线均衡主线 |

严格 Frozen 并不是把相位条件强行置零，而是保存 acquisition 阶段估计的相位条件；它只禁止读取后续帧的 Pilot、更新 CIR 和更新网络参数。所有在线方法仍不得读取 Data 标签。

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
| `Offline NN only` | 72.71% | 否 | 否 | 否 |
| `Pilot-conditioned frozen NN` | 6.75% | 是 | 否 | 否 |
| `Pilot CIR only` | 6.75% | 是 | 是 | 否 |
| `Pilot-Driven Online Adaptation` | 6.53% | 是 | 是 | 第 2 帧接受 |

该切片说明当前 checkpoint 对相位条件高度敏感，当前 Pilot 相位恢复是在线链路的主要可见收益来源；两帧不足以证明 CIR 更新或 PEFT 的稳定增益。正式论文结论必须使用统一的 60/200 帧、多 seed 主矩阵，不能引用此 smoke 代替统计结果。

## 当前结论

本阶段完成的是实验因果边界修正，不是重新训练模型。后续统一主实验应至少包含严格 Frozen、Pilot-conditioned frozen、Pilot CIR only 和完整 Pilot-driven online 四个神经条件，并同时报告 0/5/10/15 dB 的逐配置 BER、后 5 帧 BER、Reward Pilot 指标和在线审计字段。
