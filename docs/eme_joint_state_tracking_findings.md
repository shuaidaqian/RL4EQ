# EME 联合 CIR/CFO/phase 状态跟踪阶段结果

## 实现

新增 `agent/cir_estimator.py` 中的 `PilotPhysicalState` 和 `track_pilot_physical_state()`：

1. 用 acquisition/当前 CIR、上一帧 soft tail 和当前帧 Adapt Pilot 构造已知参考；
2. 在残余 CFO 范围内进行 81 点小网格搜索；
3. 每个 CFO 候选都只用 Adapt Pilot 做固定 support 的复 tap LS；
4. 选择 pilot 重构残差最小的 CFO，再估计当前 phase0；
5. 依据 pilot 数量、相位残差方差和 LS 重构误差计算 confidence；
6. 低于置信度阈值时保留上一物理状态，不用不可靠结果更新 CIR。

这不是自由输出完整高维 CIR 的神经网络，而是物理化的低维状态恢复。状态对象显式标记 `data_labels_used_online=False`。

## 与在线均衡的连接

对于 `Pilot-Driven Online Adaptation` 和 `pilot_sparse` 神经消融，处理顺序为：

```text
Adapt Pilot
  -> PilotPhysicalState
  -> 置信度门控
  -> 稀疏 tap CIR 更新
  -> phase conditioner
  -> Adapt Pilot PEFT 候选更新
  -> Reward Pilot 接受/回滚
```

严格 `Offline NN only` 仍固定 acquisition CIR，不启用该在线状态恢复。

## 15 dB、delay=116、30 帧、2 seed

| 方法 | seed=0 累计 BER | seed=1 累计 BER | 两 seed 平均 |
|---|---:|---:|---:|
| Offline NN only | 0.068713 | 0.110677 | 0.089695 |
| Pilot online，初版 phase 斜率 | 0.114360 | 0.077046 | 0.095703 |
| Pilot online，联合 LS + 置信度门控 | 0.069531 | 0.111570 | 0.090551 |

对应 60 帧、3 seed 的正式复核为：

| 方法 | 60 帧累计 BER | 后 5 帧 BER |
|---|---:|---:|
| Offline NN only | 0.140309 | 0.156994 |
| Pilot online，联合状态跟踪 | 0.138349 | 0.137723 |
| CFO+DD-Phase DFE-RLS | 0.187872 | 0.214732 |

数据目录：

```text
logs/eme_joint_phase_short_core/
logs/eme_joint_phase_main_60f_3s/
```

## 研究判断

联合状态跟踪已经改善了部分 seed 的后期稳定性，但当前证据不足以宣称它单独带来稳定的大幅 Frozen 增益。它目前最可靠的论文定位是：在线均衡链路中的物理状态恢复模块，主要贡献是把 residual CFO/phase 与 tap 更新放在同一 Pilot-only 状态估计边界内。

主目标“在线神经均衡明显优于传统非神经、非 RL baseline”仍成立；60 帧复核中在线 BER 为 13.83%，传统 DFE-RLS 为 18.79%。对于 Frozen 的增益仍需依赖更高 Pilot 资源和后续更充分的多 seed 统计，不应夸大当前联合跟踪结果。
## 不确定性 tail 模块

新增 `agent/tail_state.py` 的 `SoftTailState` 和 `update_soft_tail_state()`，维护逐符号均值、方差和置信度，并按置信度抑制低可靠 tail 跳变。200 帧诊断表明 soft/oracle tail 差异很小，因此该模块当前作为可选增强，不替换主实验递推，也不进入主结论。
