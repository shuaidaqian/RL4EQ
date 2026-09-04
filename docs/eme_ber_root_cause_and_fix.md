# EME Level B BER 根因与修正记录

## 1. 现象

旧主矩阵在 15 dB 仍出现约 14%--20% BER，容易误判为“神经网络无法处理
EME 长回波”。本记录把信道状态、物理求解器、模型权重和在线更新分开复核，避免
用增加帧数、信道编码或 PPO 调参掩盖根因。

## 2. 根因证据

### 2.1 acquisition 空档过大

旧配置在 acquisition 与首个数据帧之间设置 `30 s` 空档，而相干时间为 `120 s`。
EME 长记忆主场景的目标是慢时变，30 s 空档却会让复 tap 经过一次相关系数为
`exp(-30/120)=0.7788` 的 OU 演化，导致 acquisition-CIR 与数据 CIR 的失配被人为
放大。

在相同 `seed=50002`、15 dB、`cfo_phase_tiny` 条件下，acquisition-CIR 与首个数据
CIR 的相对误差为：

| 空档 | CIR 相对误差 | CIR 内积幅度 |
|---:|---:|---:|
| 0 s | 0.350 | 0.995 |
| 1 s | 0.399 | 0.941 |
| 30 s | 1.009 | 0.558 |

因此主配置改为 `0 s`。进一步考虑到 200 帧约覆盖 20.5 s，而 EME 主场景时变不应
严重，主配置相干时间改为 `1200 s`，对应 `rho_frame=0.99991467`；原 `120 s`
设置只保留为慢漂移压力测试。帧间慢 tap 漂移和跨帧 ISI 仍然保留。
30 s 老化只作为单独的 held-out/压力测试，不计入 Level B 主平均。

### 2.2 长记忆物理初值没有收敛

展开式神经均衡器的物理分支使用 CG 求解长记忆 LMMSE 初值。旧配置只迭代 2 次，
对于 `max_delay=116`、强径数 `12--24` 的信道不够。保持 checkpoint 权重不变，
只增加推理迭代次数，15 dB、0 s 空档、3 个困难 seed 的 Frozen NN 4 帧平均 BER 为：

| CG 次数 | seed 0 | seed 1 | seed 2 | 三 seed 平均 |
|---:|---:|---:|---:|---:|
| 2 | 4.88% | 2.93% | 3.24% | 3.68% |
| 4 | 1.42% | 0.39% | 0.92% | 0.91% |
| 8 | 0.28% | 0.08% | 0.36% | 0.24% |
| 16 | 0.14% | 0.03% | 0.31% | 0.16% |

主配置固定 `physics_warm_start_iterations=8`，这是精度和计算量的折中；16 次只作
收敛诊断，不作为主方法额外调参。

## 3. 当前可复现 checkpoint

`pretrained/eme_offline_physics8_gap0/model_best.pt` 由旧结构兼容的 checkpoint
迁移得到，网络可学习权重不变，仅使用新配置的非结构性物理推理参数。其离线验证
在 15 dB 上为 `0.1116%`（单验证 seed，1 帧）；正式统计仍以独立 seed、多帧矩阵为准。

旧 checkpoint 加载时，`compare.py` 只允许当前实验配置覆盖以下运行参数：

```text
physics_warm_start_iterations
physics_warm_start_scale
analytic_logit_skip_scale
neural_residual_scale
phase_correction_initial_scale
```

维度、层数和可学习结构仍严格服从 checkpoint，避免静默加载不兼容模型。

## 4. 在线边界

在线主方法固定为 Adapt Pilot 驱动的 phase/PEFT 更新。CIR 更新不和主在线结果混合，
使用 `--cir-update fixed`；`pilot_sparse` 只作为状态恢复消融。所有方法使用相同
Level B profile、相同 prefix Pilot（总 128，Adapt 96，Reward 32）和相同信道轨迹。

15 dB、3 个 seed、8 帧的复核结果：

| 方法 | 平均 Data BER | 结论 |
|---|---:|---|
| Pilot-conditioned frozen NN | 0.265% | 高 SNR 离线低于 10%，满足阶段性目标 |
| Pilot-Driven Online Adaptation | 0.256% | 在线低于 1%，且略低于冻结模型 |

随后使用当前 1200 s 主场景完成了 15 dB、3 seed、60 帧和 200 帧复核：

| 序列长度 | Frozen NN | Pilot 在线 | CFO+DD LMMSE | CFO+DD DFE |
|---:|---:|---:|---:|---:|
| 60 帧 | 0.216% | 0.213% | 11.856% | 12.418% |
| 200 帧 | 0.284% | 0.284% | 未纳入该次长期诊断 | 未纳入该次长期诊断 |

60 帧结果使用日志 `logs/eme_gap0_physics8_rho1200_fixedcir_15db_60f/`，200 帧结果
使用日志 `logs/eme_gap0_physics8_rho1200_fixedcir_15db_200f/`。在线在两种序列长度
上均保持低于 1%，但与 Frozen 的差距很小，因此当前证据支持“安全在线微调不破坏
离线性能并可提供微小改进”，尚不足以宣称在线适配带来大幅额外增益。

在线更新接受次数在该高 SNR 切片为 0，原因是已有模型在 Adapt/Reward Pilot 上已经
接近零损失；这说明“在线低于 1%”已经成立，但不能把该切片写成在线微调带来显著
增益。在线研究价值仍需在独立的慢状态老化/held-out 场景中用更大的 Reward Pilot
或更长窗口证明；主结果不能用 Data 标签选择动作。

## 5. 论文表述边界

当前可以写入论文的方法与阶段性结果是：

1. Level B 的主要困难来自多符号长记忆、跨帧 ISI、稀疏长回波和残余同步扰动；
2. acquisition 空档不应在慢时变主场景中设为 30 s，修正后主场景与物理假设一致；
3. 收敛的物理 warm-start 是神经残差能否发挥作用的必要条件；
4. 修正配置下高 SNR 离线 BER 已低于 10%，在线 phase/PEFT BER 已低于 1%；
5. 在线附加增益在当前高 SNR 切片很小，必须用 held-out/长期漂移矩阵补足，不应
   把接受率或单个 seed 写成普遍在线优势。

复核日志：

- `logs/eme_gap0_physics8_fixedcir_15db_8f_3seed/`
- `logs/eme_gap0_physics8_fixedcir_15db_20f/`
- `pretrained/eme_offline_physics8_gap0/`
