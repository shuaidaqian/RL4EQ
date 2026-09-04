# EME 在线均衡阶段性结果

> **当前修正结果（2026-09-04）**
>
> 本文早期章节记录了 `acquisition_to_data_gap_seconds=30` 的历史实验。该设置与
> 当前“EME 时变不严重”的主假设不一致，且会把 acquisition-CIR 老化错误地混入
> 主结果。当前主配置已固定为空档 `0 s`，仍保留帧间 `rho_frame=0.99991467` 的慢 tap
> 漂移和跨帧 ISI；随后根据 200 帧约 20.5 s 的时间跨度将主场景相干时间调整为
> `1200 s`、`rho_frame=0.99991467`；长记忆 CG warm-start 从 2 次固定为 8 次。
>
> 使用 `pretrained/eme_offline_physics8_gap0/model_best.pt`，Level B、15 dB、3 seed、
> 60 帧、prefix Pilot=128、`--cir-update fixed` 的复核结果为：Frozen NN `0.216%`，
> Pilot-Driven Online Adaptation `0.213%`，CFO+DD-Phase LMMSE-FIR `11.86%`，
> CFO+DD-Phase DFE-RLS `13.12%`。这组结果是当前主证据；后文 30 s 空档结果只作
> 历史诊断，不能与当前主平均混合。

## 研究问题

本阶段验证的问题不是“离线神经网络能否在某个固定信道上降低 BER”，而是：在同一 Level B 极端稀疏长回波 profile 内，acquisition 得到的信道状态逐渐老化后，在线接收机能否只利用当前帧前缀 Adapt Pilot 恢复状态，并在连续帧上优于 Frozen Offline NN。

主配置保持：

- `profile_name=eme_long_memory_v2`；
- `level=B`；
- `max_delay_seconds=0.0116`，符号率等效 `max_delay=116`；
- `strong_path_count=12..24`，`diffuse_energy_ratio=0.20..0.35`；
- `coherence_time_seconds=120`；
- `pilot_layout=prefix`，`pilot_total=128`，其中 Adapt Pilot 为 96 符号、Reward Pilot 为 32 符号；
- residual CFO 和慢相位扰动来自 `cfo_phase_tiny`；
- acquisition 与数据开始之间增加 `acquisition_to_data_gap_seconds=30`，按相干时间计算一次复 tap OU 状态推进。

该 gap 不是快速 Doppler。它表示 acquisition 完成到正式数据开始之间的校准老化，使在线问题具有明确的因果来源：Frozen 方法仍使用 acquisition 条件，在线方法使用当前 Adapt Pilot 恢复状态。

## 当前在线方法

在线链路为：

```text
acquisition CIR
    -> 固定主径 support 的 Pilot-only 稀疏 CIR 更新
    -> 由 Adapt Pilot 产生 PEFT 梯度候选
    -> Reward Pilot 选择候选或 skip
    -> 保留/回滚参数
```

`pilot_sparse_cir_update()` 只使用当前帧接收信号、Adapt Pilot 符号和上一帧 soft tail。它不读取 Data 标签、Reward Pilot 标签、真实 CIR 或真实 CFO。实现先使用 acquisition CFO 先验和当前 Adapt Pilot 的公共相位估计去除慢旋转，再在 acquisition 主径 support 上估计复 tap 增益。Reward Pilot 只承担候选选择和回滚，不参与梯度更新。

神经 PEFT 更新的历史默认方案是 `adapter_lora + conditioner_film`。当前 Level B 正式方案固定为独立的 `phase` 组；历史正式比较曾通过 `compare.py --cir-update pilot_sparse` 为所有方法启用共享的 Pilot-only CIR 更新，当前正式协议已将 Frozen Offline NN 的条件更新固定为 acquisition CIR，在线方法和传统 DFE-RLS 才使用当前帧 Pilot 做状态更新。

## 证据一：连续漂移恢复

实验目录（历史结果，Frozen 定义已在后文修正）：

```text
logs/eme_aged_pilotsparse_30f_2s/
```

配置为 Level B、`delay=116`、`SNR=10 dB`、2 个 seed、30 帧、prefix 128 Pilot、held-out edge 状态、`pilot_sparse` CIR 更新。逐帧 Data BER 汇总如下：

| 方法 | 前 5 帧累计 | 前 10 帧累计 | 前 20 帧累计 | 前 30 帧累计 |
|---|---:|---:|---:|---:|
| Frozen Offline NN | 0.04743 | 0.04715 | 0.04676 | 0.09604 |
| Pilot-Driven Online Adaptation | 0.05011 | 0.04766 | **0.04556** | **0.06174** |
| CFO+DD-Phase DFE-RLS | 0.22969 | 0.21847 | 0.22109 | 0.21804 |

第 30 帧单帧 BER 为：Frozen `0.27344`，Online `0.08594`，传统 DFE-RLS `0.15179`。这说明在线优势主要来自持续跟踪慢 tap 漂移，而不是首帧瞬时增益。在线适配接受次数为 30 帧中的 20 次。

## 证据二：主 SNR 切片

实验目录（历史结果，Frozen 定义已在后文修正）：

```text
logs/eme_main_aged_15f_1s/
```

该实验覆盖主 SNR `0/5/10/15 dB`，每个配置 1 个 seed、15 帧。累计平均 BER 如下：

| SNR | Frozen Offline NN | Pilot-Driven Online Adaptation | CFO+DD-Phase DFE-RLS |
|---:|---:|---:|---:|
| 0 dB | 0.22336 | 0.29874 | 0.45781 |
| 5 dB | 0.07046 | 0.07522 | 0.22314 |
| 10 dB | 0.02679 | 0.02865 | 0.22098 |
| 15 dB | 0.01488 | 0.01577 | 0.17582 |

该切片用于说明 SNR 依赖性，不作为最终显著性结论。低 SNR 下 32 符号 Reward Pilot 的方差较大，Pilot 梯度更新容易被拒绝或选错；中高 SNR 下应重点报告随帧数变化的恢复曲线。由于该切片生成时 Frozen 仍错误地执行了 Pilot CIR 更新，正式结论以文末 2026-09-01 主矩阵为准。

## 当前结论与未完成项

已经解决的主要问题：

1. acquisition 与数据状态之间具有明确、可审计的信道老化来源；
2. 老化仍由 120 秒相干时间推导，没有引入快速时变；
3. Pilot-only 稀疏 CIR 更新解决了 residual CFO/慢相位对 tap LS 的污染；
4. 修正前的历史实验曾显示在线方法在连续 20/30 帧上超过当时的 Frozen 名义结果；严格基线结论以文末主矩阵为准；
5. 在线更新的 Data 标签隔离和 Reward Pilot 回滚边界已经写入逐帧日志。

仍需在论文正式结果前完成：

- 主矩阵至少 60 帧、多个 seed 的重复实验；
- 0/5/10/15 dB 下分别报告前 1、5、10、20、30、60 帧窗口；
- 对 `fixed CIR`、`pilot_sparse CIR`、`decision-directed CIR` 做消融；
- 对 `Frozen NN`、`Pilot-only CIR`、`Pilot PEFT`、`Pilot CIR + PEFT` 做因果拆分；
- 检验 Reward Pilot 分块一致性门控是否能降低低 SNR 误更新；
- RL/Contextual Bandit 只作为动作调度消融，不能替代上述在线恢复主线。

这部分是修正前的阶段性结论，不能作为正式论文结论。严格冻结基线和 SNR 分层冻结后的正式结果见下文。

## 2026-09-01 基线边界修正与正式主矩阵

### 修正原因

此前使用 `--cir-update pilot_sparse` 时，`Offline NN only` 也会在每帧调用 Pilot-only CIR 更新。该方法虽然没有更新网络参数，但已经不再是严格的 Frozen Offline NN，因此此前的 Frozen 数值不能作为冻结基线证据。

现已在 `compare.py` 中把条件状态更新与参数更新分开：

- `Offline NN only`：模型参数冻结，CIR 固定为 acquisition CIR，唯一保留的是接收尾的正常递推；日志中的 `condition_update_mode=fixed`；
- `Pilot-Driven Online Adaptation`：使用当前帧前缀 Adapt Pilot 做稀疏 CIR 恢复，再用 Adapt Pilot 做受约束 PEFT 候选更新，Reward Pilot 只做接受或回滚；
- `CFO+DD-Phase DFE-RLS`：使用相同 acquisition/Pilot 信息边界，保持传统非神经、非 RL；
- 低于 `5 dB`：由于 Pilot 梯度和稀疏 CIR 估计的可靠性不足，在线链路整体冻结，避免把噪声当作状态变化；5 dB 及以上才开放 Pilot CIR 和 PEFT 更新。该层级由配置项 `online_adaptation_freeze_below_snr_db=5.0` 控制，日志分别记录 `fully_frozen` 与 `peft_enabled`。

### 正式实验协议

正式矩阵目录为：

```text
logs/eme_snr_freeze_main_60f_3s/
```

实验使用 `pretrained/eme_meta_from_offline_32/model_best.pt`，固定 `eme_long_memory_v2` Level B profile，`max_delay=116`，prefix Pilot 共 128 符号，`Adapt=96`、`Reward=32`，residual CFO 与慢相位扰动为 `cfo_phase_tiny`，acquisition 到数据开始的老化间隔为 30 秒。每个 SNR 使用 3 个 seed、连续 60 帧，比较 Frozen Offline NN、Pilot-driven online 和 CFO+DD-Phase DFE-RLS，共 2160 条逐帧记录。

### 主结果：累计 Data BER

下表的“前 N 帧”是在所有 seed 上合并计算的连续帧累计平均；这不是只取最后一帧，因此能够同时观察初始化和长期递推。

| SNR | 方法 | 前 1 帧 | 前 5 帧 | 前 10 帧 | 前 20 帧 | 前 30 帧 | 前 60 帧 | 后 5 帧 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 dB | Frozen Offline NN | 0.228795 | 0.333557 | 0.329613 | 0.327028 | 0.351352 | 0.361713 | 0.374405 |
| 0 dB | Pilot-driven online | 0.228795 | 0.316741 | 0.322545 | 0.334375 | 0.348624 | **0.357149** | 0.391071 |
| 0 dB | CFO+DD-Phase DFE-RLS | 0.441592 | 0.441295 | 0.457366 | 0.450856 | 0.449281 | 0.457930 | 0.472545 |
| 5 dB | Frozen Offline NN | 0.168899 | 0.193155 | 0.193043 | 0.232403 | 0.239323 | 0.261000 | 0.319568 |
| 5 dB | Pilot-driven online | 0.101935 | 0.108333 | 0.121652 | 0.143452 | 0.169891 | **0.227269** | **0.306250** |
| 5 dB | CFO+DD-Phase DFE-RLS | 0.445685 | 0.347173 | 0.337277 | 0.404799 | 0.441791 | 0.433811 | 0.359003 |
| 10 dB | Frozen Offline NN | 0.134673 | 0.169420 | 0.161161 | 0.175744 | 0.174975 | 0.200955 | 0.226488 |
| 10 dB | Pilot-driven online | 0.049851 | 0.055432 | 0.061979 | 0.116592 | 0.145015 | **0.184015** | **0.204167** |
| 10 dB | CFO+DD-Phase DFE-RLS | 0.421503 | 0.340625 | 0.328088 | 0.320368 | 0.335503 | 0.361533 | 0.401414 |
| 15 dB | Frozen Offline NN | 0.120536 | 0.204688 | 0.173698 | 0.164732 | 0.171726 | 0.198921 | 0.238690 |
| 15 dB | Pilot-driven online | 0.045387 | **0.043378** | **0.050298** | **0.051711** | **0.058346** | **0.123134** | 0.259747 |
| 15 dB | CFO+DD-Phase DFE-RLS | 0.422619 | 0.236086 | 0.259338 | 0.232031 | 0.237872 | 0.284542 | 0.307366 |

Frozen 与 Online 的逐帧配对差值定义为 `BER_Frozen - BER_Online`。在 60 帧全矩阵上，四档 SNR 的均值分别为 `0.004563`、`0.033730`、`0.016939`、`0.075787`，Online 优于 Frozen 的逐帧比例分别为 `46.11%`、`74.44%`、`74.44%`、`83.89%`。0 dB 的均值略为正，但后 5 帧出现退化，说明“低 SNR 全冻结”只能保证长期平均不劣，不能宣称低 SNR 仍有在线跟踪增益。

相对于传统 DFE-RLS，Online 在前 60 帧的 BER 降幅约为 `22.0%`、`47.6%`、`49.1%`、`56.7%`（对应 0/5/10/15 dB）。因此当前可以写入论文的核心结论是：在同一冻结的 Level B 极端稀疏长回波 profile、跨帧校准老化和 residual CFO/慢相位扰动下，Pilot-driven online receiver 在 4 个主 SNR 配置均明显优于公平传统非神经 baseline；在 5/10/15 dB 的长期累计 BER 上进一步优于严格 Frozen Offline NN。0 dB 应报告为可靠性边界，而不是在线增益配置。

### 当前研究结论边界

这组结果已经足以支撑“在线均衡有独立研究价值”：离线模型提供基础均衡能力，在线阶段利用每帧前缀 Pilot 恢复因 acquisition 老化而失配的稀疏长记忆条件，并在可靠 SNR 层对 PEFT 参数做安全微调。它尚不能支撑“任何 SNR、任何帧段都单调优于 Frozen”，尤其 15 dB 后 5 帧和 0 dB 后 5 帧仍有漂移或噪声导致的局部退化。

后续论文消融应使用严格定义：`Frozen + acquisition CIR`、`Frozen + Pilot CIR`、`Online PEFT + acquisition CIR`、`Online PEFT + Pilot CIR`，并将 RL/Contextual Bandit 保持为离散动作调度消融，而不是把 PPO 作为在线均衡本体。旧的 `logs/eme_aged_pilotsparse_30f_2s/` 和 `logs/eme_main_aged_15f_1s/` 只保留作历史诊断，不再作为正式主结果。

## 2026-09-02 长期诊断与资源消融补充

后续实验没有继续引入 Turbo、LDPC、信源编码或高阶调制。200 帧正交诊断显示，15 dB 下 `soft tail` 与 `oracle tail` 几乎重合，而 `oracle CIR` 在慢漂移 rho 下明显改善 BER，因此当前主要矛盾是跨帧 CIR 状态恢复，不是单纯 tail 表达能力。详细数据见 `docs/eme_long_episode_diagnostic_200f.md`。

已加入 Adapt Pilot 驱动的 `PilotPhysicalState`，用联合 CFO 网格 LS、phase0 拟合和置信度门控约束稀疏 tap 更新。60 帧、3 seed 的复核中，Pilot online 为 13.83%，传统 CFO+DD-Phase DFE-RLS 为 18.79%，严格 Frozen 为 14.03%；因此在线主线仍明显优于传统，但联合状态跟踪相对 Frozen 的额外优势尚不稳定，详细边界见 `docs/eme_joint_state_tracking_findings.md`。

Pilot 资源消融显示，prefix Pilot 从 128 增加到 160 后，单 seed 30 帧切片的在线 BER 从 6.95% 降至 4.53%；64/96 符号则发生严重后期退化。160 只作为待多 seed 验证的增强候选，正式主配置仍保持 128，详见 `docs/eme_pilot_resource_ablation.md`。

## 2026-09-03 安全门控统一矩阵

本文件此前的 2026-09-01/02 结果用于记录路线演进，不能覆盖随后完成的神经条件边界修正。最新、唯一可作为该轮主结果引用的矩阵位于 `logs/eme_guarded_unified_main_60f_3s/`，包含 6 种方法、4 个主 SNR、3 个 seed、60 连续帧，共 4320 条记录。完整协议、逐 SNR BER、CIR/PEFT 接受审计以及可写和不可写的论文结论见 `docs/eme_guarded_online_state_recovery_results.md`。

该矩阵确认 Proposed 在 0/5/10/15 dB 全部优于最优传统非神经基线，60 帧 BER 相对降幅为 28.52% / 42.82% / 29.37% / 23.21%。同时它否定了一个不应掩盖的结论：当前 32 符号 Reward Pilot 的 PEFT 候选选择不能稳定外推到数据段，故该版本不能将 PEFT 写成相对于 `Pilot-conditioned frozen NN` 的普适增益。下一阶段应首先验证更具代表性的 Pilot-only 窗口 reward 和离散更新调度，而不是扩大梯度更新强度。
