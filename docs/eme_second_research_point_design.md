# 第二研究点：EME 类长记忆信道下的在线均衡

## 1. 核心问题

第一研究点建立 EME 信道建模基础。第二研究点承接“长延迟和稀疏回波导致接收机存在跨帧记忆”这一物理问题，但不假设第一研究点中的离散信道参数完全准确，而采用独立的公开资料约束 Level B 信道。第二研究点研究的问题是：在 acquisition 之后 CIR、residual CFO 和慢相位仍随帧演化时，离线训练得到的整帧神经均衡器能否通过 Pilot 驱动的在线更新继续改善性能。

## 2. 接收机和信息边界

接收机是整帧缓冲、非因果块神经均衡器。离线阶段使用带 Data 标签的整帧样本训练模型；在线阶段每帧只使用前缀 Pilot，其中前 96 个 Pilot 是 Adapt Pilot，后 32 个 Pilot 是 Reward Pilot。在线适配器从 Adapt Pilot 计算监督损失，Reward Pilot 只用于动作后验收和回滚，Data 标签不进入在线 observation、loss、action、更新或回滚。

## 3. 固定的在线方法

正式方法固定为：

`Pilot 物理状态恢复 + phase 低维 PEFT 更新 + Reward Pilot 跨帧回滚`

其中：

- 物理状态恢复估计当前帧 residual CFO 和慢相位，并在低置信度时保持上一状态；
- phase 组只包含相位校正支路参数，不更新普通 conditioner、head 或 Adapter/LoRA；
- 在线更新是 Adapt Pilot 上的受限梯度步，不逐 bit 判决，不直接输出高维参数增量；
- 更新后的 Reward Pilot 损失必须达到 `1e-5` 的最小改善；
- 下一帧若当前更新后的参数在同一 Reward Pilot 上劣于更新前快照，则恢复更新前快照；
- SNR 低于 5 dB 时冻结 PEFT 更新，但仍允许物理状态观测和神经推理。

## 4. 离线与在线的唯一主对照

主结果只使用三类方法：公平的传统均衡器、`Frozen Offline NN` 和
`Pilot-Driven Online Adaptation`。Frozen 与 Online 使用同一个离线 checkpoint、
同一条 Level B 信道轨迹、同一份 Pilot 划分和同一个跨帧 soft-tail 更新规则；Frozen
冻结网络参数和 CIR 更新，但使用当前帧已知 Pilot 形成模型所需的 phase/CFO 条件；Online
在同一个离线 checkpoint 上再使用当前帧 Pilot 恢复物理状态并更新受限 PEFT 参数，再由
Reward Pilot 决定接受或回滚。这样比较才能把“离线网络直接推理”与“在线状态恢复及参数
微调”放在公平的同一信息边界上。需要单独测量 PEFT 本身时，另行使用
`--online-condition-source acquisition`，不改变主实验定义。

## 5. 更新对象消融

在相同 Level B、相同 checkpoint、相同 seed 和相同 Pilot 协议下比较 `head`、`phase`、`conditioner_film` 和 `adapter_lora`。消融的第一轮固定 CIR/phase 条件，只改变可训练组，避免把物理状态恢复效果误归因于参数对象；第二轮再把候选对象放回完整的 Pilot 状态跟踪链。

最终对象按跨 SNR、多 seed、接受率、回滚率和后期帧稳定性共同选择，不按单一 SNR 的最低 BER 选择。

## 6. 论文论证链

1. 公开资料约束的 Level B 信道具有多符号长记忆、稀疏长回波、跨帧 ISI 和同步残差。
2. 200 帧诊断显示固定 CIR 在慢 tap 漂移下会随时间退化，而 Oracle CIR 可显著降低 BER，说明在线状态恢复有明确物理空间。
3. 离线神经均衡器在完整帧监督下建立强初始化，解决“离线网络能否工作”的问题。
4. 在线阶段只用 Pilot 做物理状态恢复和受限参数更新，解决“信道状态老化后如何适配”的问题。
5. 与 CFO/慢相位补偿的传统均衡器比较，验证在线神经方法是否在每个主 SNR 层都保持优势。
6. 通过 60 帧正式矩阵和 200 帧长期诊断分别回答平均性能和长期稳定性问题。
7. Contextual Bandit 只有在固定对象和安全回滚已经证明有效、且 Reward Pilot 能稳定排序动作收益时才引入；否则不把 Bandit 作为第二研究点的必要组成。

## 7. 当前论文中必须诚实报告的风险

在线方法相对于冻结网络的增益不要求在每个 SNR 都为正，但必须报告方向翻转、回滚触发和低置信度状态比例。主成功门槛是相对于传统非神经、非 RL baseline 的逐配置优势；“在线更新一定优于冻结网络”不能在证据不足时写成结论。

## 7. 历史实验结果

以下结果来自旧的条件边界或旧的更新协议，只保留作过程记录，不能作为当前论文结果：

| SNR | CFO+DD-Phase LMMSE-FIR | CFO+DD-Phase DFE-RLS | Frozen Offline NN | Pilot phase online |
|---:|---:|---:|---:|---:|
| 0 dB | 0.459536 | 0.458761 | 0.318862 | 0.327933 |
| 5 dB | 0.407676 | 0.400372 | 0.231988 | 0.228993 |
| 10 dB | 0.237054 | 0.260534 | 0.165532 | 0.167832 |
| 15 dB | 0.182515 | 0.187872 | 0.140309 | 0.137289 |

因此不能用这组历史数字判断当前 Frozen/Online 的相对性能。当前正式结果见后文“条件边界纠错后的 60 帧复核”。

## 8. 200 帧稳定性结果

在 15 dB、3 个 seed、200 帧长期诊断中：

| 方法 | 1-16 帧 | 185-200 帧 | 1-200 帧 |
|---|---:|---:|---:|
| Frozen Offline NN | 0.126511 | 0.208380 | 0.163631 |
| Pilot phase online | 0.122559 | 0.193243 | 0.164128 |
| CFO+DD-Phase LMMSE-FIR | 0.156715 | 0.226539 | 0.209753 |
| CFO+DD-Phase DFE-RLS | 0.183501 | 0.220703 | 0.208890 |

在线方法末段 BER 低于冻结网络，说明跨帧 Reward Pilot 回滚能够限制长期失配；但 1-200 帧平均仍与冻结网络基本持平，不能据此声称在线 PEFT 已经稳定提高平均 BER。200 帧中 phase 更新接受 56 次、回滚 56 次，表明当前更新主要承担保守试探和失配抑制作用。

## 9. Contextual Bandit 决策

不引入 Contextual Bandit。原因是 Pilot replay 诊断在窗口 2、4、8 下的 Reward 与后续 Data BER Spearman 分别为 -0.041、-0.182、-0.251，均未通过预设门槛。此时增加 Bandit 只会学习一个不能可靠预测长期收益的 Reward 信号，无法形成可解释的研究贡献。若后续重新设计 Reward，应先在固定动作离线 replay 上通过排序门槛，再考虑 Bandit。

正式日志：

- `logs/eme_formal_main_60f_phase/`
- `logs/eme_long_formal_200f_phase/`
- `logs/eme_long_formal_200f_phase_replay/`

## 10. 稳定性校准后的补充结论

在加入 phase PEFT proximal trust-region 后，最终配置固定为：`learning_rate=4e-4`、`steps=1`、`max_delta_norm=1.0`、`proximal_weight=0.1`、`min_reward_improvement=5e-4`，phase 状态置信度门限保持 `0.5`。门限校准显示，放宽到 `0.15/0.30/0.40` 会把不可靠的稀疏 CIR 更新写回接收机，导致明显退化，因此不能把“更多物理状态更新”直接等同于“更好的在线均衡”。

修正版 60 帧矩阵和 200 帧长期诊断分别记录在：

- `logs/eme_formal_main_60f_phase_stable/`
- `logs/eme_long_formal_200f_phase_stable/`

修正版 200 帧中，Online phase 的前 16 帧 BER 为 `0.122652`，后 16 帧 BER 为 `0.193522`，均优于 Frozen NN 的 `0.126511` 和 `0.208380`；但全程平均 BER 为 `0.169470`，仍高于 Frozen NN 的 `0.163631`。因此第二研究点应把贡献表述为“Pilot 驱动的保守在线适配改善部分工作区间并抑制长期退化”，不能表述为“在线微调在所有 SNR 和全程平均上都超过冻结网络”。

## 2026-09-04：在线微调因果边界修正

为避免把接收机状态递推差异误算为在线参数增益，Frozen 和 Online 现在共享同一个
`tail_update_alpha`。主矩阵还要求 Frozen 使用当前帧已知 Pilot 形成推理条件，只冻结网络
参数和 CIR 更新。修正后的 Level B 主矩阵（`max_delay=116`、prefix Pilot=128、
3 seed、60 帧、`cfo_phase_tiny`）为：

| SNR | CFO+DD LMMSE-FIR | CFO+DD DFE-RLS | Frozen Offline NN | Pilot-Driven Online |
|---:|---:|---:|---:|---:|
| 0 dB | 45.477% | 44.115% | 20.152% | 20.152% |
| 5 dB | 34.663% | 31.466% | 6.040% | 6.040% |
| 10 dB | 22.319% | 17.772% | 1.079% | 1.079% |
| 15 dB | 11.856% | 12.418% | 0.213% | 0.213% |

这组结果证明主线在所有主 SNR 上超过传统 baseline，但不支持把当前 PEFT 写成稳定的
独立增益：0 dB 完全冻结，5 dB 仅 1 次短暂接受并随后回滚，10/15 dB 没有接受更新。

另外增加了 `--online-condition-source acquisition` 的参数微调因果消融。在同一 held-out
edge 信道、Reward Pilot=64 的小样本中，严格 acquisition 条件下只更新 phase 参数，4 帧、1 seed
的 BER 从严格离线约 `72.545%` 降至 `53.097%`；但帧间出现明显振荡，说明 phase-only 更新能够
修正一部分当前 Pilot 可见的状态失配，却不能单独恢复稀疏长回波结构，也不能作为最终主配置。

因此第二研究点的最终实验逻辑固定为：

1. `Frozen Offline NN` 作为离线 checkpoint 参数冻结、使用当前帧 Pilot 推理条件的强对照；
2. `Pilot-Driven Online Adaptation` 作为 Adapt Pilot 自监督、Reward Pilot 验收/回滚的主在线方法；
3. `acquisition + phase` 仅用于证明在线参数更新的因果作用及其边界；
4. 更新对象和更新间隔必须通过逐配置、逐帧配对结果确定，不能仅凭 Reward Pilot 接受率或单帧 BER 宣称在线优势。

## 2026-09-05：主对照条件边界纠错后的复核

上一轮曾错误地将 `Frozen Offline NN` 固定为 acquisition CIR/phase 条件，导致离线网络在
跨帧状态变化下严重失配；该定义把冻结网络错误变成了诊断组，不能作为主对照。现已修正为：
Frozen 使用 acquisition CIR、当前帧前缀 Pilot 的 phase/CFO 推理条件，但不更新 CIR 或
PEFT；`Pilot-Driven Online Adaptation` 才进一步使用 Adapt Pilot 做状态恢复和 PEFT 更新，
并继续使用 Reward Pilot 做验收和回滚。两种方法仍共享同一个 checkpoint、信道轨迹、帧结构
和 soft-tail 递推。

上一轮错误边界下的 `50.09%/50.43%/50.55%/50.70%` Frozen 数值以及相应 Online
优势全部作废，不能写入论文。修正后的小规模复核在 10 dB、4 帧、1 seed 下，Frozen 和
Online 均为 `0.00865`，说明 BER 回到了离线网络的正常数量级；由于该切片中没有接受 PEFT
更新，它只证明边界修正有效，不证明在线微调已经带来独立增益。后续必须在正确 Frozen
定义下重新跑 60/200 帧、多 seed 矩阵，再测量状态恢复和 PEFT 的独立贡献。

### 修正后的 60 帧、3 seed 主复核

使用 `pretrained/eme_meta_from_offline_32/model_best.pt`、`eme_long_memory_v2` Level B、
`max_delay=116`、`cfo_phase_tiny`、prefix Pilot=128、0/5/10/15 dB、3 seed、60 帧，
两种神经方法共生成 1440 条帧记录。Frozen 和 Online 的 Data BER 完全一致：

| SNR | Frozen Offline NN | Pilot-Driven Online Adaptation |
|---:|---:|---:|
| 0 dB | 20.178% | 20.178% |
| 5 dB | 6.057% | 6.057% |
| 10 dB | 1.084% | 1.084% |
| 15 dB | 0.213% | 0.213% |

Online 在 0/5/10/15 dB 的 PEFT 接受次数分别为 `0/1/0/0`，5 dB 的唯一一次接受随后
回滚。这个结果首先修复了“离线 BER 接近 50%”的错误：正确的 Frozen 是离线 checkpoint
参数冻结、使用当前帧 Pilot 推理条件，而不是固定过时 acquisition phase。其次，它诚实地
表明当前 phase-only 在线微调在该主矩阵中尚未产生可重复的独立 BER 增益；后续研究重点应
放在更合适的低维更新对象、Reward Pilot 的跨区域泛化和 200 帧稳定性，而不能把错误的
acquisition 失配差异写成在线微调收益。

进一步的 acquisition 条件参数微调探针显示，当前 `phase` PEFT 会出现 Reward Pilot
损失微小改善但 Data 段不改善的情况，说明短 Reward Pilot 容易接受不具备跨区域泛化性的
更新。本轮已加入两个子窗口一致性门控，后续必须用长帧、多 seed 重新验证其拒绝错误更新
的能力，再决定是否更换 PEFT
对象；不能通过放大步长强行制造在线增益。
