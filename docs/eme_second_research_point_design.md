# 第二研究点：EME 类长记忆信道下的在线均衡

## 1. 核心问题

第一研究点建立 EME 信道建模基础。第二研究点承接“长延迟和稀疏回波导致接收机存在跨帧记忆”这一物理问题，但不假设第一研究点中的离散信道参数完全准确，而采用独立的公开资料约束 Level B 信道。第二研究点研究的问题是：在 acquisition 之后 CIR、residual CFO 和慢相位仍随帧演化时，离线训练得到的整帧神经均衡器能否通过 Pilot 驱动的在线更新保持或恢复性能。

## 2. 接收机和信息边界

接收机是整帧缓冲、非因果块神经均衡器。离线阶段使用带 Data 标签的整帧样本训练模型；在线阶段每帧只使用前缀 Pilot，其中前 96 个 Pilot 是 Adapt Pilot，后 32 个 Pilot 是 Reward Pilot。在线适配器从 Adapt Pilot 计算监督损失，Reward Pilot 只用于动作后验收和回滚，Data 标签不进入在线 observation、loss、action、更新或回滚。

## 3. 固定的在线方法

正式方法固定为：

`Pilot 物理状态恢复 + phase 低维 PEFT 更新 + 每 8 帧调度 + Reward Pilot 跨帧回滚`

其中：

- 物理状态恢复估计当前帧 residual CFO 和慢相位，并在低置信度时保持上一状态；
- phase 组只包含相位校正支路参数，不更新普通 conditioner、head 或 Adapter/LoRA；
- 在线更新是 Adapt Pilot 上的受限梯度步，不逐 bit 判决，不直接输出高维参数增量；
- 更新后的 Reward Pilot 损失必须达到 `1e-5` 的最小改善；
- 下一帧若当前更新后的参数在同一 Reward Pilot 上劣于更新前快照，则恢复更新前快照；
- SNR 低于 5 dB 时冻结 PEFT 更新，但仍允许物理状态观测和神经推理。

## 4. 更新对象消融

在相同 Level B、相同 checkpoint、相同 seed 和相同 Pilot 协议下比较 `head`、`phase`、`conditioner_film` 和 `adapter_lora`。消融的第一轮固定 CIR/phase 条件，只改变可训练组，避免把物理状态恢复效果误归因于参数对象；第二轮再把候选对象放回完整的 Pilot 状态跟踪链。

最终对象按跨 SNR、多 seed、接受率、回滚率和后期帧稳定性共同选择，不按单一 SNR 的最低 BER 选择。

## 5. 论文论证链

1. 公开资料约束的 Level B 信道具有多符号长记忆、稀疏长回波、跨帧 ISI 和同步残差。
2. 200 帧诊断显示固定 CIR 在慢 tap 漂移下会随时间退化，而 Oracle CIR 可显著降低 BER，说明在线状态恢复有明确物理空间。
3. 离线神经均衡器在固定 acquisition 条件下建立强初始化，解决“离线网络能否工作”的问题。
4. 在线阶段只用 Pilot 做物理状态恢复和受限参数更新，解决“信道状态老化后如何适配”的问题。
5. 与 CFO/慢相位补偿的传统均衡器比较，验证在线神经方法是否在每个主 SNR 层都保持优势。
6. 通过 60 帧正式矩阵和 200 帧长期诊断分别回答平均性能和长期稳定性问题。
7. Contextual Bandit 只有在固定对象和安全回滚已经证明有效、且 Reward Pilot 能稳定排序动作收益时才引入；否则不把 Bandit 作为第二研究点的必要组成。

## 6. 当前论文中必须诚实报告的风险

在线方法相对于冻结网络的增益不要求在每个 SNR 都为正，但必须报告方向翻转、回滚触发和低置信度状态比例。主成功门槛是相对于传统非神经、非 RL baseline 的逐配置优势；“在线更新一定优于冻结网络”不能在证据不足时写成结论。

## 7. 正式实验结果

当前正式 60 帧主矩阵使用 `pretrained/eme_meta_from_offline_32/model_best.pt`，包含 4 个 SNR、3 个 seed、60 帧、前缀 Pilot 128，以及每 8 帧一次在线调度。结果如下：

| SNR | CFO+DD-Phase LMMSE-FIR | CFO+DD-Phase DFE-RLS | Pilot-conditioned frozen NN | Pilot phase online |
|---:|---:|---:|---:|---:|
| 0 dB | 0.459536 | 0.458761 | 0.318862 | 0.327933 |
| 5 dB | 0.407676 | 0.400372 | 0.231988 | 0.228993 |
| 10 dB | 0.237054 | 0.260534 | 0.165532 | 0.167832 |
| 15 dB | 0.182515 | 0.187872 | 0.140309 | 0.137289 |

因此在线方法在所有主 SNR 层均明显优于传统补偿器；相对于冻结网络，5 dB 和 15 dB 有小幅改善，0 dB 和 10 dB 有小幅退化。这个结果支持“在线状态适应具有工程和物理价值”，但不支持夸大为“在线 PEFT 在所有 SNR 都带来额外 BER 增益”。

## 8. 200 帧稳定性结果

在 15 dB、3 个 seed、200 帧长期诊断中：

| 方法 | 1-16 帧 | 185-200 帧 | 1-200 帧 |
|---|---:|---:|---:|
| Pilot-conditioned frozen NN | 0.126511 | 0.208380 | 0.163631 |
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
