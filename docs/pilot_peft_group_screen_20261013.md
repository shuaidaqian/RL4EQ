# Pilot-only PEFT 参数组筛选

> **2026-10-14 协议更正：**本页中由旧 `replay_pilot_candidate_ranking.py`
> 生成的数值使用了 legacy 默认信道、512 符号帧和 clean 扰动，不能代表真实
> EME Level B 主场景或 acquisition 老化。旧信号重构目标的相位补偿顺序也与
> 实际前向不一致。历史结果保留作问题定位，不作为新方案选型依据；统一协议与
> 后续证据见 [Pilot replay 协议复核](pilot_replay_protocol_audit_20261014.md)。

日期：2026-10-13

## 目的

在不改变离线 checkpoint 的前提下，检查普通 `pilot_encoder`、Adapter 和 LoRA
参数组是否能够被 Adapt Pilot 的真实梯度更新有效调制，并观察这种变化是否传到
Data 段输出。该实验是参数位置筛选，不是主实验统计，也不使用 Data 标签做在线
更新、候选选择或回滚。

## 固定条件

- checkpoint：`pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`
- 信道：Level B，`delay=116`
- acquisition 到 Data 的间隔：`600 s`
- Pilot：prefix，总长度 256
- SNR：5、10、15 dB
- seed：0、1、2
- 每条轨迹：2 帧
- 更新：Adapt Pilot BCE，AdamW，1 步，学习率 `1e-4`
- 候选：identity、`pilot_encoder`、`adapter`、`attention_lora`、`ffn_lora`

原始结果：
`logs/pilot_peft_group_screen_3s3snr2f_gap600_20261013.json`

## 结果

| 参数组 | Data logits 平均绝对变化 | Data soft output 变化 | Data BCE 改善 | Data BER 改善次数 |
|---|---:|---:|---:|---:|
| `pilot_encoder` | `1.76e-6` | 约 `0` | `-1.60e-9` | `0/18` |
| `adapter` | `7.51e-6` | 约 `0` | `-6.26e-9` | `0/18` |
| `attention_lora` | `4.78e-6` | 约 `0` | `+4.62e-8` | `0/18` |
| `ffn_lora` | `6.80e-6` | 约 `0` | `+4.94e-8` | `0/18` |

每个参数组包含 18 个 seed-SNR-frame 候选。所有候选的 Data 硬 BER 都没有改变，
因此无法计算有意义的候选排序相关性；排序诊断返回 `NaN`，不是通过门槛。

## 结论边界

这组结果说明：在当前 checkpoint、当前 Adapt Pilot、自监督 BCE 和受限单步更新
预算下，普通 Adapter、两类 LoRA 以及 `pilot_encoder` 的梯度路径对 Data 输出几乎
不起作用。它们不能作为下一轮正式主路线。

这不是对这些参数组在任意学习率、步数或目标下的普遍否定；它只排除了当前安全
微调预算下的低效配置。继续盲目增大学习率或步数会重新引入 Pilot 过拟合风险，
必须先用连续指标和独立 Reward Pilot 守门验证。

当前保留的可行候选仍是 `phase_trend + head`，以及在强状态老化下有信号但尚不
稳定的 `channel_residual + pilot_reconstruction`。前者已有小幅稳定增量，后者
需要改善 Pilot-only 排序后才可进入正式矩阵。

## 协议审计

- 在线更新只读取 Adapt Pilot；
- Reward Pilot 只用于验收、拒绝和回滚；
- Data 标签仅用于本文件中的事后诊断；
- `data_labels_used_online=false`；
- 离线 checkpoint 未修改。

## `phase_trend + head` 组合复核

为确认增加输出头参数是否能增强已有的 phase 路径，补做了 3 seed × 3 SNR × 4 帧
的 Pilot-only replay。配置为 AdamW、5 步、学习率 `5e-3`，比较单独 phase、
`phase_trend + head` 和 identity。

结果显示，`phase_trend + head + pilot_signal_reconstruction` 与单独
`phase_trend + pilot_signal_reconstruction` 的所有输出指标完全一致：Data BCE
平均改善 `0.003646`，Data BER 平均改善 `0.000651`，36 个候选中 10 个硬 BER
改善。组合没有带来可识别的额外作用，说明当前信号重构目标的梯度主要落在
`phase_trend`，head 没有提供新的有效方向。

单独 `phase_trend + BCE` 的 Data BCE 平均恶化 `0.001530`，Data BER 平均恶化
`0.000868`；因此当前更值得保留的是 phase 的信号重构目标，而不是把 BCE 或 head
参数加入在线更新。所有候选的 Pilot-only 排序相关仍低于 `0.6`，不能进入
Contextual Bandit 或正式主矩阵。

## `channel_residual` 复核

又做了 3 seed × 3 SNR × 2 帧的信道残差小样本 replay，比较
`channel_reconstruction`、`channel_joint` 和已有的 `phase_trend` 信号重构。
更新使用 Adapt Pilot，学习率 `1e-4`，1 步。

`channel_reconstruction` 的 Pilot 重构改善约 `1.57e-4`，Data BCE 平均改善约
`2.31e-5`，但 18 个候选中硬 BER 没有一次改善。`channel_joint` 的 Data BCE
平均改善约 `2.48e-5`，Data BER 平均恶化约 `2.17e-4`。Pilot 重构改善与 Data
BER 改善的 Spearman 相关为 `-0.041`，没有通过 `0.6` 门槛。

因此，当前信道残差参数能够拟合 Pilot 观测，却不能可靠地选择 Data 判决方向。
它暂不进入正式主矩阵；需要先改变 Pilot 覆盖或设计新的可验证状态特征，不能靠
继续放大更新步数来掩盖排序失败。

## Pilot 总长度扫描

为确认问题是否只是 Pilot 太短，诊断脚本现在支持一次扫描多个 `--pilot-total`，
并在每行记录 Pilot 总长度。固定 Reward Pilot 为 32，比较 prefix Pilot 总长度
64、128、160、256；其余条件为 Level B、gap=600 s、3 seed、5/10/15 dB、2 帧，
`phase_trend + pilot_signal_reconstruction`，AdamW，5 步，学习率 `5e-3`。

| Pilot 总长度 | Data BCE 平均改善 | Data BER 平均改善 | Data BER 正收益次数 |
|---:|---:|---:|---:|
| 64 | `-0.000662` | `-0.000248` | `5/18` |
| 128 | `+0.003545` | `-0.000434` | `4/18` |
| 160 | `+0.000256` | `-0.000473` | `4/18` |
| 256 | `+0.005314` | `-0.000217` | `3/18` |

256 个 Pilot 的 Data BCE 改善最大，但 5 dB 仍为 `-0.002604`，15 dB 没有硬 BER
改善；128 个 Pilot 在 10 dB 有正向结果，但 15 dB 的 6 个候选全部没有 BER 改善。
由于每个长度只保留一个真实更新候选，Pilot-only Spearman 不能在长度内计算，不能
把这次扫描写成排序通过。

原始结果：
`logs/pilot_length_phase_signal_screen_fixedreward_3s3snr2f_gap600_20261014.json`

结论是：单纯增加 prefix Pilot 总长度不能解决 Pilot 到 Data 的方向失配。下一步应
改变状态到参数的映射或使用能覆盖 Data 相关状态的观测方式，而不是继续扩大同一
长度扫描或接入 Bandit。

## 状态条件化 replay 的因果复核

此前候选排序脚本虽然记录了状态量，但没有把状态 embedding 传入候选模型前向，
因此补做了同一固定条件下的有/无状态条件化对照。两组均为 Level B、gap=600 s、
Pilot=256、Reward=32、3 seed、5/10/15 dB、2 帧，单一候选为
`phase_trend + pilot_signal_reconstruction`，AdamW，5 步，学习率 `5e-3`。

| 设置 | Data BCE 平均改善 | Data BER 平均改善 | 正 BER 候选 | logits 平均变化 | 参数变化范数 |
|---|---:|---:|---:|---:|---:|
| 不使用 state embedding | `+0.005314` | `-0.000217` | `3/18` | `0.3021` | `0.0283` |
| 使用真实 Adapt Pilot state embedding | `+0.003587` | `-0.000868` | `2/18` | `0.4274` | `0.0387` |

状态条件化确实让参数和 Data logits 变化更大，但没有改善硬 BER 稳定性；5 dB 下
两种设置都为负，状态条件化为 `-0.003906`，15 dB 也没有正 BER 候选。这个结果
说明“状态 embedding 已经接入梯度路径”不等于“状态 embedding 能把更新方向变成
Data 方向”，不能据此扩大正式矩阵。

原始结果：

- `logs/pilot_unconditioned_replay_3s3snr2f_gap600_20261014.json`
- `logs/pilot_state_conditioned_replay_3s3snr2f_gap600_20261014.json`

同时修正了诊断脚本的真实状态传递和设备一致性问题。在线主流程未改变，离线
checkpoint 未改变，Data 标签仍只用于事后诊断。
