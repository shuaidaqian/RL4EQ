> 状态：历史诊断/已被当前主路线取代。原始结果保留用于追溯，不代表当前默认方法。
> 当前规范：docs/state_tracking_receiver_design.md；验证：docs/state_tracking_validation_protocol.md。

# 仅更新状态映射参数的 Pilot replay（2026-10-14）

> **适用范围更正：本页下方的旧数值只属于历史诊断，不能代表 EME Level B 主配置。**
> 后续代码审查发现，旧脚本直接构造 `CommEnvConfig`，实际使用默认的
> `legacy_sparse_v1`、512 符号帧和 clean 扰动，并未加载正式配置中的 EME
> 物理时标。`gap=600` 虽然传入字段，也不能视为已实现 EME 状态老化。
> 此外，旧相位重构目标与均衡器中 Adapter 的实际输入顺序不一致。
> 更正后的统一协议和结果见
> [统一 replay 协议复核](pilot_replay_protocol_audit_20261014.md)。

## 目的

本阶段继续保持离线 checkpoint 不变，检查一个更窄的在线 PEFT 更新对象：只更新
`phase_trend` Adapter 中的 `state_to_phase_cfo` 映射层，冻结直接相位参数
`raw_phase` 和 `raw_cfo`。目标是判断 Adapt Pilot 状态 embedding 能否学出跨状态的
参数调制方向，而不是把 Pilot 的单帧误差直接写入一个全局相位参数。

## 固定条件

- checkpoint：`pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`
- 旧诊断实际信道：legacy Level B，`max_delay=116`，帧长 512；不是 EME 主信道
- Pilot：prefix，总长度 256，Reward Pilot 32
- acquisition 到数据帧间隔参数：600 秒；该旧 legacy 配置不能用于 EME 老化结论
- SNR：5、10、15 dB
- seed：0、1、2
- 每条轨迹：4 帧
- 更新目标：Adapt Pilot 复数信号重构
- 状态条件：6 维 Pilot state embedding
- Data 标签：只用于 replay 结束后的诊断，`data_labels_used_online=false`

## 工程修正

`scripts/replay_pilot_candidate_ranking.py` 新增 `--state-mapping-only`，并做了两项
协议修正：

1. 作用域参数真正传入 `apply_adapt_only_peft_update`；
2. 该模式必须同时指定 `--state-conditioned`，且只接受
   `phase_trend_signal_reconstruction` 候选。

`apply_adapt_only_peft_update` 现在会检查作用域，只让
`online_phase_trend_adapter.state_to_phase_cfo` 的参数保持可训练，其他
`phase_trend` 参数全部冻结。单元测试同时检查了直接相位参数保持不变、映射层发生更新、
缺少 embedding 和错误参数组时会报错。

## 单帧候选 replay

以下是旧版本命令，仅用于追溯历史结果。当前脚本会严格拒绝其中不一致的模型/
信道维度和 legacy 非零 gap 组合；统一协议请使用新复核报告中的命令。

```powershell
.\.venv-gpu\Scripts\python.exe scripts/replay_pilot_candidate_ranking.py `
  --config configs/continual_ppo.json `
  --pretrained pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt `
  --snrs 5 10 15 --seeds 0 1 2 --frames 4 --delay 116 `
  --pilot-total 256 --gap-seconds 600 --reward-pilot-total 32 `
  --lr 1e-4 --steps 1 --candidates phase_trend_signal_reconstruction `
  --state-conditioned --state-mapping-only `
  --output logs/state_mapping_replay_3seed_4f_20261014/state_mapping_only.json
```

结果与完整 `phase_trend` 作用域逐项相同，说明在“每帧从零初始化模型”的单帧 replay
中，映射层足以产生相同的微小输出变化，但没有带来稳定 BER 收益：

| 作用域 | Data BER 平均改善 | Data BCE 平均改善 | logits 平均变化 | 正收益 seed-frame |
|---|---:|---:|---:|---:|
| 完整 `phase_trend` | -0.000109 | +0.000154 | 0.009642 | 1/36 |
| 仅 `state_to_phase_cfo` | -0.000109 | +0.000154 | 0.009642 | 1/36 |

这里的负 BER 改善表示相对冻结模型略微变差。该结果不能支持扩大正式矩阵。

## 连续序列 replay

为了排除“单帧从零初始化掩盖状态映射泛化”的解释，新增诊断脚本
`scripts/replay_phase_state_mapping_sequence.py`。它在同一条轨迹上保留在线模型，逐帧
使用 Adapt Pilot 更新，并以同一初始模型的冻结副本作为逐帧参照。脚本还记录：

- `state_mapping` 与直接 `raw_phase/raw_cfo` 的参数变化；
- Data logits、soft output 和方向改变比例；
- Reward Pilot BCE 守门前后的候选结果；
- 回滚后最终保留的 Data BER/BCE。

### 小步长、未加 Reward 守门：`lr=1e-4, steps=1`

映射层确实更新，平均每步映射参数变化约 `3.74e-4`，直接相位参数变化为 0；但
5/10/15 dB 的平均 Data BER 改善分别为 `+0.0326/+0/+0` 个百分点，只有 5 dB 的
一个 seed 有累计正收益。Data logits 的变化很小，不能形成明确硬 BER 增益。该文件是
未加 Reward 回滚的累计序列结果，不作为安全门控效果结论。

### 能明显改变输出的步长：`lr=5e-3, steps=5`

此时每步 Data logits 平均变化约 0.97--1.02，说明参数更新路径和输出路径都有效；但
未加 Reward 守门时三个 SNR 的平均 BER 改善均为负：

| 作用域 | 5 dB | 10 dB | 15 dB |
|---|---:|---:|---:|
| 完整 `phase_trend` | -0.4883 pp | -0.0651 pp | -0.0977 pp |
| 仅 `state_to_phase_cfo` | -0.2604 pp | -0.1302 pp | -0.0977 pp |

### 加入 Reward Pilot 验收和回滚

按“Reward BCE 变好才提交，否则回滚”的最小安全规则，接受率约为 33%--42%。最终
保留更新后，平均 BER 仍没有达到稳定正收益：

| 作用域 | 5 dB | 10 dB | 15 dB | 正收益 seed 数（5/10/15 dB） |
|---|---:|---:|---:|---|
| 完整 `phase_trend` | -0.3906 pp | -0.1302 pp | -0.0977 pp | 0/3、1/3、0/3 |
| 仅 `state_to_phase_cfo` | -0.1628 pp | +0.0326 pp | -0.0651 pp | 1/3、2/3、1/3 |

Reward Pilot 能拒绝一部分坏更新，但不能把 Pilot 目标可靠地转换成 Data 目标。它仍然
适合作为安全保护，不足以作为 Data BER 排序器。

## 结论

本阶段确认了三个事实：

1. `state_to_phase_cfo` 是真正参与前向和梯度更新的 PEFT 参数，不是只写日志的状态量；
2. 小更新几乎不改变 Data 硬 BER，大更新会明显改变 Data 输出，但方向不稳定；
3. Reward Pilot 验收和回滚可以减少更新次数，却不能解决 Adapt/Reward Pilot 与 Data
   段之间的目标失配。

因此当前仍不能声称“在线微调已经稳定超过离线训练”，也不扩大正式 `5 seed × 60 帧`
矩阵，不接入 Contextual Bandit。离线 checkpoint 和默认主配置保持不变。

上述旧结果不足以决定主路线。后续先保持 prefix Pilot，统一真实 EME 信道、当前帧
Adapt Pilot 相位条件和实际 Adapter 前向顺序，再检查累计历史 Adapt 更新与独立未来
Reward Pilot 验收能否改善后续 Data。主实验不增加数据位置的 Pilot。
