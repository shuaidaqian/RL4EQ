# EME Level B 离线整帧监督训练记录

## 当前契约

离线训练使用完整接收帧生成神经均衡器的完整输出，并以所有符号等权的
`BCE_all = mean(BCE(logits_all, bits_all))` 作为默认监督目标。Pilot、Reward Pilot
和 Data 的位置掩码仍然保留，用于分区统计、验证和后续在线协议，但不会再通过
`Data BCE + 0.25 * Adapt BCE + 0.25 * Reward BCE` 改变训练样本权重。

模型输入中的已知符号来自 `frame.receiver_view().adapt_symbols`：只有 Adapt Pilot
位置保留发送符号，其余位置为零。整帧 `bits` 只作为离线监督标签，不能通过输入
上下文进入模型。因此“整帧监督”和“在线 Data 标签不可见”是同时成立的。

## 验证要求

- checkpoint 选择依据独立验证集 Data BER，而不是整帧 BCE 或 Pilot BER；
- 主实验固定 Level B、prefix Pilot、SNR 0/5/10/15 dB；
- 在线阶段仍由 Adapt Pilot 更新候选 PEFT 参数，Reward Pilot 做独立验收和回滚；
- 本文件不宣称 BCE_all 已经超过传统均衡器，正式 BER 结果必须由同一信道轨迹、
  多 seed 的实验产生。

## 配置

`configs/continual_ppo.json` 和 `configs/eme_long_memory_v2.json` 已显式设置：

```json
"offline_loss": "bce_all"
```

旧分区损失仍可通过显式设置 `"offline_loss": "partitioned"` 复现实验，
仅作为兼容对照，不作为新的主路线。
