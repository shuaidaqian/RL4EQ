# 四路在线消融阶段性结果

日期：2026-10-02

## 目的

在不修改离线训练和预训练 checkpoint 的前提下，统一比较四种在线路径：

1. `Frozen Offline NN`：冻结 CIR 和 PEFT 参数，但使用当前帧 Pilot 形成推理条件。
2. `Pilot State Only`：只使用 Adapt Pilot 更新 CIR/phase 物理状态。
3. `PEFT Only`：物理状态固定，只使用 Adapt Pilot 尝试更新 `phase` PEFT 参数。
4. `Joint State + PEFT`：同时允许物理状态更新和 `phase` PEFT 更新。

四路均使用同一 checkpoint、同一信道轨迹、prefix Pilot、固定调度，并且在线字段中 `data_labels_used_online=false`。

## Smoke 协议

- Level B，`eme_long_memory_v2`
- 最大时延：116
- `cfo_phase_tiny`
- SNR：0/5/10/15 dB
- 1 seed × 4 帧
- Pilot 总长度：256；Adapt=224；Reward=32
- checkpoint：`pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`
- 固定调度，每帧更新

逐帧结果：`logs/four_way_replay_smoke/frame_metrics.jsonl`

汇总结果：`logs/four_way_replay_smoke/summary.json`

## 审计结果

| 方法 | 记录数 | checkpoint 全部加载 | CIR 更新数 | PEFT 更新数 | Data 标签在线使用 |
|---|---:|---:|---:|---:|---|
| Frozen Offline NN | 16 | 是 | 0 | 0 | 否 |
| Pilot State Only | 16 | 是 | 16 | 0 | 否 |
| PEFT Only | 16 | 是 | 0 | 0 | 否 |
| Joint State + PEFT | 16 | 是 | 7 | 0 | 否 |

`adaptation_accepted` 表示 Adapt Pilot 内层优化没有超过参数范数限制；它不等于最终更新被保留。PEFT Only 和 Joint 的 `peft_update_applied` 均为 0，说明所有候选最终都没有通过 Reward Pilot 守门。

## Smoke BER

| SNR | Frozen | Pilot State Only | PEFT Only | Joint State + PEFT |
|---:|---:|---:|---:|---:|
| 0 dB | 16.0807% | 16.0807% | 16.0807% | 16.0807% |
| 5 dB | 5.7943% | 5.7943% | 5.7943% | 5.7943% |
| 10 dB | 0.8789% | 0.9115% | 0.8789% | 0.9115% |
| 15 dB | 0.2930% | 0.1628% | 0.2930% | 0.1628% |

这组 smoke 不是正式性能结论，但它确认了四路边界和当前因果事实：Joint 的可见收益来自物理状态更新，尚未来自神经 PEFT。

## 当前阻碍与下一步

当前阻碍不是离线 checkpoint，而是 PEFT 候选在 Reward Pilot 上没有被保留。正式扩大矩阵前需要：

1. 记录 Adapt/Reward 更新前后损失、候选组和真实参数增量；
2. 分辨“候选梯度有效但 Reward 不泛化”和“候选没有改变输出”两种情况；
3. 在不使用 Data 标签的前提下校准低维在线更新对象、学习率和相对改善阈值；
4. 通过小样本验证 `peft_update_applied` 真正出现，再重跑 3 seed × 30 帧和 5 seed × 60 帧。

在 PEFT 通过统一 Reward Pilot 验收之前，不能把 Joint 的状态更新收益写成在线神经微调收益。
