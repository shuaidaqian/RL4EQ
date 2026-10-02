# A/B/C 统一小样本 replay 结果

日期：2026-09-28

## 运行协议

- 信道：Level B，`eme_long_memory_v2`，最大物理时延 116 symbols
- impairment：`cfo_phase_tiny`
- checkpoint：`pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`
- SNR：0/5/10/15 dB
- 信道 seed：1 个（seed=0）
- 连续帧：8 帧
- Pilot：prefix，总长度 256，其中 Adapt=224、Reward=32
- 调度器：fixed；更新间隔参数为 4
- 跨帧状态：沿用 compare.py 的 soft-tail 递推
- Data 标签：只用于最终 BER 统计，不进入在线状态、更新或验收

逐帧结果和配对摘要保存在：

- `logs/abc_replay_smoke/frame_metrics.jsonl`
- `logs/abc_replay_smoke/summary.json`

## 三个候选的实际定义

| 候选 | 现有入口 | 在线行为 |
|---|---|---|
| ABC-A Pilot State Gate | `Pilot CIR only` | Adapt Pilot 驱动稀疏 CIR/相位状态更新；冻结网络参数 |
| ABC-B Pilot RLS Residual | `Pilot-Driven Online Adaptation` + `online_adaptation_algorithm=rls` | Adapt Pilot 驱动零初始化低维残差 Adapter 的 RLS 更新；不使用 RL |
| ABC-C Physics Unfolded | `Pilot CIR only` + `neural_residual_scale=0` | Adapt Pilot 驱动 CIR/相位状态更新；关闭神经 residual，不做在线 PEFT |

Frozen 参考为同一 checkpoint 的 `Frozen Offline NN`，使用同一信道轨迹和 Pilot 协议。

## 结果

表中 `Frozen - Candidate` 为逐帧配对 BER 差，正值表示候选更好。当前只有 8 帧、1 个 seed，
不计算正式 block-bootstrap 显著性结论。

| 候选 | SNR | Frozen BER | Candidate BER | 平均配对差 | 候选胜出帧比例 |
|---|---:|---:|---:|---:|---:|
| A | 0 dB | 16.3737% | 16.3737% | +0.0000 pp | 0/8 |
| A | 5 dB | 5.9570% | 5.9570% | +0.0000 pp | 0/8 |
| A | 10 dB | 0.9766% | 1.0254% | -0.0488 pp | 2/8 |
| A | 15 dB | 0.2441% | 0.1302% | +0.1139 pp | 4/8 |
| B | 0 dB | 16.3737% | 26.3346% | -9.9609 pp | 0/8 |
| B | 5 dB | 5.9570% | 8.4635% | -2.5065 pp | 2/8 |
| B | 10 dB | 0.9766% | 1.1393% | -0.1628 pp | 2/8 |
| B | 15 dB | 0.2441% | 0.2441% | +0.0000 pp | 0/8 |
| C | 0 dB | 16.3737% | 16.4388% | -0.0651 pp | 0/8 |
| C | 5 dB | 5.9570% | 5.9570% | +0.0000 pp | 0/8 |
| C | 10 dB | 0.9766% | 1.0254% | -0.0488 pp | 2/8 |
| C | 15 dB | 0.2441% | 0.1302% | +0.1139 pp | 4/8 |

## 状态边界审计

- Frozen：`cir_update_applied=false`、`peft_update_applied=false`
- A：`cir_update_applied=true`、`peft_update_applied=false`
- B：`peft_update_applied` 在 32 帧中有接受和拒绝，`uses_rl=false`
- C：`cir_update_applied=true`、`peft_update_applied=false`
- 四组所有记录：`data_labels_used_online=false`
- 四组各有 32 条记录，配对键完全一致

## 当前研究判断

1. A 在这个极小样本上只在 15 dB 显示正增益，不能支持“全 SNR 在线改善”。
2. B 在 0/5 dB 出现明显退化，当前 RLS 残差 Adapter 不能直接进入正式主线。
3. C 与 A 几乎重合，说明关闭神经 residual 后，当前代理结构没有显示独立收益；它还不能证明完整 Trellis/BCJR 方法无效。
4. 当前最合理的下一步是扩展 A/C 的多 seed replay，并单独检查 B 的更新幅度、目标 logit 和 Pilot 代理是否失配。无论下一步选择哪条，正式主实验仍需回到 5 seed × 60 帧和 3 seed × 200 帧门槛。
