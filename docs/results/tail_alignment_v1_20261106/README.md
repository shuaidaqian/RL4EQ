# 跨帧 soft-tail 训练/评估协议对齐（v1）

## 结论

离线课程训练使用 `training_tail_mode=online_soft`：用当前帧最后 `max_delay` 个位置的模型软判决递推下一帧 tail（[curriculum.py:563](../../../training/curriculum.py#L563)）。评估路径 `next_pilot_tail` 只允许在 Adapt/Reward Pilot 位置更新；当前 `prefix` Pilot 布局下帧尾 116 位全部是 Data，因此该函数是**空操作**，tail 永远停在 acquisition 尾。主在线路径 `online_adaptation.py` 的 `pilot_visible_mask[-tail_len:]` 同样为空。

本轮的 `dd_soft` 模式把评估递推对齐到离线训练：只读取接收波形和模型自身 soft 判决，不读取 Data 发送符号或 Data 比特。Frozen 与 Online 共用同一规则。它不是给 Online 开后门，也不改变信道、Pilot、SNR 或数据划分。

**它没有直接制造在线增益。** Data BER 对 tail 不敏感（除 0 dB 略好）；它的作用是移除一个长期污染 Adapt 梯度的帧边界伪影。高 SNR 的 Pilot 梯度与 Data 梯度仍然接近零/负相关，说明"数据辅助信号"仍是下一步必须解决的问题。

## 实现

- `training/pilot_transfer.py` 新增 `TAIL_UPDATE_MODES = ("pilot_only", "dd_soft")` 与 `tail_update_mode(config)`。
- `next_pilot_tail(..., mode=...)`：
  - `pilot_only`：历史语义，prefix 布局下不更新；旧结果可原样复现。
  - `dd_soft`：`visible="data"` 前向，取最后 `tail_len` 个 logits 的 `tanh(logit/2)` 作为下一帧 tail；`alpha=1.0` 与离线 `online_soft` 一致。
- `differentiable_online_trajectory` 新增 `tail_mode`；`PilotTransferTrainer`、`evaluation/pilot_transfer.py`、`scripts/prototype_head_rank1_adapt.py` 从配置读取模式。
- 新配置 [eme_long_memory_v2_online_dd_tail_v1.json](../../../configs/eme_long_memory_v2_online_dd_tail_v1.json)：`tail_update_mode="dd_soft"`、`tail_update_alpha=1.0`；其余信道/Pilot/模型字段继承原开发配置。
- 回归测试：`tests/test_pilot_transfer.py` 新增 dd_soft 软判决一致性、Data 标签翻转不变性、未知模式失败、缺省 legacy 语义；定向 128 项通过。

## 数值证据

### 1. Frozen Data BER（3 独立 seed × 16 连续帧 / SNR，48 帧）

| SNR | pilot_only | dd_soft | 绝对差（dd − pilot） |
|---:|---:|---:|---:|
| 0 dB | 0.235026 | 0.226915 | −0.008111 |
| 5 dB | 0.038032 | 0.036974 | −0.001058 |
| 10 dB | 0.003743 | 0.003743 | 0 |
| 15 dB | 0.000407 | 0.000434 | +0.000027 |

0/5 dB 的 dd_soft 略好，与"训练时模型见过正确 tail"一致；10 dB 相同；15 dB 只有极少数错误，差异应按 bit 数解读，不能当作退化证据。

命令：

```powershell
python scripts/prototype_head_rank1_adapt.py --config configs/eme_long_memory_v2_online_dd_tail_v1.json --pretrained D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --basis-seed 78001 --seed-start 92001 --num-seeds 3 --frames 16 --device cuda --branches frozen --output docs/results/tail_alignment_v1_20261106/frozen_dd_soft.json
```

### 2. Adapt Pilot 损失与梯度对齐（3 seed × 6 帧 / SNR）

| SNR | 模式 | Frozen Data BER | Adapt BCE | Adapt/Data 梯度余弦 |
|---:|---|---:|---:|---:|
| 0 dB | pilot_only | 0.228009 | 0.5188 | +0.7486 |
| 0 dB | dd_soft | 0.202402 | 0.4801 | +0.7208 |
| 5 dB | pilot_only | 0.070964 | 0.1845 | +0.4559 |
| 5 dB | dd_soft | 0.071832 | 0.1715 | +0.5011 |
| 10 dB | pilot_only | 0.007017 | 0.0282 | −0.0540 |
| 10 dB | dd_soft | 0.007234 | 0.0067 | −0.0668 |
| 15 dB | pilot_only | 0.000940 | 0.0357 | −0.0434 |
| 15 dB | dd_soft | 0.001013 | 0.0003 | −0.0370 |

关键观察：

- dd_soft 将 10/15 dB 的 Adapt BCE 从 `0.0282/0.0357` 降到 `0.0067/0.0003`。旧路径的 Pilot 损失主要由错误的帧边界 tail 造成，在线更新一直在拟合这个伪影。
- 5 dB 的 Adapt/Data 余弦从 `0.456` 升到 `0.501`；0 dB 略降（`0.749→0.721`）。
- 10/15 dB 的余弦仍接近零或负值。Pilot BCE 在模型已拟合 Pilot 后没有可迁移到 Data 的信号；这不是 tail 能修复的。

命令：

```powershell
python scripts/diagnose_tail_protocol_alignment.py --config configs/eme_long_memory_v2_online_dd_tail_v1.json --checkpoint D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --seeds 90001 90002 90003 --frames 6 --device cuda --output docs/results/tail_alignment_v1_20261106/tail_protocol_alignment.json
```

## 解释与边界

- 这是**开发机制诊断**，不是正式确认；seed 90001–90003 / 92001–92003 均属开发。
- 旧实验继续按 `pilot_only` 原语义保留；新实验必须显式使用 `dd_soft` 配置。
- dd_soft 只允许读取接收波形与模型自身输出；测试通过翻转 Data 标签验证结果不变。
- 统计功效分析见[功效与错误构成](../power_error_composition_20261106/README.md)：历史 `1e-5` 量级点估计通常只是 1 个净错误，3×16/32 设计无法检验；后续开发筛查至少需要 10 个独立 session × 128 帧，0 dB 建议 30–35 个。
- 本结果不改变"在线必须优于同源、同物理跟踪 Frozen"的核心目标，也不构成 R3 或正式确认。
