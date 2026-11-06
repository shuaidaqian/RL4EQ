# 神经 PEFT 可学性诊断：query-vs-prompt、ELBO 方向、0 dB R_t（2026-11-06）

状态：**只读诊断，不是训练结果。** 两臂共享 acquisition CIR + dd_soft tail；不引入 Bayes CIR 增益。Data 标签只用于事后 query 梯度/ BER；真实符号错误只用于 R_t oracle。

## 协议

- 强离线 checkpoint：`D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`。
- 配置：`configs/eme_long_memory_v2_online_dd_tail_v1.json`；`dd_soft` tail `alpha=1.0`。
- 开发 seed：110001–110003；每 seed/SNR 16 帧（query 诊断）；R_t oracle 用 110101–110103。
- 参数空间：当前 65 维 head（`weight`+`bias`）；梯度为诊断用途，不更新参数。

## 诊断 1：query-vs-prompt 方向（Adapt Pilot vs Data）

| SNR | cos(Adapt,Data) | Adapt 跨帧稳定 | Data 跨帧稳定 | Adapt→Data 子空间捕获 | Data→Adapt 捕获 | 梯度范数 Adapt/Data |
|---:|---:|---:|---:|---:|---:|---:|
| 0 dB | +0.7086 | +0.4497 | +0.5144 | 0.8150 | 0.6451 | 0.0282 / 0.0405 |
| 5 dB | +0.4865 | +0.5208 | +0.6701 | 0.7333 | 0.4044 | 0.00869 / 0.00997 |
| 10 dB | +0.0016 | +0.0628 | +0.1410 | 0.3572 | 0.2120 | 0.00182 / 0.00175 |
| 15 dB | +0.0150 | +0.0846 | −0.0875 | 0.2782 | 0.1298 | 8.82e-5 / 5.64e-4 |

结论：**Adapt Pilot 在 0/5 dB 有可用方向，在 10/15 dB 基本没有**（余弦近零、捕获 28%–36%、跨帧不稳定）。

## 诊断 2：ELBO 方向（Data 软似然）

| SNR | cos(ELBO,Data) | ELBO 跨帧稳定 | **ELBO→Data 子空间捕获** | 梯度范数 Adapt/Data/ELBO |
|---:|---:|---:|---:|---:|
| 0 dB | −0.0201 | −0.4077 | **0.9603** | 0.0282 / 0.0405 / 0.117 |
| 5 dB | +0.1577 | −0.4021 | **0.9629** | 0.00869 / 0.00997 / 0.0781 |
| 10 dB | +0.1158 | +0.4414 | **0.9653** | 0.00182 / 0.00175 / 0.0254 |
| 15 dB | +0.5203 | +0.1394 | **0.9871** | 8.82e-5 / 5.64e-4 / 0.00682 |

结论：

- **ELBO 的 4 维子空间在四个 SNR 都捕获 96%–99% 的 Data 梯度能量**，远高于 Adapt Pilot。
- ELBO 梯度范数在 10/15 dB 分别是 Adapt 的约 14×/77×，高 SNR 仍有可用信号。
- 但 ELBO **逐帧方向不稳定**（0/5 dB 稳定性为负，10 dB +0.44）；不能逐帧用单方向，必须用**低秩持久子空间**。

## 诊断 3：0 dB R_t oracle

| 模式 | BER | CIR NMSE |
|---|---:|---:|
| frozen（acquisition CIR） | 0.182916 | 0.1284 |
| soft（当前启发式 R_t） | 0.167535 | 0.0627 |
| oracle（真实符号错误 R_t） | 0.162842 | 0.0605 |

结论：**soft R_t 已接近 oracle**（BER 差约 0.0047）；0 dB 的主要残差是 CIR 估计本身，不是 R_t 模型。神经 PEFT 不应把 R_t 学习当作主要增益来源。

## 机制设计（根据诊断）

**Query-aligned 低秩 Adapter + ELBO 跨帧更新**，适配你的"不掺杂物理增益"约束：

- 两臂共享 acquisition CIR + dd_soft tail；Frozen 的 Adapter 系数 `c=0`，Online 的 `c` 由 ELBO 更新。
- 低秩基 `B ∈ C^{65×K}`（K=4–8）由**开发 seed** 的 ELBO 梯度 top-K 子空间得到；不重训离线模型。
- 在线：用当前帧的 ELBO 对 `c` 求梯度（`B^T g_theta`），Adam/归一化步进 + 信赖域 + Reward Pilot 门控；更新后的 `c` 用于**下一帧**（跨帧，不做同帧 TTT）。
- 不确定性条件化：把 CIR/Data 置信度/协方差统计作为 Adapter 的 context，缩放步长或调制基；这是机制 A 与 B 的结合。
- 四臂：Frozen/Online × 固定基/元训练基；第一阶段先做固定基的 Frozen/Online smoke。
- 成功判据：5/10 dB 可重复正增量（例如 ≥1e-4，10 session × 128 帧），0/15 dB 不退化；先 smoke 再扩。

## 文件与复现

- `query_alignment.json`：诊断 1+2 的逐 seed 结果。
- `rt_oracle_0db.json`：诊断 3。
- 脚本：`scripts/diagnose_peft_query_alignment.py`、`scripts/diagnose_rt_oracle.py`。

```powershell
python scripts/diagnose_peft_query_alignment.py --config configs/eme_long_memory_v2_online_dd_tail_v1.json --checkpoint D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --seeds 110001 110002 110003 --frames 16 --rank 4 --device cuda --output docs/results/peft_query_alignment_v1_20261106/query_alignment.json
python scripts/diagnose_rt_oracle.py --config configs/eme_long_memory_v2_online_dd_tail_v1.json --checkpoint D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --seeds 110101 110102 110103 --frames 16 --device cuda --output docs/results/peft_query_alignment_v1_20261106/rt_oracle_0db.json
```

## 边界

- 3 seed × 16 帧开发诊断；不是确认。
- 诊断在 65 维 head 空间；更大的 Adapter/LoRA 参数空间可能不同，需在机制 smoke 中复测。
- ELBO 子空间捕获高不等于 BER 增益；必须做 Frozen/Online 配对 smoke。
- 0 dB 的 H3 在物理确认中已不通过；本诊断不改变该结论。
