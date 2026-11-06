# 学习式 CIR 去偏映射：线性 map 过拟合诊断（2026-11-06）

状态：**开发诊断（负/中性），不是确认。** 真实 CIR 只用于离线训练/评分；在线特征只用 acquisition CIR、当前帧 Data soft 判决的加权 LS、Data 置信度与 Pilot 相位特征。

## 目的

简单 Data LS / Pilot ridge 因欠定、相位残差与错误传播失败。本诊断检验：离线学习的线性/低秩去偏映射能否把 Data LS 拉向真实 CIR，并降低 held-out NMSE。

## 协议

- 训练 seed：94001–94008；测试 seed：94009–94010；每 seed/SNR 4 帧。
- 特征：acquisition CIR（234 维实/虚）、当前帧 Data 加权 LS（234 维）、5 个标量（Data 均值置信度、置信度>0.5 比例、平均符号方差、phase0、CFO）。
- 目标：按 acquisition 相位对齐的真实 CIR（仅离线）。
- 线性 ridge map（`ridge=1e-2`）；同时报告训练集上选择的 per-SNR 简单 blend 权重。

## 结果

| SNR | NMSE acq | NMSE Data LS | NMSE 线性 map | blend 权重 | NMSE blend |
|---:|---:|---:|---:|---:|---:|
| 0 dB | 0.1132 | 0.6115 | 0.8985 | 0.05 | 0.1132 |
| 5 dB | 0.0391 | 0.0889 | 0.8825 | 0.35 | 0.0343 |
| 10 dB | 0.0132 | 0.0207 | 0.8768 | 0.45 | 0.0090 |
| 15 dB | 0.0048 | 0.0067 | 0.8794 | 0.45 | 0.0030 |

## 判断

1. **线性 map 严重过拟合**：128 个训练样本 vs 约 110k 参数，held-out NMSE 约 0.88，比 acq/LS 都差。不能据此支持"学习式去偏"。
2. **简单 blend 在 NMSE 上有效**：5/10/15 dB 相对 acq 降低约 12%/32%/37%，权重随 SNR 从 0.05 升到 0.45；0 dB 几乎不用 Data LS。
3. **但 NMSE 改善不等于 BER 改善**：简单 Data LS 的 3×16 BER 在 5 dB 中性、10 dB 区间跨零；NMSE 的改善未等比例转化为判决。
4. **贝叶斯递归 CIR 是更好的路线**：它用先验协方差与符号方差传播，3×16/5×32 在 5/10 dB 取得可重复 BER 增益；本线性 map 不保留为方法。
5. 若以后仍要做学习式去偏，需要更大训练集、低秩/结构化 map，并以 held-out BER 而非 NMSE 选型。

## 复现

```powershell
python scripts/diagnose_cir_learned_map.py --config configs/eme_long_memory_v2_online_dd_tail_v1.json --checkpoint D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --train-seeds 94001 94002 94003 94004 94005 94006 94007 94008 --test-seeds 94009 94010 --frames 4 --device cuda --output docs/results/cir_learned_map_v1_20261106/learned_map.json
```
