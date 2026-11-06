# 共享 Bayes CIR 下的在线 head 增量（2026-11-06）

状态：**开发机制诊断（1 seed × 8 帧 smoke），不是确认。** Frozen 与 Online 共用同一 Bayes CIR、同一 dd_soft tail 与同一波形；Online 每帧用 Adapt Pilot BCE 更新 head（Reward Pilot BCE 门控），再对同一帧 Data 判决。Data 标签只用于事后 BER。

## 目的

贝叶斯 CIR 已相对 acquisition CIR 取得 3/3 seed 同向正的物理增益。本诊断检验：把 Bayes CIR 作为**共享物理跟踪**后，在线神经 head 微调是否还能提供额外增量。只有 Online 相对共享 Bayes Frozen 仍为正，才能支持"神经微调"核心主张。

## 协议

- 3×16 Bayes CIR 的同一来源 checkpoint 与配置；开发 seed 96001；每 SNR 8 帧。
- Bayes CIR：`acquisition_prior` + 每帧 `pilot_posterior` → `data_posterior`（Data soft 判决，不读标签），`Q=1e-4`、`ridge=1e-2`。
- Online head：Adapt Pilot BCE，Adam `lr=0.01`，4 步，累计半径 `0.05`；Reward BCE 不劣于 Frozen 才提交。
- 每帧重置 head，不跨帧持久；物理状态（CIR/tail）两臂共享。

## 结果

| SNR | 共享 Bayes Frozen BER | Online BER（提交后） | 增益 | 接受帧 |
|---:|---:|---:|---:|---:|
| 0 dB | 0.155924 | 0.155762 | +0.000163 | 7/8 |
| 5 dB | 0.032552 | 0.032389 | +0.000163 | 5/8 |
| 10 dB | 0.002767 | 0.002767 | 0 | 3/8 |
| 15 dB | 0.000651 | 0.000651 | 0 | 7/8 |

0/5 dB 的 `+0.000163` 是 1 个净比特；10/15 dB 无变化。

## 判断

1. **共享 Bayes CIR 后，Adapt-Pilot head 的神经增量约为 1 个比特，不构成可重复研究增益。**
2. Bayes CIR 已捕获 5/10 dB 的 CIR 相关 BER 头room（Bayes 增益与该 3×16 因果 oracle 真实 CIR 相当）；剩余非 CIR 残差没有通过 head 转化为 BER。
3. 这与之前"同帧 oracle head 显著、留出帧消失"的结论一致：head 的 per-frame 标签拟合不能稳定迁移，Adapt Pilot 信号不足以驱动有效数据侧修正。
4. **归因结论**：当前最强的在线增益是**物理状态（CIR）增益**，不是神经参数微调增益。AGENTS 契约明确"纯 Kalman/RLS CIR 跟踪不能单独完成核心目标"，因此不能把 Bayes CIR 单独写成研究点完成。
5. 下一步候选：
   - 在共享 Bayes CIR 之上做**学习式可靠性/噪声模型**（离线外层用 Data BER 训练 Data 行噪声尺度、Q、先验协方差），再检验是否有额外增益；
   - 或做**后验不确定度条件化的 FiLM/LoRA**（用 Bayes 协方差作为神经输入）；
   - 若两者仍无增量，需要如实报告"物理在线增益成立、神经增量未成立"，并调整核心主张或结构。
6. 本 smoke 每 SNR 仅 8 帧、1 seed；不能用于确认或模型选择。

## 复现

```powershell
python scripts/diagnose_bayes_head.py --config configs/eme_long_memory_v2_online_dd_tail_v1.json --checkpoint D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --seeds 96001 --frames 8 --steps 4 --lr 0.01 --radius 0.05 --gate 0.0 --device cuda --output docs/results/bayes_head_diag_v1_20261106/bayes_head_smoke.json
```
