# 不确定性条件化 FiLM 开发 smoke（2026-11-06）

状态：**开发机制 smoke，不是确认。** 两臂共享 acquisition CIR + dd_soft tail；Frozen 的 FiLM 为零（严格等价原模型）；Data 标签只用于事后 BER。

## 协议

- Online：在 head 之前对 64 维隐藏特征做 FiLM
  `h' = h * (1 + tanh(ctx @ Ws + bs)) + (ctx @ Wv + bv)`
  `ctx` 为 Data 置信度与 ELBO 残差统计（合法，stop-gradient）。
- FiLM 参数由当前帧 ELBO 梯度更新（Adam + 信赖域 + Reward BCE 门控），作用于下一帧。
- 评估 seed 120101，8 帧；`lr=0.02`、半径 `0.5`、4 步；另做了 `radius=0.05`、1 步的预 smoke。

## 结果

强预算（radius 0.5，4 步）：

| SNR | Frozen BER | Online BER | 增益 | 接受步数 | 参数范数 |
|---:|---:|---:|---:|---:|---:|
| 0 dB | 0.243652 | 0.243490 | +0.000163 | 24 | 0.500 |
| 5 dB | 0.041992 | 0.041992 | 0 | 12 | 0.500 |
| 10 dB | 0.002604 | 0.002604 | 0 | 9 | 0.500 |
| 15 dB | 0.000326 | 0.000326 | 0 | 11 | 0.273 |

弱预算（radius 0.05，1 步）：0/5/10/15 dB 增益均为 0。

## 判断

1. **FiLM 参数确实更新到信赖域（|p|=0.5），但硬 BER 在 5/10/15 dB 完全不变，0 dB 只 +1 比特。**
2. 两个允许的神经机制——ELBO 低秩 Adapter（B）与不确定性 FiLM（A）——都没有产生可重复的 5/10 dB BER 增益。
3. 这与三个诊断和文献 v2 一致：Adapt Pilot 在 10/15 dB 无可用方向；ELBO 子空间捕获高但逐帧方向不稳定；CIR 相关头room 已被物理跟踪吃掉；剩余误差低于硬判决分辨率。
4. **在"共享 acquisition CIR + 跨帧 + 不允许同帧 TTT / 离线重训 / Bayes CIR 增益"的约束下，神经 PEFT 不成立。** 这不是"没调好学习率/门控/半径"，而是信息与可辨识性上限。

## 下一步选项（需要用户决定）

- **接受负结果**：保留已确认的物理 Bayes CIR 结果；神经 PEFT 作为"未成立"归档。
- **允许同帧 TTT**：同一帧内用 Data 波形 + ELBO 更新再重新判决；同帧 oracle head 在 5/10 dB 曾显著（`−2.34e-3`/`−7.81e-4`），但它不是跨帧增长，且延迟/算力更高。
- **允许 Bayes CIR 共享**：物理增益对两臂相同，神经增量仍可归因；但基线更强，神经增量更小；可提供后验协方差 context。
- **允许离线重训/增广**：用 Bayes CIR/协方差特征重训离线模型；这是表示变化，不是纯在线 PEFT，需要四臂归因。

## 复现

```powershell
python scripts/diagnose_film_uncertainty_adapter.py --config configs/eme_long_memory_v2_online_dd_tail_v1.json --checkpoint D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --seeds 120101 --frames 8 --lr 0.02 --radius 0.5 --steps 4 --device cuda --output docs/results/peft_film_uncertainty_smoke_v1_20261106/smoke.json
```

## 边界

- 1 seed × 8 帧开发 smoke；不是确认。
- FiLM 只作用在 head 之前的隐藏特征；Transformer 块内 LoRA/更大 Adapter 未测试，但信息上限不变。
- 同帧 oracle head 的增益是上界，不等于合法可部署收益。
