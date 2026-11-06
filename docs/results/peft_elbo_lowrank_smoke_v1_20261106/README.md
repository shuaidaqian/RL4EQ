# ELBO 低秩 Adapter 开发 smoke（2026-11-06）

状态：**开发机制 smoke，不是确认。** 两臂共享 acquisition CIR + dd_soft tail；不引入 Bayes CIR。Data 标签只用于事后 BER；ELBO 只读接收波形、模型 soft 输出与估计 CIR。

## 协议

- 基 seed：110001–110003，8 帧；基 = 这些 seed 的 ELBO 梯度 top-4 子空间。
- 评估 seed：120001–120002，16 帧；0/5/10/15 dB。
- Frozen：`c=0`；Online：`theta = theta0 + B c`，`c` 由当前帧 ELBO 梯度更新（Adam `lr=0.01`、半径 `0.05`、Reward BCE 门控），作用于下一帧。
- 低秩 Adapter 只改 65 维 head 参数；物理状态（CIR/tail）两臂共享。

## 结果

| SNR | Frozen BER | Online BER | 增益 | 接受步数 | \|c\| 均值 |
|---:|---:|---:|---:|---:|---:|
| 0 dB | 0.171183 | 0.170980 | +0.000203 | 13 | 0.0225 |
| 5 dB | 0.038981 | 0.038859 | +0.000122 | 16 | 0.0385 |
| 10 dB | 0.002726 | 0.002726 | 0 | 19 | 0.0413 |
| 15 dB | 0.000651 | 0.000651 | 0 | 17 | 0.0433 |

逐帧增益在 0/5 dB 为 ±1–2 个比特的波动，10/15 dB 全为零。1 seed × 4 帧预 smoke 在 5 dB 还出现 1 比特退化。

## 判断

1. **参数确实在变**（|c| 到 0.04），但**硬 BER 没有可重复改善**；0/5 dB 的点估计是 1–2 个比特，10/15 dB 为零。
2. 这与诊断一致：ELBO 子空间捕获 96%–99% 的 Data 梯度，但**逐帧方向不稳定**；持久低秩系数在连续损失上移动，却没有跨越判决边界。
3. 机制 B（ELBO 低秩 Adapter）在当前作用点/参数化下**不成立**；不扩 10×128。
4. 下一步只剩机制 A（不确定性条件化 FiLM/LoRA，作用在 head 之前的隐藏特征），它能比 1 维 logit 残差更大幅度地改变 Data 判决；但文献先验仍不支持大增益。

## 复现

```powershell
python scripts/diagnose_elbo_lowrank_adapter.py --config configs/eme_long_memory_v2_online_dd_tail_v1.json --checkpoint D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --basis-seeds 110001 110002 110003 --basis-frames 8 --eval-seeds 120001 120002 --frames 16 --rank 4 --lr 0.01 --radius 0.05 --steps 1 --device cuda --output docs/results/peft_elbo_lowrank_smoke_v1_20261106/smoke.json
```

## 边界

- 2 seed × 16 帧开发 smoke；不是确认。
- 只在 65 维 head 空间；更大 Adapter/FiLM 的参数空间可能不同，需 A 的 smoke。
- ELBO 子空间捕获高不等于 BER 增益；连续损失下降没有转化为硬判决改善。
