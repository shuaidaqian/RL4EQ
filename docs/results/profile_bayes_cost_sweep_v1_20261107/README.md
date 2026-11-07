# warm-start / AMP 成本-精度扫描（2026-11-07）

状态：**开发性能/精度扫描，不是正式确认。** 1 seed × 4 帧 × 4 SNR；已确认的 Bayes CIR 逻辑与 tail 复用固定。

## 协议

- 只改变 `physics_warm_start_iterations`（8/4/2）与模型前向是否 autocast（CUDA fp16 / CPU bf16）。
- 其他：CIR 先验、Pilot/Data 后验、tail 复用、Reward 门控、信道、Pilot、SNR 不变。
- 每 SNR 报告 Data BER、逐帧总耗时、相对 102.4 ms 预算比、相对 `ws8_amp0` 的不退化检查，以及 tail 复用一致性。

## 结果

| SNR | ws8_amp0 | ws8_amp1 | ws4_amp0 | ws4_amp1 | ws2_amp0 | ws2_amp1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 dB | 0.308594 | 0.308594 | 0.278646 | 0.278646 | **0.188802** | 0.188802 |
| 5 dB | 0.070313 | 0.070313 | **0.067383** | 0.067383 | 0.092448 | 0.092448 |
| 10 dB | **0.012695** | 0.012695 | 0.021484 | 0.021484 | 0.056641 | 0.056641 |
| 15 dB | **0.003581** | 0.003581 | 0.008464 | 0.008464 | 0.042318 | 0.042318 |

成本与不退化：

| 变体 | 总耗时 | 预算比 | 四 SNR 不退化 |
|---|---:|---:|---|
| ws8_amp0 | ≈430 ms | 4.2× | 是（基线） |
| ws8_amp1 | ≈440 ms | 4.3× | 是 |
| ws4_amp0 | ≈320 ms | 3.1× | 否（10/15 dB 退化） |
| ws4_amp1 | ≈327 ms | 3.2× | 否 |
| ws2_amp0 | ≈270 ms | 2.6× | 否（5/10/15 dB 退化） |
| ws2_amp1 | ≈270 ms | 2.6× | 否 |

tail 复用与独立 dd_soft 在全部 24 个记录中逐位一致（`tail_match=True`）。

## 判断

1. **AMP 在 GTX 1650 上没有加速，也没有 BER 变化**（autocast 开销抵消了 fp16 收益）；不保留为加速手段。
2. **warm-start 4 在 0/5 dB 更好，但 10/15 dB 明显退化**（10 dB `0.0127→0.0215`、15 dB `0.0036→0.0085`）；违反四 SNR 不退化护栏，不能作为默认。
3. **warm-start 2 进一步变差**，只有 0 dB 因早期帧而受益；同样不满足护栏。
4. **只有 warm-start 8 满足四 SNR 不退化**；当前确认配置应保留 8。
5. **tail 复用是唯一零精度 2×**；它已把 ~830 ms 降到 ~430 ms，但在 102.4 ms 预算下仍约 4.2×。
6. **模型前向仍是瓶颈**；单纯减少 warm-start 迭代数会牺牲高 SNR 精度，下一步需要蒸馏/剪枝/重训、batch/更大的 GPU，或在结构上减少迭代展开。

## 复现

```powershell
python scripts/profile_bayes_cost_sweep.py --config configs/eme_long_memory_v2_online_dd_tail_v1.json --checkpoint D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --seeds 160201 --frames 4 --device cuda --output docs/results/profile_bayes_cost_sweep_v1_20261107/sweep.json
```

## 边界

- 1 seed × 4 帧开发扫描；0 dB 的前 4 帧处于早期收敛，绝对 BER 不代表稳态；相对比较用于成本-精度取舍。
- 未测试 batch>1、剪枝、蒸馏、重训；这些是后续工作。
- warm-start 4/2 的 0/5 dB 改善是开发观察，不能作为按 SNR 调度的依据。
- 未把扫描结论改入确认协议；确认仍用 warm-start 8 + tail 复用后的逻辑（tail 复用只改实现，不改数值）。
