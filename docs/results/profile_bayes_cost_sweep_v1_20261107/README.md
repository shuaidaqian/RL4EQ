# warm-start / AMP 成本-精度扫描（2026-11-07，2026-11-19 修正时序）

状态：**开发性能/精度扫描，不是正式确认。** 1 seed × 4 帧 × 4 SNR；已确认的**跨帧因果** Bayes CIR 逻辑与 tail 复用固定。

> **2026-11-19 修正**：旧版用本帧更新后的 CIR 做同帧重解码，BER/tail 取自第二次前向，但 `total_ms` 只计第一次前向，导致 BER 属同帧重解码、成本属跨帧因果。现改为与 `diagnose_bayesian_cir.py` 一致的因果时序：本帧判决/BER/tail 只用更新前 CIR，本轮后验只作用于下一帧，每帧一次前向。旧表（同帧 BER）见 git 历史。

## 协议

- 只改变 `physics_warm_start_iterations`（8/4/2）与模型前向是否 autocast（CUDA fp16 / CPU bf16）。
- 其他：CIR 先验、Pilot/Data 后验、tail 复用、Reward 门控、信道、Pilot、SNR 不变。
- 时序：`causal`——本帧判决早于本帧后验更新；`total_ms = forward + pilot + data + tail_reuse`，与实际部署同一口径。
- 每 SNR 报告 Data BER、逐帧总耗时、相对 102.4 ms 预算比、相对 `ws8_amp0` 的不退化检查，以及 tail 复用一致性。

## 结果

| SNR | ws8_amp0 | ws8_amp1 | ws4_amp0 | ws4_amp1 | ws2_amp0 | ws2_amp1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 dB | **0.214844** | 0.214844 | 0.216146 | 0.216146 | 0.228516 | 0.228516 |
| 5 dB | 0.071289 | 0.071289 | **0.069336** | 0.069336 | 0.090169 | 0.090169 |
| 10 dB | **0.013346** | 0.013346 | 0.021159 | 0.021159 | 0.056315 | 0.056315 |
| 15 dB | **0.003255** | 0.003255 | 0.007487 | 0.007487 | 0.042643 | 0.042643 |

成本与不退化（4 SNR 平均）：

| 变体 | 总耗时 | 预算比 | 四 SNR 不退化 |
|---|---:|---:|---|
| ws8_amp0 | ≈473 ms | 4.6× | 是（基线） |
| ws8_amp1 | ≈456 ms | 4.5× | 是 |
| ws4_amp0 | ≈328 ms | 3.2× | 否（0/10/15 dB 退化） |
| ws4_amp1 | ≈332 ms | 3.2× | 否 |
| ws2_amp0 | ≈268 ms | 2.6× | 否（四 SNR 全部退化） |
| ws2_amp1 | ≈274 ms | 2.7× | 否 |

tail 复用与独立 dd_soft 在全部 24 个记录中逐位一致（`tail_match=True`）。

## 判断

1. **AMP 在 GTX 1650 上没有加速，也没有 BER 变化**（autocast 开销抵消了 fp16 收益）；不保留为加速手段。
2. **warm-start 4 只在 5 dB 略好**（`0.0713→0.0693`），但 0 dB 略差（`0.2148→0.2161`）、10/15 dB 明显退化（10 dB `0.0133→0.0212`、15 dB `0.0033→0.0075`）；违反四 SNR 不退化护栏，不能作为默认。
3. **warm-start 2 四个 SNR 全部退化**；不满足护栏。
4. **只有 warm-start 8 满足四 SNR 不退化**；当前确认配置应保留 8。
5. **tail 复用是唯一零精度 2×**；它已把 ~830 ms 降到 ~470 ms，但在 102.4 ms 预算下仍约 4.6×。
6. **模型前向仍是瓶颈**；单纯减少 warm-start 迭代数会牺牲高 SNR 精度，下一步需要蒸馏/剪枝/重训、batch/更大的 GPU，或在结构上减少迭代展开。

## 与旧版差异（时序修正的影响）

- 旧版 0 dB 曾显示 ws4 `0.2786`、ws2 `0.1888` 都优于 ws8 `0.3086`——那是**同帧重解码**的收益：更新后的 CIR 重解码本帧，warm-start 越少反而越能从更新中获益。
- 因果时序下同一让步消失：0 dB ws4/ws2 均不优于 ws8。说明旧表的 0 dB 排序是时序口径造成的，不能用于“按 SNR 降 warm-start”的调度依据。
- 高 SNR 的排序（ws8 最好）在两版中一致；AMP 结论不受影响。

## 复现

```powershell
python scripts/profile_bayes_cost_sweep.py --config configs/eme_long_memory_v2_online_dd_tail_v1.json --checkpoint D:/Research/02_研究点二_神经均衡/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --seeds 160201 --frames 4 --device cuda --output docs/results/profile_bayes_cost_sweep_v1_20261107/sweep.json
```

## 边界

- 1 seed × 4 帧开发扫描；绝对 BER 不代表稳态；相对比较用于成本-精度取舍。
- 未测试 batch>1、剪枝、蒸馏、重训；这些是后续工作。
- warm-start 4 的 5 dB 改善是开发观察，不能作为按 SNR 调度的依据。
- 未把扫描结论改入确认协议；确认仍用 warm-start 8 + tail 复用后的逻辑（tail 复用只改实现，不改数值）。
