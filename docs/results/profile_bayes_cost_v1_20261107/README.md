# Bayes CIR 成本 profile（2026-11-07）

状态：**开发性能 profile，不是新算法结论。** 1 seed × 4 SNR × 8 帧；当前 GPU；batch=1。

## 拆解

每帧记录：
- `forward_ms`：带 CIR/tail 条件的模型前向（4 展开 + 8 物理 warm-start）；
- `pilot_ms` / `data_ms`：Pilot/Data 后验（117×117 逆、108/768 行 Gram）；
- `tail_forward_ms`：`next_pilot_tail` 的独立模型前向；
- `tail_reuse_ms`：直接用当前帧 logits 最后 116 位递推；
- `total_forward_ms` 与 `total_reuse_ms`：两种 tail 实现的逐帧总耗时。

## 结果

| SNR | forward | pilot | data | tail_fwd | total_fwd | total_reuse | 预算比 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 dB | 413 ms | 8.6 ms | 19.3 ms | 402 ms | 842 ms | 441 ms | 4.3× |
| 5 dB | 400 ms | 9.6 ms | 21.2 ms | 395 ms | 826 ms | 431 ms | 4.2× |
| 10 dB | 401 ms | 9.7 ms | 20.4 ms | 400 ms | 831 ms | 431 ms | 4.2× |
| 15 dB | 402 ms | 10.4 ms | 21.7 ms | 408 ms | 841 ms | 434 ms | 4.2× |

- **tail 复用与独立前向逐位一致**（`tail_match=True`），把每帧总耗时从 ~830 ms 降到 ~435 ms；这是零精度损失的 2×。
- **模型前向 ~400 ms 是剩余瓶颈**（4 展开 + 8 warm-start）；Pilot/Data 后验合计仅 ~30 ms；acquisition prior 每 session 一次。
- 102.4 ms 帧预算下，tail 复用后仍约 **4.2×**；模型前向需要约 4× 加速才能进入实时范围。

## 优化优先级

1. **立即采用 tail 复用**：当前 `next_pilot_tail` 重新做一次完整前向；直接用本帧已算的 logits 最后 116 位即可，零精度损失。
2. **模型前向**：warm-start 迭代数、展开层数、batch/AMP、蒸馏/剪枝；这是进入 102.4 ms 的主要路径。
3. **后验**：~30 ms，占比小；可后续用 Cholesky/Woodbury 进一步压缩，但不是当前瓶颈。
4. **acquisition**：每 session 一次，可忽略。

## 公平性与复现

- `tail_match=True` 在四个 SNR 全部成立；tail 复用与独立前向的结果逐位一致。
- 1 seed × 8 帧，GPU 与缓存状态会影响绝对数值；比例与瓶颈判断稳定。

```powershell
python scripts/profile_bayes_cost.py --config configs/eme_long_memory_v2_online_dd_tail_v1.json --checkpoint D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --seeds 160001 --frames 8 --device cuda --output docs/results/profile_bayes_cost_v1_20261107/profile.json
```

## 边界

- 单 seed、8 帧的开发 profile；不是正式性能基准。
- 未测试 batch>1、AMP、剪枝/蒸馏；这些是下一步。
- 未把 tail 复用改入已确认的 `diagnose_bayesian_cir`/传统对照脚本；当前只记录 profile 结论。
