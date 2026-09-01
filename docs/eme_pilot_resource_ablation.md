# EME prefix Pilot 资源消融

## 协议

固定 Level B、`max_delay=116`、15 dB、30 帧、seed=0、`eme_long_memory_v2`，只改变 prefix Pilot 总长度。每帧仍按 3:1 划分 Adapt/Reward，三种方法使用完全相同的 Pilot 资源：

- `Offline NN only`；
- `Pilot-Driven Online Adaptation`；
- `CFO+DD-Phase DFE-RLS`。

## 结果

| 总 Pilot | 方法 | 累计 BER | 前 5 帧 BER | 后 5 帧 BER | 在线接受率/phase confidence |
|---:|---|---:|---:|---:|---:|
| 64 | Offline NN only | 0.070625 | 0.069583 | 0.068125 | - |
| 64 | Pilot online | 0.289931 | 0.067917 | 0.414792 | 0.500 / 0.227 |
| 64 | CFO+DD-Phase DFE-RLS | 0.393542 | 0.481458 | 0.388542 | - |
| 96 | Offline NN only | 0.069397 | 0.068103 | 0.068750 | - |
| 96 | Pilot online | 0.330172 | 0.649353 | 0.362500 | 0.400 / 0.472 |
| 96 | CFO+DD-Phase DFE-RLS | 0.204203 | 0.239224 | 0.210991 | - |
| 128 | Offline NN only | 0.068713 | 0.068080 | 0.066518 | - |
| 128 | Pilot online | 0.069531 | 0.093080 | 0.064509 | 0.467 / 0.514 |
| 128 | CFO+DD-Phase DFE-RLS | 0.121317 | 0.185938 | 0.310268 | - |
| 160 | Offline NN only | 0.069869 | 0.066898 | 0.066667 | - |
| 160 | Pilot online | **0.045332** | 0.063657 | 0.075000 | 0.533 / 0.546 |
| 160 | CFO+DD-Phase DFE-RLS | 0.076350 | 0.063657 | 0.270602 | - |

原始数据位于：

```text
logs/eme_pilot_ablation_p64_30f/
logs/eme_pilot_ablation_p96_30f/
logs/eme_pilot_ablation_p128_30f/
logs/eme_pilot_ablation_p160_30f/
```

## 结论

1. 64/96 符号时，在线状态估计方差足以造成严重的后期错误积累；增加帧数不能解决该问题。
2. 128 符号是当前可用的最低主配置，在线不再明显劣于 Frozen，并已明显优于传统 baseline。
3. 160 符号在该切片中把在线 BER 降至 4.53%，同时优于 Frozen 的 6.99% 和传统 baseline 的 7.64%。
4. 160 符号应作为后续主结果候选配置，但在修改正式主矩阵前需要至少 3 seed、0/5/10/15 dB 重复验证。当前主矩阵仍保持 128，避免用单 seed 消融结果替换既有协议。

该结果说明 Pilot 不是附属工程参数，而是在线均衡研究问题的一部分：prefix 中 Adapt Pilot 负责状态恢复，Reward Pilot 负责安全选择，两者过短时无法可靠区分真实信道漂移与噪声扰动。
