# EME Level B 200 帧长期诊断

## 目的

本实验用于定位连续帧 BER 退化的来源，不作为主方法性能表。实验固定 Level B、`max_delay=116`、15 dB、prefix Pilot 128，并比较：

- `fixed CIR`：始终使用 acquisition CIR；
- `oracle CIR`：诊断时使用仿真器暴露的当前真值 CIR；
- `soft tail`：按模型输出递推历史尾部；
- `oracle tail`：诊断时使用真实跨帧尾部。

每个条件使用 3 个 seed，分别测试 `rho=0.9991470306520146`（`eme_long_memory_v2` 的慢漂移值）和 `rho=1.0`（无 tap 漂移对照），总计 4800 条逐帧记录。

## 结果

| rho | tail | CIR | 1-16 帧 BER | 185-200 帧 BER | 196-200 帧 BER | 1-200 帧 BER |
|---:|---|---|---:|---:|---:|---:|
| 0.999147 | soft | fixed | 0.03060 | 0.04943 | 0.05243 | 0.04803 |
| 0.999147 | oracle | fixed | 0.03044 | 0.04986 | 0.05330 | 0.04773 |
| 0.999147 | soft | oracle | 0.02403 | 0.02094 | 0.02188 | 0.02613 |
| 0.999147 | oracle | oracle | 0.02387 | 0.02111 | 0.02222 | 0.02596 |
| 1.0 | soft | fixed | 0.02989 | 0.02989 | 0.03299 | 0.02972 |
| 1.0 | oracle | fixed | 0.02979 | 0.03006 | 0.03316 | 0.02973 |
| 1.0 | soft | oracle | 0.02572 | 0.02550 | 0.02656 | 0.02583 |
| 1.0 | oracle | oracle | 0.02539 | 0.02545 | 0.02639 | 0.02574 |

原始逐帧数据和汇总位于：

```text
logs/eme_long_diagnostic_200f_core/frame_metrics.jsonl
logs/eme_long_diagnostic_200f_core/summary.json
```

## 结论

1. `soft tail` 与 `oracle tail` 在两种 rho 下几乎重合。因此当前 15 dB 主配置的首要问题不是历史 tail 的均值递推，而是长记忆 CIR 状态失配。
2. 在慢漂移 `rho=0.999147` 下，`fixed CIR` 的长期 BER 约 4.8%，`oracle CIR` 约 2.6%，说明当前信道状态恢复具有明确的可优化空间。
3. `rho=1.0` 下 fixed/oracle CIR 差异明显缩小，验证 BER 差异来自 tap 漂移，而不是诊断脚本或固定网络的随机误差。
4. 200 帧不会自动改善 BER；它的作用是把 acquisition 老化和跨帧状态递推造成的退化显现出来。

## 对研究路线的约束

当前应优先做 Adapt Pilot 驱动的联合 `CIR + residual CFO + phase` 状态恢复。信道编码、Turbo 迭代和信源编码暂不进入主实验，因为它们会掩盖均衡器状态恢复问题。
