# RL4EQ：离线神经均衡器 + 在线贝叶斯 CIR 适配

本分支 `physical-bayes-cir-v1` 是物理算法快照；历史 PEFT/RL 实验在 `codex/online-peft` 与 `docs/route-v4-20261106` 分支。

## 结论（独立确认 v1）

独立 seed 100001–100010，10 session × 128 帧；开发 seed 全部排除；配置/超参数/checkpoint 运行前冻结。

| SNR | Frozen BER | Bayes BER | ΔBER | 单侧 95% 下界 | Holm p |
|---:|---:|---:|---:|---:|---:|
| 0 dB | 0.2081 | 0.1713 | **+0.03680** | +0.03029 | 1e-5 |
| 5 dB | 0.0479 | 0.0399 | **+0.00803** | +0.00627 | 2e-5 |
| 10 dB | 0.00623 | 0.00426 | **+0.00197** | +0.00146 | 6e-5 |
| 15 dB | 0.000985 | 0.000724 | **+0.000260** | +0.000149 | 0.001 |

- **H1**：四 SNR 等权 `+0.011767`，单侧下界 `+0.010188`，p=`1.26e-7` → 通过。
- **H2**：四 SNR 下界均 >0，10/10 seed 无负向 → 通过。
- **H3**：5/10/15 dB 增长通过；0 dB 斜率 `-0.001238/bin` → 不通过（增益大但早期饱和/波动）。

## 方法

- **共享物理状态**：`dd_soft` tail、Pilot 条件 CIR/CFO/相位；Frozen 与 Bayes 一致。
- **贝叶斯递归 CIR**：acquisition 协方差先验 → Adapt Pilot 后验 → Data soft 判决后验；Data 行噪声含符号方差传播；后验协方差跨帧持久并加过程噪声。
- **信息边界**：在线不读 Data 标签/真实 CIR/相位/CFO；Data 标签只做事后评分；Reward Pilot 只做动作后验收。

## 文档

- [研究路线（物理版 v1）](docs/research_route_physical_v1_20261106.md)
- [独立确认归档](docs/results/bayesian_cir_confirmation_v1_20261106/README.md)
- [开发筛查归档](docs/results/bayesian_cir_diag_v1_20261106/README.md)
- [传统基线公平对照](docs/results/traditional_comparison_v1_20261107/README.md)
- [0 dB Data 后验变体](docs/results/diagnose_0db_data_posterior_v1_20261107/README.md)
- [确认协议](docs/protocols/confirmation_physical_v1_20261106.md)
- [全仓文档索引](docs/document_index_20261106.md)

## 复现

```powershell
$env:PYTHONUTF8=1
python -m pytest -q -p no:cacheprovider

python scripts/diagnose_bayesian_cir.py --config configs/eme_long_memory_v2_online_dd_tail_v1.json --checkpoint D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --seeds 100001 100002 100003 100004 100005 100006 100007 100008 100009 100010 --frames 128 --process-noise 1e-4 --ridge 1e-2 --device cuda --output docs/results/bayesian_cir_confirmation_v1_20261106/confirmation.json
```

## 边界

- 神经参数微调子主张未成立，不在本快照的确认范围。
- 0 dB 的 H3 不通过；2-seed Data 后验/相位变体筛查显示 0 dB 是早期收敛/后段饱和+晚窗为正，不是增长；5/10/15 dB 才支持增长。
- 同帧 EM/block_phase 的 BER 使用同帧更新后的 CIR，与已确认的跨帧因果不同，需单独标注或改为跨帧。
- 逐帧耗时：Frozen ≈795 ms、Bayes ≈829 ms；P1 变体 soft/block ≈1.7 s、EM2 ≈2.1 s、EM3 ≈2.8 s；均未完整插桩，当前不实时。
- 传统基线公平比较为 3×16 开发；Frozen/Bayes 四 SNR 显著优于 CFO+DD-Phase LMMSE/DFE-RLS；`SC-FDE-MMSE` 无相位补偿，仅作弱对照。
- 确认集只运行一次；不得重测或换 seed。
