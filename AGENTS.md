# AGENTS.md

我是中文语境开发者，所有对话和代码注释都用中文。

## 当前规范

本项目当前是**物理算法快照** `physical-bayes-cir-v1`；唯一事实源是[研究路线（物理版 v1）](docs/research_route_physical_v1_20261106.md)。

- 核心：离线神经均衡器 + 合法在线贝叶斯物理状态（CIR）适配；相对同源 acquisition-CIR Frozen 的 Data BER 增益。
- 独立确认 v1（seed 100001–100010，10 session × 128 帧）：H1/H2 通过；H3 在 5/10/15 dB 通过、0 dB 不通过。
- 传统基线公平比较（3 seed × 16 帧，seed 130001–130003）：Frozen/Bayes 四 SNR 显著优于 CFO+DD-Phase LMMSE/DFE-RLS；成本约 795/829/268/761/2 ms/帧，当前不实时；见 `docs/results/traditional_comparison_v1_20261107/`。
- 0 dB Data 后验/相位变体（2 seed × 16 帧，seed 140001–140002）：0 dB 是早期收敛/后段饱和+晚窗为正，不是增长；5/10/15 dB 才支持增长；0 dB H3 判据待定。EM2 的 0 dB 增益跨 seed 异质，block_phase 同向但仅比 soft 高约 0.0018；见 `docs/results/diagnose_0db_data_posterior_v1_20261107/`。
- 成本：模型前向 ≈400 ms、tail 独立前向 ≈400 ms、后验 ≈30 ms；tail 复用逐位一致，总耗时 ≈830→≈435 ms（4.2× 预算）；模型前向是剩余瓶颈，当前不实时。见 `docs/results/profile_bayes_cost_v1_20261107/`。
- **不宣称在线神经参数微调**；该子主张在开发中未成立，不在本快照的确认范围。
- 历史 PEFT/RL/子空间实验保留在 `codex/online-peft` 与 `docs/route-v4-20261106` 分支；本快照不保留其脚本与文档。

## 不可放松的约束

- 在线只读：已接收波形、Adapt Pilot 已知符号、模型 soft 输出、Pilot 驱动的 CIR/CFO/相位/tail 估计。
- 禁止在线读取：Data 标签、真实 CIR/相位/CFO、Reward Pilot 回流内循环、未来帧标签。
- Adapt Pilot 用于更新；Reward Pilot 只做动作后验收；Data 标签只做事后评分。
- 不改信道配置：1024 符号帧、116 符号记忆、prefix Pilot=256（Adapt=224/Reward=32）、0/5/10/15 dB。
- Frozen 与 Bayes 同源 checkpoint、同 dd_soft tail 规则、同波形；神经权重两臂冻结。

## 复现与测试

```powershell
$env:PYTHONUTF8=1
python -m pytest -q -p no:cacheprovider
```

物理确认命令见[路线第 6 节](docs/research_route_physical_v1_20261106.md#6-复现)。

传统基线公平对照：

```powershell
python scripts/compare_traditional_baselines.py --config configs/eme_long_memory_v2_online_dd_tail_v1.json --checkpoint D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --seeds 130001 130002 130003 --frames 16 --device cuda --output docs/results/traditional_comparison_v1_20261107/comparison.json
```

0 dB Data 后验变体：

```powershell
python scripts/diagnose_0db_data_posterior.py --config configs/eme_long_memory_v2_online_dd_tail_v1.json --checkpoint D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --seeds 140001 140002 --frames 16 --device cuda --output docs/results/diagnose_0db_data_posterior_v1_20261107/variants_2x16.json
```

成本 profile：

```powershell
python scripts/profile_bayes_cost.py --config configs/eme_long_memory_v2_online_dd_tail_v1.json --checkpoint D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --seeds 160001 --frames 8 --device cuda --output docs/results/profile_bayes_cost_v1_20261107/profile.json
```

## 证据边界

- 开发 seed 90001–99010、130001–130003、140001–140002 与确认 seed 100001–100010 已使用；不得重测或并入确认。
- 0 dB 的 H3 不通过；2-seed 变体筛查显示 0 dB 是早期收敛/后段饱和+晚窗为正，不是增长；5/10/15 dB 才支持增长。
- 同帧 EM/block_phase 的 BER 使用同帧更新后的 CIR，与已确认的跨帧因果不同，需单独标注或改为跨帧。
- 成本 profile：模型前向 ≈400 ms、tail 独立前向 ≈400 ms、后验 ≈30 ms；tail 复用逐位一致，总耗时 ≈830→≈435 ms（4.2× 预算）；模型前向是剩余瓶颈，当前不实时。
- 传统基线公平对照为 3×16 开发（seed 130001–130003）；传统方法使用代码默认实现，未加载 clean 分支未包含的 frozen calibration；`SC-FDE-MMSE` 无相位补偿。
- 开发筛查不等于确认；不得把开发结果写成确认。
