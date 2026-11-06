# AGENTS.md

我是中文语境开发者，所有对话和代码注释都用中文。

## 当前规范

本项目当前是**物理算法快照** `physical-bayes-cir-v1`；唯一事实源是[研究路线（物理版 v1）](docs/research_route_physical_v1_20261106.md)。

- 核心：离线神经均衡器 + 合法在线贝叶斯物理状态（CIR）适配；相对同源 acquisition-CIR Frozen 的 Data BER 增益。
- 独立确认 v1（seed 100001–100010，10 session × 128 帧）：H1/H2 通过；H3 在 5/10/15 dB 通过、0 dB 不通过。
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

## 证据边界

- 开发 seed 90001–99010 与确认 seed 100001–100010 已使用；不得重测或并入确认。
- 0 dB 的 H3 不通过，不能用四 SNR 平均掩盖。
- 逐帧延迟/GPU 显存未插桩，成本报告不完整。
- 开发筛查不等于确认；不得把开发结果写成确认。
