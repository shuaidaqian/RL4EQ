# RL 信道均衡研究分析

## 1. 为什么离开常规信道

在常规无线信道中，LMMSE、DFE、RLS、Kalman tracking 和块检测等传统均衡方法已经非常强。若只在短时延、密集但温和的多径上使用神经网络均衡，很容易出现以下问题：

- 传统 baseline 已接近可达上界，神经方法难以稳定超过。
- 神经模型需要大量训练和调参，工程复杂度高。
- 若优势只来自更长训练或更多标签，论文结论不够清晰。

因此当前研究转向 20–40 符号极端时延扩展、稀疏强长回波、传统均衡器病态性更明显的场景。这里的 EME 是启发来源：关注长相对多径、稀疏强回波和慢漂移，不模拟完成同步后的绝对地月传播时延。

## 2. 研究对象

研究对象是整帧缓冲、非因果块神经均衡器。在线不是逐符号即时判决式输出，而是在信道运行期间按帧持续适配接收机与 PPO policy。

主场景为 Level B：

- 最大相对时延：20、30、40 符号。
- SNR：10、15、20 dB。
- 慢漂移：`rho=0.99`。
- 成功目标：每个主配置 `BER_data < 0.01`。

Level A 用于课程与校准；Level C 仅压力测试，不混入主平均。

## 3. 为什么全程 Pilot 条件课程预训练

当前最终方案是全程 Pilot 条件监督预训练，并使用课程学习。理由是：

- 在线阶段一定依赖 Pilot 条件；离线阶段如果长期无 Pilot，预训练目标和部署分布不一致。
- 多 Pilot 开销与布局消融可以直接暴露 Pilot 资源、goodput 与 BER 的权衡。
- 从 Level A 到 Level B 的课程更利于先学习可达接收，再进入极端长回波。
- first-order meta 需要明确的 Adapt support 和 Reward/Data query 隔离。

Pilot 标签不能直接拼接到当前位置的均衡主干输入。Adapt Pilot 只能进入 conditioner 或 support loss；Reward Pilot 标签动作前不可见，只能用于动作后的 reward 和留出评估；Data 标签只用于离线 outer loss 与仿真评估。

## 4. 模型职责分解

- Hybrid CIR Estimator：从 acquisition 和 Adapt Pilot 得到显式稀疏 CIR、support、噪声、置信度和 latent residual。
- Physics-Guided Unfolded Equalizer：用 `H/H^H` 物理算子进行块级迭代，用 Transformer denoiser 处理长程依赖。
- Adapter/LoRA/head：承担在线参数高效适配。
- Fixed Gate：确认在进入 PPO 前，接收机和动作空间在 Level B 主配置上至少可用。
- Continual PPO：选择何时更新、更新哪些 PEFT 组、用几步、用什么学习率和检测迭代，不直接判决 Data bit。

## 5. PPO 的贡献边界

PPO 的核心成功目标是进一步降低 BER，而不是证明 RL 能从零替代接收机。合理贡献应表述为：

- 在已通过 Best Fixed 前置门槛的可达接收机上，Continual PPO 可以利用 Reward Pilot 与历史状态选择更合适的在线适配策略。
- 若相对 Best Fixed、规则控制和 Bandit baseline 在配对设置下稳定降低 `BER_data`，则说明策略学习有贡献。
- 若前置门槛未过，应回到信道、接收机或训练阶段，不应让 PPO 掩盖不可达问题。

## 6. 公平 baseline

正式比较保留：

- Perfect-CSI Block：只作为可达性基线。
- Sparse CIR + Kalman/RLS。
- Block LMMSE/CG。
- DFE-RLS。
- Analytic Iterative BPSK。
- Legacy LMMSE-FIR。
- Legacy DFE。
- No Adapt。
- Best Fixed。
- Drift-Aware Pilot Rule。
- Contextual Bandit。
- Continual PPO。

除明确标注的 Perfect-CSI 可达性基线外，所有可部署方法只能使用 acquisition/Adapt Pilot。Reward Pilot 不进入动作前 observation，Data 标签不进入在线流程。本项目不使用数据标签上界。

## 7. Pilot 布局消融

Pilot 开销与布局按 12 个候选初筛：

```text
total_pilot ∈ {64, 96, 128, 160}
layout ∈ {prefix, two_block, multi_block}
```

先按九配置门槛、Reward/Data Spearman、effective goodput、最坏 seed 筛出 2–3 个候选；最终 3→1 必须等 Continual PPO 开发结果产生后冻结，避免在 PPO 前提前选择对 RL 不公平的布局。

## 8. 成功标准与风险

成功标准：

- Perfect-CIR / unfolded 接收机在主配置可达。
- Best Fixed 每个主配置低于 0.1。
- Reward/Data Spearman 不低于 0.6。
- Continual PPO 每个主配置 `BER_data < 0.01`。
- 至少相对 Best Fixed、规则控制、Bandit baseline 有配对改善。

主要风险：

- Level B 个别配置本身过于病态，Perfect-CSI 或 Perfect-CIR 不可达。
- Reward Pilot 与 Data BER 改善相关性不足。
- Pilot 开销过大导致 BER 下降但 effective goodput 不占优。
- 轻量 smoke 结果不能代表正式结论。

论文可以声称的内容必须来自新分支真实重跑结果。旧分支历史数字、旧 checkpoint 和旧实验图不迁移为当前结果。
