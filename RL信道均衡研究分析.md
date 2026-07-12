# 强化学习做信道均衡：代码架构、文献实现与后续路线分析

生成日期：2026-07-12  
项目目录：`D:\Research\RL4EQ`

> 说明：本次只读梳理代码，没有修改任何 `.py` 代码文件。新增内容仅包括本报告和下载到 `reference/rl_equalization_review/` 的论文 PDF/文本摘录。

## 1. 当前项目代码架构

当前项目已经对齐根目录 `开发框架.md`，主线是：

```text
Offline Stage:
  随机无线信道生成
  -> AdapterEqualizer 监督预训练
  -> 输出与在线阶段同构的 checkpoint

Online Stage:
  接收当前帧 Pilot + Data
  -> Neural Equalizer 初始推理
  -> 基于 Pilot 构造 RL observation
  -> PPO Agent 选择少量参数更新策略
  -> Adapter/输出头/同步头参数高效更新
  -> Equalize Data
  -> 与 MMSE baseline 对比
```

### 1.1 主入口

| 文件 | 当前职责 |
|---|---|
| `pretrain.py` | 离线监督预训练入口，训练 `AdapterEqualizer`，保存 `model_best.pt`、`model_final.pt`、`model_config.json`、预训练曲线。 |
| `online_train.py` | 在线自适应入口，加载预训练模型，使用 PPO 控制参数高效更新策略，并输出 PEFT 与 MMSE 对比指标。 |
| `compare.py` | SNR 扫描入口，循环调用 `run_online_adaptation()`，输出 PEFT 与 MMSE 的 BER 曲线。 |
| `tests/test_parameter_efficient_adaptation.py` | 主线约束测试，覆盖 Adapter 可训练参数比例、帧结构、状态构造、采样信道、在线指标、MMSE baseline、checkpoint 加载等。 |

### 1.2 环境与信道

| 文件 | 实现 |
|---|---|
| `env/frame_structure.py` | 固定帧结构：`Training(128) + Pilot(64) + Data(128) + Pilot(64) + Data(128)`。训练序列与导频使用确定 PN pattern，数据位随机。 |
| `env/comm_env.py` | 生成帧、过信道、构造整帧状态。状态维度为 `2*(2K+1)+3`，默认 `K=10` 时为 45 维。 |
| `env/rayleigh_channel.py` | 频率选择性 Rayleigh 多径信道，支持可变 tap、delay spread、AWGN、可选 Doppler 更新接口。 |
| `env/rician_channel.py` | Rician 多径信道，含 LOS 分量和散射分量。 |
| `env/_3gpp_channel.py` | 3GPP EPA/EVA/ETU profile 信道。 |
| `baseline/mmse_equalizer.py` | 基于训练序列估计信道，使用频域 MMSE 均衡，作为强 baseline。 |

当前状态构造为：

```text
state_t = [
  rx_window_IQ,        # 2*(2K+1)
  known_symbol,        # train/pilot 位置为真实 BPSK 符号，data 位置为 0
  known_mask,          # train/pilot 为 1，data 为 0
  data_mask            # data 为 1，train/pilot 为 0
]
```

这比旧版“位置 one-hot”更合理，因为已知符号被显式注入状态，同时 data 段不泄漏真实 bit。

### 1.3 Neural Equalizer

实现文件：`agent/neural_equalizer.py`

核心模型是 `AdapterEqualizer`：

```text
states (B, T, 45)
  -> input_proj: Linear + LayerNorm
  -> position embedding
  -> optional NeuralChannelEncoder + FiLM
  -> optional SyncPhaseDelayHead + FiLM
  -> TransformerEncoder backbone
  -> ResidualAdapter
  -> output_head
  -> bit logits / probabilities
```

关键接口：

- `forward(states) -> (logits, probs)`
- `enable_parameter_efficient_tuning(train_adapter, train_output, train_sync)`
- `set_trainable_targets(...)`
- `trainable_parameters()`
- `trainable_parameter_count()`

离线阶段训练完整模型；在线阶段默认冻结主干，只开放 Adapter、输出头，必要时开放同步头。

### 1.4 在线 RL 控制器

实现文件：

- `agent/adaptation_controller.py`
- `agent/adaptation_policy.py`

当前 RL 不是直接输出均衡比特，也不是直接输出滤波器系数，而是帧级策略控制器：

```text
RL action_id -> AdaptationStrategy
```

离散 action 表：

| action | 更新模块 | lr | steps |
|---|---:|---:|---:|
| `skip` | 不更新 | 0 | 0 |
| `head-slow` | output_head | 1e-4 | 1 |
| `head-fast` | output_head | 5e-4 | 1 |
| `adapter-slow` | adapter | 1e-4 | 1 |
| `adapter-fast` | adapter | 5e-4 | 1 |
| `both-slow` | adapter + output_head | 1e-4 | 1 |
| `both-fast` | adapter + output_head | 5e-4 | 1 |
| `both-deep` | adapter + output_head | 5e-4 | 3 |

RL observation 当前为 8 维：

```text
pilot_loss
BER_pilot
pilot_conf_mean
pilot_conf_std
SNR/30
loss_ema
ber_ema
last_reward
```

reward：

```text
reward = pilot_loss_before - pilot_loss_after - 0.01 * adapt_steps
```

符合 `开发框架.md` 的约束：在线 reward 只来自 pilot，不使用 data 标签。

### 1.5 当前实验输出指标

`online_train.py` 已输出：

- `BER_data`
- `BER_pilot`
- `pilot_loss`
- `adapt_params`
- `adapt_steps`
- `latency_ms`
- `generalization`

这些指标符合 AGENTS.md 和 `开发框架.md` 要求。

## 2. 已下载并分析的论文

本次检索时，当前会话没有暴露专门的 academic-search skill，因此使用网页检索 + arXiv/出版社开放 PDF 作为替代。已下载 PDF 到：

```text
reference/rl_equalization_review/
```

并提取了方法片段到：

```text
reference/rl_equalization_review/extracted_method_snippets.txt
```

### 2.1 论文列表

| 论文 | 链接 | 本地文件 |
|---|---|---|
| Jeon, Lee, Poor, 2019, Robust Data Detection for MIMO Systems with One-Bit ADCs: A Reinforcement Learning Approach | https://arxiv.org/abs/1903.12546 | `1903.12546-jeon2019-onebit-mimo-rl-detection.pdf` |
| Kim, Jeon, Li, Tavangaran, Poor, 2020, Data-Aided Channel Estimator for MIMO Systems via Reinforcement Learning | https://arxiv.org/abs/2003.10084 | `2003.10084-kim2020-data-aided-ce-rl.pdf` |
| Kim et al., 2022, Semi-Data-Aided Channel Estimation with Reinforcement Learning | https://arxiv.org/abs/2204.01052 | `2204.01052-kim2022-semi-data-aided-ce-rl.pdf` |
| Mo, Chang, Kan, 2021, Deep Reinforcement Learning Aided Monte Carlo Tree Search for MIMO Detection | https://arxiv.org/abs/2102.00178 | `2102.00178-mo2021-drl-mcts-mimo-detection.pdf` |
| Bereketoglu, 2025, Composite Reward Design in PPO-Driven Adaptive Filtering | https://arxiv.org/abs/2506.06323 | `2506.06323-bereketoglu2025-ppo-adaptive-filtering.pdf` |
| Obeed, Jian, 2026, Learning During Detection: Continual Learning for Neural OFDM Receivers via DMRS | https://arxiv.org/abs/2602.20361 | `2602.20361-obeed2026-learning-during-detection.pdf` |
| Ben-Itzhak, Ayanoglu, 2026, RIS-Enabled Wireless Channel Equalization: Adaptive RIS Equalizer and Deep Reinforcement Learning | https://arxiv.org/abs/2603.02489 | `2603.02489-benitzhak2026-ris-equalizer-drl.pdf` |
| Katwal, Bhatia, 2021, Improved Channel Equalization using Deep Reinforcement Learning and Optimization | https://doi.org/10.4108/eai.28-10-2021.171685 | `katwal2021-improved-channel-equalization-drl-optimization.pdf` |

项目原有 `reference/` 下还已有：

- `P-FTNet Pilot-Conditioned and Feature-Augmented Transformer Network for Equalization Under Extreme Delay Spread-0423-zhang.pdf`
- `Reducing Pilots in Channel Estimation with Predictive Foundation Models.pdf`
- `reference/README.md` 中整理的 24 篇相关论文清单。

## 3. 文献实现范式对比

### 3.1 Jeon et al. 2019：RL 学 likelihood function，而不是直接判 bit

对象：one-bit ADC MIMO data detection。

实现逻辑：

- 问题不是直接用 RL 预测符号，而是修正由粗量化和信道估计误差造成的 likelihood mismatch。
- 检测后的 input-output sample 带有伪标签不确定性，不能全部当作可靠训练样本。
- MDP state 包含已选样本、权重和当前样本索引。
- action 是二值选择：是否使用当前检测样本更新经验 likelihood。
- reward 是 likelihood 估计 MSE 改善。

启示：

该类工作没有让 RL 端到端替代通信算法，而是让 RL 做“样本选择/估计修正”。这是更稳的范式：RL 控制一个低维、可解释、对最终误差有明确物理关联的过程。

### 3.2 Kim et al. 2020/2022：RL 选择 data-aided channel estimation 样本

对象：MIMO data-aided channel estimation。

实现逻辑：

- 常规 LMMSE 只用 pilot，pilot 长度受限导致 CSI 不准。
- data detection 后的符号可作为额外 pseudo-pilot，但错误检测会导致 error propagation。
- RL 的 action 控制“是否接受当前检测符号作为额外训练样本”。
- reward 定义为 channel estimate MSE 改善。

启示：

这类工作把 RL 放在“是否信任伪标签”的决策层，而不是让 RL 直接做均衡。对你的项目很重要：当前在线阶段只用 pilot loss 驱动，而没有利用 data 段的置信度/一致性来安全扩展监督信号，所以 online adaptation 的信息量明显不足。

### 3.3 Mo et al. 2021：DRL + MCTS 做 MIMO detection

对象：MIMO 符号检测。

实现逻辑：

- 将 MIMO detection 表述为树搜索问题。
- DRL 网络输出 policy 和 state value，用于指导 MCTS 扩展搜索。
- 训练方式更接近离线 self-play / search-guided learning。
- 推理成本较高，但检测性能可以超过 ZF/MMSE/DNN detector。

启示：

这类方法的优势来自“搜索 + 值函数”的组合，不是简单 PPO 更新参数。它适合 MIMO 符号空间，不一定适合你当前一维 BPSK 序列均衡，但说明如果 RL 直接参与检测，通常需要结构化搜索或显式序列决策机制。

### 3.4 Bereketoglu 2025：PPO 用于 adaptive filtering，但 reward 是复合物理指标

对象：自适应滤波/信号增强。

实现逻辑：

- PPO 控制滤波器更新。
- reward 不是单一分类 loss，而是复合目标，例如输出质量改善、误差项、平滑/稳定惩罚等。
- 强调 reward design 对收敛和鲁棒性很关键。

启示：

你当前 reward 只有 `pilot_loss_before - pilot_loss_after - step_penalty`，过于局部。它能优化 pilot，但不保证 data 段 ISI、相位偏移、同步误差或泛化 BER 同步改善。需要引入更多 pilot 可观测的物理一致性项。

### 3.5 Obeed & Jian 2026：Learning During Detection，DMRS 驱动在线学习

对象：神经 OFDM receiver 的在线 continual learning。

实现逻辑：

- 不主打 RL，而是用 DMRS/pilot 在检测过程中持续更新 receiver。
- 关键在于 pilot 设计和在线更新流程，使 pilot 同时服务 demodulation 和 learning。
- 更接近你的当前架构：当前项目也是 pilot loss 驱动在线微调。

启示：

如果目标是工程有效性，这类“pilot-driven online learning”比硬套 RL 更直接。你的 RL Agent 必须证明它比固定 PEFT 策略更好，否则论文贡献会被质疑为“用 PPO 选择学习率/步数，但收益不稳定”。

### 3.6 Ben-Itzhak & Ayanoglu 2026：RIS equalizer 的 DRL 控制连续物理动作

对象：RIS 辅助脉冲响应均衡。

实现逻辑：

- action 是 RIS 相位等连续物理控制量。
- 比较 DDPG、TD3、SAC 等连续控制算法。
- reward 与接收脉冲响应、主径能量、ISI 能量、SNR 提升有关。

启示：

该类 DRL 任务的 action 与信道物理机制直接耦合，因此 RL 有发挥空间。你当前 action 是离散训练策略，不直接控制均衡器系数/相位/滤波响应，RL 信号更间接，效果上限自然更受限。

### 3.7 Katwal & Bhatia 2021：WOA + Q-learning 均衡

对象：传统数字通信信道下的 channel equalization。

实现逻辑：

- 先用 Whale Optimization Algorithm 优化 bit stream 或均衡相关参数。
- 再用 Q-learning 选择低干扰 bit stream。
- 在 AWGN、Rician、Rayleigh、Nakagami 下对比 BER。

启示：

论文声称 BER 改善，但实现更偏启发式优化 + Q-learning，不是严格的现代在线 receiver 架构。可作为相关工作，不建议作为你论文方法主依据。

## 4. 你的当前方案为什么接近或弱于 MMSE

### 4.1 MMSE baseline 本身很强，而且使用了正确的物理先验

`baseline/mmse_equalizer.py` 使用训练序列估计信道，再构造频域 MMSE：

```text
W = H* / (|H|^2 + 1/SNR)
```

在当前仿真设置中：

- 调制是 BPSK。
- 信道是线性多径 + AWGN 为主。
- 训练序列和 pilot 占 50% 帧长。
- MMSE 与信道模型高度匹配。

因此 MMSE 是强 baseline，不是弱传统算法。若信道主要是线性时不变或准静态多径，神经/RL 方法没有明显额外信息，反而会因为训练误差、泛化误差和在线更新噪声而弱于 MMSE。

### 4.2 当前 Neural Equalizer 没有显式信道估计，只从短窗口隐式学习逆信道

当前模型输入是 `rx_window_IQ + known_symbol/meta`。它没有显式拿到：

- 信道冲激响应估计 `h`
- 频域响应 `H(f)`
- 噪声方差估计
- CFO/相位/定时偏移估计
- MMSE soft output

Transformer 只能从整帧相关性中隐式推断这些量。离线训练覆盖信道族不足或模型容量/损失设计不足时，面对某些信道会不如 MMSE 的显式估计-求逆流程。

### 4.3 RL action 太间接，优化的是“如何微调模型”，不是“如何均衡信道”

当前 PPO action 是：

```text
skip / head-slow / head-fast / adapter-slow / adapter-fast / both-slow / both-fast / both-deep
```

这类 action 只决定学习率、步数和可训练模块。它没有直接控制：

- 均衡滤波器系数
- channel estimate 的样本选择
- pseudo-label 接受/拒绝
- 判决反馈强度
- 相位/CFO/时延修正

因此 RL 的作用链路很长：

```text
action -> 微调策略 -> pilot loss 变化 -> 模型参数变化 -> data BER
```

链路越长，pilot reward 与 data BER 的相关性越弱。

### 4.4 reward 只看 pilot loss 的单步下降，容易过拟合 pilot

当前 reward：

```text
pilot_loss_before - pilot_loss_after - 0.01 * steps
```

局限：

- pilot loss 下降不等于 data BER 下降。
- 两个 64 长度 pilot block 对复杂多径/时变信道覆盖不足。
- 若模型对 pilot 局部过拟合，data 段可能变差。
- 没有参数漂移惩罚，只有 step penalty。
- 没有置信度校准、ISI 残差、判决一致性、平滑性等物理指标。

这也是“pilot 上看起来更新有效，但 data 段弱于 MMSE”的常见原因。

### 4.5 `reset_each_frame=True` 限制了跨帧学习

`online_train.py` 默认每帧恢复 `θ_pre`：

```text
每帧恢复 θ_pre: True
```

优点：

- 避免上一帧错误更新污染下一帧。
- 评估更稳定，符合“当前帧在线自适应”设定。

缺点：

- PPO policy 可以跨帧学习，但 equalizer 的参数适配不跨帧积累。
- 对慢时变信道、相邻帧相关信道，不能利用历史连续性。
- 若当前帧 pilot 信息不足，模型无法通过多帧统计改善。

这会让方法更像“每帧少量梯度搜索”，而不是完整在线学习 receiver。

### 4.6 离线训练目标和在线目标存在分布差异

离线：

```text
data_loss_weight * BCE(data) + known_loss_weight * BCE(train/pilot)
```

在线：

```text
只用 pilot BCE 做更新与 reward
```

这会带来目标错配：

- 离线模型最重视 data 位；
- 在线更新只看到 pilot；
- RL policy 只根据 pilot 局部收益学习；
- 最终评估却看 `BER_data`。

如果 pilot distribution 和 data distribution 在 ISI 影响、位置、上下文上不一致，在线更新可能对主指标帮助有限。

### 4.7 当前 PPO 数据效率偏低

当前 `DiscretePPOPolicy` 是帧级单步 PPO。每帧只有一个 action 和一个 reward，rollout 短，reward 噪声大。PPO 更适合有足够交互样本的策略优化；在这里策略学习的数据量可能不够，容易退化成随机探索离散微调策略。

如果没有固定策略 baseline 对比，例如：

```text
always skip
always head-slow
always both-fast
oracle best action on pilot
oracle best action on data，仅离线评估用
```

很难证明 PPO 本身贡献了稳定收益。

## 5. 在保持 `开发框架.md` 不变的前提下，下一步应该怎么做

### 5.1 第一优先级：把实验闭环做严谨

必须先回答一个问题：

```text
PPO 控制 PEFT 是否稳定优于固定 PEFT 策略和 MMSE？
```

建议增加对比矩阵：

| 方法 | 目的 |
|---|---|
| MMSE | 强传统 baseline |
| no adaptation | 预训练 equalizer 上限/下限 |
| fixed head-slow | 固定轻量更新 |
| fixed adapter-slow | 固定 adapter 更新 |
| fixed both-fast | 固定强更新 |
| PPO policy | 当前方法 |
| pilot-oracle action | 每帧选 pilot loss 最小的 action，检验 action space 上限 |
| data-oracle action | 只用于仿真诊断，检验 pilot reward 和 data BER 的错配程度 |

如果 PPO 不优于固定策略，优先改 reward/observation/action，而不是继续加模型复杂度。

### 5.2 第二优先级：量化 pilot reward 与 data BER 的相关性

建议输出散点和相关系数：

```text
Δpilot_loss vs ΔBER_data
BER_pilot vs BER_data
pilot_conf_mean/std vs BER_data
action_id vs BER_data
```

如果 `Δpilot_loss` 与 `ΔBER_data` 相关性很低，说明当前 reward 不适合作为在线优化目标。此时继续 PPO 没有意义。

### 5.3 第三优先级：增强 observation

在不使用 data 标签的前提下，建议加入：

```text
pilot_loss_before
pilot_loss_after_candidate，若做候选动作试探
pilot_conf_quantiles
pilot error burst length
pilot residual autocorrelation
estimated SNR
estimated CFO / phase drift
estimated delay spread
MMSE pilot BER / MMSE pilot loss
neural-vs-MMSE disagreement on pilot
history: last_action, last_adapt_params, last_latency
```

其中最重要的是：

```text
neural-vs-MMSE disagreement
pilot residual autocorrelation
estimated delay spread / CFO
```

这些特征能让策略知道当前失败模式，而不仅知道“pilot loss 大不大”。

### 5.4 第四优先级：增强 reward，但仍只用 pilot 可观测量

建议 reward 改成复合形式：

```text
reward =
  λ1 * (pilot_loss_before - pilot_loss_after)
  + λ2 * (pilot_ber_before - pilot_ber_after)
  + λ3 * (confidence_after - confidence_before)
  - λ4 * parameter_delta_norm
  - λ5 * update_steps
  - λ6 * latency_ms
  - λ7 * pilot_residual_correlation
```

参数漂移惩罚很关键：

```text
parameter_delta_norm = ||θ_adapt - θ_pre|| / ||θ_pre||
```

否则模型可能为降低 pilot loss 做过强局部更新。

### 5.5 第五优先级：让 RL 控制更有物理意义的动作

当前 action 只控制训练策略。建议逐步扩展为：

```text
action = {
  update_target: head / adapter / sync / lora / none,
  lr_id: low / mid / high,
  steps_id: 0 / 1 / 3,
  regularization_id: weak / medium / strong,
  pseudo_label_gate: off / high_conf_only / consistency_with_mmse
}
```

更推荐增加“伪标签门控”：

```text
只把 neural 与 MMSE 高置信一致的数据位置作为 pseudo-pilot
```

这直接借鉴 Kim/Jeon/Poor 的 data-aided RL 思路：RL 不直接学 bit，而是控制哪些 data 决策可被信任。

### 5.6 第六优先级：把 MMSE 从对手变成教师/特征

目前 MMSE 只作为 baseline。若目标是在强传统算法上改进，应利用它：

```text
输入特征加入：
  raw rx window
  MMSE soft output window
  MMSE residual
  estimated channel h 或 H(f)
  neural-MMSE disagreement
```

训练目标加入：

```text
L = BCE(data)
  + λ_distill * KL(neural_soft, mmse_soft)
  + λ_residual * residual_consistency
```

这样神经网络学习的是 MMSE 的非线性残差和错误修正，而不是从零学习线性均衡。

### 5.7 第七优先级：减少过强的“50% 已知位”设定依赖

当前已知位为 256/512，占比 50%。这对 MMSE 很有利，也使“在线学习”的研究价值不够突出。建议设置多组 pilot overhead：

```text
known ratio: 50%, 25%, 12.5%, 6.25%
```

你的方法如果要有论文价值，最好在低 pilot overhead、非线性硬件失真、CFO、Doppler、模型错配场景下超过 MMSE。

## 6. 如果保持当前整体架构，我建议的短期实验路线

### 阶段 A：诊断

1. 固定同一批 channel seeds，比较 MMSE / no adaptation / fixed PEFT / PPO。
2. 输出每帧 action、pilot loss 变化、data BER 变化。
3. 计算 `Δpilot_loss` 与 `ΔBER_data` 的 Pearson/Spearman 相关。
4. 做 data-oracle action，只用于诊断 action space 是否本身有潜力。

判断标准：

- 如果 data-oracle action 明显优于固定策略，但 PPO 不行，问题在 observation/reward/policy。
- 如果 data-oracle action 也不优于固定策略，问题在 action space 或 equalizer 本身。
- 如果 neural no-adaptation 已弱于 MMSE 很多，先修离线预训练/模型输入，而不是在线 RL。

### 阶段 B：改 reward 和 observation

1. 加入参数漂移惩罚。
2. 加入 pilot residual correlation。
3. 加入 MMSE pilot 指标和 neural-MMSE disagreement。
4. 引入 candidate action probing：对每个 action 做一小步虚拟更新，取 pilot 可观测 summary 给 policy。

### 阶段 C：引入 pseudo-pilot data-aided adaptation

规则：

```text
data 位置只有同时满足以下条件才能参与在线更新：
  neural confidence > τ1
  MMSE confidence > τ2
  neural decision == MMSE decision
  邻域判决稳定
```

RL action 控制：

```text
τ1 / τ2 / pseudo-label 使用比例 / 更新模块 / 更新步数
```

这会把当前方法从“只用 128 个 pilot 更新”扩展到“安全利用 data 段高置信伪标签”，但仍不使用真实 data 标签。

### 阶段 D：加入困难信道

优先加入：

```text
CFO
phase noise
timing offset
IQ imbalance
PA nonlinearity
Doppler time-varying channel
longer delay spread > window_K
```

这些是 MMSE 容易模型错配的场景，也是神经/RL 方法更可能体现价值的场景。

## 7. 更推荐的替代架构：Model-Assisted RL Equalizer

如果允许调整研究实现架构，但仍保持“强化学习做信道均衡”这一研究点，我建议从当前：

```text
RL controls PEFT strategy
```

升级为：

```text
Model-Assisted RL controls sample selection + residual equalization + adaptation strength
```

### 7.1 核心思想

不要让 RL 直接替代 MMSE，也不要只让 RL 选择学习率。更稳的做法是：

```text
MMSE / LS 给出物理可解释初始均衡
Neural Equalizer 学残差修正
RL Agent 决定：
  1. 哪些 data 判决可以作为 pseudo-pilot
  2. 当前帧是否需要更新
  3. 更新哪些 PEFT 模块
  4. 更新强度和正则强度
```

这与文献趋势一致：

- Jeon/Kim 系列：RL 选择可靠样本，降低伪标签传播错误。
- Bereketoglu：PPO reward 要包含物理质量和稳定性项。
- Obeed/Jian：pilot/DMRS 驱动在线 receiver 更新。
- RIS DRL：action 与物理通道控制更直接时，RL 更有效。

### 7.2 推荐系统流程

```text
Offline:
  多信道生成
  -> LS/MMSE baseline 计算 h_hat、mmse_soft、residual
  -> Neural Residual Equalizer 监督训练
  -> 训练一个 pilot/data 置信度校准头
  -> 训练/预训练 RL policy 或 imitation policy

Online:
  当前帧 rx
  -> LS/MMSE 得到 h_hat、mmse_soft、mmse_conf
  -> Neural Equalizer 初始输出 neural_soft
  -> 构造 observation:
       pilot metrics
       MMSE metrics
       neural-MMSE disagreement
       residual statistics
       channel summary
  -> RL Agent 选择:
       update or skip
       pseudo-label threshold
       trainable module
       lr / steps / regularization
  -> 用 pilot + selected pseudo-pilots 做 PEFT 更新
  -> 输出 data decision
  -> 评估 BER_data
```

### 7.3 推荐 action space

```text
action = (
  update_target,
  lr_id,
  steps_id,
  pseudo_label_gate_id,
  regularization_id
)
```

其中：

```text
update_target:
  none / output_head / adapter / lora_qv / lora_ffn / sync_head

lr_id:
  1e-5 / 1e-4 / 5e-4

steps_id:
  0 / 1 / 3

pseudo_label_gate_id:
  off
  neural_conf_0.9
  neural_mmse_agree_0.8
  neural_mmse_agree_0.9

regularization_id:
  weak / medium / strong
```

### 7.4 推荐 reward

在线真实 reward 仍不能使用 data label。建议：

```text
reward =
  + Δpilot_loss
  + α * Δpilot_BER
  + β * Δpilot_confidence
  - γ * parameter_delta_norm
  - η * pseudo_label_disagreement
  - μ * latency_ms
  - ν * update_steps
```

其中：

```text
pseudo_label_disagreement =
  mean(|neural_soft - mmse_soft| on selected pseudo-labels)
```

这个项不需要真实 data 标签，但能抑制错误伪标签扩散。

### 7.5 推荐论文卖点

如果按该架构推进，论文贡献可以写成：

1. 提出一种 pilot-constrained RL adaptation 框架，严格避免 data label 泄漏。
2. 将 MMSE 作为 model-assisted teacher，而非仅作为 baseline。
3. 设计可靠伪标签门控的 RL action，使 data 段高置信判决参与在线自适应。
4. 引入物理一致性 reward，缓解 pilot overfitting。
5. 在低 pilot overhead、CFO/Doppler/nonlinear impairment 下超过 MMSE。

## 8. 结论

当前项目架构方向是合理的：它已经避免了“RL 直接逐 bit 判决”的高方差问题，把 RL 放在在线参数高效更新控制层，并且严格限制 reward 来自 pilot。这符合通信 receiver 的在线可观测约束。

但当前方法的主要不足是：

1. MMSE baseline 与当前线性多径仿真高度匹配，传统算法很强。
2. Neural Equalizer 没有充分利用显式信道估计和 MMSE soft 信息。
3. PPO action 只控制学习策略，和均衡物理过程耦合较弱。
4. reward 只看 pilot loss 单步改善，与 data BER 可能弱相关。
5. 每帧恢复 `θ_pre` 限制了慢时变信道下的连续学习收益。
6. 缺少固定 PEFT、oracle action、pilot-data reward correlation 等关键消融，导致很难证明 RL 策略本身的贡献。

接下来不建议盲目增加 PPO 复杂度。优先级应是：

```text
先诊断 pilot reward 是否真的预测 data BER
-> 再增强 observation/reward
-> 再加入 MMSE-assisted 特征和 pseudo-pilot 门控
-> 最后扩展到 CFO/Doppler/非线性等 MMSE 模型错配场景
```

如果目标是形成硕士第二个研究点，最稳的题目方向是：

```text
基于 Pilot 约束强化学习的参数高效神经信道均衡：
从固定 PEFT 到模型辅助的伪标签门控在线自适应
```

