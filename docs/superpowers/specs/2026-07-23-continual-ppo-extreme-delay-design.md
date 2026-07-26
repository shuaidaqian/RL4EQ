# RL4EQ Continual PPO 极端长时延均衡设计规范

**日期：** 2026-07-23  
**状态：** 已由用户逐项确认  
**架构约束：** 本规范批准后用于重写根目录 `开发框架.md`，后者仍是生产代码实施时的唯一架构标准。

## 1. 研究目标

唯一主路线是：在 EME 启发的 Level B 稀疏强长回波慢变信道中，先建立物理引导、可少步适配的块神经均衡器，再使用部署期间持续更新的分层 recurrent PPO 联合控制信道估计、参数微调和展开检测。

主工作矩阵为：

```text
Level B
max_delay = 20 / 30 / 40 symbols
SNR = 10 / 15 / 20 dB
Gauss-Markov rho = 0.99
```

第一主贡献是 Continual Hierarchical Recurrent PPO。物理引导均衡器、课程训练和 meta-training 是进入 PPO 的技术底座，不是并列研发路线。

系统固定为整帧缓冲、非因果、单载波、未编码 BPSK。粗同步已完成；不建模 CFO、相位噪声、CP、LDPC、MIMO、RIS、非线性和绝对地月传播时延。

## 2. 阶段门槛

1. Perfect-CSI 强块检测器在 Level B 九配置分别具备 `BER_data < 0.01` 的可达性。
2. Perfect-CIR Unfolded Neural Equalizer 在九配置分别达到 `BER_data < 0.01`。
3. Best Fixed Adaptation 在九配置分别达到 `BER_data < 0.1`；开发阶段每配置至少 4/5 seeds 达标。
4. Reward Pilot loss 改善与 Data BER 改善的验证集 Spearman 相关系数至少为 0.6。
5. Continual PPO 在九配置分别达到 `BER_data < 0.01`，并配对显著优于 Best Fixed、Drift-Aware Rule 和 Contextual Bandit。

不实现任何数据标签上界、轨迹标签上界、Frozen PPO 或 PPO-from-random-init。

## 3. 信道族

共同规则：总 tap 功率归一化，时延必含 0 和最大时延，其余时延无放回采样，复相位随机，episode 内 support 固定，跨帧保留 ISI。

### 3.1 Level A

- 路径数 3–5；
- 最强/次强功率差 6–15 dB；
- 最大时延路径相对最强路径为 -10 至 -20 dB；
- 总延迟路径能量占比 10%–35%；
- 用于课程学习和消融，不用于主结论。

### 3.2 Level B

- 路径数 3–7；
- 最强/次强功率差 0–6 dB；
- 最大时延路径不低于最强路径 -10 dB；
- 总延迟路径能量占比 35%–75%；
- 排除冻结阈值定义的病态深谱零点；
- 用于主训练、门槛和论文结论。

### 3.3 Level C

- 路径数 3–10；
- 总延迟路径能量占比 50%–90%；
- 近等功率远时延回波、深谱零点或高条件数；
- 单独报告压力测试，不混入 Level B 平均，不要求全部低于 0.01。

在神经训练前生成至少 10,000 个候选信道，使用 2048 点频谱最小增益、条件数 proxy 和 Perfect-CSI 结果冻结 A/B/C 阈值。分类边界不得根据神经 BER 事后移动。

SNR 使用 `Es/N0` 口径；信道 tap 总能量和 BPSK 符号能量均为 1，复噪声总方差为 `10**(-SNR_dB/10)`，I/Q 每维方差为其一半。禁止按单帧偶然接收功率重新缩放。

跨帧增益围绕 episode 基准 tap 做均值回归 Gauss-Markov 演化。主场景 `rho=0.99`，`0.999` 和 `0.95` 分别用于慢变与快变消融；第一版不加入路径出生/消失。

## 4. 帧、Acquisition 与状态

每个 episode 开始发送 `max_delay` 个已知 warm-up BPSK 符号和一个 512 符号全已知 acquisition frame，用于初始化 CIR、噪声、tracker、Conditioner、previous-tail、PEFT anchor 和策略状态。它们不计入 Data BER，但计入 goodput 开销。

普通帧长固定为 512。候选 Pilot 开销为：

| Total | Adapt | Reward | Data |
|---:|---:|---:|---:|
| 64 | 48 | 16 | 448 |
| 96 | 72 | 24 | 416 |
| 128 | 96 | 32 | 384 |
| 160 | 120 | 40 | 352 |

候选布局为 Prefix、Two-block 和 Multi-block。Pilot 使用连续子块、随帧变化且可复现的 PN 序列；至少保留一个长度不小于 48 的主要 Adapt anchor，剩余 Adapt/Reward 子块分布到帧内。

主均衡器只区分 Adapt-known 与 Unknown。Reward Pilot 和 Data 使用相同 unknown region；Reward mask、当前位置和标签不进入动作前模型输入或 PPO observation。动作完成后才揭示 Reward 标签。Reward 标签不得用于均衡器梯度、CIR 更新或下一帧 Adapt。

每种方法独立维护上一帧最后 `max_delay` 个 soft symbols。第一普通帧由 acquisition 已知尾部精确初始化；后续只使用本方法自身软判决，不得用 Data 标签纠正。

最终帧结构在满足 Reward/Data Spearman、Fixed Adaptation 和 Continual PPO 门槛后，以包含 warm-up/acquisition 的 effective goodput 最大者为准。

## 5. 接收机架构

### 5.1 Hybrid CIR Conditioner

Adapt Pilot 子块输入 Sparse CIR Estimator，输出：

- 长度为 `max_delay+1` 的复 `h_hat`；
- tap support probability；
- noise/SNR estimate；
- confidence；
- latent residual embedding。

离线损失包含复 CIR NMSE、support loss、Pilot reconstruction 和 noise estimation。在线没有真实 CIR，只使用 Adapt Pilot BCE、Pilot reconstruction 和 proximal regularization。

### 5.2 Physics-Guided Unfolded Equalizer

用 `h_hat` 构造可微块卷积算子 `H_hat` 和伴随 `H_hat^H`。第 k 层执行：

```text
r_k = y - H_hat x_k
z_k = x_k + alpha_k H_hat^H r_k
x_(k+1) = NeuralDenoiser_k(z_k, channel_embedding, noise_hat)
```

第一版展开 2/4/6/8 层可选。网络输出 BPSK bit logits 和 probabilities。禁止使用 MMSE 判决作为输入或教师；允许在核心网络内部使用显式 `H/H^H` 物理算子。

在线参数组包括 Conditioner/FiLM、Adapter、最后一层 Attention LoRA、FFN LoRA 和输出头。最终单帧动作最多影响总模型参数的 10%，优先不超过 5%；Full Fine-tune 仅作模块诊断，不进入 PPO。

## 6. 离线训练

训练采用 Stage 0–6：

0. 校准信道族与 Perfect-CSI 可达性；
1. 独立训练 Sparse CIR Estimator；
2. 使用 true CIR 训练 Unfolded Equalizer并通过 `<0.01` 门槛；
3. Estimated-CIR 联合训练，课程为 Level A→B、Pilot 160→64、SNR 20→10、delay 20→40；
4. 对 12 种 Pilot 结构做共享模型初筛，只对前 2–3 个候选做独立对齐；
5. First-order episodic meta-training；
6. 冻结接收机、Pilot、PEFT 参数组、动作边界和 reward 后训练 PPO。

Meta-training 中 Support 为 Adapt Pilot，inner-loop 只更新允许的参数组；Query 为 Reward Pilot + Data，Data 标签只用于离线 outer loss。Data 标签不得用于 PPO、在线更新、伪标签或动作搜索。

## 7. Continual Hierarchical Recurrent PPO

PPO observation 使用结构化编码器：CIR 时延轴、support/confidence、帧间 CIR 差、展开残差序列、Adapt Pilot loss/BER、Pilot reconstruction、噪声、Pilot配置、previous-tail 置信度、历史动作、更新次数、参数 delta、历史 reward 和帧编号。禁止当前 Reward 标签、Reward loss、Data 标签和 BER_data。

策略使用单层 GRU（第一版 hidden size 128），episode 内持续、episode 间清零。动作空间：

- 离散 mode：skip、update-channel、update-equalizer、joint-update、detector-refine、rollback；
- 离散 parameter group：Conditioner/FiLM、Adapter、Attention LoRA、FFN LoRA、Adapter+LoRA、Conditioner+PEFT；
- steps：1/2/4；
- detector iterations：2/4/6/8；
- 连续 learning rate `[1e-6,1e-3]`（对数映射）；
- proximal strength `[1e-6,1e-2]`；
- reconstruction weight `[0,1]`；
- damping `[0,0.95]`；
- CIR trust `[0,1]`。

动作先经过预算和数值安全投影。NaN、Inf、发散残差或越界参数变化触发拒绝与分状态回滚。

Reward 只使用动作后揭示的 Reward Pilot：

```text
r_immediate = log((loss_before+eps)/(loss_after+eps))
r_cumulative = log((shadow_loss+eps)/(loss_after+eps))
reward = r_immediate + beta * r_cumulative
```

不加入普通参数量、步数或延迟惩罚。shadow model 是 episode-start 冻结均衡器。

PPO先在仿真中离线预训练，部署期间继续更新 policy。开发 episode 为 1 acquisition + 300 普通帧；正式为 1 acquisition + 1000 普通帧。每 32 帧在线 PPO update 2–4 epochs，使用小学习率和 KL 安全回滚。每个测试 seed 都从相同离线 PPO、optimizer规则和接收机状态重置，禁止跨 seed 继承。

主指标是完整 1000 帧 prequential BER；同时报告 1–100、101–300、301–1000 分段结果。只实现 Continual PPO；不实现 Frozen PPO。小规模消融为无 GRU、无累计 shadow reward、无 detector-control。

## 8. Baseline 与公平比较

保留但不作为独立研发路线：

- Perfect-CSI Block LMMSE/强迭代检测（诊断）；
- Adapt Pilot Sparse CIR + Kalman/RLS tracker；
- Sparse-CIR Block LMMSE/CG；
- Sparse-CIR DFE-RLS；
- Sparse-CIR 解析 Iterative BPSK Detector；
- 当前 LMMSE-FIR、DFE-RLS（简单参考）；
- No Adapt；
- Best Fixed Adaptation；
- Drift-Aware Pilot Rule；
- Contextual Bandit；
- Continual Hierarchical Recurrent PPO。

所有可部署方法使用相同 acquisition、Adapt Pilot、接收帧、噪声、previous-tail规则和 episode状态。Reward 标签不得被传统方法或固定方法用于估计和更新。Perfect-CSI 方法不进入可部署排名。

## 9. 统计与结论

开发阶段每配置 5 seeds × 200 帧用于均衡器/Pilot/Fixed门槛；PPO开发使用 5 seeds × 300 帧。正式冻结后使用 10 个未见 seeds × 1000 帧。

正式统计使用 seed/episode 分层 block bootstrap，帧块长度 10。报告均值、seed标准差、95% CI、每配置成功 seed比例、最坏 seed、中位数、四分位数和配对差。

Continual PPO 成功要求：九配置分别平均 `<0.01`，每配置至少 8/10 seeds 达标，配对显著优于 Best Fixed、Pilot Rule 和 Contextual Bandit，至少 7/9 配置取得最低可部署 BER。任一配置退化超过 0.002 必须单列。

## 10. 工程约束

核心实验必须在 GTX 1650 4 GB 上可运行：主模型优先不超过 2M 参数，策略优先不超过 1M 参数，峰值显存目标不超过 3.5 GB。使用 AMP、gradient accumulation、first-order meta-learning和 truncated BPTT。

单模块开发训练目标小于 2 小时，单候选和 PPO开发训练小于 12 小时。所有长入口必须保存模型、optimizer、scheduler、RNG和阶段状态，支持 `--resume`，按 method/delay/SNR/seed 增量写出并跳过已完成分片。

## 11. 文档与清理

实施时从零重写根目录 `开发框架.md` 和 `RL信道均衡研究分析.md`，同步更新 README、AGENTS、配置和测试。旧 checkpoint 与新架构不兼容，不做转换。旧数据标签上界、固定动作表、latent-only生产主干和旧全量结果只保留在带时间戳迭代记录中，不继续作为当前结论。
