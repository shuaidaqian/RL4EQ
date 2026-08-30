# EME 长记忆信道在线元适配设计

## 目标

第二研究点研究：在 Level B 极端稀疏长回波信道中，当前帧只有前缀 Adapt Pilot 和 Reward Pilot 可用、Data 标签不可见时，在线适配能否恢复离线模型在合法但未见过的慢状态轨迹上的性能，并逐配置超过公平的传统非神经均衡器。

## 研究假设

EME 主场景不是快速 Doppler 信道。在线收益来自长记忆、跨帧 ISI、稀疏长回波、残余 CFO、慢相位扰动和慢 tap-gain 漂移共同造成的有限状态失配。匹配分布作为 sanity check；主实验使用同一 Level B profile 内、离线训练未覆盖但物理范围合法的状态轨迹。

## 主算法

主算法命名为“Pilot 驱动两时间尺度约束在线元适配”。模型由物理长记忆 warm-start 和神经残差块组成。

1. 每帧前缀 Adapt Pilot 估计 residual CFO、公共相位和可见长回波状态。
2. CFO/相位使用逐帧快速递推；稀疏 tap 使用固定 support 下的慢速递推；上一帧输出形成 soft tail，按真实帧顺序更新。
3. Adapt Pilot 只更新 head、FiLM 和小规模 adapter/LoRA 等 PEFT 参数，更新步数为一到两步，限制增量范数。
4. Reward Pilot 只用于更新后的留出 guard/reward。变差则恢复 PEFT 和慢状态快照，但不回退已经产生的最新 soft tail。
5. 离线阶段采用序列化一阶元训练：外层目标直接优化“Adapt Pilot 更新后”的 Reward Pilot 损失和跨帧稳定性，而不是只优化初始离线损失。
6. RL 不直接生成高维参数增量。后续只允许选择离散的更新强度、PEFT group 和跨帧更新率；固定规则版本是主消融，contextual bandit 优先于 PPO。

## 信息边界

在线适配可以读取接收信号、profile-level 先验、Adapt Pilot 及其已知符号、Reward Pilot 及其已知符号。在线 observation、动作、参数更新和 soft-tail 更新不得读取 Data 标签、真实 CIR、真实 CFO、真实相位或仿真上界。

## 对照和成功判据

至少比较 Frozen Offline NN、普通 Pilot PEFT、元适配主方法、CFO+DD LMMSE-FIR、CFO+DD DFE-RLS、稀疏 CIR Kalman/RLS，以及完整记忆 Pilot LMMSE/CG 诊断项。主结果按 SNR 0/5/10/15 dB、channel seed 和帧序列报告，不能只报告总体平均。

成功要求主方法在有限状态失配和慢漂移主实验中稳定优于 Frozen Offline NN，并逐配置优于最强公平传统 baseline；匹配分布中至少不能因在线更新退化。若在线适配没有超过 Frozen Offline NN，只能报告安全稳健适配，不能声称在线带来持续增益。

## 失败保护

低 SNR、Reward Pilot 样本不足、非有限梯度、参数增量超限或 Reward Pilot 变差时拒绝更新并降低更新强度。所有更新前后损失、接受率、参数增量、状态置信度、tail 更新率和标签使用审计字段必须落盘。
