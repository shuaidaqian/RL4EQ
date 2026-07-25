# reference 文献索引

本目录只索引当前 Level B 极端长时延块均衡路线直接相关资料：

1. P-FTNet：Pilot-conditioned Transformer equalization under extreme delay spread。用于说明长上下文、Pilot 条件和极端 delay spread 的神经均衡背景。
2. Learning During Detection：在线持续学习接收机，强调 Pilot/DMRS 在部署期间更新接收机的作用。
3. Online Modular Bayesian Receiver：低延迟在线接收机适配参考，作为非 RL 在线学习对照。
4. PPO-driven adaptive filtering reward：复合 reward 与 PPO 控制滤波器更新的参考，支撑 Continual PPO reward 设计。

当前研究合同：

- Level B 是主论文场景。
- Continual PPO 是第一主贡献。
- 接收机是整帧缓冲、非因果块均衡器。
- 正式目标为每个主配置 `BER_data < 0.01`。
- 在线流程不使用数据标签上界。

未索引 MIMO MCTS、RIS 空间均衡、多用户资源分配等与当前 SISO 极端长回波主线无直接关系的资料。
