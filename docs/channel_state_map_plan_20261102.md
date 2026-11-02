# 保留 acquisition 先验的在线目标筛查

当前状态方向是 ridge/MAP 解，但前一阶段在线损失只有 likelihood。
这会把“后验方向中的正则”和“优化目标中的正则”混成一件事。新增目标为
`mean(|A h-y|²)/noise + sum(|h-h_acq|²/prior_variance)/rows`，
与方向构造共享 acquisition 行数、噪声和相干时间的对角先验。
Reward 仍只用留出信号 loss/BER，不加入先验惩罚，不参与 Adapt 梯度。

先固定 channel_state B256，不重新训练，不改变 guard。新的 offline_train
78000–78002、四 SNR、五帧龄 0/7/31/63/127，60 当前上下文。
对前阶段预设的 small/medium/large 三档，同帧比较 likelihood 与 MAP 两目标，
共 360 候选；全部候选和验收后才读 Data 标签。不据 Data 排序动作或选择权重。
统计按 seed，记录实际 CIR 位移和 BER/BCE；同当前帧的两目标是配对比较。

先用数学单元验证：未截断的后验方向、常数 gate=1 时 MAP 梯度应接近零，
而 likelihood 一般不是驻点；再验证 AdamW 一致、隐藏标签隔离和外层有限差分。
通过并不证明 BER 收益。只有机制有正向证据，才固定新的匹配 A/B 训练计划；
否则不扩大同一目标的留出矩阵。原四臂结论不变。
