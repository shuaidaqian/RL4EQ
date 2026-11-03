# 跨帧观测下的离线适配目标配对计划

## 触发依据

116 样本共同后缀完成环境、算子、神经前向、传统接收机、在线入口和标签隔离验证。原 checkpoint 的 10 dB、3 seed、128 帧探测中，Proposed 相对独立 Frozen 平均改善为 −0.02950 个百分点，95% seed 级配对区间跨零；后期减早期同样未通过。不能把增加接收窗口本身写成在线学习成功，也不直接扩大到数百秒主矩阵。

## 固定训练协议

- 配置为 `configs/eme_long_memory_v2_cross_frame_memory_rl.json`，保留 Level B、0/5/10/15 dB、1024 样本帧、116 样本记忆、1200 秒相干时间、前缀 Adapt224/Reward32 和会话开始一次 acquisition。允许等待后缀的时间为 11.6 ms。
- 从同一个 `pretrained/channel_memory_map_control64_20261102/model_best.pt` 恢复两次，分别训练 `baseline` 和 `post_adapt`，各 **64 steps、batch size 1**；不从测试 Data 选 checkpoint，终点固定为预算终点。
- 离线实际环境 seed 为 **98100–98163**，SNR 按 0/5/10/15 轮转，历史帧龄按 0/127/1023/2047/4095 轮转。外层同时评分当前帧和 +8 帧；这些未来外层标签只供离线监督。
- 复用同一个带配置指纹的离线样本缓存，A/B 的当前波形、后缀波形和帧龄必须一致。新缓存目录 `logs/cross_frame_offline_samples_20261103` 不复用旧无后缀样本。
- 外层只更新现有 `channel_state_memory_scale`；内层仍为三步 AdamW、MAP Pilot 信号目标、记忆参数组、半径 .8，反馈规则、动作和 Pilot 预算不变。B 分支加入更新后 Data 外层目标，A 分支只用普通离线 BCE_all。两者均可在离线外层使用 Data 标签；实际在线更新始终禁用。
- 模型前向的物理 warm start 使用联合连续卷积；当前和后缀在同一坐标上作相位补偿。内层信号重构仍只用当前帧已知 Adapt 方程，不把下一帧接收样本当作纯尾部残差。

## 先验检查与接续门槛

先完成 2 step GPU smoke，确认高阶反传、后缀输入、缓存和保存路径；不根据 smoke 的 Data 值改训练预算或学习率。然后顺序运行 A/B，避免 GPU 并发，保存训练元数据及 checkpoint SHA256。

训练结束先用固定的新开发 seed、同波形四臂检查：A Frozen、A 固定 Pilot 更新、B Frozen、B 固定 Pilot 更新。该检查用于检验离线目标是否改善在线更新的 Data 泛化，不单独证明 RL 归因，也不替代真实连续会话。只在机制获得支持后，再冻结新 checkpoint 和独立 seed 规则，进行 128 帧正式 RL/Frozen 连续复核。

后缀改变了传统接收机的可用观测；旧传统校准参数在当前短 smoke 中只作冻结设置诊断。最终主比较之前必须在同一新窗口下完成独立传统校准，不能宣称旧设置已是新协议最优设置。最终目标仍是四 SNR 逐配置优于校准传统 baseline、相对同 checkpoint Frozen 的在线改善有独立 seed 区间支持、后期更明显。
