# 正式离散记忆动作接入完成，统计门槛仍未通过

本轮完成[接入计划](channel_memory_rl_bridge_plan_20261102.md)的工程契约：两个正式入口可显式选择复记忆离散安全动作，梯度只更新 234 个记忆参数，34 个尺度及其余离线参数固定。动作在 8 帧窗口复用；Reward Pilot 的 BER 与信号损失改善构成窗口反馈；保持每步、窗口与会话累计投影和回滚。

## 修复的实际协议问题

`online_train.py` 的离散 runner 原先丢失 acquisition CFO，未遵循配置中的 PPO 学习率和状态摘要开关；`compare.py` 的 RL 条件原先未传 CFO。这些问题现在已修复。同 checkpoint 的独立 Frozen 与 identity RL，在 prefix 多帧、零/非零 acquisition CFO 条件下输出一致。

新增 `rl_reward_receiver_input=full_rx` 显式配置，让 Reward 判决读取整帧已接收波形。其标签仍只来自 Reward，信号方程仍只使用已知 Pilot；策略 observation 与梯度分别只用 Adapt。跨两个 PPO 窗口污染隐藏 Data 发送符号/标签，不改变动作概率、Reward、PPO 更新、记忆参数或策略参数。

旧默认动作表及 `pilot_only` 语义保留；记忆动作需显式启用，缺少记忆分支、非 prefix 实际布局或非法接收协议会在入口拒绝。正式 checkpoint 和动作表仍严格校验。

## 已完成验证

- 全量默认 pytest：**807 passed，416.82 秒**；完整日志 `logs/channel_memory_rl_protocol_pytest_20261102.txt`。这次使用默认 direct 后端，且和一段研究运行有并发，不用于生产时延比较。
- 此后补充的汇总测试：**8 passed，3.92 秒**，覆盖 full_rx 声明、混合协议拒绝、独立 seed 配对、PPO/窗口计数与验收后保留范数。未记录范数的历史日志明确返回 null。
- GPU 统一入口四 SNR × 8 帧 × 五方法，共 **160 条记录、32 个配对帧**，同帧接收波形 SHA 一致，传统方法使用冻结校准协议。该次只抽到 identity，Proposed 与独立 Frozen BER 完全一致。

8 帧同步检查的 Frozen/Proposed BER 均为：0 dB 0.303060，5 dB 0.074056，10 dB 0.003418，15 dB 0.000977。它没有在线收益证据；0/5 dB 还未超过冻结 FDE。完整[机器记录](results/channel_memory_rl_bridge_sync_8f_20261102.json)保留各方法比较和协议摘要。

## 接续阶段

按[修复后计划](channel_memory_rl_protocol_plan_20261102.md)，从新的训练范围实际 seed 100010 开始，四 SNR 各 128 帧，固定控制 checkpoint、固定传统协议，从首帧持续更新。显式使用既有 FFT 后端并保存完整进程墙钟，修复前及未完成的 direct 日志不混入有效统计。

接入成功不等于统计成功。整体目标仍未完成：需要四 SNR 逐配置超过校准传统 baseline，相对独立 Frozen 的在线改善有独立 seed 区间支持，且后期平均优势更明显，最终覆盖数百秒连续会话。
