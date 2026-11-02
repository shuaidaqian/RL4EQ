# 连续历史没有确认统一收益，先修通正式接入契约

按[前阶段登记](channel_memory_history_stage_20261102.md)，固定尺度控制、offline_train 100000–100002、四SNR、年龄1023/4095，连续Adapt1/8 × Reward1/8，signal/MAP分别评分。24个当前上下文、192行，两年龄各自从零记忆产生候选，不是数百秒持续学习。

**64项逐年龄改善区间、32项跨年龄平均改善区间均包含零**。更多已知Pilot可以补足代数约束，但未建立统一Data判决收益。研究墙钟 **502.17秒**，包含连续物理轨迹模拟，不等同生产时延。

MAP在Adapt8/Reward8的跨年龄改善均值为 +.21701/+.82465/−.19531/−.04340 pp；全部区间跨零。signal同组合为+.65104/+.08681/+.10851/+.02170 pp，也都跨零。没有根据这些均值改选动作、改guard或增加留出seed。

[机器摘要](results/channel_memory_contiguous_stage_20261102.json)保存所有组合、逐年龄区间、因素差值、来源SHA。原始 `logs/channel_memory_contiguous_windows_20261102.json`。

完成入口修复后的全量CPU FFT回归 **799 passed，245.99秒**；PPO/机制定向两文件135项通过，新物理帧龄端到端测试单独通过。GPU一次acquisition 8帧CLI smoke已跑通，不是学习后策略证据。

## 下一条依赖：正式接入

已有独立持续保留参数的8帧机制在复记忆10/15 dB成立；在线动作异质、单一通用候选与窗口扩大均未解决所有SNR。下一步应检验离散安全RL是否能利用合法Pilot区分这些候选，不能继续把“所有候选都先在四SNR成功”作为RL研究的必要前提。

这不是宣称RL必然解决问题。先修复已有正式入口的两个可复现问题：

1. `online_train.py --method ppo`传入`state_split`，runner原签名不接受；现在显式转交环境构造。
2. 正式PPO与Frozen前向未传真实物理帧龄，状态分支默认为0；现在从`Frame.frame_index`转交，refinement重复前向也保持相同物理年龄，日志区分局部与物理计数。启用状态分支时缺少物理年龄必须失败，旧无状态测试桩保留兼容。

此处只修入口契约，没有添加新RL动作或改其输入/reward语义。已有独立机制始终显式传物理年龄，不受此问题影响。

定向PPO与机制文件 **135 passed，32.40秒**；新增实际状态分支在局部frame=1、物理age=1023时的Frozen/PPO前向测试另 **1 passed，4.42秒**。GPU正式CLI `--method ppo --state-split offline_train`、116时延、prefix256、10dB、8帧跑通；该smoke仍是原head动作，不当作复记忆RL结果。

后续按[接入计划](channel_memory_rl_bridge_plan_20261102.md)完成动作、目标、接收输入与Reward语义一致性核验，再做小预算训练范围窗口机制。整体目标、传统主比较与长期优势仍未完成。
