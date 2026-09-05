# EME 在线动作跨帧延迟效应诊断

## 目的

Contextual Bandit 假设动作的收益主要可以由当前帧上下文和当前 Reward Pilot
反馈解释。如果一次 Adapter、FiLM 或 phase 更新要经过后续多个帧才产生收益，
问题就具有 delayed credit assignment，此时才有必要考虑安全约束的 Recurrent
Double DQN。

## 诊断方法

`scripts/diagnose_action_delay.py` 按 `seed`、delay、SNR、Pilot 配置和 state
split 对齐帧，从每个动作发生帧开始，统计 horizon `0/1/2/4/8` 的
`bandit_reward`。该 reward 只由 Reward Pilot 的前后 loss 改善、更新成本和回滚
惩罚组成，不读取 Data 标签。Data BER 可以保留在原始记录中作为最终仿真评估，
但不能参与控制器选择或诊断计算。

运行示例：

```powershell
.\.venv-gpu\Scripts\python.exe scripts/diagnose_action_delay.py `
  logs/final_online_smoke `
  --output logs/final_online_smoke/action_delay.json
```

## 决策规则

- 若非 `skip` 动作的延迟 reward 没有稳定高于 horizon 0，保留安全 Contextual
  Bandit；
- 若 horizon 1/2/4/8 的收益在多个 seed、多个 SNR 和 heldout edge 上稳定高于
  即时收益，标记为需要进一步验证的 delayed effect；
- 单个 seed 或单个动作出现延迟峰值，不足以支持引入 Recurrent Double DQN；
- 当前实现要求至少覆盖 2 个 seed 和 2 个 SNR 才会把延迟峰值标记为
  `delayed_effect_detected=true`，输出中同时记录 `support` 计数；
- 即使进入后续 DRQN 研究，动作仍只能来自安全离散集合，不能直接输出高维参数
  增量，也不能使用 Data 标签训练控制器。

当前文档只固化诊断协议，不预先宣称存在 delayed effect。正式结论必须引用
脚本输出和配对多 seed 实验。
