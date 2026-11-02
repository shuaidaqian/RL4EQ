# 把已验证的神经状态动作接入正式离散窗口RL

本轮不再要求所有候选先在四SNR独立成功；RL研究的问题正是能否根据合法Pilot状态选择不同安全候选。现有复记忆持续机制在10/15dB成立，0/5dB、尺度学习额外价值与后期优势未确认。最终门槛保持不变。

先完成以下可审查工程契约，不以接入成功替代统计成功：

- 新动作必须显式配置选择，旧默认动作表不变；identity、记忆signal、记忆MAP、小步/medium以及rollback只输出离散索引，绝不输出234维参数增量。
- 复记忆动作更新与`differentiable_adapt`精确一致，使用Adapt信号目标；固定34尺度与全部旧权重，保留累计半径。
- 动作选择observation只用Adapt及历史反馈；每个动作在短窗口复用，reward只用动作后窗口Reward Pilot BER/loss改善。
- 实验显式使用整帧接收波形计算Reward BER，信号loss仍仅用Reward方程；Data发送符号/标签不进入任何在线控制。旧配置的输入语义不静默改变。
- 正式前向必须传真实物理帧龄，不能用局部评估帧号代替；同Frozen使用相同acquisition、条件估计和接收波形。
- 正式`online_train.py`与`compare.py`都要能选择动作与配置，严格checkpoint加载及动作表校验。更新/rollback、污染隔离与实际参数范数必须验证。

完成工程接入后，先在新的offline_train seed用8/32帧GPU smoke核对实际动作与reward，再登记128帧持续机制；通过后再数百秒与冻结传统协议比较。不能把初始策略8帧一次窗口的结果充作已学习策略，不能根据Data选动作或挑checkpoint。

未有新的用户约束需要澄清；沿用一次会话开始acquisition、每帧prefixPilot、先统计后实时优化的授权。
