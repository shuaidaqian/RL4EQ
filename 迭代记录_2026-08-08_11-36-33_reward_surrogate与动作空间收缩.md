# 迭代记录：reward surrogate 重设计与动作空间收缩

时间戳：2026-08-08 11:36:33

## 本轮目标

1. 重设计 reward surrogate，让它更贴近 Data BER 改善。
2. 收缩 / 重构离散动作空间，去掉污染选择的动作。
3. 重新跑一轮 Spearman 诊断，验证 surrogate 与 Data BER 的关系。

## 本轮实际修改

### 1）诊断脚本补齐了真实 PEFT 参数变化范数

在 `evaluation/research_diagnostics.py` 中，新增并回传了：

- `peft_delta_norm`
- `action_delta_norm`

这样可以把真实 PEFT 更新幅度纳入离线诊断，而不是把 PEFT 动作当成“零成本”操作。

### 2）动作空间从“带 aggressive 项”收缩为“轻量 PEFT + 回滚”

在 `agent/discrete_safe_policy.py` 和 `scripts/diagnose_research_assumptions.py` 中，动作表已经收缩为：

- `identity`
- `peft_head_light`
- `peft_adapter_lora_light`
- `peft_adapter_lora_head_light`
- `rollback_identity`

之前加入的更激进动作，例如：

- `peft_adapter_lora_fast`
- `peft_adapter_lora_head_deep`

在这一轮被移除，因为它们在这批 Level B 主配置上更容易把 Data BER 往坏的方向推。

### 3）窗口级 online reward 现在会计入真实 PEFT 参数变化

在 `training/windowed_discrete_ppo.py` 中，窗口 reward 的参数惩罚现在会把实际 PEFT 更新幅度计进去，而不是只看 modulation 的幅度。

这一步的意义是：如果动作确实引起了真实参数变化，reward 不再把它视作“零代价扰动”。

## 本轮测试结果

### 1）代码测试

完整测试通过：

```text
127 passed in 305.01s
```

说明：

- 诊断脚本接口没有破坏；
- 离散动作表的新结构是可运行的；
- PEFT 更新范数和窗口 reward 的改动没有引入回归。

### 2）全量 Spearman 诊断结果

本轮先用“含 aggressive 动作”的版本做过一轮诊断，再收缩到当前的轻量动作表后又跑了一轮。

#### 收缩前的关键现象

- `reward_ber_improvement` 基本全是 0，`reward_ber_delta` 没有信息量。
- `reward_loss_delta` 和 `data_ber_improvement` 的 Spearman 仍然很弱，甚至是负的。
- selected action 常常落到 `peft_adapter_lora_fast` / `peft_adapter_lora_head_deep`，但它们在这批主配置上的平均 Data BER 改善是负的。

#### 收缩后的关键现象

当前轻量动作表下：

- `reward_ber_delta` 仍然没有有效方差；
- `reward_loss_delta` 对 `Data BER` 的全量 Spearman 依然没有达到门槛；
- 但 selected action 的平均 Data BER 改善已经回到 `0.0`，不再是负值；
- 这说明 aggressive 动作确实是拖累当前诊断的主要来源，动作空间收缩是对的；
- 但同时也说明：仅靠当前这组轻量 PEFT 动作，仍然没有足够强的 Data 改善信号来支撑 `Spearman >= 0.6`。

## 这轮结论

### 结论 1：reward surrogate 还没有被真正“校准”到 Data BER

当前可见事实是：

- `reward_ber_delta` 在这批样本上几乎恒为 0；
- `reward_loss_delta` 虽然有变化，但和 `Data BER` 的方向关系仍不稳定；
- 即使加入真实 PEFT 参数变化范数，单帧 / 单动作的相关性依然不够强。

所以这轮不能把“reward surrogate 已经贴近 Data BER”当成成立结论。

### 结论 2：动作空间收缩是正确的，但收缩后太保守

删掉 `fast/deep` 之后：

- 选择不再明显伤害 Data BER；
- 但也几乎失去了正向 Data 改善样本；
- 说明当前轻量动作空间更安全，但还不足以把数据性能真正拉起来。

这意味着后续要么：

- 再做更细的窗口级 reward 设计；
- 要么重新引入一个“受控的、更强动作”，但必须加更严格的门控；
- 要么回到 offline 接收机本体，先把基础模型做强，再谈 RL。

## 对下一步的直接建议

1. 不要直接进入正式 PPO 训练。
2. 先做窗口级、按 episode 聚合的 reward / Data 相关性诊断。
3. 若仍然没有正相关，不要继续扩大 RL 训练规模。
4. 如果要恢复更强动作，必须把它们放进门控路径里，而不是直接放进主动作表。

## 本轮修改文件

- `evaluation/research_diagnostics.py`
- `scripts/diagnose_research_assumptions.py`
- `agent/discrete_safe_policy.py`
- `training/windowed_discrete_ppo.py`
- `tests/test_research_diagnostics.py`
- `tests/test_windowed_discrete_ppo.py`

