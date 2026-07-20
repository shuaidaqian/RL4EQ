# AGENTS.md

我是中文语境开发者，所有对话和代码注释都使用中文。

## 唯一架构标准

RL4EQ 后续开发以根目录 `开发框架.md` 为唯一架构标准。新功能、重构、实验、测试和清理必须先与该文档对齐。

当前唯一研究主线：

```text
Offline Stage:
  EME 启发稀疏长回波信道
  -> 长记忆神经均衡器监督预训练
  -> 随机 Pilot 条件对齐训练
  -> pretrained/model_best.pt

Online Stage:
  Adapt Pilot + Reward Pilot + Data
  -> Adapt Pilot observation
  -> PPO 选择 Adapter/输出头更新策略
  -> Reward Pilot 留出奖励
  -> Data 均衡与仿真评估
```

## 主入口

```bash
python pretrain.py --config configs/extreme_delay.json --save_dir pretrained
python online_train.py --config configs/extreme_delay.json --pretrained pretrained/model_best.pt --output_dir logs/online
python compare.py --config configs/extreme_delay.json --pretrained pretrained/model_best.pt --policy logs/online/policy.pt --output_dir logs/compare
python -m pytest tests/test_extreme_delay_adaptation.py -q
```

## 代码边界

保留并维护：

- `pretrain.py`
- `online_train.py`
- `compare.py`
- `agent/neural_equalizer.py`
- `agent/adaptation_controller.py`
- `agent/adaptation_policy.py`
- `env/extreme_delay_channel.py`
- `env/frame_structure.py`
- `env/comm_env.py`
- `baseline/mmse_equalizer.py`
- `baseline/traditional_equalizers.py`
- `tests/test_extreme_delay_adaptation.py`

不得恢复旧的逐比特 Actor-Critic、旧 PPO、LDPC、3GPP 多信道并行实现、MMSE 辅助特征、伪标签、CFO 或同步实验分支。

## 在线信息边界

- Adapt Pilot 可用于模型条件、observation 和梯度更新。
- Reward Pilot 只能在动作执行后用于 reward 和留出评估。
- `BER_data` 只能作为仿真评估指标，绝不能进入在线 reward、observation 或动作选择。
- 所有在线实验至少报告 `BER_data`、`BER_adapt_pilot`、`BER_reward_pilot`、`pilot_loss`、`adapt_params`、`adapt_steps`、`latency_ms`、`parameter_delta_norm` 和 `generalization`。

