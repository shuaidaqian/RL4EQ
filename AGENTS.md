# AGENTS.md

我是中文语境开发者，所有对话和代码注释都用中文。

## 当前项目标准

RL4EQ 后续开发以根目录 [开发框架.md](开发框架.md) 为唯一架构标准。任何新功能、重构、实验入口、测试和清理工作都应先对齐该文档。

当前主线是：

```text
Offline Stage:
  随机无线信道生成
  -> Neural Equalizer 监督预训练
  -> 输出 AdapterEqualizer 同构 checkpoint: pretrained/model_best.pt

Online Stage:
  接收 Pilot + Data
  -> Pilot Equalization
  -> Pilot Loss / SNR / CFO / 历史状态
  -> RL Agent 选择少量参数更新策略
  -> Adapter/LoRA/输出头参数高效更新
  -> Equalize Data
  -> 与 MMSE baseline 对比
```

## 主入口

```bash
python pretrain.py --num_steps 1000 --batch_size 4 --save_dir pretrained
python online_train.py --num_frames 300 --snr 10 --pretrained pretrained/model_best.pt --output_dir logs/peft_online
python compare.py --snr_min 0 --snr_max 20 --snr_step 5 --num_frames 50 --pretrained pretrained/model_best.pt --output_dir logs/peft_snr_compare
python -m pytest tests/test_parameter_efficient_adaptation.py -q
```

## 代码边界

保留并维护：

- `pretrain.py`
- `online_train.py`
- `compare.py`
- `agent/neural_equalizer.py`
- `agent/adaptation_controller.py`
- `agent/adaptation_policy.py`
- `env/*.py`
- `baseline/mmse_equalizer.py`
- `tests/test_parameter_efficient_adaptation.py`

不再恢复旧的 `actor_critic.py`、`actor_critic_v2.py`、`agent/ppo.py`、`pretrain_v8.py`、`online_train_v8.py` 等历史并行实现。

## 指标标准

所有在线实验至少报告：

- `BER_data`
- `BER_pilot`
- `pilot_loss`
- `adapt_params`
- `adapt_steps`
- `latency_ms`
- `generalization`

`BER_data` 只能作为仿真评估指标，不能作为真实在线 reward。
