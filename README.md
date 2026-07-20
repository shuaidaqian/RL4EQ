# RL4EQ：极端长时延信道的离线预训练与在线强化学习微调

RL4EQ 研究一种面向 EME 启发稀疏长回波信道的神经均衡框架。项目不尝试在常规无线信道上盲目替代已经很强的传统均衡器，而是聚焦最大相对时延达到 20–40 个符号、传统滤波器面临长程 ISI 和高维估计困难的极端场景。

## 核心流程

```text
离线阶段：
  随机完整序列长记忆预训练
  -> 随机 Pilot 开销条件对齐训练
  -> 生成通用神经均衡器 checkpoint

在线阶段：
  | Adapt Pilot 48 | Reward Pilot 16 | Data 448 |
  -> PPO 根据 Adapt Pilot 选择微调动作
  -> 只更新 Adapter/输出头
  -> Reward Pilot 计算留出奖励
  -> 均衡 Data 并与 LMMSE-FIR、DFE-RLS 比较
```

绝对地月传播时延在粗同步完成后不会直接形成符号间干扰。本项目所称“EME 启发”特指稀疏、强回波、长相对时延扩展和跨帧慢变信道记忆。

## 环境

- Python 3.12
- PyTorch 2.x
- NumPy
- Matplotlib（仅用于曲线输出）
- Pytest

当前环境安装的是 CPU 版 PyTorch，因此论文级多种子实验建议在 CUDA 版 PyTorch 环境运行。

## 快速开始

离线预训练：

```bash
python pretrain.py --config configs/extreme_delay.json --save_dir pretrained
```

在线 PPO 参数高效微调：

```bash
python online_train.py \
  --config configs/extreme_delay.json \
  --pretrained pretrained/model_best.pt \
  --output_dir logs/online
```

统一比较：

```bash
python compare.py \
  --config configs/extreme_delay.json \
  --pretrained pretrained/model_best.pt \
  --policy logs/online/policy.pt \
  --delays 20 30 40 \
  --snrs -5 0 5 10 15 20 \
  --num_seeds 5 \
  --num_frames 200 \
  --output_dir logs/compare
```

测试：

```bash
python -m pytest tests/test_extreme_delay_adaptation.py -q
```

## CPU Smoke

```bash
python pretrain.py --config configs/extreme_delay.json --stage_a_steps 2 --stage_b_steps 2 --batch_size 1 --save_dir pretrained/smoke
python online_train.py --config configs/extreme_delay.json --pretrained pretrained/smoke/model_best.pt --num_episodes 1 --frames_per_episode 2 --output_dir logs/smoke_online
python compare.py --config configs/extreme_delay.json --pretrained pretrained/smoke/model_best.pt --policy logs/smoke_online/policy.pt --delays 20 40 --snrs 0 10 --num_seeds 1 --num_frames 2 --output_dir logs/smoke_compare
```

## 目录

```text
agent/       长上下文均衡器、在线适配控制器和 PPO 策略
baseline/    LMMSE-FIR 与 DFE-RLS
configs/     统一实验参数和固定评估种子
env/         极端稀疏长回波信道、帧结构和帧级环境
reference/   与极端时延、Pilot 在线学习和 PPO 奖励直接相关的资料
tests/       新主线唯一契约测试
```

## 结论纪律

单次或单种子实验只能说明可行性趋势。只有完成配对的多种子实验并给出 95% 置信区间后，才能声称神经均衡器优于传统算法，或 PPO 优于固定微调。
