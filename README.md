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

- NVIDIA GeForce GTX 1650（4 GB）
- Python 3.12.7
- PyTorch 2.11.0+cu128
- CUDA Runtime 12.8 / cuDNN 9.19
- NumPy 1.26.4
- Matplotlib 3.11.1
- Pytest 9.1.1

本机已创建并验证 `.venv-gpu`。Windows PowerShell 下可重新创建同构环境：

```powershell
python -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install --upgrade pip
.\.venv-gpu\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv-gpu\Scripts\python.exe -m pip install numpy==1.26.4 matplotlib==3.11.1 pytest==9.1.1
.\.venv-gpu\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 快速开始

离线预训练：

```powershell
.\.venv-gpu\Scripts\python.exe pretrain.py --config configs/extreme_delay.json --save_dir pretrained --device cuda
```

在线 PPO 参数高效微调：

```powershell
.\.venv-gpu\Scripts\python.exe online_train.py `
  --config configs/extreme_delay.json `
  --pretrained pretrained/model_best.pt `
  --output_dir logs/online `
  --device cuda
```

统一比较：

```powershell
.\.venv-gpu\Scripts\python.exe compare.py `
  --config configs/extreme_delay.json `
  --pretrained pretrained/model_best.pt `
  --policy logs/online/policy.pt `
  --delays 20 30 40 `
  --snrs -5 0 5 10 15 20 `
  --num_seeds 5 `
  --num_frames 200 `
  --output_dir logs/compare `
  --device cuda
```

测试：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv-gpu\Scripts\python.exe -m pytest tests/test_extreme_delay_adaptation.py -q -p no:cacheprovider
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
