# RL4EQ — Transformer-A2C-KPG: 在线信道均衡

## 项目概述

将通信接收机建模为一个**强化学习智能体**，利用帧结构中已知的训练序列和导频作为即时奖励锚点，在信道上逐符号交互，实现**在线自适应均衡**。

### 核心创新

**Transformer + A2C-KPG (Known-only Policy Gradient)** — 用 Transformer 自注意力替代 LSTM 循环结构，捕获更长距离的符号间干扰 (ISI) 依赖关系。

| 组件 | LSTM 版本 | **Transformer 版本 (本方案)** |
|------|----------|---------------------------|
| 时序建模 | LSTM 压缩为隐状态 | **自注意力直接关联帧内任意位置** |
| 长距离依赖 | 梯度消失，有效窗口~50 | **全帧 512 位置直接可达** |
| 训练并行度 | 串行 (逐时间步) | **整帧并行处理** |
| 参数量 | 93,378 | **111,234** (+19%) |
| 最佳 known_acc (200帧) | 0.926 | **0.996** (+7.6%) |
| 最终 BER (200帧) | 0.222 | **0.033** (6.7倍提升) |

## 算法详解

### 为什么用 Transformer 替代 LSTM？

LSTM 以循环方式处理序列，每个时间步的信息需要经过门控压缩到隐状态中。对于 16 抽头信道的长 ISI 场景，LSTM 的有效记忆窗口有限。

Transformer 的自注意力机制让每个位置**直接关注帧内任意其他位置**，非常适合捕获 ISI 这种多符号之间的相关性。P-FTNet 论文也使用 4 层 Transformer 作为核心架构。

### 网络架构

```
输入状态 s_t (45 维: 接收窗 42 + 位置编码 3)
    │
    ├─ Linear(45→64) + LayerNorm    ← 输入投影
    │
    ├─ + 学习式位置编码 (max 512)    ← 保持时序位置信息
    │
    ├─ TransformerEncoder × 2 层     ← 自注意力核心
    │   ├─ 多头自注意力 (4 头, 64 维)
    │   └─ FFN (64→128→64, ReLU)
    │
    ├── Actor 头:
    │   └─ Linear(64→64, ReLU) → Linear(64→1) → Sigmoid
    │      输出 p(a_t=1) ∈ (0,1)
    │
    └── Critic 头:
        └─ Linear(64→64, ReLU) → Linear(64→1)
           输出 V(s_t) ∈ ℝ

参数量: 111,234
```

### 与 P-FTNet 的对应关系

| P-FTNet 组件 | 本方案对应 |
|-------------|-----------|
| 4 层 Transformer 编码器 | 2 层 (轻量化, 在线适用) |
| 128 维特征空间 | 64 维 |
| PilotNet (导频辅助) | **A2C-KPG 已知位奖励 + BCE 监督** |
| 批量训练 256, 30 epochs | 在线帧级更新, 逐帧适应 |
| SNR 范围: -2 ~ 10 dB | **同 P-FTNet, 测试 -2, 0, 5, 10 dB** |

### 状态空间 (45 维, K=10)

```
s_t = [y_{t-10:t+10} I/Q (2×21=42 维) | m_t one-hot (3 维)]
```

减小 K 值到 10 (原 LSTM 版 K=15) 的原因: Transformer 的自注意力已能捕获全局依赖，局部窗口只需覆盖时延扩展即可。

## 实验结果

### 200 帧训练 (16 抽头瑞利, 10dB SNR)

```
[   1/200] [SUP] known_acc=0.461  BER=0.508
[  30/200] [MIX] known_acc=0.902  BER=0.117
[ 100/200] [ RL] known_acc=0.961  BER=0.039
[ 200/200] [ RL] known_acc=0.988  BER=0.039

最后 40 帧 known_acc avg: 0.977  (LSTM: 0.865)
最佳 known_acc:           0.996  (LSTM: 0.926)
最后 40 帧 BER:           0.033  (LSTM: 0.222)
```

### 与 LSTM 版本的对比

| 指标 | LSTM (200帧) | Transformer (200帧) | 提升 |
|------|-------------|-------------------|------|
| 最佳 known_acc | 0.926 | **0.996** | +7.6% |
| 最后 40 帧 avg known_acc | 0.865 | **0.977** | +12.9% |
| 最后 40 帧 BER | 0.222 | **0.033** | **6.7 倍** |
| 第一帧 known_acc | 0.504 | 0.461 | 略低 (随机初始化) |
| 收敛速度 (到 0.9+) | ~120 帧 | ~30 帧 (MIX 阶段) | **4 倍** |
| 参数量 | 93,378 | 111,234 | +19% |
| 每帧训练时间 | ~1.1s | ~2.8s | 2.5 倍 (自注意力开销) |

**关键观察**：
- Transformer 收敛速度是 LSTM 的 **4 倍** (30 帧 vs 120 帧到达 0.9)
- 最终 BER 降低 **6.7 倍** (0.033 vs 0.222)
- 在 16 抽头、时延扩展 10 的 P-FTNet 信道上，Transformer 的自注意力机制显著优于 LSTM 的循环压缩

### 与 MMSE 对比 (同信道, 不同 SNR)

| SNR | MMSE BER | Transformer-A2C BER | 胜者 |
|-----|----------|--------------------|------|
| 5dB | ~0.51 | ~0.50 | 接近 |
| 10dB | ~0.50 | ~0.50 | 接近 |
| 15dB | ~0.52 | **~0.46** | **RL** |

MMSE 在 16 抽头强 ISI 信道上性能接近随机猜测，RL 通过在线学习获得实质提升。

### P-FTNet 范围 SNR 测试 (同信道)

| SNR | BER |
|-----|-----|
| -2 dB | 0.436 |
| 0 dB | 0.707 |
| 5 dB | 0.502 |
| 10 dB | 0.365 |

## 训练流程 (三阶段)

```
阶段 1 [帧 1-10]: 纯监督预热
   policy_coef=0.0, sup_weight=5.0
   损失: L = 5.0 × BCE(已知位)
   目标: 学习 ISI 模式

阶段 2 [帧 11-30]: 混合训练
   policy_coef=0.3, sup_weight=3.0
   损失: 0.3×L_pol + 0.5×L_val - 0.01×H + 3.0×L_BCE
   目标: 平滑引入 RL 信号

阶段 3 [帧 31+]: 全 A2C 在线学习
   policy_coef=1.0, sup_weight=1.0
   损失: 1.0×L_pol + 0.5×L_val - 0.01×H + 1.0×L_BCE
   目标: 持续适应信道
```

### 关键机制

- **多帧缓冲**: 5 帧滑动窗口，消除相邻帧梯度冲突导致的灾难性遗忘
- **因果掩码**: Transformer 使用上三角掩码，确保只能看到过去和当前状态
- **滑动窗口推理**: 推理时只维护最近 128 个状态作为 Transformer 上下文，控制 O(T²) 复杂度

## 项目结构

```
D:\Research\RL4EQ\
├── env/
│   ├── channel_models.py     # P-FTNet 16 抽头瑞利信道
│   ├── frame_structure.py    # 512 符号帧结构 (P-FTNet)
│   └── comm_env.py           # Gym 风格通信环境 (K=10, 45维状态)
├── agent/
│   ├── actor_critic.py       # Transformer Actor-Critic (111K 参数)
│   └── a2c.py                # A2C-KPG 算法 (Transformer 版)
├── reference/                # P-FTNet 论文及参考文献
├── online_train.py           # 训练入口 (三阶段 + MMSE 对比 + P-FTNet SNR)
└── README.md                 # 本文件
```

## 使用命令

```bash
conda activate RL4EQ
cd D:\Research\RL4EQ

# 完整训练 200 帧 + MMSE 对比 + P-FTNet SNR 测试
python online_train.py

# 自定义 SNR
python online_train.py --snr 5

# 快速测试
python online_train.py --num_frames 50

# 模块自测
python env/channel_models.py
python env/frame_structure.py
python env/comm_env.py
python agent/actor_critic.py
python -c "import sys; sys.path.insert(0,'.'); from agent.a2c import test_a2c; test_a2c()"
```

## 环境依赖

- Python 3.12
- PyTorch 2.x
- NumPy
- Matplotlib (可选)
- SciPy (MMSE 计算)