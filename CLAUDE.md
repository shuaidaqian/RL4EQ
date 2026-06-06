# CLAUDE.md

使用说明：此文件指导 Claude Code (claude.ai/code) 在该仓库中工作时的行为。

## 项目概览

RL4EQ 研究**基于强化学习的在线自适应信道均衡**。使用 Transformer Actor-Critic 网络和 A2C-KPG（已知位策略梯度）算法，在无需信道状态信息的情况下逐符号均衡频率选择性衰落信道。

## 常用命令

```bash
conda activate RL4EQ

# 训练 200 帧，默认 SNR=10dB
python online_train.py

# 自定义 SNR / 帧数 / 设备
python online_train.py --snr 5 --num_frames 50 --device cpu

# 带 LDPC 信道编码的训练 (256,128) + BP 解码
python online_train_ldpc.py --snr 10 --num_frames 200

# 模块自测（单独验证各组件）
python env/channel_models.py
python env/frame_structure.py
python env/comm_env.py
python agent/actor_critic.py
python agent/a2c.py
python env/ldpc_coding.py
```

## 架构说明

### 目录结构

```
env/                        # 通信环境
  channel_models.py         # 16抽头瑞利多径信道（频率选择性，时延扩展=10）
  frame_structure.py        # 帧结构：|训练(128)|导频(64)|数据(128)|导频(64)|数据(128)| = 512 符号
  comm_env.py               # Gym风格环境：state(45维)=接收窗(42)+位置编码(3)，已知位奖励±1
  ldpc_coding.py            # (3,6)-规则 LDPC (256,128)，对数域 BP(Sum-Product) 解码

agent/                      # RL 智能体
  actor_critic.py           # Transformer编码器(2层,4头,64维) + Actor头 + Critic头，111K参数
  a2c.py                    # A2C-KPG算法：仅已知位计算策略梯度，多帧缓冲

online_train.py             # 主训练入口：三阶段(监督→混合→纯RL) + MMSE对比 + P-FTNet SNR扫描
online_train_ldpc.py        # 同上，但数据位携带LDPC码字，评估BP解码后BER

reference/                  # 参考文献（P-FTNet, PPO自适应滤波, 贝叶斯接收机等）
logs/                       # 训练可视化（training_curve.png, pftnet_snr.png, mmse_comparison.png）
```

### 核心设计

- **A2C-KPG（已知位策略梯度）**：仅训练序列和导频位提供奖励（±1）和策略梯度信号。数据位通过共享的 Transformer 参数间接学习，无需人工设计数据位奖励。
- **状态表示**：45维向量 = 以当前符号为中心的接收IQ窗（2×21）+ one-hot位置编码（训练/导频/数据）。
- **因果掩码 Transformer**：自注意力覆盖整帧512位置，但每个时间步只能看到过去和当前状态（上三角掩码）。使用可学习位置编码（最大512）。
- **三阶段训练**：SUP（1-10帧，纯BCE监督）→ MIX（11-30帧，混合RL+BCE）→ RL（31帧+，纯A2C）。阶段切换控制 `policy_coef` 和 `sup_weight`。
- **多帧缓冲**：5帧滑动窗口累积梯度，降低相邻帧梯度冲突导致的方差。
- **推理**：维护最近128个状态的滑动窗口，控制 O(T²) 复杂度。
- **LDPC 集成**：数据位承载一个(256,128)LDPC码字。RL均衡后，对数域BP解码进一步降低BER。

### 关键参数

| 配置项 | 值 |
|--------|------|
| 帧长 | 512 符号 |
| 已知位 | 256（50%：训练128 + 导频×2 64） |
| 信道抽头 | 16抽头瑞利，指数衰减PDP |
| 时延扩展 | 10 符号 |
| 状态维度 | 45（窗长 K=10） |
| Transformer | 2层，4头，d_model=64，FFN=128 |
| 参数量 | 111,234 |
| 优化器 | Adam, lr=3e-4 |
| Gamma | 0.97 |
| 缓冲大小 | 5 帧 |

### 关键编辑位置

- `online_train.py` / `online_train_ldpc.py`：训练循环、阶段调度、评估、绘图。各自包含独立的 `A2CAgent` 类和 `train_on_buffer` 方法（与 `agent/a2c.py` 中的 A2C 类为两套独立实现）。
- `agent/actor_critic.py`：Transformer 架构——在此修改 d_model、n_layers、n_heads 等。
- `agent/a2c.py`：A2C-KPG 算法——配置 actor_lr、gamma、entropy/value 系数。
- `env/comm_env.py`：状态表示、奖励函数、环境接口。
- `env/channel_models.py`：信道模型（抽头数、PDP、SNR、时变选项）。
- `env/ldpc_coding.py`：LDPC 码构造、BP 解码器参数。

**注意**：`online_train.py` 和 `online_train_ldpc.py` 各自包含一份独立的 `A2CAgent` 类（代码重复，非从 `agent/a2c.py` 导入）。`agent/a2c.py` 模块是另一套重构后的 A2C 实现。`ppo.py` 实现的是标准的 LLM RLHF PPO 训练流程，与通信均衡任务完全无关。
