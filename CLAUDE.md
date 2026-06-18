# CLAUDE.md

使用说明：此文件指导 Claude Code (claude.ai/code) 在该仓库中工作时的行为。

## 项目概览

RL4EQ 研究**基于强化学习的在线自适应信道均衡**。使用 Transformer Actor-Critic 网络，支持 A2C-KPG 和 PPO+GAE 两种算法，在无需信道状态信息的情况下逐符号均衡频率选择性衰落信道。新增**离线预训练 + 在线微调**流水线。

## 系统架构

### 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RL4EQ 系统架构                                │
└─────────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
   │ 离线预训练    │    │ 在线 PPO 训练 │    │ 基线对比评估   │
   │ pretrain.py  │    │ online_train │    │ --eval 模式   │
   │              │    │ .py          │    │              │
   │ 固定信道      │    │ PPO/A2C 双   │    │ MMSE/DFE     │
   │ 监督 BCE     │    │ 模式         │    │ + 可视化      │
   │ 双向注意力    │    │ GAE + Clip   │    │              │
   └──────┬───────┘    └──────┬───────┘    └──────────────┘
          │                   │
          ▼                   ▼
   ┌─────────────────────────────────────────┐
   │           共享模块                        │
   │                                          │
   │  env/    环境层 (信道/帧/状态/奖励)        │
   │  agent/  智能体层 (Actor-Critic/PPO)     │
   └─────────────────────────────────────────┘
```

### 模块依赖关系

```
pretrain.py (standalone)
  └── env.frame_structure  → FrameConfig, FrameGenerator
  └── env.channel_models   → RayleighMultipathChannel
  └── env.comm_env         → CommunicationEnv, EnvConfig
  └── agent.actor_critic   → ActorCritic, TransformerConfig

online_train.py (main entry)
  └── env.frame_structure  → FrameConfig
  └── env.channel_models   → RayleighMultipathChannel
  └── env.comm_env         → CommunicationEnv, EnvConfig
  └── agent.actor_critic   → ActorCritic, TransformerConfig
  └── agent.ppo            → PPOTrainer
  └── utils.baselines      → mmse_equalize, dfe_equalize
  └── (optional) env.ldpc_coding → LDPC
```

### 数据流路径

```
离线预训练:
  固定信道 → reset_taps() → 帧数据生成 → 信道卷积
    → get_all_states() → Transformer(双向/无mask)
    → BCE(已知位) → AdamW更新 → 保存权重

在线PPO训练:
  每帧:
    固定信道 → reset_taps() → 帧数据生成 → 信道卷积
      → get_all_states() → batch_act(因果mask)
      → 采样action → 计算reward → 存储traj
      → GAE优势计算 → PPO-Clip更新
  每隔log_interval:
    输出 known_acc, BER

评估(可选):
  同一组固定信道抽头 → 不同SNR
    → PPO vs MMSE vs DFE
    → 绘制semilogy对比曲线 → baseline_comparison.png
```

## 常用命令

```bash
conda activate RL4EQ

# PPO 训练（推荐）
python online_train.py --algo ppo --snr 10 --num_frames 500

# A2C 训练（对比）
python online_train.py --algo a2c --snr 10 --num_frames 200

# 带 LDPC 信道编码
python online_train.py --algo ppo --ldpc --snr 10 --num_frames 200

# 离线预训练 + 在线微调
python pretrain.py --snr 10 --num_steps 600                # 预训练600步
python online_train.py --finetune pretrained/actor_critic_final.pt    # 微调

# 训练 + 基线评估 + 绘图
python online_train.py --algo ppo --eval --snr 10 --num_frames 500

# 模块自测
python env/frame_structure.py
python env/channel_models.py
python env/comm_env.py
python agent/actor_critic.py
python -c "import sys; sys.path.insert(0,'.'); from agent.ppo import test_gae, test_ppo; test_gae(); test_ppo()"
python env/ldpc_coding.py
```

## 架构说明

### 目录结构

```
RL4EQ/
├── CLAUDE.md                  # 本文件
├── README.md                  # 项目README
├── reference/                 # 参考文献和设计文档
│   ├── rl-design.md           # RL 信道均衡设计方案
│   ├── Reducing Pilots*.pdf   # IEEE TCOM 论文
│   └── P-FTNet*.pdf           # P-FTNet 论文
│
├── env/                       # 通信环境
│   ├── channel_models.py      # 16抽头瑞利多径信道
│   ├── frame_structure.py     # 帧结构 |训练(128)|导频(64)|数据(128)|导频(64)|数据(128)|
│   ├── comm_env.py            # Gym风格环境：状态(45维)=接收窗(42)+位置编码(3)
│   └── ldpc_coding.py         # (3,6)-规则 LDPC (256,128)，对数域 BP 解码
│
├── agent/                     # RL 智能体
│   ├── actor_critic.py        # Transformer编码器(2层,4头,64维) + Actor头 + Critic头
│   └── ppo.py                 # PPO+GAE 训练器
│
├── utils/
│   ├── baselines.py           # MMSE/DFE 基线均衡器
│   └── __init__.py
│
├── pretrain.py                # 离线预训练（固定信道监督学习）
├── online_train.py            # 统一训练入口（PPO/A2C/LDPC/微调）
│
├── pretrained/                # 预训练模型输出目录
│   ├── actor_critic_best.pt   # 最佳验证模型
│   ├── actor_critic_final.pt  # 最终模型
│   └── pretrain_curve.png     # 预训练损失曲线
│
└── logs/                      # 训练可视化
    ├── training_curve.png
    ├── pftnet_snr.png
    ├── mmse_comparison.png
    └── baseline_comparison.png
```

### 核心设计

- **PPO+GAE**：GAE(λ=0.95) 优势估计 + PPO-Clip(ε=0.2) 限制步长 + 优势归一化 + KL早停
- **状态表示**：45维 = 接收IQ窗(42) + one-hot位置编码(3)
- **因果掩码 Transformer**：2层,4头,d_model=64,FFN=128；上三角因果掩码保持在线推理时序一致性
- **A2C-KPG（已知位策略梯度）**：仅训练序列和导频位提供奖励(±1)和策略梯度信号，数据位通过共享参数间接学习
- **三阶段训练（A2C专用）**：SUP(1-10帧,纯BCE) → MIX(11-30帧,混合RL+BCE) → RL(31+帧,纯A2C)
- **离线预训练+微调**：固定信道双向注意力监督预训练 → 在线 PPO 微调 100 帧达 BER < 0.01
- **推理**：维护最近128个状态的滑动窗口，控制 O(T²) 复杂度
- **LDPC 集成**：数据位承载(256,128)LDPC码字，RL均衡后BP解码进一步降低BER

### 最新训练结果

| 算法 | SNR | 参数量 | 帧数 | Best BER | < 0.01 |
|------|-----|--------|------|----------|--------|
| PPO+GAE | 10 dB | 111K | 500 | 0.01367 | ❌ |
| A2C-KPG | 10 dB | 111K | 200 | ~0.049 | ❌ |
| **预训练+PPO微调** | **10 dB** | **111K** | **100** | **0.00977** | **✅** |

### 关键参数

| 配置项 | 值 |
|--------|------|
| 帧长 | 512 符号 |
| 已知位 | 256（50%：训练128 + 导频×2 64） |
| 信道抽头 | 16抽头瑞利，指数衰减PDP |
| 时延扩展 | 10 符号 |
| 状态维度 | 45（窗长 K=10） |
| Transformer | 2层，4头，d_model=64，FFN=128 |
| 参数量 | 111,234 (+ 预训练权重 ~450KB) |
| 优化器 | Adam, lr=3e-4 (在线) / AdamW, lr=3e-4 (预训练) |
| PPO γ/λ/ε | 0.95 / 0.95 / 0.2 |
| PPO k_epochs | 8 |
| 预训练 | 600步, 双向注意力, 余弦退火 |
| 微调帧数 | 100（从零需500） |

### 关键编辑位置

- `online_train.py`：训练循环、阶段调度、评估、--finetune 加载预训练权重、--eval 基线对比
- `pretrain.py`：离线预训练流程、保存权重到 pretrained/
- `agent/actor_critic.py`：Transformer 架构——所有模型共享此网络定义
- `agent/ppo.py`：PPO 训练器——GAE计算、PPO-Clip 更新、KL早停
- `env/comm_env.py`：状态表示、奖励函数、环境接口
- `env/channel_models.py`：信道模型（抽头数、PDP、SNR、时变选项）
- `env/frame_structure.py`：帧结构配置（训练/导频/数据长度）
- `utils/baselines.py`：MMSE/DFE 基线算法
