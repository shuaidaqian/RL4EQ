# RL4EQ — 基于强化学习的在线自适应信道均衡

将通信接收机建模为一个**强化学习智能体**，利用帧结构中已知的训练序列和导频作为即时奖励锚点，在时变信道上逐符号交互，实现在线自适应均衡。

## 项目概况

### 研究目标

在**不依赖离线大量训练**、**信道可能时变**的条件下，接收端直接从接收符号流中恢复数据比特，使系统能够边工作边学习，持续适应信道变化。

### 核心创新

- **A2C-KPG（已知位策略梯度）**：仅训练序列和导频位提供奖励信号，数据位通过共享网络参数间接学习
- **Transformer Actor-Critic**：自注意力替代 LSTM，捕获整帧（512符号）长距离 ISI 依赖
- **PPO+GAE 在线学习**：PPO-Clip 约束策略更新步长 + GAE 传播稀疏导频奖励 + 优势归一化

## 算法路线

### 路线 A：A2C-KPG（原始）

三阶段训练演进：
```
SUP [1-10帧] → MIX [11-30帧] → RL [31+帧]
纯BCE监督        RL+BCE混合      纯A2C在线学习
```

`online_train.py --algo a2c`

### 路线 B：PPO+GAE（推荐，更优）

每帧独立 PPO-Clip 更新 + GAE 优势估计：
```
GAE(λ=0.95) 传播稀疏奖励 → 优势归一化 → PPO-Clip(ε=0.2) → KL早停
```

`online_train.py --algo ppo`

关键改进项对比：

| 维度 | A2C-KPG | PPO+GAE |
|------|---------|---------|
| 优势估计 | MC回报 G_t 或单步TD | GAE(λ=0.95) 多步加权 |
| 策略更新 | 无约束，可能一步破坏 | clip(ρ, 1-ε, 1+ε) |
| 数据效率 | 每帧1 epoch | 每帧K=4~8 epoch |
| KL控制 | 无 | KL早停 |
| 梯度平衡 | 无 | 优势归一化 |

### 路线 C：LDPC 编码增强

数据段承载 (256,128) LDPC 码字 → RL 均衡后对数域 BP 解码 → 进一步降低 BER。

`python online_train_ldpc.py`

## 最新训练结果

| 算法 | SNR | 参数量 | 帧数 | Best BER | BER < 0.01 |
|------|-----|--------|------|----------|------------|
| **PPO+GAE** | **10 dB** | **111K** | **500** | **0.00586** | ✅ |
| **PPO+GAE** | **5 dB** | **486K** | **500** | **0.00781** | ✅ |
| PPO+GAE | 5 dB | 111K | 1000 | 0.01367 | ❌ |
| A2C-KPG | 10 dB | 111K | 200 | ~0.033 | — |
| A2C-LSTM（历史对比） | 10 dB | 93K | 200 | ~0.222 | — |

## 项目结构

```
RL4EQ/
├── env/                          # 通信环境
│   ├── channel_models.py         # 16抽头瑞利多径信道（频率选择性，时延扩展=10）
│   ├── frame_structure.py        # 帧结构：|训练(128)|导频(64)|数据(128)|导频(64)|数据(128)|
│   ├── comm_env.py               # Gym风格环境：45维状态（接收窗42+位置编码3）
│   └── ldpc_coding.py            # (3,6)-规则 LDPC (256,128)，对数域 BP 解码
│
├── agent/                        # RL 智能体
│   ├── actor_critic.py           # Transformer编码器 (2层,4头,64维) + Actor头 + Critic头
│   ├── a2c.py                    # A2C-KPG 算法（独立实现）
│   └── ppo.py                    # PPO+GAE 训练器（GAE计算、PPO-Clip、KL早停、优势归一化）
│
├── online_train.py               # 主训练入口（PPO/A2C 双模式，MMSE对比，SNR扫描）
├── online_train_ldpc.py          # LDPC 编码版训练入口（含 BP 解码后 BER 评估）
├── BER_IMPROVEMENT_ANALYSIS.py   # BER 改进方案分析文档
│
├── utils/
│   ├── metrics.py                # SER/BER 计算
│   ├── complex_ops.py            # I/Q ↔ 复数转换
│   └── matplotlib_zh.py          # 中文绘图配置
│
├── reference/                    # 参考文献（24篇 RL for 通信/在线学习论文）
│   └── rl-design.md              # On-Policy RL 信道均衡设计方案
│
├── configs/
│   └── default.yaml              # 配置模板
│
└── logs/                         # 训练可视化
```

## 核心设计

### 状态空间（45维）

```
s_t = [ y_{t-K:t+K} I/Q (2×(2K+1)=42) | m_t one-hot (3=训练/导频/数据) ]
```

K=10 的接收窗覆盖 ISI 扩展范围 + one-hot 帧内位置指示。

### 网络架构

```
输入 s_t (45维)
  → Linear(45→64) + LayerNorm
  → + 学习式位置编码 (max 512)
  → TransformerEncoder × 2层（因果掩码，4头，d_model=64，FFN=128）
  → Actor头：Linear(64→ReLU→1) → Sigmoid → p(a=1)
  → Critic头：Linear(64→ReLU→1) → V(s)
```

参数量：111,234（小网络）/ 485,762（大网络，d_model=128, 3层）

### 奖励函数

```
r_t = +1 正确判决（训练/导频段）
r_t = -1 错误判决（训练/导频段）
r_t =  0 数据段（无标签，通过价值自举传播信号）
```

### 知训序列与导频角色

| 组件 | 作用 | 说明 |
|------|------|------|
| 训练序列（帧头128） | 冷启动 | 首帧行为克隆预训练让策略初步生效 |
| 导频（2×64） | 奖励锚点 | 价值函数将正确判决信号沿时间轴传播到数据段 |
| 数据段（2×128） | 盲恢复 | 依靠策略泛化和价值自举进行无监督学习 |

## 使用命令

```bash
conda activate RL4EQ
cd D:\Research\RL4EQ

# PPO 训练（推荐）
python online_train.py --algo ppo --snr 10 --num_frames 500

# A2C 对比
python online_train.py --algo a2c --snr 10 --num_frames 200

# LDPC 编码训练
python online_train_ldpc.py --snr 10 --num_frames 200

# 模块自测
python env/channel_models.py
python env/frame_structure.py
python env/comm_env.py
python agent/actor_critic.py

# PPO 模块测试（需从项目根运行）
python -c "import sys; sys.path.insert(0,'.'); from agent.ppo import test_gae, test_ppo; test_gae(); test_ppo()"
```

## 关键参数

| 配置项 | 值 |
|--------|------|
| 帧长 | 512 符号 |
| 已知位 | 256（50%：训练128 + 导频×2 64） |
| 信道抽头 | 16抽头瑞利，指数衰减PDP |
| 时延扩展 | 10 符号 |
| 状态维度 | 45（窗长 K=10） |
| Transformer | 2层，4头，d_model=64，FFN=128 |
| 参数量 | 111,234 / 485,762 |
| 优化器 | Adam, lr=3e-4 |
| PPO γ/λ/ε | 0.95 / 0.95 / 0.2 |
| 每帧 PPO epoch | 4~8 |
| MMSE 基线 | ~50% BER（16抽头信道） |

## 参考文献

主要技术参考：P-FTNet 论文（`reference/` 目录内 PDF）以及 24 篇 RL for Communication 相关文献（详见 `reference/README.md`）。

更详细的设计文档请参见 `reference/rl-design.md`。
