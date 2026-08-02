# reference 文献索引

更新日期：2026-07-31

本目录用于保存 RL4EQ 当前论文写作需要引用的外部文献。整理原则：

- 文件名按文章题名命名，去掉 Windows 文件名不允许使用的冒号、斜杠等字符。
- 优先保存开放可获取 PDF：arXiv、开放期刊 PDF、作者/项目主页 PDF。
- 当前环境无法直接下载 PDF 的文献，保留 DOI/网页元数据说明，不伪造 PDF。
- `references.bib` 只作为写作初稿用的 BibTeX 草稿，投稿前还需要按目标期刊格式复核。

## 已保存 PDF

### 神经接收机与在线学习

1. `DeepRx Fully Convolutional Deep Learning Receiver.pdf`
   - 来源：arXiv:2005.01494
   - 用途：深度学习接收机代表工作，说明常规神经接收机背景。

2. `Learning During Detection Continual Learning for Neural OFDM Receivers via DMRS.pdf`
   - 来源：arXiv:2602.20361
   - 用途：部署期间利用 pilot/DMRS 持续学习接收机，作为“非 RL 在线学习”最近邻参考。

3. `Online Learning of Modular Bayesian Deep Receivers Single-Step Adaptation.pdf`
   - 来源：arXiv:2511.06045
   - 用途：在线 Bayesian/deep receiver 单步适配参考。

4. `P-FTNet Pilot-Conditioned and Feature-Augmented Transformer Network for Equalization Under Extreme Delay Spread.pdf`
   - 来源：本地已有 PDF
   - 用途：Pilot 条件、长上下文 Transformer、极端 delay spread 神经均衡背景。

### RL / PPO / 动作控制相关

5. `Proximal Policy Optimization Algorithms.pdf`
   - 来源：arXiv:1707.06347
   - 用途：PPO 原始方法引用。

6. `Composite Reward Design in PPO-Driven Adaptive Filtering.pdf`
   - 来源：arXiv:2506.06323
   - 用途：PPO 控制滤波器更新与复合 reward 设计参考。

7. `Improved Channel Equalization using Deep Reinforcement Learning and Optimization.pdf`
   - 来源：EUDL DOI:10.4108/eai.28-10-2021.171685
   - 用途：深度强化学习用于信道均衡的直接相关工作。

8. `RIS-Enabled Wireless Channel Equalization Adaptive RIS Equalizer and Deep Reinforcement Learning.pdf`
   - 来源：arXiv:2603.02489
   - 用途：RIS 场景下 DRL 控制均衡器，作为“RL equalization 已有人研究”的边界参考；不作为本项目研发路线。

### EME / 极端长回波场景

9. `Modeling and Characterization of Broadband Earth-Moon-Earth Communication Channels.pdf`
   - 来源：Journal of Electronics & Information Technology，DOI:10.11999/JEIT251028
   - 用途：EME 宽带信道建模与表征参考。当前项目是 EME-inspired，不做完整物理 EME 仿真。

10. `Frequency-Dependent Characteristics of the EME Path.pdf`
    - 来源：K1JT / WSJT 项目 PDF
    - 用途：EME 路径频率选择性与传播特征背景。

## 当前仅保存元数据的文献

1. `Cluster Channel Equalization using Adaptive Sensing and Reinforcement Learning for UAV Communication System`
   - DOI：10.7717/peerj-cs.2557
   - 网页：https://peerj.com/articles/cs-2557/
   - 状态：PeerJ PDF 在当前命令行环境返回 403，未保存 PDF；已在 `download_manifest.json` 中记录。
   - 用途：UAV 信道均衡 + 强化学习相关边界参考。

## 与当前研究路线的关系

当前 RL4EQ 主线仍然是：

- Level B 极端稀疏长回波信道；
- 整帧缓冲、非因果块均衡器；
- Pilot 条件监督预训练；
- 在线按帧持续适配；
- Reward Pilot 与 Data 标签严格隔离；
- Continual PPO 作为在线控制器。

这些文献用于支撑“相关工作”和“创新边界”：

- 不能声称神经接收机、RL equalization、PPO 或 EME 信道建模本身首次提出；
- 可以讨论当前项目的组合差异：Level B EME-inspired 极端稀疏长回波基准、整帧非因果块均衡、Adapt/Reward Pilot 隔离、严格 Data-label isolation、以及在线控制器与强 fixed/rule/bandit baseline 的公平配对比较。
