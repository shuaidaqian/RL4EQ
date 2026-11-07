# 公开文献下载目录

本目录记录本轮为 RL4EQ 研究补充的公开资料。arXiv 论文保存为 PDF，并使用
`pdftotext -layout` 生成同名 `.txt`，便于全文检索。文件大小、版本、来源 URL
和 SHA-256 摘要见 `download_manifest.json`；arXiv 元数据见 `arxiv_metadata.json`。

## 与 A/B/C replay 直接相关的文献

| 方向 | 文献 | 本项目可借鉴之处 | 与当前契约的关系 |
|---|---|---|---|
| A：Pilot 条件化状态与漂移门控 | [Obeed & Jian, 2026](2602.20361-obeed2026-learning-during-detection.pdf) | DMRS 同时承担检测和持续学习；支持 Pilot 驱动在线条件化 | 使用 Pilot 更新，但原文允许 SGD；本项目仍保持 Adapt/Reward/Data 隔离 |
| A：漂移检测 | [Uzlaner et al., 2024](2407.09134-uzlaner2024-asynchronous-drift-adaptation.pdf) | 先检测 drift，再决定是否更新模块 | 可用于低 SNR 下的更新门控 |
| B：单步 Bayesian 适配 | [Gusakov et al., 2025](2511.06045-gusakov2025-online-bayesian-receiver.pdf) | 用后验递推替代多轮 SGD，降低在线延迟 | 非 RL，可作为低维递推适配参照 |
| B：连续环境适配 | [Sun et al., 2020](2011.07782-sun2020-continuous-wireless-optimization.pdf) | 处理随 episode 变化的无线环境并减轻遗忘 | 任务是资源优化，不是符号均衡 |
| C：Trellis/BCJR | [Yang et al., 2022](2202.10635-yang2022-online-trellis-learning.pdf) | 只用 Pilot 学习长记忆 Trellis 观测模型 | 直接对应 Level B 长回波，但会改变接收机结构 |
| C：模型驱动展开 | [Zhao et al., 2022](2207.04478-zhao2022-model-driven-unfolding-equalizer.pdf) | 把 MMSE 结构展开成低维可训练均衡器 | 可作为当前 UnfoldedEqualizer 的结构参照 |
| C：软 SIC 展开 | [Baumgartner et al., 2023](2308.12591-baumgartner-sicnn.pdf) | 低参数软干扰抵消和可解释展开 | 可用于长 ISI 的替代结构比较 |
| C：Pilotless/VAE 旁路线 | [Caciularu & Burshtein, 2019](1905.08795-caciularu-vae-equalization.pdf) | 无 Pilot 的无监督均衡 | 改变当前 Pilot-only 契约，仅保留为压力测试参考 |

## 在线接收机、RL 与相关旁路线

| 文献 | 主要内容 | 当前项目判断 |
|---|---|---|
| [Jiang et al., 2018](1812.06638-jiang-online-adaptive-ofdm.pdf) | OTA 在线自适应 OFDM 接收机 | 依赖少量在线参数；原文使用可恢复标签，不能直接作为主线协议 |
| [Fischer et al., 2022](2203.13571-fischer-adaptive-neural-ofdm.pdf) | 基于 FEC 恢复标签的神经 OFDM 在线适配 | 需要信道编码标签，当前主实验暂不采用 |
| [Raviv et al., 2022](2203.14359-raviv-meta-learning-hybrid-receivers.pdf) | 混合模型接收机的预测式元学习 | 可作为后续快速适配基线，当前不新增独立 meta 阶段 |
| [Bereketoglu, 2025](2506.06323-bereketoglu2025-composite-reward-ppo.pdf) | PPO 控制自适应滤波器 | RL 控制滤波更新，但不是 Pilot-only 符号均衡 |
| [Jeon et al., 2019](1903.12546-jeon2019-robust-data-detection-rl.pdf) | 一比特 ADC MIMO 检测中的 RL 似然学习 | 依赖 MIMO/检测设定，不进入 Level B 主矩阵 |
| [Mo et al., 2021](2102.00178-mo2021-deep-rl-mcts-mimo.pdf) | DRL 引导 MCTS 的 MIMO 检测 | 在线延迟和 MIMO 设定不符合当前主线 |
| [Ben-Itzhak & Ayanoglu, 2026](2603.02489-benitzhak2026-ris-equalizer-drl.pdf) | DRL 控制 RIS 空间均衡 | 动作作用于 RIS，不是接收端均衡参数 |
| [Giwa et al., 2025](2507.10619-giwa2025-meta-rl-spectrum.pdf) | 动态无线资源分配的 Meta-RL | 可参考安全快速适配，但任务不同 |
| [Liu et al., 2025](2502.17168-liu2025-neuromorphic-continual-learning.pdf) | 神经形态持续学习 | 提供抗遗忘和低能耗方向，不直接对应当前接收机 |
| [Mehlhose et al., 2019](1911.04291-mehlhose2019-adaptive-receive-filtering.pdf) | SDR 平台自适应接收滤波 | 工程实现参考，使用监督式检测 |
| [Awan et al., 2017](1711.00355-awan2017-online-adaptive-noma.pdf) | 5G-NOMA 在线自适应检测 | 场景和多用户设定不同 |
| [Korpi et al., 2023](2312.05158-korpi2023-pilotless-spatial-multiplexing.pdf) | Pilotless 空间复用 | 改变 Pilot-only 问题定义 |
| [Shao et al., 2020](2012.13523-shao2020-feature-adaptive-tuning.pdf) | 特征辅助自适应调参 | 可参考低维条件化，但任务是大规模设备检测 |

## 创新性边界

当前检索到的工作分别覆盖 Pilot/DMRS 持续学习、单步 Bayesian 适配、漂移检测、
模型驱动展开、Trellis 学习和 RL 滤波控制，但尚未发现一篇公开工作同时具备下列
全部条件：

1. Level B 极端稀疏长记忆、跨帧 ISI 与 residual CFO/慢相位扰动；
2. 离线整帧神经块均衡器预训练；
3. 在线只使用前缀 Adapt Pilot，Reward Pilot 留出验收，Data 标签完全隔离；
4. 非因果整帧接收机和跨帧 soft-tail 递推；
5. 受限低维安全动作调制 PEFT，而不是逐 bit RL 或直接输出高维参数增量；
6. 在 0/5/10/15 dB 上逐配置与传统非神经、非 RL baseline 和 Frozen checkpoint 做配对统计比较。

这可以作为论文的**候选创新组合**，但不能仅凭“没有检索到同时满足”就宣称绝对首创。
论文需要报告检索范围、排除条件和统一 replay 结果；如果 A/B/C 中只有某个组合在当前
契约下成立，创新点应准确表述为“在这些约束组合下的安全 Pilot-only 在线适配框架”。
