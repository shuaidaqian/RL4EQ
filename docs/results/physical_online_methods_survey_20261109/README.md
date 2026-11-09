# 非神经、非 RL 在线均衡方法调研（2026-11-09）

状态：**只读调研，不改变现有实现。** 分支 `physical-online-methods-search` 从 `bayes-cir` 的 `2082ed1` 拉出；
本报告只整理文献、开源实现与候选方案，不接在线路径、不改信道/导频/checkpoint。

## 0. 目标与筛选标准

在“离线训练适应信道 + 在线物理状态适配”的框架内，寻找能改进当前 Bayes CIR 的**非神经、非 RL**方法。

三个维度都纳入筛选：

1. **同等成本下 BER 更低**；
2. **同等 BER 下成本更低**；
3. **收敛更快/更稳**（更早进入饱和、跨 seed 方差更小）。

当前系统固定条件：单载波 BPSK、1024 符号帧、116 符号长回波、前缀 Pilot=256（Adapt=224 / Reward=32）、
准静态信道（1200 s 相干）、0/5/10/15 dB。**准静态**意味着多普勒/BEM 类方法的收益预期要打折。

## 1. 当前 Bayes CIR 的定位

实现：[training/bayesian_cir.py](../../../training/bayesian_cir.py)

- 高斯先验 + Pilot 后验 + Data soft 后验；全 117 维协方差，`posterior_update` 每帧两次 `O(N³)` 求逆；
- 过程噪声 `Q=1e-4` 固定；Data 行噪声用启发式 `R_t = σ² + Σ_l |h_l|² Var_{t-l}`；
- 已确认：四 SNR 显著优于 Frozen，5/10/15 dB 随帧数增长、0 dB 早期收敛后饱和。

残余空间（来自 11-06/11-07 诊断）：

- 0 dB soft R_t BER `0.1675` vs oracle `0.1628`（差 `0.0047`）；
- 5/10 dB 换真实 CIR 后 BER 还有约 20% 相对改善空间；
- block_phase 相位变体增量约 `0.0018`，可归因增量不成立；
- 后验约 30 ms/帧，模型前向约 400 ms/帧，在线算法不是成本瓶颈。

**结论：剩余优化空间主要在 CIR 估计方差与噪声/相位建模，不在“换一个更大的优化器”。**

## 2. 候选方法分类

### A. 稀疏贝叶斯学习（SBL）——优先级最高

**动机**：116 符号长回波里非零抽头通常远少于 117；当前高斯先验对全部抽头一视同仁，
SBL 用稀疏先验可以降低有限导频下的估计方差。

| 文献 | 要点 |
|---|---|
| Prasad & Murthy, *Bayesian Learning for Joint Sparse OFDM Channel Estimation and Data Detection*, GLOBECOM 2010 | 稀疏抽头 + SBL 联合估计信道/统计量；EM-SBL 变体 |
| Prasad, Murthy & Rao, *Joint Channel Estimation and Data Detection in MIMO-OFDM: A SBL Approach*, IEEE TSP 2015 | 组稀疏/簇稀疏 + MMV + 时间相关 |
| Barbu et al., *OFDM Receiver for Fast Time-Varying Channels Using Block-Sparse Bayesian Learning*, IEEE TVT 2016, DOI `10.1109/TVT.2016.2554611` | BEM + block-sparse + 变分贝叶斯联合估计 |
| Sionna `SBLChannelEstimator`（Wei/Zhang/Liu ICC 2017） | 延迟域稀疏、共享延迟支持方差、EM 更新；Python 参考实现 |

**开源**：
- Sionna `sionna.bench.mimo_ofdm_chest.papers.wei2017.SBLChannelEstimator`（Python，MIMO-OFDM，需改造成单载波/延迟域）；
- `GAMPTeam/vampyre`（Python，GAMP/AMP，可承载 SBL 消息传递）；
- `ITU-AI-ML-in-5G-Challenge/ML5G-PHY-ChannelEstimation-ICARUS`（MATLAB，块稀疏 SBL）。

**收益/成本**：NMSE 可能下降、导频需求可能降低；代价是每帧 EM 迭代（10–20 次），
117 维下若每次 `O(N³)` 会明显超过当前 30 ms；可用对角/低秩近似压到可接受范围。
**与现实现兼容性高**：只替换先验/后验更新，不动观测模型与在线边界。

### B. 软输入 Kalman / 自适应 AR 信道跟踪——与当前最接近

**动机**：当前 Bayes CIR 已经是“AR 过程 + 过程噪声 + soft 后验”的递归贝叶斯；
可借鉴的是**自适应估计 `F/Q`**、软输入更新和长延迟处理。

| 文献 | 要点 |
|---|---|
| Hong Liu, PhD dissertation, *Frequency-domain equalization of single carrier transmissions over doubly selective channels*, Ohio State 2007（导师 Schniter） | 单载波 + 软输入 Kalman + 长延迟扩展 + IFDE；与本系统最接近 |
| Schniter & Liu, *Iterative frequency-domain equalization for single-carrier systems in doubly-dispersive channels*, Asilomar 2004 | 上述工作的会议版 |
| Pedrosa, Dinis et al., *Channel Tracking and Equalization for UAV Communications*, Springer 2022 | SC-FDE + IB-DFE + EKF |
| Muralidhar & Li, *A low-complexity Kalman approach for channel estimation in doubly-selective OFDM*, IEEE SPL 2009 | 低复杂度 Kalman |
| Hijazi & Ros, *Joint data QR-detection and Kalman estimation for OFDM time-varying Rayleigh channel complex gains*, IEEE TCOM 2010 | QR 检测 + Kalman |

**开源**：
- `rlabbe/filterpy`、`pykalman`（Python 通用 KF/EKF/EM；需自建 AR 信道状态模型）；
- 没有找到“无线信道跟踪 + Kalman”的成熟 Python 仓库，需在 filterpy 上自建。

**收益/成本**：估计 `Q/F` 的成本低（复用现有 `O(N³)` 求逆），收益中低；
在准静态信道下 `F≈I`，主要收益来自 `Q` 的逐帧自适应而非多普勒跟踪。
**兼容性最高**：可直接在现有 `add_process_noise`/`posterior_update` 上替换 `Q`。

### C. 变分贝叶斯 / 期望传播 / 消息传递——联合 CIR+相位+噪声

**动机**：当前 `R_t` 是启发式、相位是分块 DD；VB/EP 可以联合估计 CIR、噪声方差和相位，
并避免 DD 误差传播。

| 文献 | 要点 |
|---|---|
| Nissilä & Pasupathy, *Adaptive iterative detectors for phase-uncertain channels via variational bounding*, IEEE TCOM 2009 | VB 联合估计/检测，含 Wiener 相位模型 |
| Şahin, Cipriano, Poulliat, Boucheret, *Joint Frequency Domain Channel Estimation and Equalization Based on Expectation Propagation for Single Carrier Transmissions*, ICASSP 2020, DOI `10.1109/ICASSP40776.2020.9054739` | 单载波 EP 联合信道估计/均衡，FFT 处理 |
| Wang et al., *A BP-MF-EP based iterative receiver for joint phase noise estimation, equalization, and decoding*, IEEE SPL 2016 | BP/mean-field/EP 混合 |
| arXiv `2305.09226`, *Joint Channel Estimation and Turbo Equalization of Single-Carrier Systems over Time-Varying Channels* | 单载波 EP 消息传递更新方程 |

**开源**：没有找到成熟的单载波 EP/VB 接收机 Python 仓库；需按论文自实现。

**收益/成本**：可能改善相位/噪声建模与低 SNR 收敛；代价是迭代消息传递，成本高（可能数倍于当前 30 ms）。
**兼容性中**：需要新接收机结构，不宜直接塞进现有 `posterior_update`。

### D. EM / 双迭代接收机——软信息精化

| 文献 | 要点 |
|---|---|
| Pham, Liang, Nallanathan, *Doubly Iterative Receiver with EM-Based Channel Estimation*, IEEE TWC 2009 | BI-GDFE + EM，软信息反馈，近 ML、低复杂度 |
| *Symbol MAP detection aided with first channel estimate for TDMA systems*, IEEE 2009 | EM 联合均衡/信道估计，每迭代线性复杂度 |
| Nissilä PhD thesis, *Iterative receivers for digital communications via variational inference and estimation*, Oulu 2008 | BEM/软判决 DD 信道估计的理论参考 |

**收益/成本**：EM 的 E 步与当前 Data soft 后验同构；增量主要在联合噪声方差/软信息调度，成本中低。
**兼容性高**。

### E. 自适应滤波（RLS/LMS/DFE）——低成本基线

| 文献/实现 | 要点 |
|---|---|
| Haykin, *Adaptive Filter Theory* | RLS/LMS 基础 |
| `gokhanntosun/channel-estimation`（MATLAB） | RLS/LMS/ML + MMSE 均衡，Monte Carlo BER |
| `karel/gr-adapt`（C++/Python） | LMS/NLMS/RLS/QRD-RLS + CFO 补偿，GNU Radio 模块 |
| `ChrisZonghaoLi/mmse_dfe`（Python） | FFE+DFE 联合优化 |
| `nuniz/ParaFilt`（PyTorch） | 批量 LMS/NLMS/RLS |
| `opticommpy` `adaptiveFiltering`（Python） | `rlsUpBlock` / `ddrlsUpBlock`（DD-RLS） |
| Berkeley EECS-2017-112 | DFE + LMS/RLS 的完整实现参考 |

**收益/成本**：`O(N²)`，成本低于当前 `O(N³)`；但性能通常低于贝叶斯后验，适合作为成本对照/基线。
**兼容性高**，可作为“同等 BER 下成本更低”的候选。

### F. DD 相位跟踪 / 最小二乘相位扰动补偿

| 文献 | 要点 |
|---|---|
| Dagres & Polydoros, *Decision-directed least-squares phase perturbation compensation in OFDM systems*, IEEE TWC 2009 | 时域窗化最小二乘估计相位扰动，抑制残余 ICI |
| *Using phase correction to improve the decision-directed frequency tracking algorithm for OFDM systems*, IEEE 2009 | 相位旋转校正，单符号内补偿 |
| Du Yan et al., *A New Efficient OFDM Frequency Offset Tracking Algorithm Based on Decision-directed*, JEIT 2005 | 低复杂度 DD CFO 跟踪 |

**收益/成本**：低复杂度；但项目 block_phase 10×64 筛查已显示可归因增量不成立，预期收益低。
**兼容性高**，优先级低。

### G. BEM（基扩展模型）——时变信道

| 文献 | 要点 |
|---|---|
| Tang, Leus et al., *Pilot-assisted time-varying channel estimation for OFDM systems*, IEEE TSP 2007 | BEM 基础 |
| *Subspace-Based Estimation of Rapidly Varying Mobile Channels for OFDM*, IEEE TSP 2021, DOI `10.1109/TSP.2020.3045562` | 低复杂度子空间 BEM |

**收益/成本**：准静态信道（1200 s 相干）下多普勒跟踪收益低；除非未来引入快变信道，否则不建议。
**优先级最低**。

## 3. 对比与推荐

相对当前 Bayes CIR 的定性比较：

| 方法 | BER 收益 | 在线成本 | 收敛/稳定性 | 兼容性 | 建议 |
|---|---|---|---|---|---|
| SBL 稀疏先验 | 中 | 中高（需近似） | 中 | 高 | **先做只读诊断** |
| 自适应 Q/F Kalman | 低中 | 低 | 中 | 最高 | 第二优先 |
| EM 噪声方差 | 低中 | 中低 | 中 | 高 | 可与 SBL 合并 |
| VB/EP 联合相位 | 中 | 高 | 中 | 中 | 暂缓 |
| RLS/LMS/DFE | 低 | 低 | 低 | 高 | 成本对照基线 |
| DD 相位 LS | 低 | 低 | 低 | 高 | 低优先 |
| BEM | 低（准静态） | 中 | 低 | 中 | 暂缓 |

**推荐顺序**：

1. **SBL 稀疏 CIR 先验：已诊断，不成立。** 见第 4 节：四 SNR 的 NMSE/BER 都差于
   LMMSE/当前 Bayes，成本最高。关闭该路线。
2. **自适应 `Q`：已诊断，不成立。** 第 5 节确认固定 Q 次优，但第 6 节的 Pilot 交叉验证
   选 Q 没有显著 BER 改善，0 dB NMSE 反而显著恶化；保持固定 `Q=1e-4`。
3. **EM 联合噪声方差**：替换启发式 `R_t`，与 SBL 共用迭代。
4. **RLS/DFE 成本对照**：验证“同等 BER 下成本更低”这一维度。
5. **VB/EP、BEM 暂缓**：成本高或与准静态信道不匹配。

## 4. 只读诊断结果：SBL vs 当前 Bayes（2026-11-09）

**协议**：3 seed × 16 帧 × 4 SNR（seed 110001–110003）；四臂共享同一 acquisition 先验、
同一 dd_soft tail 与同一波形；Data 标签只事后算 BER，真实 CIR 只事后算 NMSE。

- `ls`：Pilot-only 最小二乘；
- `lmmse`：Pilot-only + acquisition 先验；
- `bayes`：Pilot + Data soft 后验（当前实现，Q=1e-4）；
- `sbl`：Pilot + Data soft + 稀疏残差 EM（10 次迭代，先验均值=acquisition）。

真实 CIR 稀疏度（1 seed × 4 帧）：top-10 抽头占 **79%** 能量，90% 能量约需 **24/117** 抽头。

| SNR | 臂 | BER（均值 ± 95% CI） | CIR NMSE（均值 ± 95% CI） | 更新耗时 |
|---:|---|---:|---:|---:|
| 0 dB | LS | 0.4361 ± 0.0781 | 0.8111 ± 0.1159 | 6.8 ms |
| 0 dB | LMMSE | 0.2010 ± 0.0734 | 0.1258 ± 0.0143 | 9.8 ms |
| 0 dB | Bayes | 0.2081 ± 0.0998 | **0.0688 ± 0.0148** | 35.9 ms |
| 0 dB | SBL | 0.2497 ± 0.0990 | 0.2004 ± 0.0789 | 48.1 ms |
| 5 dB | LMMSE | 0.0510 ± 0.0109 | 0.0418 ± 0.0016 | 9.6 ms |
| 5 dB | Bayes | **0.0467 ± 0.0065** | **0.0233 ± 0.0046** | 36.5 ms |
| 5 dB | SBL | 0.0559 ± 0.0087 | 0.0589 ± 0.0120 | 48.7 ms |
| 10 dB | LMMSE | 0.00559 ± 0.00180 | 0.0148 ± 0.0007 | 9.9 ms |
| 10 dB | Bayes | **0.00491 ± 0.00187** | **0.0077 ± 0.0016** | 36.8 ms |
| 10 dB | SBL | 0.00616 ± 0.00210 | 0.0158 ± 0.0019 | 48.7 ms |
| 15 dB | LMMSE | 0.000841 ± 0.000540 | 0.0060 ± 0.0006 | 9.9 ms |
| 15 dB | Bayes | 0.000922 ± 0.000552 | **0.0041 ± 0.0007** | 37.4 ms |
| 15 dB | SBL | 0.000922 ± 0.000577 | 0.0061 ± 0.0010 | 49.7 ms |

**结论：**

- **SBL 在四个 SNR 的 NMSE 都差于 LMMSE 和 Bayes，BER 也差，成本最高。**
  稀疏先验没有带来收益。可能原因：acquisition 先验协方差已经捕捉了抽头相关性，
  SBL 的对角稀疏先验丢失了这部分信息；在有限 Pilot/Data 观测下，EM 的 `gamma`
  更新更容易过拟合噪声。
- **当前 Bayes 的 NMSE 在四个 SNR 都是最优**；BER 在 5/10 dB 点估计最优，
  15 dB 与 LMMSE 持平，0 dB 与 LMMSE 的 CI 重叠（BER 差异不显著）。
- 0 dB 出现“NMSE 显著更低但 BER 无显著改善”，提示 0 dB 的 BER 受错误传播/噪声主导，
  CIR 估计改善没有线性转化为 BER；这与 11-07 的 0 dB 变体筛查一致。
- **SBL 路线关闭**：不值得在在线路径上实现。

## 5. 过程噪声 Q 网格诊断（2026-11-09）

**协议**：3 seed × 16 帧 × 4 SNR（seed 110001–110003）；只改变跨帧过程噪声 Q，
其余与当前 Bayes 实现一致；Data 标签只事后 BER，真实 CIR 只事后 NMSE。

| SNR | Q=0 | 1e-6 | 1e-5 | **1e-4（当前）** | 1e-3 | 1e-2 | 1e-1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 dB NMSE | 0.0721 | 0.0720 | 0.0712 | **0.0688** | 0.0944 | 0.2509 | 0.4661 |
| 0 dB BER | 0.2029 | 0.2030 | 0.2000 | 0.2081 | **0.1911** | 0.2496 | 0.3161 |
| 5 dB NMSE | **0.0196** | **0.0196** | 0.0197 | 0.0233 | 0.0380 | 0.0572 | 0.0683 |
| 5 dB BER | **0.04603** | **0.04603** | 0.04609 | 0.04666 | 0.04932 | 0.05214 | 0.05545 |
| 10 dB NMSE | 0.0046 | **0.0045** | 0.0048 | 0.0077 | 0.0134 | 0.0162 | 0.0171 |
| 10 dB BER | 0.00461 | **0.00453** | 0.00456 | 0.00491 | 0.00543 | 0.00583 | 0.00591 |
| 15 dB NMSE | 0.0022 | **0.0020** | 0.0025 | 0.0041 | 0.0057 | 0.0061 | 0.0063 |
| 15 dB BER | **0.000814** | **0.000814** | **0.000814** | 0.000922 | 0.000949 | 0.000895 | 0.000895 |

**结论：**

- **当前固定 `Q=1e-4` 只在 0 dB 的 NMSE 上最优；5/10/15 dB 的最优 Q 在 0–1e-6。**
  相对最优值，当前配置的 NMSE 分别差约 19%/71%/105%，BER 差约 1.3%/8.4%/13.3%。
- **最优 Q 随 SNR 单调下降**：低 SNR 需要过程噪声帮助跟踪，高 SNR 下过程噪声反而破坏
  已经收敛的后验。固定 Q 不是最优。
- **0 dB 的 NMSE 与 BER 不一致**：NMSE 最优在 `1e-4`，BER 最优在 `1e-3`（`0.1911` vs `0.2081`）。
  这说明 0 dB 的 BER 受错误传播/符号错误主导，CIR 更准不一定 BER 更低；与 11-07 的
  0 dB 判据一致，0 dB 应单独判断。
- 这是一个**开发筛查**（3×16，点估计未计算 CI）；正式采用前需要扩样本并做配对区间。

**下一步（低成本）**：用合法观测（Pilot 残差/innovation、从 Pilot 估计的噪声方差）
做自适应 Q，而不是按真实 SNR 调度。若自适应 Q 不能稳定复现上述增益，则保持固定 Q。

## 6. 自适应 Q 只读实现（2026-11-09）

**协议**：3 seed × 16 帧 × 4 SNR；每帧把有效 Adapt Pilot 行分成前一半（更新）与后一半（评分），
在候选 Q ∈ {0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2} 中选后半 Pilot 预测误差最小者；
该 Q 只用于下一帧先验协方差。全程只用已知 Pilot 与接收波形，不读 Data 标签或真实 SNR。

三臂：`fixed_1e-4`（当前）、`fixed_0`（参考）、`adaptive`（交叉验证选 Q）。

| SNR | 臂 | BER（均值 ± 95% CI） | CIR NMSE（均值 ± 95% CI） | 平均 Q |
|---:|---|---:|---:|---:|
| 0 dB | fixed_1e-4 | 0.2001 ± 0.0854 | **0.0673 ± 0.0150** | 1e-4 |
| 0 dB | fixed_0 | 0.2025 ± 0.0849 | 0.0721 ± 0.0134 | 0 |
| 0 dB | adaptive | **0.1854 ± 0.0676** | 0.1207 ± 0.0447 | 1.4e-3 |
| 5 dB | fixed_1e-4 | 0.04674 ± 0.00650 | 0.0235 ± 0.0046 | 1e-4 |
| 5 dB | fixed_0 | **0.04603 ± 0.00678** | **0.0196 ± 0.0055** | 0 |
| 5 dB | adaptive | 0.04842 ± 0.00672 | 0.0293 ± 0.0048 | 4.4e-4 |
| 10 dB | fixed_1e-4 | 0.004937 ± 0.001887 | 0.0078 ± 0.0016 | 1e-4 |
| 10 dB | fixed_0 | **0.004612 ± 0.001773** | **0.0046 ± 0.0023** | 0 |
| 10 dB | adaptive | 0.004666 ± 0.001973 | 0.0072 ± 0.0022 | 1e-4 |
| 15 dB | fixed_1e-4 | 0.000922 ± 0.000552 | 0.0042 ± 0.0007 | 1e-4 |
| 15 dB | fixed_0 | **0.000814 ± 0.000544** | **0.0022 ± 0.0009** | 0 |
| 15 dB | adaptive | 0.000895 ± 0.000532 | 0.0032 ± 0.0010 | 3.6e-5 |

**结论：自适应 Q 不成立。**

- **所有 SNR 的 BER CI 都跨零/重叠**：adaptive 相对 fixed_1e-4 没有显著 BER 改善；
  0 dB 的点估计更低（`0.1854` vs `0.2001`），但 CI 重叠。
- **0 dB 的 NMSE 显著恶化**（`0.1207 ± 0.0447` vs `0.0673 ± 0.0150`，CI 不重叠）；
  5 dB NMSE 也轻度恶化（`0.0293` vs `0.0235`）。
- **交叉验证选出的 Q 偏大**：0 dB 平均 `1.4e-3`、5 dB `4.4e-4`，而 Q 网格的 NMSE 最优是
  `1e-4`（0 dB）和 0（5 dB）。Pilot 后半的预测误差不等于 CIR NMSE，选 Q 目标错位。
- 10/15 dB 的 adaptive 介于 fixed_1e-4 与 fixed_0 之间，没有独立增益。

**建议：保持固定 `Q=1e-4`。** 若未来还要做按 SNR 调度，需要更可靠的合法 Q 估计
（innovation 全协方差、噪声方差估计），但当前证据不支持继续投入。

## 7. 最小只读验证实验（不改现有实现）

- 数据：`make_episode` 3 seed × 16 帧 × 4 SNR，开发 seed 110001–110003（与既有诊断隔离）。
- 离线比较臂：
  1. LS（无先验）；
  2. LMMSE（当前 acquisition 先验）；
  3. 当前 Bayes CIR（Pilot + Data soft 后验）；
  4. SBL（稀疏先验，EM 10–20 次，延迟域）；
  5. RLS（遗忘因子扫描）。
- 指标：CIR NMSE、Data BER（真实 CIR/标签只事后评分）、每帧耗时。
- 判据：SBL 的 NMSE 在 5/10 dB 显著低于当前 Bayes，且 BER 不退化；否则记录“稀疏先验不是瓶颈”。
- 该实验只读，不接在线路径，不改 `bayes-cir`。

## 8. 开源实现索引

| 仓库 | 语言 | 用途 | 适配性 |
|---|---|---|---|
| `nvlabs/sionna`（SBLChannelEstimator） | Python | SBL 信道估计（MIMO-OFDM） | 需改造成单载波延迟域 |
| `GAMPTeam/vampyre` | Python | GAMP/AMP | 可承载 SBL 消息传递 |
| `rlabbe/filterpy` | Python | KF/EKF/UKF | 需自建 AR 信道模型 |
| `pykalman` | Python | KF + EM | 可估计 `F/Q` |
| `gokhanntosun/channel-estimation` | MATLAB | RLS/LMS/ML + MMSE | 算法参考 |
| `karel/gr-adapt` | C++/Python | LMS/NLMS/RLS/QRD-RLS + CFO | 自适应滤波参考 |
| `ChrisZonghaoLi/mmse_dfe` | Python | FFE+DFE | DFE 结构参考 |
| `nuniz/ParaFilt` | PyTorch | LMS/NLMS/RLS | 批量/GPU |
| `opticommpy` | Python | RLS/DD-RLS | 自适应滤波参考 |
| `ITU-AI-ML-in-5G-Challenge/...-ICARUS` | MATLAB | 块稀疏 SBL | 算法参考 |

## 9. 证据边界

- 本报告是**只读调研**，未实现、未训练、未改在线路径。
- 文献为**摘要级核验**：通过检索页/开放摘要确认标题、作者、年份与要点，未逐篇精读全文；
  引用前应再核对 DOI/arXiv 与具体实验条件。
- 文献收益均为原文报告值，不保证迁移到本项目的单载波、长回波、准静态设置。
- 成本为基于 117 维矩阵运算的粗估，未实测。
- 准静态信道下 BEM/多普勒类方法收益预期低；SBL 的收益取决于真实 CIR 的稀疏度，需先做只读 NMSE 诊断确认。
- 第 4 节 SBL 诊断是**开发筛查**（seed 110001–110003，3×16），不是确认；CI 宽，0 dB 差异不显著。
- 所有候选方法都必须遵守现有在线只读约束：不看 Data 标签、真实 CIR、未来帧标签。

## 10. 收口结论（2026-11-10）

**第二个研究点保持当前 Bayes CIR 实现，主线回到 `bayes-cir`。**

理由：

- 当前 Bayes CIR 已达成确认目标（四 SNR 显著优于 Frozen，5/10/15 dB 随帧数增长、
  0 dB 早期收敛后饱和）；
- 非神经、非 RL 的替代/增强方法中，SBL 与自适应 Q 均被只读诊断否定；
- DD 相位增量不成立，RLS 只能作成本对照，BEM 与准静态信道不匹配，VB/EP 成本高且未实现；
- **没有找到优于当前 Bayes CIR 的实现**。

本分支保留文献、脚本与三项诊断归档，不合并回 `bayes-cir`；主线不引入任何新在线算法。
