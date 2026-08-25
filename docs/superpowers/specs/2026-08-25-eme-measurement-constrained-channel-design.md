# 权威文献约束的 EME 稀疏长回波信道设计

## 1. 研究目标

第二个研究点面向宽带 EME 接收端在线均衡。信道设计不继承第一篇论文中的具体功率时延分布、散射分段公式或仿真参数，只继承“月面分布式反射会形成跨多符号长回波”这一研究方向。

本阶段先建立一个可追溯、可统计验证的 Level B 主信道。它必须同时满足：

- 绝对回波支撑由公开月球雷达文献约束；
- 离散等效信道表现为少数强径与弱弥散长尾并存；
- 抽头支撑在在线 episode 内固定，复增益只缓慢变化；
- 整个发送符号流连续卷积，帧边界不清空历史，因此自然产生跨帧 ISI；
- 同步后仅保留小 residual CFO 与慢相位扰动；
- Level C 只增加弥散能量、增益漂移或残余同步压力，不混入主论文平均。

## 2. 权威证据边界

### 2.1 可直接作为 EME 物理约束的证据

1. J. V. Evans, “Radar Studies of the Moon,” *Journal of Research of the National Bureau of Standards, Section D*, 1965, DOI: `10.6028/jres.069d.195`。
   - 月球完整雷达深度约为 11.6 ms。
   - 短脉冲实测给出回波功率随时延的衰减曲线。
   - 足够灵敏时，回波可以从前沿一直检测到月缘。

2. D. F. Winter, “A Theory of Radar Reflections From a Rough Moon,” *Journal of Research of the National Bureau of Standards, Section D*, 1962, DOI: `10.6028/jres.066d.025`。
   - 用准镜面和弥散两类独立散射成分解释强前沿与慢衰减长尾。
   - 给出短脉冲回波时间分布并与 Pettengill 的 440 MHz 实测比较。

3. G. H. Pettengill and J. C. Henry, “Enhancement of Radar Reflectivity Associated with the Lunar Crater Tycho,” *Journal of Geophysical Research*, 1962, DOI: `10.1029/JZ067i012p04881`。
   - 在延迟域存在显著高于局部均值的离散强散射中心，为稀疏长回波提供直接观测依据。

4. S. J. Fricker et al., “Computation and Measurement of the Fading Rate of Moon-Reflected UHF Signals,” *Journal of Research of the National Bureau of Standards, Section D*, 1960, DOI: `10.6028/jres.064d.055`。
   - 412 MHz 实测 fading rate 约为 0.005 至 3--4 fades/s。
   - 主场景选取其中的慢变化区间；快速区间只用于 Level C 压力测试。

### 2.2 只能作为统计建模方法的类 EME 证据

- Stojanovic and Preisig, IEEE Communications Magazine, 2009, DOI: `10.1109/MCOM.2009.4752682`：长时延通信信道的传播与统计建模。
- Qarabaqi and Stojanovic, IEEE Journal of Oceanic Engineering, 2013, DOI: `10.1109/JOE.2013.2278787`：连续时变长多径信道的高效随机生成。
- Li and Preisig, IEEE Journal of Oceanic Engineering, 2007, DOI: `10.1109/JOE.2007.906409`：固定或缓变稀疏支撑与时变复增益的状态表达。
- Michelusi et al., IEEE Transactions on Signal Processing, 2012, DOI: `10.1109/TSP.2012.2205681`：离散强径与弥散成分并存的稀疏/弥散模型。

这些文献不得用于宣称具体 EME 参数。它们只决定随机过程和数据结构的表达方式。

## 3. 信道数学模型

### 3.1 连续符号流

发送端首先生成连续复符号流 `x[n]`，然后通过信道，最后再按 frame 切分接收样本。不得逐帧独立卷积或在帧边界补零。

接收模型为：

\[
y[n] = e^{j(2\pi\epsilon n + \phi[n])}
       \sum_{\ell=0}^{D} h_{f(n)}[\ell]x[n-\ell] + w[n].
\]

其中 `f(n)` 表示样本 `n` 所属帧，`D` 是最大离散时延，`epsilon` 是同步后的 residual CFO，`phi[n]` 是跨帧连续的慢相位状态。

### 3.2 稀疏强径与弱弥散尾

每帧使用的离散 CIR 为：

\[
h_f[\ell] = \sum_{q=1}^{K} a_{q,f}\delta[\ell-d_q] + g_f[\ell].
\]

- `d_q`：episode 级固定的强径支撑，必须包含前沿强径，并允许在较大延迟处出现异常强散射径；
- `a_{q,f}`：强径复增益，跨帧高相关缓慢变化；
- `g_f[ell]`：低功率弥散尾，Level B 中能量受限，Level C 中增强；
- 强径和弥散尾总功率归一化为 1，SNR 使用归一化前的统一定义。

### 3.3 绝对时间与离散时延

最大物理回波支撑固定为：

\[
T_{\max}=2R_{\text{moon}}/c \approx 11.6\ \text{ms}.
\]

离散最大时延由采样率唯一换算：

\[
D=\lceil T_{\max}F_s\rceil.
\]

配置必须显式记录 `sample_rate_hz`、`symbol_rate_hz`、`samples_per_symbol` 和 `max_delay_seconds`。主实验不再把无量纲 `max_delay` 当作独立物理参数。

### 3.4 慢复增益状态

Level B 的强径支撑不变化。复增益采用以实际帧时长为输入的高相关状态：

\[
a_{q,f+1}=\rho_f a_{q,f}+\sqrt{1-\rho_f^2}\,u_{q,f},
\qquad
\rho_f=e^{-T_{\text{frame}}/T_c}.
\]

`T_c` 是相干时间。Level B 使用慢变化区间，并通过 `rho_frame` 和对应秒数共同记录，避免只给出无法解释的无量纲 `rho`。

### 3.5 同步残差

- residual CFO 在 episode 开始时采样一次，跨帧连续；
- 慢相位为小步长随机游走，状态在帧边界连续；
- 所有方法共享相同的 profile-level 上界和 Pilot；
- 接收端不得读取真实 CFO、真实相位状态或真实 tap innovation。

## 4. Level A/B/C 定义

### Level A：静态可通性检查

- EME 时延支撑；
- 固定稀疏强径；
- 极弱或关闭弥散尾；
- 无 residual CFO 与相位漂移；
- 只用于验证连续卷积、离线网络容量和传统算法实现。

### Level B：主论文场景

- 最大物理支撑 11.6 ms；
- 少数强径加受限弱弥散尾；
- 支撑固定，复增益慢变；
- 小 residual CFO 与慢相位；
- SNR 固定为 0/5/10/15 dB；
- Pilot 只在前缀；
- 所有主结果逐配置报告。

### Level C：压力测试

- 更高弥散能量或更多有效抽头；
- 更短相干时间；
- 更大的同步残差；
- 可加入轻度差分多普勒；
- 不混入 Level B 主平均和主成功门槛。

## 5. 信道校准与冻结门槛

进入任何神经网络训练前，必须通过以下独立验证：

1. **物理时延**：配置换算出的 `max_delay_seconds` 与 11.6 ms 一致，误差不超过一个采样周期。
2. **跨帧连续性**：构造上一帧末尾单脉冲，验证其回波按 CIR 精确进入下一帧；逐帧独立卷积实现必须失败。
3. **支撑稳定性**：Level B 在完整 episode 中强径 delay support 完全不变。
4. **慢变化**：经验 frame-lag 自相关与配置 `rho_frame` 的误差在预定置信区间内。
5. **稀疏度**：报告强径数、有效抽头数、90% 能量支撑数和弥散能量比，不只报告总 tap 长度。
6. **包络约束**：Monte Carlo 平均 PDP 与数字化 EME 回波曲线在预设时延网格上比较；主能量前置且保留可检测长尾。
7. **同步连续性**：CFO 相位和慢相位在帧边界不跳变。
8. **信息边界**：receiver view 不暴露真实 CIR、真实 CFO、真实相位或 Data 标签。

校准阶段只运行信道统计和传统非神经 baseline。候选 profile 冻结后，离线 NN 与在线 RL 使用完全相同的 profile，不再按结果修改信道。

## 6. 算法阶段顺序

1. 冻结 Level A/B/C 信道 profile；
2. 在公开单载波长记忆信道基准上复现已发表的离线神经均衡结果；
3. 在冻结的 EME Level B 上证明 Offline NN 超过最佳传统非神经 baseline；
4. 比较冻结 NN、固定学习率微调、规则调制和 RL 调制；
5. 只有 Online RL 在 Level B 每个 SNR 配置上超过最佳传统方法，且优势随帧数增加，才判定第一目标完成。

## 7. 公平性与禁止项

- 传统方法允许跨帧保持状态，允许稀疏自适应、变遗忘因子 RLS 和基于 Pilot 的同步补偿；
- Proposed 是唯一使用神经网络和 RL 的方法；
- Perfect-CSI 只用于诊断；
- Reward Pilot 只用于在线 reward 和留出评估；
- Data 标签只用于离线监督和仿真评估；
- 不使用逐 bit RL、高维参数动作、Data Oracle、多载波、MIMO、RIS、非线性或信道编码扩展。

## 8. 当前明确不采用的设计

- 每帧独立重抽全部 CIR；
- 把路径相关大多普勒扩展作为 Level B 主变量；
- 用第一篇论文中的具体 PDP 或 100 路径参数作为事实依据；
- 为达到目标 BER 反向调节信道；
- 在帧边界清零卷积历史；
- 只比较普通 LMS/DFE 而遗漏强传统长信道跟踪方法。
