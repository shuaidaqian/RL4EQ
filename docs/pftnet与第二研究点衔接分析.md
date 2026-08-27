# P-FTNet 与第二研究点的衔接分析

## 1. 文献定位

`P-FTNet: Pilot-Conditioned and Feature-Augmented Transformer Network for
Equalization Under Extreme Delay Spread` 是一篇与本项目 Offline 阶段高度相关的
参考论文。论文研究单载波 BPSK 在 20/30/40 符号级严重时延扩展下的均衡，使用
前置 Pilot、PilotNet、FiLM、多尺度卷积和 Transformer，并通过改变测试时延来
验证信道失配鲁棒性。

本文只把 P-FTNet 作为 Offline 神经均衡器的可复现参考，不把其 Rayleigh/Rician
城市多径信道当作 EME 信道，也不把论文的静态单帧实验解释成在线跨帧适配证据。

## 2. 可以直接借鉴的设计

### 2.1 原始 Pilot 条件分支

P-FTNet 的 PilotNet 直接输入已知 Pilot 和接收 Pilot，生成供主干网络使用的
FiLM 条件。这个设计比只把估计 CIR 压缩成少量向量更适合本项目，因为它能保留
Pilot 中的局部相位、噪声和残差结构。

### 2.2 局部与全局特征并行

多尺度卷积适合提取局部 ISI 结构，Transformer 适合捕捉跨多个符号的依赖。本项目
已经有显式长时延线性算子，因此不必照搬完整 MFE，而应使用轻量 Pilot 条件分支
补充物理展开网络的条件表达能力。

### 2.3 训练测试信道失配

论文训练最大时延为 20、测试时改变时延。这一思想适合本项目的 Offline 泛化验证，
但本项目的失配应改为：路径支撑、弱弥散能量、CIR 复增益、residual CFO、慢相位
和 soft tail 状态失配，并且测试信道 seed 必须独立于训练 seed。

## 3. 不能直接迁移的内容

论文没有验证以下因素：

- 跨帧历史符号导致的 boundary ISI；
- 由前一帧软判决递推得到的 soft tail 不确定性；
- residual CFO 与慢相位状态跨帧连续变化；
- 慢变 tap gain 或 libration-like 复增益漂移；
- 动作后的 Reward Pilot 反馈和在线权重适配。

因此 P-FTNet 不能作为本研究 Online RL 结果的替代证据。

## 4. 两个研究点的承接关系

第一研究点提供 EME 相关的宏观先验：回波延迟范围、散射包络、稀疏长回波和
慢变化时间尺度。第二研究点使用独立的 EME-inspired 研究信道，把这些宏观先验
离散化为可控的长记忆信道：

```text
文献/第一点的 EME 特性
    -> 独立的 Level B 稀疏长时延信道抽象
    -> Offline Pilot-conditioned physical NN
    -> 跨帧 residual mismatch
    -> Pilot 驱动的安全在线适配
    -> RL 选择校正强度和更新率
```

`0.0116 s` 和 `2000 symbol/s` 只用于得到 `D=24` 的离散时延上界；强径数量、
弥散能量比例和复路径演化属于研究抽象参数，不能写成 EME 的直接测量值。

## 5. 对当前实现的决定

保留当前物理展开式网络作为主研究实现，增加原始 Adapt Pilot 条件分支作为
Offline 增强，而不是复制完整 P-FTNet。Offline 模型必须先在相同 prefix Pilot
条件下超过最强传统 baseline；之后才允许研究在线动作。

Online 层的核心仍然是 Pilot 驱动的跨帧适配。RL 只调度 phase correction strength、
soft-tail 更新率和后续 CIR/tap-gain 更新率，不直接生成完整高维参数增量，也不
使用 Data 标签。

## 6. 论文中建议的对照组

```text
传统：CFO+DD-Phase LMMSE-FIR、CFO+DD-Phase DFE-RLS、SC-FDE-MMSE
Offline：当前 physical NN、P-FTNet-inspired Pilot-conditioned NN
Online：Offline NN only、固定规则适配、RL 调度适配
诊断：Perfect-CSI、oracle tail、oracle CIR、Fixed CG detector
```

P-FTNet 结果只作为公开文献/组内参考，不能与本项目的 EME Level B BER 直接合并
平均，因为 Pilot 长度、信道族、指标和是否存在跨帧状态均不同。
