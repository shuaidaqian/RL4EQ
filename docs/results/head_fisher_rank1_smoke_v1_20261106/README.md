# 强离线模型上的 Fisher rank-1 在线更新 smoke

## 目的与边界

本次只检查强离线 checkpoint 上的固定 Pilot-Fisher 主方向候选是否能走完 Adapt 更新、Reward 硬门控和 Data 事后评分流程。使用现有 Level B `eme_long_memory_v2`、116 符号长回波、Pilot=256（Adapt=224、Reward=32）及 0/5/10/15 dB，不改信道与接收协议。`result.json` 保存配置、checkpoint、波形和方向来源指纹，以及逐帧候选记录。

该 smoke 使用一个新评估 session seed（63011）、每 SNR 两帧。Pilot-Fisher 方向由独立训练角色 seed 63000 的四 SNR Adapt Fisher 平均主特征向量构成；在线标量 Adam 每帧重置，接受后的 head 参数跨帧持久，Reward 不通过则不提交。Data 标签只在守门与提交完成后评分。

## 观察

| SNR | 分支 | Reward 接受数 / 候选数 | 两帧平均 Online−Frozen BER 收益 |
|---:|---|---:|---:|
| 0 dB | 全 head Adam | 1/2 | +0.001302 |
| 0 dB | Fisher rank-1 | 1/2 | +0.001302 |
| 5 dB | 全 head Adam | 1/2 | 0 |
| 5 dB | Fisher rank-1 | 0/2 | 0 |
| 10 dB | 全 head Adam | 1/2 | 0 |
| 10 dB | Fisher rank-1 | 1/2 | 0 |
| 15 dB | 全 head Adam | 0/2 | 0 |
| 15 dB | Fisher rank-1 | 0/2 | 0 |

样本量只有每 SNR 一个会话、两个 Data 帧，0 dB 的差异仅对应单个帧上的一次硬判决变化；其余三个 SNR 没有观察到 BER 增益。该结果只能证明代码路径可运行且部分候选可被 Reward 接受，不能证明 Online 优于 Frozen，也不能检验收益随帧数增长。

## 解释限制与下一步

只读复审发现 Fisher rank-1 与全 head 对照同时改变了子空间、优化器细节和局部信赖域：全 head 路径有逐帧裁剪及 AdamW 衰减，rank-1 标量 Adam 没有对应设置，且 `projected_delta_norm` 当前没有反映最终累计投影量。两臂不是单因素对照。Frozen 配对也应明确区分独立 Frozen 轨迹与沿 Online 软尾状态重放的 Frozen 权重反事实，避免把两种估计混为一列。

因此不扩大 seed 矩阵。先统一并记录一维候选更新器与局部/累计约束、补充门控状态转移及标签隔离测试，再比较固定 Fisher 方向上的阻尼一维 Gauss–Newton 与标量 Adam。若候选质量能稳定通过 Reward 守门，再让多帧离线外层在同一个一维在线更新器下学习方向，并以新的独立 session 检验全程收益与晚早差/斜率。
