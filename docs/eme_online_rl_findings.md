# EME 在线均衡阶段性结果

## 研究边界

本阶段固定使用 `eme_measurement_v1` 的 Level B B-core 配置，不根据 proposed
结果反向修改信道。主配置为：最大相对时延 `0.0116 s`，等效采样率/符号率
`2000`，`max_delay=24`，帧长 `512`，相干时间 `120 s`，`rho_frame=0.9978689`，
稀疏强路径数 `3..7`，扩散能量比 `0.05..0.15`，仅加入 tiny residual CFO 和慢
相位扰动。Pilot 固定为 prefix，主 SNR 为 `0/5/10/15 dB`。

传统校准已经先于 proposed 冻结。B-core 在 5 个 seed、每个 seed 100 帧上的
结果为：

| SNR | CFO+DD-Phase LMMSE-FIR | CFO+DD-Phase DFE-RLS |
|---:|---:|---:|
| 0 dB | 0.350552 | 0.361729 |
| 5 dB | 0.187854 | 0.199255 |
| 10 dB | 0.066364 | 0.073193 |
| 15 dB | 0.026505 | 0.031989 |

证据文件为
`data/eme/calibration/eme_measurement_channel_calibration_2026-08-26/summary.json`，
SHA-256 为 `db174fa0d988ae24c2e2109ed6385f8baf56db5701328d573afcd603e0bd3961`。

## 已确认的问题

### 1. 训练和部署的跨帧状态分布不一致

旧版 `CurriculumTrainer._step_loss` 始终使用真实 `frame.tail_symbols`，而在线
运行从第 2 帧开始使用模型产生的 soft tail。这使得离线验证看起来可能正常，
但在线递推会在某一帧发生错误后把错误传播到后续帧。

现在新增 `training_tail_mode`：

- `oracle`：兼容旧版 teacher-forcing 训练；
- `online_soft`：先无梯度地跑过历史帧，再使用模型 soft tail 训练当前帧；
- `corrupted_oracle`：用可复现的符号翻转模拟不完全可靠的历史判决。

序列验证在 `model_validation_sequence_state=true` 时也使用接收机 tail 计算
phase 特征，避免输入 tail 和 conditioner tail 不一致。

### 2. 动态 tail 不能随窗口 rollback 回退

旧版在窗口奖励非正时把 `soft_tail` 恢复到窗口起点。对于随机 BPSK 帧，这等于
把 8 帧前的符号重新当作当前帧历史，物理上是错误的。现在 rollback 只恢复
PEFT 参数和慢 CIR 状态，动态 tail 始终保留当前帧最新判决。

### 3. RL 原来没有控制跨帧更新率

动作表原来只有 CIR 更新速率和 PEFT 动作，没有实现已经确定的“先控制校正强度，
再控制跨帧更新率”。现在增加：

- `tail_alpha_slow = 0.25`；
- `tail_alpha_nominal = 0.5`；
- `tail_alpha_fast = 0.8`。

每个动作的实际 `selected_tail_update_alpha` 会写入 frame metrics。默认配置可以
用 `tail_update_alpha` 设置 identity 动作的更新率；主 EME 配置暂设为 `0.5`。

### 4. CUDA augmentation 存在设备错误

`_noisy_cir` 和 `_dd_like_cir` 使用 CPU generator 生成随机量后直接与 CUDA
张量计算，GPU pretrain smoke 会失败。现在随机量先在 CPU 按固定 seed 生成，再
迁移到目标设备；正式 GPU smoke 已通过。

## 阶段性实验记录

以下结果均使用相同 B-core 信道，不改变传统 baseline 的实现。

| 实验 | 关键设置 | 结果 |
|---|---|---:|
| 旧版全分布离线 NN | 200 steps，`lr=1e-3` | 内部离线验证 `0.27018` |
| 旧版真实在线对照 | 10 dB，3 seeds，30 帧 | NN `0.31846`；LMMSE `0.02193`；DFE `0.03348` |
| online soft 从头训练 | 200 steps，跨帧预热 | 内部离线验证 `0.4375`，未达标 |
| 旧 checkpoint online-tail 续训 | 200 steps，`lr=1e-3` | 内部离线验证 `0.41146`，未达标 |
| 旧 checkpoint 低学习率续训 | 20 steps，`lr=1e-4` | 内部离线验证 `0.25521`；20 帧在线 `0.1637`，未超过传统 |
| 同帧 fixed-point refinement | 旧 checkpoint，1 次 refinement | 20 帧在线 `0.2095`，未作为默认方案 |
| 全 SNR oracle-tail 训练 | 0/5/10/15 dB，1000 steps | 内部离线验证 `0.13216`，未超过传统 |

可复查输出目录包括：

- `logs/eme_online_rl_snr10_update8`：旧版 60 帧、多 seed 在线诊断；
- `logs/eme_online_online_tail_finetune_lr1e4_20`：低学习率 online-tail 续训；
- `logs/eme_online_tail_refine_old_model`：fixed-point 诊断；
- `logs/eme_online_tail_alpha05_old_model`：固定 tail 更新率诊断。

## 当前结论

截至本阶段，`RL-Modulated Neural Block Equalizer` 还没有在 Level B 主配置上
逐配置明显超过传统均衡器，因此不能宣称第二个研究点已经完成。已经证明的是：

1. EME 物理信道、跨帧历史和 tiny CFO/慢相位条件已能稳定透传到实验入口；
2. 传统 baseline 的 Pilot 相位/CFO 补偿和校准结果稳定；
3. 旧 NN 的首帧低 BER 不能代表在线能力；
4. 主要失败模式是离线模型跨 channel seed/SNR 泛化不足，以及 NN soft tail
   错误对非因果 Transformer 的全帧传播；
5. 窗口 rollback 不能回退动态符号状态，已按物理时序修正；
6. RL 的作用边界现在包含 PEFT/CIR/tail 更新率，但尚未获得有效的正 reward
   策略，不能把动作分布当作在线微调成果。

## 下一阶段的严格门槛

下一次正式实验必须同时满足：

- 训练和验证覆盖 `0/5/10/15 dB`，至少 3 个独立 channel seed；
- 先报告 offline NN 的每个 SNR BER，再报告在线 30/60 帧结果；
- 在线 observation、reward、动作选择继续不使用 Data 标签；
- proposed 与传统 baseline 使用同一 profile-level 先验和同一 prefix Pilot；
- 不只看平均 BER，还要报告前 5 帧、后 5 帧、窗口 rollback、tail 更新率和
  Reward Pilot 改善；
- 只有 proposed 逐配置超过传统 baseline，且优势随在线帧数增加而保持，才进入
  主论文结果表。
