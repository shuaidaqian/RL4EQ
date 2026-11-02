# 协议修复后的正式离散记忆 RL 机制验证

延续已授权路线：Level B、116 符号长回波、0/5/10/15 dB、prefix256（Adapt224 / Reward32）、会话开始一次 acquisition，随后每帧适配；先验证统计收益并报告墙钟。不得把接入、通过测试或单 seed 均值当作目标完成。

## 本轮修复边界

两个入口保留 acquisition CFO 并传入相同条件；显式 `rl_reward_receiver_input=full_rx` 允许整帧接收波形进入 Reward 判决，梯度仍只用 Adapt，验收和 PPO reward 只用 Reward。Data 发送符号和标签仅在动作、回滚与策略更新完成后用于评估。旧配置默认保留 `pilot_only`，新增记忆动作必须显式启用。

复记忆动作只选择 signal/MAP × small/medium，另有 identity/rollback，固定尺度与其余离线权重；累计半径 0.8，窗口 8 帧、PPO 更新间隔 32 帧。独立 Frozen 保持零记忆，同 acquisition、条件和波形。

此前 `channel_memory_discrete_rl_128f_1s_20261102` 与已停止的 `channel_memory_discrete_rl_128f_3s_20261102` 使用修复前不同步协议，**排除有效机制证据**，不与修复后结果混合。

## 预先固定的顺序

1. 完成全量回归及跨两个 PPO 窗口的 Data 标签/发送符号污染测试。
2. 已完成四 SNR × 8 帧统一入口同步检查；该次仅抽到 identity，Proposed/Frozen BER 完全一致，不算适配证据。
3. 修复后统一 `compare.py`，控制 checkpoint 为 `pretrained/channel_memory_map_control64_20261102/model_best.pt`（SHA256 `153ccc68515fe77996f822924f0056d25a283ca0fd7b185c19a030bfe2871f08`），冻结传统协议 `configs/traditional_calibration_20261101.json`。使用 `offline_train` 实际环境 seed 100010，四 SNR 各 128 帧，从首帧连续运行；`--seed-start 50010` 显式映射到环境 seed。策略重新初始化，不加载既有策略文件。
4. 若出现真实记忆更新、窗口保留和 PPO 更新，扩到预先固定的 100011–100012，逐 seed 计算对独立 Frozen 的全程及后 1/3 减前 1/3 改善，并同时报告各传统方法。这里只是开发机制诊断，后续需新的独立验证 seed。
5. 若门槛不通过，先根据 Pilot 控制记录定位更新、验收或奖励的问题；不得按 Data 选动作、阈值或 checkpoint，不直接扩大数百秒主矩阵。

执行时显式设置 `$env:RL4EQ_LINEAR_OPERATOR_BACKEND = 'fft'`。交接后的首次完整运行没有继承前阶段的环境变量，已停止；保留 `logs/channel_memory_rl_protocol_128f_seed100010_20261102` 的 direct 部分日志作为执行记录，不纳入完整统计。有效输出为 `logs/channel_memory_rl_protocol_fft_128f_seed100010_20261102`，从相同 seed 首帧重新开始；同名 `_timing.json` 记录完整进程墙钟。已有 FFT/逐 tap 的线性算子与模型输出一致性检查仍适用，不混用两批轨迹。

正式第一目标仍为四 SNR 逐配置超过校准传统 baseline，在线相对同 checkpoint Frozen 的改善有独立 seed 区间支持，且后期平均优势更明显。最终需覆盖数百秒连续会话。
