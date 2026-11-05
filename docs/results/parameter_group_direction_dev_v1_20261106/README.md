# 在线参数组方向传输诊断 v1

## 目的与边界

本诊断检查合法 Adapt Pilot 的 BCE 梯度是否能沿所选神经参数组传递到后续 Data 检测输出，并记录 Reward Pilot 硬守门后的事后 Data 变化。它用于筛选下一轮表示，不是在线增益或持续增长的证明。

固定使用 Level B `eme_long_memory_v2`、116 符号长回波、前缀 Pilot=256（Adapt=224、Reward=32）、0/5/10/15 dB，以及同一强离线 checkpoint。未改变信道、Pilot、接收窗口、同步或状态跟踪协议。每组使用 3 个共同的新开发会话 `61000–61002`，每个 SNR、每个会话连续 8 帧。候选更新只使用 Adapt Pilot；Reward Pilot 独立决定接受；Data 标签只在候选与门控结束后读取，用作离线梯度和 BER 诊断。

离线权重仍冻结，逐帧候选参数从该帧离线初始化生成，未跨帧累积。本轮重点是单参数组梯度/作用筛查，不能替代真实的持久在线轨迹比较。统计单位为 3 个会话；`mean` 与区间由会话级配对汇总。稀疏硬 BER 的 0 点或点估计不能解释为总体增益。

## 主要结果

`cos` 为 Adapt BCE 与 Data BCE 梯度余弦均值；`接受`是 24 个帧候选中通过 Reward 守门的数量；`Online-Frozen` 是候选经门控保留后的事后 Data BER 差，负值表示候选更好。区间是 3 个独立会话级汇总的 95% 区间。

| 参数组 | SNR | 参数数 | Adapt/Data 梯度余弦 | Reward接受 | Online−Frozen BER | 接受后有害帧 |
|---|---:|---:|---:|---:|---:|---:|
| `head` | 0 dB | 65 | +0.650 [+0.246, +1.053] | 19/24 | +0.0109 pp [-0.0733, +0.0950] pp | 3 |
| `head` | 5 dB | 65 | +0.405 [+0.015, +0.796] | 10/24 | +0.0054 pp [-0.0179, +0.0288] pp | 1 |
| `head` | 10 dB | 65 | +0.048 [+0.027, +0.068] | 3/24 | 0 | 0 |
| `head` | 15 dB | 65 | +0.047 [-0.131, +0.225] | 1/24 | 0 | 0 |
| `phase` | 0 dB | 6469 | +0.419 [-0.552, +1.390] | 14/24 | -0.4883 pp [-2.0795, +1.1030] pp | 3 |
| `phase` | 5 dB | 6469 | +0.086 [-0.606, +0.778] | 11/24 | +0.0488 pp [+0.0084, +0.0893] pp | 5 |
| `phase` | 10 dB | 6469 | +0.001 [-0.944, +0.945] | 5/24 | -0.0054 pp [-0.0672, +0.0563] pp | 1 |
| `phase` | 15 dB | 6469 | -0.262 [-1.673, +1.149] | 7/24 | 0 | 0 |
| `phase_trend` | 0 dB | 16 | +0.411 [-0.282, +1.105] | 19/24 | -0.4720 pp [-1.7734, +0.8294] pp | 4 |
| `phase_trend` | 5 dB | 16 | +0.074 [-1.248, +1.395] | 9/24 | -0.0163 pp [-0.3215, +0.2889] pp | 4 |
| `phase_trend` | 10 dB | 16 | +0.185 [-0.679, +1.050] | 3/24 | -0.0271 pp [-0.1802, +0.1259] pp | 1 |
| `phase_trend` | 15 dB | 16 | +0.050 [-0.706, +0.806] | 9/24 | +0.0054 pp [-0.0179, +0.0288] pp | 1 |

注意：BER 符号采用 `candidate_gain = BER_Frozen - BER_Online` 的相反数，表中已按 `Online−Frozen` 表达。`phase` 的 5 dB 是明确的退化方向；其它多数区间跨零。`head` 的低 SNR 梯度余弦较高，但硬 BER 收益仍没有明确正证据。`phase_trend` 在多个 SNR 上明显改变 Data logits，却未稳定改变硬判决，说明“参数影响输出”并不等于“改善检测”。没有任何参数组在本轮建立全 SNR 在线增益，更没有帧数增长证据。

## 解释与下一步

这轮结果排除了“当前 `phase_trend` 组参数太少、因此不能影响 Data 输出”的简单解释；同时也表明只扩大到 6469 个 `phase` 参数会放大 logit 位移并在 5 dB 造成误码增加。`head` 是更小且低 SNR 方向更一致的候选，但高 SNR 的接受率和有效误码数很低，仍需避免把梯度余弦误当成收益。

下一项应转向**离线学习低维神经更新状态空间先验**，而不是再增加在线步长、历史长度或半径：固定在线硬守门和 Adapt/Reward 协议，离线内循环学习少量共享更新方向及其跨帧不确定度/过程噪声；对照为相同数据预算的普通监督续训与固定先验更新器。先做小规模配对开发，再按完整会话窗口测早/晚窗收益和斜率。若没有稳定的 `Online−Frozen` 改善及晚窗增长，不扩大实验，也不进入正式确认。

## 文件与复现

- 配置：[eme_long_memory_v2_update_direction_diagnostic_v1.json](../../../configs/eme_long_memory_v2_update_direction_diagnostic_v1.json)
- 诊断脚本：[diagnose_head_transfer.py](../../../scripts/diagnose_head_transfer.py)
- 原始逐帧 JSON：[head.json](head.json)、[phase.json](phase.json)、[phase_trend.json](phase_trend.json)
- 每份 JSON 内含 checkpoint SHA256、配置、输入波形指纹、逐帧结果和会话汇总。
- 诊断入口示例：

```powershell
python scripts/diagnose_head_transfer.py --config configs/eme_long_memory_v2_update_direction_diagnostic_v1.json --pretrained D:\Research\RL4EQ\pretrained\eme_bce_all_32_20260905_pilot256\model_best.pt --seed-start 61000 --num-seeds 3 --frames 8 --groups phase_trend --device cuda --output docs/results/parameter_group_direction_dev_v1_20261106/phase_trend.json
```

本诊断的候选更新属于事后机制检查；线上 Data 标签控制标志为 false。3 会话仅用于开发筛查，不作正式确认或跨总体结论。

配置 SHA256：`9919d0dd35232ba1c30e8c7149073477c5f63893ac39489ae390d73637bb9160`。逐帧结果 SHA256：`head.json`=`36e160d700ed766283ba844e2a1adbeb2f5c6d153345091a4ba8ec9e21d2585f`；`phase.json`=`375276bc266d20bde9bcb47599e857ce962ff8d95aea349689c5bb2e329855e9`；`phase_trend.json`=`8640b31884eea5a968f3d871076e5f803489a881535c255a174f33abcf4d0e79`。checkpoint SHA256 为`8cd3912d1c957c9689c51730aff9d27a8f74942186c802a7483e5781af65aec4`。
