# 强离线 checkpoint 的 Pilot-transfer 元训练开发筛查

## 问题与实验

启发式 residual-RLS 在强离线 checkpoint 上能更新参数，却未带来可重复 BER 收益。下一条候选是离线训练表示，让合法 Pilot 更新后更容易泛化到 Data。本轮用已有 `pilot_transfer` 可微内外循环做了一个小规模匹配筛查，不把它说成与 residual-RLS 相同的更新算法。

A/B均从已验证强离线 checkpoint 初始化，使用完全相同的32个 `offline_train` 样本和训练预算：A继续优化 `BCE_all`；B优化`BCE_all + 当前Pilot更新后的Data BCE`。训练内循环是每帧Adapt Pilot驱动的 `phase_trend` AdamW，5步、学习率0.0025、增量半径0.05；Data标签只用于离线外层梯度。在线评估保留PEFT参数，采用同一每帧Adapt更新与Reward Pilot守门；评估Data标签不进入更新或门控。

原四臂评估器过去只允许 `heldout_edge`，这会改信道状态范围。为符合当前固定同分布路线，本轮只扩展其允许传入既有 `offline_train`；没有更改任何信道、Pilot、窗口、状态生成或接收行为。完整测试验证两种状态范围都能安全运行。

## 复现命令

训练A：

```powershell
& 'D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe' pretrain.py --config configs/eme_long_memory_v2_pilot_transfer.json --stage pilot_transfer --steps 32 --batch-size 1 --transfer-objective baseline --resume D:\Research\RL4EQ\pretrained\eme_bce_all_32_20260905_pilot256\model_best.pt --save-dir pretrained/strong_checkpoint_transfer_baseline32_20261105 --device cuda
```

训练B：

```powershell
& 'D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe' pretrain.py --config configs/eme_long_memory_v2_pilot_transfer.json --stage pilot_transfer --steps 32 --batch-size 1 --transfer-objective post_adapt --resume D:\Research\RL4EQ\pretrained\eme_bce_all_32_20260905_pilot256\model_best.pt --save-dir pretrained/strong_checkpoint_transfer_post32_20261105 --device cuda
```

持久适配评估：

```powershell
& 'D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe' compare.py --config configs/eme_long_memory_v2_pilot_transfer.json --pretrained pretrained/strong_checkpoint_transfer_baseline32_20261105/model_best.pt --transfer-checkpoint pretrained/strong_checkpoint_transfer_post32_20261105/model_best.pt --transfer-persistent --delays 116 --snrs 0 5 10 15 --num-seeds 3 --seed-start 45400 --frames 8 --state-split offline_train --output-dir logs/strong_checkpoint_transfer_post32_dev_20261105 --device cuda
```

主模型 checkpoint SHA256=`8cd3912d1c957c9689c51730aff9d27a8f74942186c802a7483e5781af65aec4`；A模型=`07f3a13b5b0d20a57d10e972da94101407fb9970c118eef2d57542a9bc0deb37`；B模型=`e8ea322a2279dd8b45876aebb50feefb0cd0dece38611fd7f8e27c97ebe2b71e`；训练配置 `configs/eme_long_memory_v2_pilot_transfer.json` SHA256=`5898fe3a9f1433743ab8ddf0e668ed7f039bfbb6d367c68ad9a530b3f8e1de73`。

## 结果和解释

每个SNR只有3个开发session、每个8帧；区间以session为单位。A/B在各SNR的Frozen BER逐点完全相同。A/B在线表现与配对交互为：

| SNR | A/B Online−Frozen增益 | B−A在线增益交互 | 末窗−首窗增益 |
|---:|---:|---:|---:|
| 0 dB | `+0.00304`，95%区间`[-0.00875, +0.01482]` | `0` | `-0.00065`，区间跨零 |
| 5 dB | 约`0`，95%区间`[-0.00246, +0.00246]` | `0` | `+0.00217`，95%区间`[+0.00093, +0.00341]` |
| 10 dB | `+0.000326`，95%区间`[-0.000375, +0.001026]` | `0` | `-0.000434`，区间跨零 |
| 15 dB | A=`+0.000054`，B=`+0.000109`，两者区间跨零 | `+0.000054`，95%区间`[-0.000179, +0.000288]` | A=`+0.000109`、B=`0`，区间跨零 |

唯一较清晰的时间趋势出现在5 dB，但A和B完全相同，不能归因于 post-adapt 离线目标；5 dB全程平均增益约为零。四点中没有证据表明post-adapt离线训练创造了独有、稳定的Online收益，也没有达到多窗口长期增长验收。

这次小筛查不值得扩成大 seed 矩阵。它定位到方法差距：现有 `pilot_transfer` 只在一个帧的Adapt更新后计算外层目标，虽可评估持久参数，但训练没有反向传播穿过一段连续的在线更新轨迹。下一步应实现可微/可校验的多帧持续适配内循环：参数跨帧保留，逐帧只使用合法Adapt Pilot更新，Reward Pilot仅作动作后守门，隔离Data外层目标评估多个后续窗口；再以匹配普通继续训练A臂和同源Frozen在四SNR新session评估收益随帧增长。当前结果仅为开发筛查，未开始正式确认。

训练指标、逐帧评估、汇总与session趋势分别保存在本目录的`training_*_metrics.json`、`frame_metrics.jsonl`、`compare_summary.json`和`session_trends.json`。
