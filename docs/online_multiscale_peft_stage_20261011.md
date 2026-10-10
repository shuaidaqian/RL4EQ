# Reward 选择多尺度 PEFT 阶段结果

日期：2026-10-11

## 研究问题

固定一个学习率和步数时，Pilot 上的改善有时会过冲，有时又太小。当前阶段测试三个
不同幅度的同类 `phase_trend` Adapter 更新，让 Reward Pilot 在动作后选择其中一个；
如果三个候选都不能通过 Reward Pilot，就保持 Frozen 参数并回滚。三个候选都只更新均衡器
内部参数，不改变离线 checkpoint，也不读取 Data 标签。

方法名称：`Reward-selected multi-scale PEFT`

## 固定设置

- checkpoint：`pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`
- 信道：Level B，`delay=116`
- Pilot：prefix，总长度 256，Adapt 224，Reward 32
- acquisition 到数据帧间隔：600 s（状态老化压力测试）
- SNR：5、10 dB
- Adapt 更新：`phase_trend`，Pilot 复数信号重构，AdamW，5 步
- 候选幅度：学习率比例 `0.125/0.25/0.5`，参数上限比例 `0.25/0.5/1.0`
- 漂移门：Pilot CIR 相对残差 `0.7`
- Reward 验收：两个 Reward 子窗口、Reward 硬 BER 不增加、Reward 复数信号重构误差不增加
- 跨帧回滚：开启

可复现配置：[`configs/online_phase_signal_multiscale.json`](../configs/online_phase_signal_multiscale.json)

## 原始候选的 3 seed × 10 帧结果

逐帧日志：[`logs/phase_signal_multiscale_gap600_3s2snr10f_20261011`](../logs/phase_signal_multiscale_gap600_3s2snr10f_20261011)

| SNR | seed | Frozen 平均 BER | Online 平均 BER | 配对改善 | 保留更新次数 | 结论 |
|---:|---:|---:|---:|---:|---:|---|
| 5 dB | 0 | 35.737% | 34.448% | +1.289 pp | 4 | 正收益 |
| 5 dB | 1 | 35.737% | 34.761% | +0.977 pp | 4 | 正收益 |
| 5 dB | 2 | 35.737% | 35.776% | -0.039 pp | 1 | 轻微退化 |
| 10 dB | 0 | 34.961% | 34.818% | +0.143 pp | 3 | 正收益 |
| 10 dB | 1 | 34.961% | 34.961% | 0.000 pp | 0 | 持平 |
| 10 dB | 2 | 34.961% | 34.063% | +0.898 pp | 5 | 正收益 |

短窗口平均改善为 5 dB `+0.742 pp`、10 dB `+0.347 pp`。单帧候选选择记录显示，
Reward Pilot 会在不同帧选择 `conservative`、`standard` 或 `fast`，而不是始终使用
最大更新；这正是固定学习率方案没有的自适应幅度控制。

## fast 候选失败复核与安全幅度修正

补充的 5 seed × 4 帧 replay 发现，原始 `fast` 候选在 5 dB、seed 4 的第 3 帧虽然通过
Reward Pilot，却把该帧 Data BER 从约 21% 推到约 50%；下一帧触发跨帧回滚，但无法撤销
已经发生的 Data 错误。因此原始三档候选不能作为推荐配置。

随后把三档改为 `conservative/standard/moderate`，对应学习率比例
`0.125/0.25/0.5` 和参数上限比例 `0.25/0.5/1.0`。安全幅度版本的 5 seed × 4 帧结果为：

| SNR | seed 正收益数 | 平均配对改善 | 最坏 seed | 判断 |
|---:|---:|---:|---:|---|
| 5 dB | 4/5 | +0.9375 pp | 0 pp | 短窗口通过 |
| 10 dB | 2/5 | +0.2018 pp | -0.2604 pp | 仍需长窗口复核 |

该结果只说明限制更新幅度能消除已发现的灾难性坏帧，不能替代正式 5 seed × 60 帧门槛。

## 对照结果

### 固定 `physics_residual` 候选的正式主配置短矩阵

在 gap=0、5 seed × 60 帧、0/5/10/15 dB 下，`physics_residual` 能改变 Data 连续 BCE，
但 BER 平均改善为：5 dB `+0.026 pp`、10 dB `-0.001 pp`、15 dB `0 pp`；5 dB 只有
3/5 seed 正收益，10 dB 只有 1/5 seed 正收益。因此不把它作为主候选。

### 双视图 Adapt Pilot 一致性

前后半段参数方向门控已经记录在
[`docs/dual_view_pilot_consistency_replay_20261010.md`](dual_view_pilot_consistency_replay_20261010.md)，
不能可靠预测 Data BER，也不接入主流程。

## 当前判断

多尺度候选是目前最符合目标的在线方法：离线模型和 checkpoint 不变，在线只根据 Adapt
Pilot 更新 Adapter 参数，Reward Pilot 负责候选选择和回滚。它在状态老化压力测试的小
矩阵中同时改善了连续 Data BCE 和部分硬 BER，但还没有通过正式统计门槛，不能写成最终
完成的创新结果。

## 下一步

1. 使用安全幅度配置运行 gap=600 s 的 `5 seed × 60 帧 × 5/10 dB` 正式矩阵。
2. 要求每个 SNR 至少 4/5 seed 正收益，且平均改善为正；同时检查最坏 seed 的 Data BCE、
   Reward 回滚和 `data_labels_used_online=false`。
3. 通过后，再在 gap=0 的 0/5/10/15 dB 主矩阵验证是否仍有额外收益；不通过则把结果
   明确限定为“对状态老化有效”，不把短窗口收益夸大为普遍在线增益。
4. 在上述门槛通过前不接入 Contextual Bandit，也不重新训练离线 checkpoint。

## 复现命令

```powershell
.\.venv-gpu\Scripts\python.exe compare.py `
  --config configs/online_phase_signal_multiscale.json --method-group proposed `
  --pretrained pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt `
  --delays 116 --snrs 5 10 --num-seeds 3 --frames 10 --pilot-total 256 `
  --pilot-layout prefix --acquisition-to-data-gap-seconds 600 `
  --scheduler fixed --online-condition-source pilot_cir_phase `
  --online-optimizer adamw --online-drift-gate-threshold 0.7 `
  --online-require-reward-signal-reconstruction --online-reward-signal-tolerance 0 `
  --online-min-reward-improvement 0 --online-reward-windows 2 `
  --online-cross-frame-tolerance 0 --update-interval 1 --resume `
  --output-dir logs/phase_signal_multiscale_gap600_3s2snr10f_20261011
```
