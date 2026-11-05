# 多帧持久 Adapt 轨迹开发筛查 v1

本轮把离线训练内循环扩展到连续 4 帧：每帧用 Adapt Pilot 更新 `phase_trend` 神经参数，用独立 Reward Pilot 按现有硬规则守门，并将被接受的参数状态传至下一帧；离线 Data BCE 只用于外层梯度，并增加末窗相对首窗的平滑趋势目标。目的在于检验“在线更新学习是否需要显式匹配跨帧持久状态”，不是正式确认。

信道与通信协议沿用既有 `eme_long_memory_v2` 配置：Level B、116 符号长回波、`cfo_phase_tiny`、前缀 Pilot=256（Adapt=224、Reward=32）、0/5/10/15 dB。没有改变 PDP、漂移/同步分布、Pilot、窗口或 acquisition。A 与 B 均从强离线 checkpoint `eme_bce_all_32_20260905_pilot256` 初始化，SHA256=`8cd3912d1c957c9689c51730aff9d27a8f74942186c802a7483e5781af65aec4`，各训练 16 步、使用相同 seed/SNR 序列与每步相同信道采样。

## 结果

这是 3 个新会话、每会话 16 帧的开发筛查。四个 SNR 全部保留。BER 改善定义为 Frozen BER 减 Online BER，正值代表在线较好；区间以独立信道会话为单位。四臂 A/B 分别是普通 `BCE_all` 续训与多帧轨迹外层训练，每臂同时测 Frozen 和 persistent Online。

| SNR | A 在线−Frozen BER改善（95% CI） | B 在线−Frozen BER改善（95% CI） | B−A 在线增益交互（95% CI） | B 末窗−首窗改善（95% CI） |
|---:|---:|---:|---:|---:|
| 0 dB | 0.00578 `[-0.00351, 0.01507]` | 0.00551 `[-0.00273, 0.01374]` | -0.00027 `[-0.00149, 0.00095]` | 0.00156 `[-0.02439, 0.02751]` |
| 5 dB | 0.00130 `[-0.00308, 0.00568]` | 0.00130 `[-0.00290, 0.00550]` | 0.00000 `[-0.00020, 0.00020]` | 0.00061 `[-0.00201, 0.00322]` |
| 10 dB | 0.000081 `[-0.000727, 0.000890]` | 0.000081 `[-0.000727, 0.000890]` | 0.00000 `[0, 0]` | 0.000174 `[-0.001173, 0.001520]` |
| 15 dB | 0.000027 `[-0.000090, 0.000144]` | 0.000027 `[-0.000090, 0.000144]` | 0.00000 `[0, 0]` | 0.000087 `[-0.000287, 0.000460]` |

四帧窗口的 session-level 趋势斜率区间在全部 SNR 均跨零；轨迹训练相对普通续训的逐窗在线增益交互也都没有建立正向证据。0/5 dB 的在线点估计虽为正，但 A 与 B 基本相同，不能归因于多帧轨迹训练；10/15 dB 的可用错误数太少，BER 多数窗口量化为并列。累计共有 384 帧/训练臂评估；候选接受数为 0/5/10/15 dB 的 38/28/23/15（每点 48 次），接受更新仍有少量当帧 Data BER 有害案例。

因此，本轮证明了连续轨迹训练和持久在线参数更新可以运行，也确认 0/5 dB 有少量在线点估计优势；但**没有证明多帧元训练增加在线收益，也没有证明收益随传输帧增长**。研究目标未完成，不进入正式确认。

## 实现与审计边界

- `training/pilot_transfer.py` 新增 trajectory 目标；训练轨迹必须连续且共享 acquisition CIR/CFO，Reward 前向决策复用线上硬守门，拒绝时参数状态不推进。
- 反向传播使用 Pilot-only 的平滑门控代理。该代理将 BCE 改善与软错误改善相乘，属于保守 AND 近似；线上仍用原始硬门控。这一近似的梯度质量仍需独立检验。
- `next_pilot_tail` 的前向状态按部署逻辑截断梯度；本配置为前缀 Pilot，Pilot 不覆盖尾部历史区域。评估器现在在持久模式下用已提交参数递推尾部状态，Frozen 使用冻结权重。
- 增加 Data 标签毒化测试，验证它不会改变在线候选、Reward 决策和持久参数；增加非连续帧拒绝测试。Data BCE 与趋势损失仅参与离线外层优化。
- 强离线 checkpoint 的训练seed与评估seed隔离；评估所有比较分支逐帧共享接收波形，并冻结于相同物理 acquisition 协议。

## 产物与复现

- 配置：[eme_long_memory_v2_trajectory_transfer_dev_v1.json](../../../configs/eme_long_memory_v2_trajectory_transfer_dev_v1.json)，SHA256=`094fd44962044ba71c8d251c086609b32bde68c1654279f4684b6ac3519ef77`。
- 在线轨迹训练 A/B 指标、逐帧四臂结果及 4 帧窗口趋势见本目录中的 JSON/JSONL 文件。
- 评估帧日志 SHA256=`8a0430fd06d7d5903cbb06e58c654b99c51129a10b6a5f9833d8110355645281`。A checkpoint SHA256=`5676d73069074d42ae405820ced3a269cf8859160b2bf150ac72b1511fbac694`；B checkpoint SHA256=`816f89d4dee533f40f20545355178ae26638733b7739f3fee1b9862929fd424c`。
- 汇总命令：`python scripts/summarize_trajectory_transfer_dev.py --input <frame_metrics.jsonl> --window-frames 4 --output <window_trends.json>`。
- 训练入口：`pretrain.py --stage pilot_transfer --transfer-objective baseline|trajectory`。两臂使用同一配置、训练步数和来源 checkpoint。
