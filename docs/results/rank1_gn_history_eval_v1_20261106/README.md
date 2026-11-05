# 当前 Adapt Pilot 与最近8帧历史的固定 Fisher-GN 对照

## 设置

固定 Level B `eme_long_memory_v2`、Pilot和接收协议，以同一个 Pilot-Fisher rank-1 basis和阻尼GN更新比较仅当前帧Adapt（history=1）与最近8帧Adapt（history=8）。history=8 臂使用7帧预热，只构建同会话、帧龄连续的合法Adapt历史；预热帧不进入BER窗口。评估为一个独立开发session 75002，每SNR16帧、四个等长窗口，完整波形级结果见 `result.json`。数据用途仅为开发筛查。

## 观察

| SNR | 方法 | Reward接受 / 16 | Online−Frozen平均BER收益 | 晚窗−早窗 |
|---:|---|---:|---:|---:|
| 0 dB | history=1 | 4 | +3.26e-4 | +6.51e-4 |
| 0 dB | history=8 | 4 | 约0 | 0 |
| 5 dB | history=1 | 3 | -1.63e-4 | 0 |
| 5 dB | history=8 | 3 | -1.63e-4 | 0 |
| 10 dB | history=1/8 | 1/2 | 0 | 0 |
| 15 dB | history=1/8 | 1/1 | 0 | 0 |

在这一session里，8帧历史没有提高候选接受率或BER收益。5 dB两臂都出现同样的负点估计；history=8未改变它。该结果只能作为开发方向筛查，不能代表session级统计推断。

## 实现注意

离线非缓存采样路径已改为在每个历史帧的真实帧前状态下保存Adapt上下文，再递推tail；避免把最终帧tail错误复用到所有历史帧。在线评估也只把经过预热的同session历史输入Adapt损失，Reward仍只看当前帧，Data仍只用于动作后评分。
