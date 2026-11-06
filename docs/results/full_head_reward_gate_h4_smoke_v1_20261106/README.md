# 最近4帧 Reward Pilot 聚合门控 smoke

固定 trajectory 训练 checkpoint 和 Adapt-only Adam 候选，仅把 Reward 验收从当前帧扩展为当前帧与最近3帧的 Reward Pilot 聚合。开发 session 76130，四个SNR各8帧。窗口逐步从1帧填充到4帧；仅合法Reward Pilot参与窗口评分。

| SNR | 当前帧门控收益 | 4帧门控收益 | 接受数当前/4帧 | 有害接受当前/4帧 |
|---:|---:|---:|---:|---:|
| 0 dB | -3.26e-4 | -3.26e-4 | 7/8 | 1/3 |
| 5 dB | 0 | 0 | 2/1 | 0/0 |
| 10 dB | 0 | 0 | 3/0 | 0/0 |
| 15 dB | 0 | 0 | 0/1 | 0/0 |

这是单session、8帧早期筛查，不具统计效力；聚合门控没有改善BER，0 dB有害接受更多，因此不扩大该机制。完整逐帧指标见 eval.json。
