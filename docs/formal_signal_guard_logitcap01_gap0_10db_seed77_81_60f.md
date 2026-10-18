> 状态：历史诊断/已被当前主路线取代。原始结果保留用于追溯，不代表当前默认方法。
> 当前规范：docs/state_tracking_receiver_design.md；验证：docs/state_tracking_validation_protocol.md。

# 10 dB 长窗口 logits 守门复核（2026-10-15）

## 固定条件

本轮沿用 `online_logit_guard_replay_20261015.md` 的全部设置，只将验证扩展到
新的 seed 77--81、每条轨迹 60 帧、SNR=10 dB。离线 checkpoint、Level B 信道、
prefix Pilot、`phase_trend` Adapter、Adapt 信号重构、Reward 双窗口守门和
`Reward logits 平均改变量 <= 0.1` 均保持不变。

原始日志：
`logs/formal_signal_guard_logitcap01_gap0_10db_seed77_81_60f/`

## 结果

改善定义为 `(Frozen BER - Online BER) * 100`，单位为百分点。

| SNR | 帧数 | Frozen BER | Online BER | 平均改善 | 正收益 seed | 最差 seed | 接受更新总数 | 回滚总数 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 dB | 60 | 0.7027% | 0.6901% | +0.0126 pp | 2/5 | -0.0087 pp | 14 | 9 |

| seed | Frozen BER | Online BER | 改善 (pp) | 接受更新 | 回滚 |
|---:|---:|---:|---:|---:|---:|
| 77 | 1.1567% | 1.0959% | +0.0608 | 4 | 3 |
| 78 | 0.4102% | 0.4102% | 0 | 2 | 1 |
| 79 | 0.4536% | 0.4536% | 0 | 1 | 1 |
| 80 | 0.8529% | 0.8420% | +0.0109 | 2 | 2 |
| 81 | 0.6402% | 0.6489% | -0.0087 | 5 | 2 |

## 事后连续指标诊断

该诊断只读取日志，未参与在线更新或验收。10 dB 的 8 次候选更新均通过当前
Reward 守门；其中 5 次使当帧 Data BCE 下降，但只有 1 次改变了当帧硬 BER。说明
连续输出方向有一定信息，但 32 个 Reward 符号和 768 个 Data 符号之间仍存在
排序误差，不能把 Reward BCE 或信号重构的微小改善当成硬 BER 改善保证。

seed=81 的负收益很小，但部分被接受的更新在 Reward 信号误差上的改善只有约
`0.00018--0.00028`。下一轮将验证要求 Reward 信号误差至少改善 `0.0005` 是否能
减少这类低幅度误放行；该门仍只使用 Reward Pilot。

## 判断

延长到 60 帧没有让 `logits <= 0.1` 变成稳定方案：平均收益仍为正，但仅 2/5 seed
有正收益且出现 1 个负收益 seed。当前方法已经证明 PEFT 更新能够影响 Data 输出，
但尚未证明在 10 dB 上能稳定超过冻结离线模型。
