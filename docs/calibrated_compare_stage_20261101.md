# 校准传统方法与当前神经/RL 接收机的同轨迹诊断

本阶段完成冻结传统协议接入和真实 GPU 配对验证。当前 head checkpoint 的
RL 与 Frozen 在四个 SNR 上的 Data BER 完全相同；相对较强的同步/边界校准
FDE，神经优势也没有在全部 SNR 上获得配对区间支持。因此整体目标仍未达成。

## 协议与接入

设置在运行前记录于[计划](calibrated_compare_plan_20261101.md)。新增显式参数
`--traditional-calibration`，通过
[`traditional_calibration_20261101.json`](../configs/traditional_calibration_20261101.json)
加载三类传统方法的冻结设置；不改变主方法组，不纳入额外线性块参考。

冻结协议核验实际环境/Pilot、校准 seed 隔离，记录来源和算法版本；拒绝混用
不同协议 resume。统一比较额外记录 `rx_sha256` 和真实 `environment_seed`，
本轮 160 个帧组内的 5 种方法接收波形完全匹配。共 800 条唯一逐帧记录，
其中 480 条为应用冻结设置的传统方法记录。

信道仍为 Level B、116、prefix 256（Adapt=224、Reward=32）、原扰动范围。
heldout_edge、CLI seed=47001–47005，对应实际环境 seed=97001–97005，每条 8 帧。
checkpoint 为先前固定的 head post-adapt 256 步终点，SHA256：
`b8281a8132759865b1b9c303aed60067eaae72a6917f1a0ace6457c4f5b4b7e5`。
未加载额外策略 checkpoint；PPO 从安全先验开始。窗口和更新间隔均为 8。

## 逐 SNR 结果

下表单位为百分数，全部来自相同测试轨迹。

| SNR | 校准 FIR | 校准 DFE | 校准零填充 FDE | Frozen | RL |
|---:|---:|---:|---:|---:|---:|
| 0 dB | 22.04102% | 23.90300% | 23.24544% | 22.83854% | 22.83854% |
| 5 dB | 12.24284% | 13.37240% | 5.64779% | 5.58919% | 5.58919% |
| 10 dB | 6.05143% | 6.90430% | 0.98958% | 0.93750% | 0.93750% |
| 15 dB | 2.85482% | 4.33268% | 0.28971% | 0.20182% | 0.20182% |

相对同 checkpoint Frozen，RL 的四个 SNR 平均改善和后段减前段改善均为 0。
这只是样本内无 BER 变化，不证明总体零差异或零风险。

相对校准 FDE，RL 的 seed 级配对改善如下，单位为百分点：

| SNR | BER 改善 | 95% 配对区间 |
|---:|---:|---:|
| 0 dB | +0.40690 | [−5.42568, +6.23948] |
| 5 dB | +0.05859 | [−0.10781, +0.22500] |
| 10 dB | +0.05208 | [−0.06867, +0.17283] |
| 15 dB | +0.08789 | [+0.01701, +0.15877] |

在 0 dB，RL 相对 FIR 的均值反而较差 −0.79753 pp，区间跨零；不能声称
该档超过全部传统方法。5/10/15 dB 对 FIR/DFE 的较大优势，也不能替代对较强
FDE 的比较。所有区间以 5 个独立 seed 的轨迹均值为统计单位。

## 动作作用与结论边界

0/5 dB 的全部帧选择 identity；10 dB 为 32 帧 identity 和 8 帧 rollback_identity。
15 dB 有 16 帧选择 peft_head_fast，其中 9 帧通过 Pilot 验收；其余 24 帧为
identity。head 确实改变部分连续输出，逐帧 Data logit 平均绝对变化最高约 0.01250，
但最终 BER 仍与 Frozen 一致。identity 上的接受标记不算 PEFT 更新。

每条仅一个 8 帧窗口，策略在窗口结束才更新，没有下一个窗口执行更新后的
动作分布。因此这轮验证的是初始策略/动作及公平对照差距，不能据此断言长期
PPO 学习无效，也不能证明长期学习增益。没有因为这组结果改变传统参数或扩大失配。

## 验证与复现

统一入口在四 SNR 上复现接入前存档的前三帧传统结果；隐藏符号/标签污染、
实际信道预算不匹配、协议混用 resume 测试通过。完整回归 **669 passed，
273.67 秒**；随后新增配对统计守卫，与接入测试合跑 **13 passed，16.09 秒**。
两组存在重叠，不能相加为独立测试总数。
汇总工具拒绝重复/缺失方法、不同波形、不同协议和在线信息隔离声明失败的输入。
声明须结合既有隔离测试理解，不能单凭日志字段自证。

机器可读统计见[结果摘要](results/calibrated_compare_head256_20261101.json)，
包含全部方法的 seed 区间、动作计数、原始日志 SHA256 和比较 manifest。
运行命令见计划，汇总入口为：

```powershell
& D:/Research/RL4EQ/.venv-gpu/Scripts/python.exe scripts/summarize_calibrated_compare.py --input logs/calibrated_compare_head256_5s8f_20261101 --output logs/calibrated_compare_head256_summary_new.json
```

## 下一步

保留本轮作为同步、边界与当前 checkpoint 的统一诊断，不扩大同一 head 配置的
长期矩阵。下一轮核心仍是神经调制的作用能力和更新后 Data 泛化；调制位置/结构
选择等待用户已发出问题的答复。确定选择后，先用训练范围的机制实验检查主误差
能否被合法 Pilot 更新修正，再决定离线训练和长期 PPO 验证。
