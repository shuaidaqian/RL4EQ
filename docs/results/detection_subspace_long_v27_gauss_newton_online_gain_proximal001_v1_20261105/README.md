# 第二十七轮：prox=0.01下匹配配对增益离线训练

## 假设与控制

v26显示把GN近端权重降至0.01会扩大更新步幅，但沿用旧表示仍没有稳定收益。本轮固定v25的配对 `online_gain` 外层目标、2维Pilot上下文网络、阻尼GN、验证Pilot守门、锚点回退与收缩；唯一实验因素是将A/B离线训练与该prox=0.01在线内循环匹配。A为普通监督，B为配对Online增益元训练；二者使用相同64步预算、训练/选择数据及初始化。使用新的开发seed `121700/121701/121702`，四个既定SNR均完整评估，未改变信道、PDP、同步/漂移、Pilot、接收窗口或在线信息边界。固定清单为[配置](../../../../configs/audits/detection_subspace_long_v27_gauss_newton_online_gain_proximal001_v1.json)。

## 配对开发结果

增益定义为 `BER_frozen - BER_online`，正值代表在线更好。

| SNR | Frozen BER | A Online BER | A增益（95%区间） | B Online BER | B增益（95%区间） | B−A交互 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 dB | 0.17107477 | 0.17110189 | -2.71e-05 [-3.52e-04, +2.98e-04] | 0.17103408 | +4.07e-05 [-3.24e-04, +4.05e-04] | +6.78e-05 |
| 5 dB | 0.031467015 | 0.031453451 | +1.36e-05 [-1.97e-04, +2.24e-04] | 0.031412761 | +5.43e-05 [-6.25e-05, +1.71e-04] | +4.07e-05 |
| 10 dB | 0.001397027 | 0.001397027 | 0 | 0.001397027 | 0 | 0 |
| 15 dB | 8.1380e-05 | 8.1380e-05 | 0 | 8.1380e-05 | 0 | 0 |

B在0/5 dB有同向正点估计，且相对普通监督A的交互也为正；但B的95%区间均跨零，三个开发seed不足以证明可重复收益。10/15 dB无在线BER变化。故本轮是值得继续验证的低SNR机制信号，不是核心目标已达成，也不是正式确认。开发集每臂每SNR 96帧；所有逐帧记录、汇总和输入指纹见[`summary.json`](summary.json)及[`frame_metrics.jsonl.gz`](frame_metrics.jsonl.gz)。

## 只读更新诊断

最终接受的更新在0 dB，B臂有8帧Data BER下降、26帧不变、6帧上升；5 dB有5帧下降、26帧不变、2帧上升；10 dB的17帧全不变，15 dB无接受更新。该离线事后评分不参与在线控制，详见[`gate_alignment.json`](gate_alignment.json)。

在更新前系数处重建梯度后，B臂Adapt与Data梯度平均余弦在0/5/10/15 dB约为`0.285/-0.018/0.039/0.064`；GN候选的Data BCE平均下降量（正值表示下降）约为`+4.98e-5/-7.93e-5/-1.22e-5/接近0`。5 dB的Adapt方向与Data目标基本不对齐，且候选平均提高Data BCE；15 dB Adapt梯度均值约`2.62e-6`，而验证梯度近零、几乎不更新。A/B完整结果分别见[`gradient_alignment_A.json`](gradient_alignment_A.json)和[`gradient_alignment_B.json`](gradient_alignment_B.json)。Data梯度与标签仅用于离线事后诊断。

## 结论与下一步

匹配离线训练使B臂在0/5 dB相对A出现正交互点估计，是近期GN系列中首次呈现两个低SNR同向的开发信号；然而Online−Frozen区间跨零、逐帧获益稀少，且5 dB Adapt方向没有有效指向Data损失下降。主要瓶颈仍是合法Pilot更新方向与Data均衡目标不一致，而不是Pilot Jacobian秩不足（四SNR几何满秩）。下一步保留v27 checkpoint作冻结来源，先设计一项新的独立开发会话验证0/5 dB信号能否复现，并保留10/15 dB护栏；若复现失败，不继续扩展同一GN配置，转向可由合法Pilot辨识的更新目标/表示。Data标签不得用于该在线选择、更新、门控或回退。正式确认仍不启动。

## 运行与验收状态

- 768条配对开发记录，四SNR、A/B、三个开发seed完整。
- A/B训练均64步；在线更新2个神经上下文参数，增量更新平均约93–97 ms（不含完整接收链路与物理状态跟踪）。
- 清单、源码、checkpoint、缓存和逐帧结果指纹记录在[`summary.json`](summary.json)；为独立复核现已归档A/B checkpoint，训练特征缓存未复制。
- 修改后的完整pytest运行结果为`957 passed in 335.62s`；新增v27清单约束测试另经定向验证`1 passed`。
- `core_confirmed=false`、`confirmation_ready=false`；本轮不是独立确认。
