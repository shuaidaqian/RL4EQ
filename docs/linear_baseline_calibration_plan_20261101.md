# 传统线性接收机的记忆、同步和块边界校准

## 目的与边界

当前代码把 FIR 长度限制为 41、DFE feedback 固定为 5，而主信道记忆为 116；
SC-FDE 直接采用整帧循环卷积，但环境没有 CP。这些事实说明还需校准，不能单凭
维度较小就断言增加 taps 必然降低 BER：有限前缀 Pilot 还限制估计的自由度。

本阶段固定信道与资源，不使用神经接收机结果选择传统参数。所有参考只读取
acquisition CIR/CFO、Adapt Pilot、接收信号和自身历史判决。Data 标签只在接收机
全部更新完成后用于离线评分。Reward Pilot 不作为已知输入。代码不包含神经网络
或 RL，也没有真实 CIR 或真实 tail 注入。

新增的纯线性块 MMSE 只作校准诊断，尚不改变主方法组。它与既有
`Fixed CG-BPSK-DD Block Detector` 不同：仅求线性高斯估计，不作硬判决迭代、
局部 bit 搜索或 decision-directed CIR 更新。Perfect-CSI 和旧 DD 块检测器仍只
作诊断参考。是否采用新增传统方法进入最终主对照，应在正式比较前明确固定。

## 预先列出的候选

所有候选采用与当前神经接收机相同的 acquisition CFO 先验，并从当前 Adapt
估计公共相位。跨帧历史由各自接收估计生成，不使用真实前帧发送符号。

- FIR：长度 41/117/233，各配 ridge 倍率 1/16/224；不假设越长越好。
- DFE-RLS：原 41+5 与记忆覆盖的 117+116；均按现有传统 RLS 规则更新。
- 原循环 SC-FDE 与去除已估计历史贡献后的零填充 Wiener FDE 近似。
  零填充近似不声称等于有限观测的精确 MMSE 解。
- 精确线性卷积块 MMSE：复数先验与实 BPSK 线性先验，已知 Adapt 坐标约束。
  实先验正规方程为 Re(HᴴH)+sigma²/2，不包含非线性 BPSK 判决反馈。
  CG 最多 128 步，容差 1e-6；重算真实正规方程残差并报告未收敛数。

首轮只在 `offline_train`、seed=36000–36002、每条 4 帧、0/5/10/15 dB 校准。
单 seed/单帧 smoke 只检查入口，seed=36000 的 smoke 样本会在校准中重复，
不能另作独立证据。不使用 `heldout_edge` 选择候选或改变上述网格。
若需冻结参数，按各 SNR 的校准 seed 平均 Data BER 选择 FIR 参数和 DFE 配置，
在新的 heldout_edge seed 上核验；校准成绩不能充当最终主实验成绩。

校准完成后，按上述规则已生成
[`linear_baseline_frozen_selection_20261101.json`](results/linear_baseline_frozen_selection_20261101.json)。
后续核验在运行前固定为 heldout_edge、seed=46001–46005、每条 8 帧。
核验只执行冻结的 FIR/DFE 和四个不调参的频域/线性块参考。代码在 heldout_edge
拒绝没有冻结设置的网格搜索，并校验设置的配置 SHA256。

## 验证与入口

5 项数值测试已通过：独立稠密 Toeplitz 最优解、复数/实先验、非零历史、已知
Pilot 约束、污染未知发送符号不变、direct/FFT 两算子以及 116 长回波。
现有主比较行为暂时不变。

```powershell
$env:RL4EQ_LINEAR_OPERATOR_BACKEND = 'fft'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& D:/Research/RL4EQ/.venv-gpu/Scripts/python.exe scripts/calibrate_linear_baselines.py --config configs/eme_long_memory_v2_head_transfer.json --seed-start 36000 --num-seeds 3 --frames 4 --output logs/linear_baseline_calibration_3s4f_20261101.json
& D:/Research/RL4EQ/.venv-gpu/Scripts/python.exe scripts/calibrate_linear_baselines.py --config configs/eme_long_memory_v2_head_transfer.json --seed-start 46001 --num-seeds 5 --frames 8 --state-split heldout_edge --selection docs/results/linear_baseline_frozen_selection_20261101.json --output logs/linear_baseline_confirmation_5s8f_20261101.json
```
