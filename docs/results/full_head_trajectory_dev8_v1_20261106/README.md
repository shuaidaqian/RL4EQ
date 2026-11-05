# 全 head 多帧 trajectory 与普通续训配对开发筛查

## 固定条件与方法

固定 Level B `eme_long_memory_v2`、116符号长回波、Pilot=256（Adapt=224、Reward=32）、0/5/10/15 dB。两训练臂均从相同强离线 checkpoint `8cd3912d1c957c9689c51730aff9d27a8f74942186c802a7483e5781af65aec4` 开始，使用相同配置、训练 seed 76000、SNR/连续帧 episode 采样顺序及8个外层步。

- **普通续训 A**：全 head 常规监督 BCE 续训。
- **trajectory 续训 B**：只对全 head 做可微外层更新；内循环复现逐帧 Adam 的 Adapt-only BCE、32个独立 Reward Pilot 的硬门控、通过后跨帧保留参数、总半径投影和 soft-tail。轨迹长度4帧；优化器状态每帧重置，Adapt历史长度1，与评估器一致。外层损失包含4帧的更新后 Data BCE 和早晚趋势项。

两臂都在新 session 76120、76121上评估，每个SNR连续16帧；同一checkpoint内Frozen和Online使用相同波形及相同物理状态跟踪协议。Data标签只在离线外层目标及在线动作完成后的事后BER评分中出现，不参与在线更新、门控或回滚。逐帧原始数据、权重、配置和来源指纹保存在各臂的结果目录；每个方法结果为 `eval16.json`。

## 结果

增益定义为 `Frozen BER − Online BER`，正数表示在线更好。早/晚窗差为最后4帧窗增益减最初4帧窗增益。表内均值按两个独立session计算，描述性95%区间不具备正式推断效力。

| SNR | Trajectory平均增益 | Trajectory描述性95%区间 | Trajectory晚−早 | 普通续训平均增益 | 普通续训晚−早 | 接受数 B/A | 有害接受数 B/A |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 dB | -1.63e-4 | [-1.20e-3, 8.71e-4] | -3.26e-4 | +2.44e-4 | [-1.82e-3, 2.31e-3] | 25/25 | 5/4 |
| 5 dB | +1.22e-4 | [-3.95e-4, 6.39e-4] | +1.63e-4 | -4.07e-5 | [-1.59e-3, 1.51e-3] | 15/15 | 0/1 |
| 10 dB | -4.07e-5 | [-5.58e-4, 4.76e-4] | 0 | 0 | [0,0] | 3/3 | 0/0 |
| 15 dB | 0 | [0,0] | 0 | 0 | [0,0] | 4/4 | 0/0 |

Trajectory相对普通续训的 Online−Frozen 增益差（B−A）依次为 **-4.07e-4、+1.63e-4、-4.07e-5、0**。没有跨四个SNR一致的在线增量。两个session在0 dB的trajectory总收益都为负；5 dB均为正但幅度很小，且接受候选没有事后Data有害提交。早晚窗趋势不稳定：0 dB为负，5 dB略正，高SNR无误码差异。两session下的区间很宽/退化为0，不能据此下统计结论。

Reward可接受不等于神经更新产生Data收益；高SNR几乎没有BER变化，即使更新被接受也不代表持续适配收益。2个开发session不能验证跨会话重复性，且窗口误码离散，帧不是独立统计单位。

## 判定

这轮全 head trajectory 训练和普通监督续训均未建立核心在线增益或随传输帧数扩大收益的证据。Trajectory没有稳定优于普通续训，故**停止扩展当前4帧趋势损失配置的seed矩阵**。结果仅否定本轮具体训练/更新搭配，不否定目标本身。下一轮优先检查Adapt→Data梯度/方向与Reward→Data门控一致性；然后固定候选更新器，测试近期Adapt Pilot加权重放，或小维度预条件更新。不得更换信道/Pilot协议，也不得用Data标签在线控制。

## 同一累计轨迹的连续 Data BCE 复核

修订后的评估器在已完成 Reward 门控和参数提交后记录 Data BCE，只用于离线评分；用相同checkpoint、相同session和相同seed重跑，BER和参数提交结果与原始 A/B 评估相同。下面列出每个session的 Frozen BCE − Online BCE 均值及最后4帧减最初4帧变化，正值表示 BCE 改善。该项并非新的训练或控制输入。

| SNR | Trajectory BCE增益 | Trajectory晚−早 | 普通续训BCE增益 | 普通续训晚−早 | 两臂BER观察 |
|---:|---:|---:|---:|---:|---|
| 0 dB | +8.03e-4 | +6.25e-4 | +8.10e-4 | +6.18e-4 | Trajectory两个session BER总收益均为负；普通续训点估计为正 |
| 5 dB | +2.22e-4 | +2.91e-4 | +2.22e-4 | +2.91e-4 | 两臂BER点估计略正，session间差异小 |
| 10 dB | +1.07e-6 | +1.59e-6 | +1.06e-6 | +1.58e-6 | BER近乎完全不变 |
| 15 dB | +8.85e-7 | +1.17e-6 | +8.85e-7 | +1.17e-6 | BER完全不变 |

这说明低SNR下在线更新会改善大多数 Data 符号的连续置信损失，并且收益在后窗更大；少数靠近判决边界的符号仍可能翻错，使0 dB整体 BER 无改善或退化。连续损失不能替代用户要求的 BER 验收。Trajectory与普通续训的BCE收益几乎相同，故当前未显示 trajectory 学到额外在线能力。高SNR几乎没有可测的更新效应。两个session不足以推断趋势或总体效应。

据此下一项单因素机制优先测试跨帧 Reward Pilot 验收：固定Adapt候选、参数范围和步长，将当前帧32个 Reward 标签门控与最近固定窗口内历史Reward Pilot聚合门控比较。它可能降低单帧小样本验证的波动；由于信道可能变化，窗口长度必须预先固定并报告过时风险。Data只用于动作后 BER/BCE 评分，不得参与窗口选择、接受、回滚或超参数挑选。

## 复现

```powershell
python pretrain.py --config configs/eme_long_memory_v2_full_head_trajectory_dev_v1.json --stage pilot_transfer --steps 8 --batch-size 1 --transfer-objective trajectory --resume <离线checkpoint> --save-dir <trajectory目录> --device cuda
python pretrain.py --config configs/eme_long_memory_v2_full_head_trajectory_dev_v1.json --stage pilot_transfer --steps 8 --batch-size 1 --transfer-objective baseline --resume <同一离线checkpoint> --save-dir <普通续训目录> --device cuda
python scripts/prototype_head_rank1_adapt.py --config configs/eme_long_memory_v2_full_head_trajectory_dev_v1.json --pretrained <续训checkpoint> --basis-seed 78001 --seed-start 76120 --num-seeds 2 --frames 16 --device cuda --branches frozen,full_head_adam --output <结果JSON>
```
