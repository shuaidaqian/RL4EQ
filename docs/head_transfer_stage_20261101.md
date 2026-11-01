# 神经 head 更新后泛化：256 步四臂结果

本阶段完成实现验证，但没有通过在线 BER 收益门槛。新增更新后 Data 目标没有
在四个主 SNR 上可靠地改善同 checkpoint 的在线表现，也没有可靠优于等样本量
的普通监督分支。因此不把这一 checkpoint 接入正式大矩阵。

## 固定协议

预先固定设置见[实验计划](head_transfer_plan_20261101.md)。A 为整帧 BCE，B 为
整帧 BCE 加上 Adapt 更新后的 Data BCE；两组从相同历史 checkpoint 出发，各
训练 256 步，batch=1、外层 lr=1e-4，训练 seed=34000–34255。动作沿用已有
`peft_head_fast`：head、Adapt-only BCE、AdamW、lr=5e-4、1 步、增量上限 0.05。

真实 Level B，记忆 116、帧长 1024、prefix Pilot=256（Adapt=224、Reward=32），
原 CFO/相位范围不变。两组使用 FFT 和 math attention；后者支持内循环的高阶
反传。测试为 heldout_edge、seed=43001–43005、每条 8 帧，共 320 条四臂记录。
每帧重置参数；固定动作只作组件消融，不代表正式 PPO 或累计学习。

## 独立测试

BER 以百分数表示；改善和区间以百分点表示。正改善表示 Online 优于自身 Frozen。
区间先对每个 seed 的 8 帧求均值，再按 5 个独立 seed 计算配对 t 区间。

| SNR | B Frozen BER | B Online BER | B 在线改善（95% CI） | B Online 相对 A Online 改善 |
|---:|---:|---:|---:|---:|
| 0 dB | 21.37044% | 21.36068% | +0.00977（−0.02405, +0.04358） | −0.02604 pp |
| 5 dB | 7.17773% | 7.17448% | +0.00326（−0.00578, +0.01229） | 约 0 pp |
| 10 dB | 1.04167% | 1.04167% | 0（样本内无变化） | +0.01302 pp |
| 15 dB | 0.22461% | 0.22461% | 0（样本内无变化） | −0.00326 pp |

B Online 相对 A Online 的所有 seed 级区间均包含零。B 在 0 dB 接受 39/40 次
更新，其中 8 次事后 Data BER 恶化；5 dB 接受 22/40 次；10/15 dB 分别接受
22/40、23/40 次，但没有改变任何 Data 硬判决。更新被接受不能作为 BER 收益证据。

0 dB 的 B 分支后段减前段增益区间为正，但本实验每帧重置，没有跨帧参数累计。
这个片段差异不构成“随在线学习积累而改善”的因果证据，且总体增益未通过门槛。
10/15 dB 的零宽经验区间也不能证明总体零风险。

## 事后作用幅度诊断

使用训练范围 offline_train 的新 seed=35000–35002、每条 4 帧，诊断 B checkpoint。
在线候选及 Reward 验收结束后才计算 Data 梯度；分析结果不进入在线控制。

| SNR | Pilot/Data 梯度余弦均值 | Data logit 平均绝对变化 | 修复/引入错误数 |
|---:|---:|---:|---:|
| 0 dB | 0.963 | 0.00302 | 1 / 2 |
| 5 dB | 0.853 | 0.00287 | 0 / 0 |
| 10 dB | 0.212 | 0.00256 | 0 / 0 |
| 15 dB | 0.135 | 0.00235 | 0 / 0 |

至少在该诊断内，低 SNR 的 Pilot/Data 梯度方向多为一致，但既有 head 动作对
Data 判决的影响很弱；中高 SNR 的方向还存在不确定性。基础模型的
`neural_residual_scale=0.1`，物理 warm-start 直接参与最终输出。因此，“仅换
离线损失便能使这个 head 动作产生明显 BER 收益”的假设没有得到支持。
这不是证明提高尺度就一定有效，也不能据此盲目放大步幅。

## 实现和验证

- 可微内循环支持已有 PEFT 分组和 BCE，与正式 PPO 的 head、adapter_lora_head
  AdamW 候选逐参数数值对照通过。
- 改动隐藏 Reward/Data 标签和波形不改变内循环更新；head 更新后的外层梯度
  通过有限差分检查。在线累计快照及投影也使用配置选择的参数组。
- 修复 CUDA 高效 attention 不支持二阶导数的问题，两训练分支统一 math attention。
  外层损失或梯度非有限时直接停止，避免导出坏模型。
- 全量回归 **656 passed，293.06 秒**；真实 1024/116 CUDA post-adapt 两步 smoke
  和事后诊断 CPU smoke 通过。A/B 训练分别约 193.8/287.0 秒；CPU 回归曾并行，
  耗时仅作资源记录，不作为严格速度基准。

机器可读结果见[阶段摘要](results/head_transfer_stage_20261101.json)。原始日志和
checkpoint 在本机 `logs/`、`pretrained/`，由 Git 忽略；版本库保留配置、脚本、
SHA256、完整实验 manifest 和精简统计。复现命令见计划文档；事后诊断命令为：

```powershell
$env:RL4EQ_LINEAR_OPERATOR_BACKEND = 'fft'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& D:/Research/RL4EQ/.venv-gpu/Scripts/python.exe scripts/diagnose_head_transfer.py --config configs/eme_long_memory_v2_head_transfer.json --pretrained pretrained/head_transfer_post256_20261101/model_best.pt --seed-start 35000 --num-seeds 3 --frames 4 --device cpu --output logs/head_diagnostic_post256_3s4f_new.json
```

## 下一步

暂停扩大该 head 配置的训练或评估矩阵。先验证神经更新能否在合法 Pilot 信息
边界内影响主要均衡误差，再确定是否调整调制位置/表示。任何新表示均需与普通
监督、相同 checkpoint Frozen 配对；若新增外层目标有收益，还需加入不含内循环
的同权重 Data 目标对照，排除单纯重加权的解释。

并行的必要工作是校准传统 baseline：现有 FIR 最多 41 taps、DFE feedback=5，
不能直接视作 116 记忆下的充分校准；SC-FDE 的整帧循环边界也与无 CP 线性信道
不一致。校准完成前，不用其较弱 BER 宣布系统目标完成。
