# 在线微调阶段进展（2026-11-06）

## 研究主张与当前结论

本阶段沿用冻结的 Level B `eme_long_memory_v2` 长回波配置：最大时延 116 符号、Pilot=256（Adapt=224、Reward=32）、0/5/10/15 dB。目标仍是离线学习可由合法 Pilot 适配的神经均衡器，并在线真实更新神经参数；只有独立 session 上 Online 相对同 checkpoint、相同物理状态跟踪协议的 Frozen 有可重复 Data BER 改善，且预先定义的时间窗支持总体增益扩大，才算支持研究主张。

当前离线模型相对所列传统 DFE-RLS/LMMSE-FIR 的强基线优势已在[离线训练与基线报告](offline_equalizer_training_and_baseline_20261105.md)归档。它建立的是当前开发矩阵内的离线优势，不自动证明在线增益。

此前 rank-1 Fisher-GN 机制在两个独立复核 session 上没有获得整体 Online−Frozen 正收益，接受更新中也有事后 Data 退化。因此不再扩大同一 rank-1 机制矩阵；失败结果保留为机制筛查证据，不推断一般在线微调不可行。详细数值见[rank-1复核结果](results/rank1_gn_meta_fisher_eval32_rep_v1_20261106/README.md)。

## 已完成：全 head trajectory 端到端 smoke

新增配置 `configs/eme_long_memory_v2_full_head_trajectory_dev_v1.json`，冻结来源强离线 checkpoint 后只允许全 head 在线/外层更新；离线 trajectory 训练展开连续4帧的同一会话在线过程，内循环使用当前帧 Adapt-only BCE、逐帧 Adam、独立 Reward Pilot 硬门控、参数跨帧保留及固定总半径。训练端关闭 Adam 状态跨帧持久化，历史长度为1，和当前评估器逐帧重建 Adam 的行为匹配。离线外层可以使用这些训练 episode 的 Data 标签优化多帧目标；部署更新与 Reward 门控不读取 Data 标签。

采用同一来源 checkpoint、同一训练配置和训练 seed，分别以 `--transfer-objective trajectory` 与 `--transfer-objective baseline` 跑2步烟雾测试。两臂均成功保存权重；随后在相同未见开发 session 76110、全部四个 SNR 上运行8帧 Frozen/Online 配对评估。结果文件分别位于：

- `results/full_head_trajectory_smoke_v1_20261106/eval.json`
- `results/full_head_supervised_smoke_v1_20261106/eval.json`

| SNR | Trajectory Online−Frozen收益（Frozen−Online BER） | 普通续训收益 | Trajectory晚窗−早窗 | 普通续训晚窗−早窗 | Trajectory接受/8 | 普通续训接受/8 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 dB | +6.51e-4 | +4.88e-4 | +1.30e-3 | -1.30e-3 | 8 | 8 |
| 5 dB | -1.63e-4 | -1.63e-4 | -6.51e-4 | -6.51e-4 | 4 | 4 |
| 10 dB | 0 | 0 | 0 | 0 | 0 | 0 |
| 15 dB | 0 | 0 | 0 | 0 | 0 | 0 |

**这些是单 session、每 SNR 8 帧的流程 smoke，不能用作增益证据或模型选择依据。** 0 dB两臂都出现正点估计，5 dB都为负，10/15 dB没有候选通过门控；数据量不足以判断 trajectory 是否优于普通续训，也不能证明随帧数增加的收益趋势。Smoke 的用途仅为确认训练、checkpoint载入、四点推理、在线更新、Reward守门与逐帧结果归档可以端到端运行。

## 已完成：matched A/B 开发筛查与决策

8步训练 A/B 与新 session 76120、76121 的16帧四 SNR 评估已完成，详细结果见[全 head trajectory 开发筛查](results/full_head_trajectory_dev8_v1_20261106/README.md)。Trajectory的四 SNR平均 Online−Frozen 收益依次为 `-1.63e-4/+1.22e-4/-4.07e-5/0`，普通监督续训为 `+2.44e-4/-4.07e-5/0/0`；trajectory相对普通续训的增益差没有跨SNR稳定为正。0 dB两个trajectory session的整体收益均为负；5 dB正点估计很小，且两会话CI很宽。早晚窗差为0 dB负、5 dB略正、高SNR为0，不能证明收益随帧推进扩大。共接受47/128个候选，5次trajectory接受后的Data BER事后恶化；这些值只作开发机制诊断。

因此不扩大当前4帧趋势损失配置，也不进入正式确认。随后对两个已训练 checkpoint 做了逐帧单步事后梯度诊断，细节见[全 head 梯度方向诊断](results/full_head_trajectory_dev8_v1_20261106/head_gradient_alignment_diagnostic.md)：0/5 dB Adapt/Data余弦约0.75/0.65且候选连续Data BCE下降，说明更新方向并非完全错误；但0 dB存在Reward接受后的Data有害候选，BER收益不稳定。10/15 dB余弦接近零、更新极少且Data变化很小，说明高SNR缺少有效更新信号。诊断每点仅2个session，且逐帧候选从起点重建，没有复现跨帧参数累计，所以只用于机制筛查。

下一步需要把当前评估轨迹逐帧记录扩充为动作后Data BCE（仅事后评分），并将Reward BCE/BER变化与Data BCE/BER变化配对，定位低SNR的门控失配和高SNR的低可辨识/低灵敏度。随后只选一个修复：先测试基于连续Reward统计量的安全滞回门控，或近期Adapt Pilot固定加权重放，不能同时改目标和更新器。每个候选仍与匹配普通监督续训和同源Frozen比较；所有在线控制只读Adapt/Reward Pilot，Data只用于离线外层训练和事后评分。趋势损失只是训练目标，不是趋势证据；统计单位仍为独立session，不能将帧/bit当作独立样本。

## 复现命令

```powershell
# trajectory 外层训练
python pretrain.py --config configs/eme_long_memory_v2_full_head_trajectory_dev_v1.json --stage pilot_transfer --steps 8 --batch-size 1 --transfer-objective trajectory --resume <离线checkpoint> --save-dir <输出目录> --device cuda

# 匹配的普通监督续训控制
python pretrain.py --config configs/eme_long_memory_v2_full_head_trajectory_dev_v1.json --stage pilot_transfer --steps 8 --batch-size 1 --transfer-objective baseline --resume <同一离线checkpoint> --save-dir <输出目录> --device cuda

# 每个续训checkpoint的相同 Frozen/Online 评估
python scripts/prototype_head_rank1_adapt.py --config configs/eme_long_memory_v2_full_head_trajectory_dev_v1.json --pretrained <续训checkpoint> --basis-seed 78001 --seed-start <未见session起始seed> --num-seeds 2 --frames 16 --device cuda --branches frozen,full_head_adam --output <结果JSON>
```

阶段修改后全仓测试为 `992 passed`；A/B归档之后还需做定向测试、diff检查和配置验证。本 worktree 尚未将阶段文件提交或推送；通过这些检查后将阶段文件提交到当前分支并推送远端 `RL4EQ`。
