# 持续均衡 v2 执行记录

目标依据：[用户确认的完整研究路线](研究点二_第一性原理审视与研究路线_20261119.md)。本文件只记录新分支事实与下一步，不替换历史物理快照的证据。

## 已完成的仓库操作

- `bayes-cir`：当前修正与研究路线已提交为 `9516241` 并推送到 `RL4EQ/bayes-cir`。
- 提交前全量测试：`python -m pytest -q -p no:cacheprovider`，160 passed。
- 已创建并推送 `continual-equalization-v2`，跟踪同名远端。
- 实验环境使用 `.venv-gpu/Scripts/python.exe`：PyTorch 2.11.0+cu128，GTX 1650。系统默认 Python 是 CPU 版本。

## 需求与进度

| 阶段 | 需求 | 当前状态 |
|---|---|---|
| P0 | 第一研究点到离散信道的物理映射、状态尺度和信息契约 | 已恢复原 MAT/代码并完成 PDP 数值及理想解扩映射；时间统计映射仍待补足 |
| P1 | acquisition 合法估噪、A/B/C 匹配对照、早期计分 | 四臂 3×32 开发完成，估噪尺度合理；尚未建立四 SNR 持续增长 |
| P2 | Q/持久化/soft Data 贡献与可靠性 | 圆周同步四臂已完成；低效 Q 筛查保留部分结果，向量化验证后用新 seed 继续 |
| P3 | 闭环离线重训、多训练重复、神经贡献与协同 | 三重复同源微调完成；低效验证保留部分结果，尚未建立完整验证收益 |
| P4 | 物理边界与长会话、功效与饱和容差冻结 | 未开始 |
| P5 | 四 SNR 自身改善、优势增长、饱和与保持的新独立确认 | 未开始 |
| P6 | 成本报告与实现优化；明确预算后的实时验收 | 未开始 |

## 实验规则

- 所有新实验先在 `docs/protocols/continual_v2_seed_registry.json` 登记，检查已有结果 seed 和保守历史保留范围。
- 同一实验内同波形 A/B/C/D 比较属于预先登记的配对设计。不得把已经运行的开发 seed 变成独立确认。
- 输出记录开发身份、配置/源码/checkpoint 指纹、噪声来源、时序和前向次数；中断保留部分结果，不覆盖重跑。
- 新接收模块只接受已剥离标签、真实状态和未来帧的接收视图。评分在接收状态更新之外完成。
- 新方案先使用未归一化有效 CIR，避免直接投影后协方差的解释问题；旧归一化方法保持为历史锚点。初始相位由 acquisition CIR 吸收，当前公共相位由 Adapt 估计。

## 最近决策

先用既有 checkpoint 检验合法估噪与递归物理状态，不立即扩网络。当前 Pilot 先更新再单次解码；Data 后验只作用于后续帧，禁止重复计入同一批 Pilot 信息。尚未证实新研究目标。

## 当前实现与验证

- [物理映射审计](continual_v2_physical_mapping_20261119.md)区分第一研究点、雷达包络与当前构造性假设；核查生成器创新与历史均匀 Q 的量级差异，尚不据此更换超参数。
- `training/continual_receiver.py`：严格接收视图、acquisition CFO 粗网格＋局部细化、未归一化 LS 残差估噪、有效复 CIR、A/B/C/D 前缀 Pilot 时序及一次前向。
- `evaluation/seed_registry.py`：扫描历史结果并登记实验，拒绝种子冲突和隐式重跑。
- `evaluation/continual_summary.py`：以会话为单位分别汇总自身改善、配对增量和时间变化；拒绝未完成/缺帧/波形不匹配的完整结论。
- 新增噪声尺度、线性高斯递归一致性、隐藏标签隔离、种子重复和相对优势误归因测试。
- 本阶段全量测试：`python -m pytest -q -p no:cacheprovider`，**170 passed**。不以单元测试替代研究收益验证。

## 首轮已完成的开发实验

实验名 `p1_four_arms_estimated_noise_3x32`，seed 210001–210003，32 帧，四 SNR、四接收臂，全部是开发数据。精确设定噪声只保存为事后比值诊断；模型条件和所有后验使用 acquisition 估计。

```powershell
$env:PYTHONUTF8=1
& ./.venv-gpu/Scripts/python.exe scripts/develop_continual_receiver.py --checkpoint pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --seeds 210001 210002 210003 --frames 32 --experiment p1_four_arms_estimated_noise_3x32 --output docs/results/continual_v2_p1_20261119/four_arms_3x32.json --device cuda
```

**上述为已登记且已完成的命令，不能再次执行。** 以下只读汇总已生成结果：

```powershell
python scripts/summarize_continual_receiver.py --input docs/results/continual_v2_p1_20261119/four_arms_3x32.json
```

原工具 exec session `51194` 已结束；完整结果 1536 行，状态 completed，逐会话与窗口统计见同目录 `_summary.json` 和 `_report.md`。

截至本次中间记录，0 dB 的前两个完整会话已出现相反方向的 Pilot 递归增量，不能据首个会话升级方案；噪声估计均接近生成方差。先完成全部预定会话，再结合状态误差/协方差诊断决定修正，不删除负向 seed。

完整结果平均 Data BER：

| SNR | acquisition | 当前 Pilot | 递归 Pilot | 递归 soft |
|---|---:|---:|---:|---:|
| 0 | 0.232042 | 0.181546 | 0.172499 | 0.169800 |
| 5 | 0.048164 | 0.045573 | 0.043783 | 0.041802 |
| 10 | 0.004869 | 0.004829 | 0.004801 | 0.004055 |
| 15 | 0.000583 | 0.000583 | 0.000705 | 0.000570 |

估噪/真值比四 SNR 均值约 1.005–1.006。0 dB 递归 Pilot 相对当前 Pilot 的增量异质，95% 会话区间跨零；10 dB soft 相对递归 Pilot 的增量在三个会话同向且未校正区间为正，但不能升级为确认。15 dB 递归 Pilot 平均变差，soft 自身早期减晚期 BER 为 -0.000543，区间 [-0.001009, -0.000076]，当前实现不支持四 SNR 自身改善目标。**不通过增加样本掩盖方法问题；先诊断同步、过程噪声和网络条件。**

## 同步诊断与下一步实现

新增 `training/continual_phase.py`，在 acquisition CFO 给定时用复圆周相关估计公共相位；并提供全合法 Adapt 参考与不依赖 tail 的完整已知卷积行两种候选。没有使用 Reward 或 Data 标签拟合同步。旧解缠绕中位数仍是默认锚点，不据理论最优性宣称实际接收收益。

已完成首帧诊断 `p1_phase_first_frame_12sessions`，全新 seed 211001–211012 × 四 SNR。所有相位估计结束后，才用真实符号计算 Reward/Data 预测误差。0 dB 的 Data 预测 MSE：旧方法 1.403319，圆周相关全 Adapt 1.175302；配对下降 0.228017，未校正 95% t 区间 [-0.072148, 0.528182]。5/10/15 dB 差异很小且区间跨零。**尚不能认定 BER 改进或递归稳定性成立**。结果见 `docs/results/continual_v2_phase_20261119/first_frame_12sessions.json`。

新增 `scripts/compare_continual_variants.py` 以清单预先声明 checkpoint、状态臂、同步方法、线性算子后端和指定配对差；保存实际前向调用计数及配对 logits/判决差。物理分支消融保留同一已学习同步，仅移除 Transformer 修正，不能称完全非神经基线。

`p1_backend_full_receiver_1x8` 已在 CPU 完成，seed 214001、四 SNR、8 帧，配对 acquisition/递归 Pilot 两种接收轨迹的 direct/FFT 实现。不减 CG 迭代数，128 行结果中已记录 Data 判决全部一致，配对 logits 最大绝对差 0.000309；FFT 平均耗时约为 direct 的 55%–64%，仍不实时。该筛查与 GPU 实验并发，成本仅作初步诊断，需后续 GPU 核查；一会话不能证明任意轨迹数值或判决等价。启动源码按原记录 SHA 逐字节校验后归档在同目录 `source_snapshot.json`，没有把启动后的实现反写进该实验。

已实现 `training/continual_offline.py` 与 `scripts/train_continual_offline.py`：匹配 acquisition 条件训练与递归 Pilot 条件训练，同会话波形、同初始化、同训练帧与梯度次数；每会话接收时固定权重，轨迹完成后用独立保存的 Data 标签训练。实际模型输入被快照，监督不进入接收视图；历史状态/soft-tail 截断梯度。**已完成三重复微调，尚未证明协同**。训练权重保存在忽略目录，报告及权重 SHA 保存到研究结果目录。

全量测试 **179 passed**；定向验证覆盖原接收机契约、圆周相位恢复与未知标签隔离、输入快照、离线梯度更新、物理支路等价及真实调用计数。新增相同实际上下文的物理支路评分，复用神经臂已经生成的 CIR/相位/tail，避免把独立 soft-tail 轨迹差异误当纯 Transformer 增量。后续修改未注入正在运行的 P1 进程；P1 使用启动时导入的实现，源码 SHA 已记录，准确对应提交 `0929c06` 中的接收模块。

GPU 后端核查 `p1_backend_gpu_full_receiver_1x4` 已完成，seed 214002、四 SNR、4 帧；64 行中已记录 Data 判决全部一致，最大 logit 差 0.000439。FFT 总耗时约 544–624 ms/帧，direct 约 962–1310 ms/帧。这是短程开发成本，仍不实时；CPU 小筛查更快，后续机制筛查先用 CPU，正式成本另作无并发测量。

## 新完成结果与计算优化

1. `p2_circular_phase_four_arms_3x32` **已完成**：seed 215001–215003、四 SNR、32 帧，CPU；清单 `docs/protocols/continual_v2_phase_ber_screen_20261119.json`，输出 `docs/results/continual_v2_phase_ber_20261119/paired_3x32.json`，2304 行。圆周 C 相对旧同步 C 平均 BER 改善为 +0.046183/+0.000705/-0.000068/-0.000014；10 dB 有小幅退化，不能宣称四 SNR 一致改进。圆周 C 相对 B 的增量在 0/10 dB 的未校正会话区间为正；15 dB 不支持递归增量。0 dB 在共享相同实际输入时，Transformer 修正平均反而增加 0.000095 BER；10/15 dB 未观察到判决差，**没有等价精度保证**。该结果支持继续开发同步，但未完成增长至饱和目标。
2. `p3_matched_offline_finetune_3repeats_32updates` **已完成**：环境 seed 212001–212024、训练顺序 seed 401001–401003；每重复 8 会话 × 8 帧、每会话训练 0/1/3/7 帧，四 SNR 各两会话。两训练方式每重复均为 64 次 rollout 前向＋32 次梯度前向，同一既有 checkpoint 起点、独立自身 tail；圆周同步、FFT，GPU。报告 `docs/results/continual_v2_offline_20261119/training.json` 共 48 训练行、6 权重，权重在忽略目录 `pretrained/continual_v2_p3_20261119/`。只属于同源微调开发，不等同从头重训多初始化。

发现 `_pilot_reference_from_cir` 逐 tap 循环与设备同步是新的成本瓶颈。圆周候选已经改为批量构造合法参考，保留原实现作显式比较。合成输入 CPU 小基准：相位特征由约 59.4 ms 降至 2.38 ms；长记忆、稀疏 tap、非零 soft-tail 的定向数值检查通过，尚待完整新会话轨迹验证。未减少 Pilot、状态维度或 CG 次数。最新全量测试 **182 passed**。

以下两项开发实验因计算开销主动中止，已停止对应进程，原数据与全部登记 seed 均保留，不生成完整统计结论：

- `p2_q_physics_10x64`：seed 220001–220010，输出 `docs/results/continual_v2_q_20261119/physics_10x64.json`，status=interrupted。已保存会话显示 0 dB 小 Q 值得进一步检验，不能据部分数据升级结论。
- `p3_matched_finetune_validation_2x16`：seed 217001–217002，输出 `docs/results/continual_v2_offline_20261119/validation_2x16.json`，status=interrupted。保留已保存的 0 dB 等数据，不并入下一次验证。

中止由确定的计算瓶颈触发，没有按效应方向停止；两项后续都使用新 seed，不能覆盖文件或复用旧 seed。下一步先按 `docs/protocols/continual_v2_vectorized_reference_check_20261119.json` 验证完整接收轨迹；通过后再用新 seed 继续 Q 开发与匹配微调验证。原 exec session 68402/52612 已停止，不应继续等待它们作为活实验。

以上成本部分来自并发资源条件，不能当作最终独占资源成本。均未使用新确认数据。每次续接先检查同一活进程和结果，不只凭 JSON running 状态判断；完整汇总只接受 completed。两臂训练所见波形/预算预先匹配，不能把模型整体变好直接归因于离线在线协同。统计汇总对会话样本方差为零的差值不再给出零宽 t 区间，明确标记无法由此证明总体等价。

## 向量化核查完成后的后续入口

`p2_vectorized_reference_full_trajectory_1x16` 已完成：新 seed 221001、四 SNR、16 帧，递归 Pilot/soft 各自旧参考构造与批量构造配对，共 256 行。本样本 logits 最大差 **0**、Data 判决差 **0**，通过事先设置的 max|Δlogit|≤0.005 且 Data 判决相同的开发门槛。不能将本样本一致性外推成任意输入逐位保证。CPU 神经整臂耗时约由 237–305 ms 降为 164–219 ms，物理支路的主要同步开销也已消除。

新增 `evaluation/continual_training_summary.py` 及汇总脚本，计算 `(B-C)_闭环训练-(B-C)_参照训练`，分别以接收会话、训练重复为统计轴，不把三个模型在同一会话上的结果算作三个独立会话。只生成开发描述，没有联合总体置信区间。

后续新实验增加严格事后评分的幅度、相位对齐 NMSE、BCE/Brier 与高置信错误计数，避免最优复幅度对齐 NMSE 掩盖增益错误，或把高置信输出直接当作可靠 soft 反馈。真值只传入评分函数，不进入接收模块；新增代码有对应隔离/统计回归检查。

接下来使用新 seed 继续相同 Q 开发和匹配微调验证；此前中断结果保留。离线训练本身已经完成，不再次训练或复用其 seed。P4/P5 的长会话、判据、样本量与独立确认均尚未完成。

## 2026-11-19 续接记录

`p3_matched_finetune_validation_vectorized_2x16` 已完成：使用新 seed 222001–222002，四 SNR、16 帧、14 个候选，共 1792 行；完整结果为 `docs/results/continual_v2_offline_20261119/validation_vectorized_2x16.json`，SHA-256 `627a64bcbb1902582d88b9b35157b4e3aaf92f745752e3986ec731d6c1a04764`。清单 `docs/protocols/continual_v2_offline_validation_20261119.json` 和结果身份已登记到 seed registry。分别按训练重复轴与接收会话轴统计后，离线 acquisition 条件训练和递归 Pilot 条件训练的交互很小、方向不一致，未建立稳定协同证据；15 dB 的递归增量仍多为负。该结论只适用于同源 checkpoint 的小预算微调验证，不能排除充分重训或新结构。

同批结果的推理成本约 63–75 ms/帧，运行时有并发负载，只作开发测量，不视为实时验收。完成本轮结果后全量测试为 186 passed。

`p2_q_physics_vectorized_10x64` 正在用 CPU 运行，seed 223001–223010、10 会话×64 帧×四 SNR×六臂；使用 `physics_branch` 的物理机制筛查，不等价于神经整臂确认。续接时进程仍活跃，SNR=10 dB 正在输出，0/5 dB 已完成。初步小 Q 结果优于旧 Q，soft 相对 Pilot-only 的增量仍需结合四 SNR 完整统计和失败会话诊断；不据此提前升级结论。当前 exec session 为 21468，不得重启或复用这些 seed。最近一次推送已将分支 `continual-equalization-v2` 推至 `150cf17`；本段、微调验证结果和其登记尚待提交，运行中的 Q 结果不应纳入提交。
