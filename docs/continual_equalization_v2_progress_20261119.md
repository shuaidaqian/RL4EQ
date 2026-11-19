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
| P0 | 第一研究点到离散信道的物理映射、状态尺度和信息契约 | 已完成来源/抽象审计，精确 PDP 和时间统计映射仍待补足 |
| P1 | acquisition 合法估噪、A/B/C 匹配对照、早期计分 | 四臂 3×32 开发完成，估噪尺度合理；尚未建立四 SNR 持续增长 |
| P2 | Q/持久化/soft Data 贡献与可靠性 | 圆周同步四臂与共享上下文神经分支筛查运行中，Q 尚待独立诊断 |
| P3 | 闭环离线重训、多训练重复、神经贡献与协同 | 已启动同源微调三重复的匹配 acquisition/递归 Pilot 训练，尚无验证收益 |
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

已实现 `training/continual_offline.py` 与 `scripts/train_continual_offline.py`：匹配 acquisition 条件训练与递归 Pilot 条件训练，同会话波形、同初始化、同训练帧与梯度次数；每会话接收时固定权重，轨迹完成后用独立保存的 Data 标签训练。实际模型输入被快照，监督不进入接收视图；历史状态/soft-tail 截断梯度。**尚未启动重训或证明协同**。训练权重保存在忽略目录，报告及权重 SHA 保存到研究结果目录。

全量测试 **179 passed**；定向验证覆盖原接收机契约、圆周相位恢复与未知标签隔离、输入快照、离线梯度更新、物理支路等价及真实调用计数。新增相同实际上下文的物理支路评分，复用神经臂已经生成的 CIR/相位/tail，避免把独立 soft-tail 轨迹差异误当纯 Transformer 增量。后续修改未注入正在运行的 P1 进程；P1 使用启动时导入的实现，源码 SHA 已记录，准确对应提交 `0929c06` 中的接收模块。

GPU 后端核查 `p1_backend_gpu_full_receiver_1x4` 已完成，seed 214002、四 SNR、4 帧；64 行中已记录 Data 判决全部一致，最大 logit 差 0.000439。FFT 总耗时约 544–624 ms/帧，direct 约 962–1310 ms/帧。这是短程开发成本，仍不实时；CPU 小筛查更快，后续机制筛查先用 CPU，正式成本另作无并发测量。

## 当前活实验与续接

1. `p2_circular_phase_four_arms_3x32`：seed 215001–215003、四 SNR、32 帧，CPU；清单 `docs/protocols/continual_v2_phase_ber_screen_20261119.json`。exec session **6537**，输出 `docs/results/continual_v2_phase_ber_20261119/paired_3x32.json`。包括旧同步递归 Pilot、圆周同步 A/B/C/D、共享 C 实际模型输入的物理支路评分。不能重复执行。
2. `p3_matched_offline_finetune_3repeats_32updates`：环境 seed 212001–212024、训练顺序 seed 401001–401003；每重复 8 会话 × 8 帧、每会话训练 0/1/3/7 帧，四 SNR 各两会话。两训练方式每重复均为 64 次 rollout 前向＋32 次梯度前向，同一既有 checkpoint 起点、独立自身 tail；圆周同步、FFT，GPU。exec session **30089**，报告 `docs/results/continual_v2_offline_20261119/training.json`，权重在忽略目录 `pretrained/continual_v2_p3_20261119/`。只属于同源微调开发，不等同从头重训多初始化，也未评价验证集。不能重复执行。

P2 CPU 筛查与 P3 GPU 微调同时运行，性能结论可分别评价，当前耗时不能当作最终独占资源成本。两者都未使用确认数据。每次续接先检查同一活进程和结果，不只凭 JSON running 状态判断；完整汇总只接受 completed。两臂训练所见波形/预算预先匹配，不能把模型整体变好直接归因于离线在线协同。
