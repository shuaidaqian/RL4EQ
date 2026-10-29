# PEFT 同帧排序与未见 seed 诊断（2026-10-28）

## 目的和边界

目标是固定离线 checkpoint，仅用 Adapt Pilot 更新均衡器 PEFT 参数，Reward Pilot 仅用于验收和回滚，Data 波形和标签均不进入在线更新及决策。此处使用的历史比较入口在候选通过 Reward 守门后，**还按 Reward 分数选择候选**（见 `compare.py` 的 `reward_ranking_score > best_reward_ranking_score`）。因此下面的两组结果只用于诊断历史实现，不能作为满足目标边界的方法验证。日志中的 Data 隔离标记是自报告；相关扰动测试进一步检查了隔离，标记本身不构成证明。

原汇总将不同帧的候选混合计算 Spearman：即使同帧候选排序全部相反，也可能出现正相关。新增同帧候选两两排序，平分不算正确或错误；同时分别统计帧数、候选对数和 seed。此指标不含不更新的 identity，只是事后诊断，候选对也不是独立信道轨迹。原混合相关性字段仅供历史对照，其 `gate_pass` 不能用作方法的排序准入结论。汇总中 `actual_selected` 仅覆盖有候选记录的帧，整条轨迹的 BER 应配对原始逐帧记录。

## 固定条件

- Level B `eme_long_memory_v2`，delay 116，prefix Pilot 256（Adapt 224、Reward 32）。
- checkpoint `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`，SHA256 `8CD3912D1C957C9689C51730AFF9D27A8F74942186C802A7483E5781AF65AEC4`。
- seed 5、6、7，各 4 帧；SNR 0 和 5 dB。每档 3 条轨迹、12 帧，只有每条轨迹第 1 帧记录了三个候选。
- 四窗口 Reward、`fixed` 调度、`pilot_cir_phase`、固定 CIR；`freeze_below_snr_db=-1` 是为了检查 0 dB 而临时解除冻结的诊断设置，不能视为默认协议。
- gap 0 s 是本次零间隔参考；gap 600 s 是强失配诊断，不能合并为主场景平均。

运行模板（将 `GAP` 替换为 `0` 或 `600`，输出目录分别命名）：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe compare.py --config configs/online_phase_signal_state_embedding_formal_moderate.json --pretrained D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --methods 'Frozen Offline NN' 'Pilot-Driven Online Adaptation' --delays 116 --snrs 0 5 --seed-start 5 --num-seeds 3 --frames 4 --pilot-total 256 --reward-pilot-total 32 --pilot-layout prefix --acquisition-to-data-gap-seconds GAP --online-freeze-below-snr-db -1 --online-reward-windows 4 --scheduler fixed --cir-update fixed --online-condition-source pilot_cir_phase --device cuda --output-dir OUTPUT_DIR
D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe scripts/summarize_pilot_ranking_replay.py --input OUTPUT_DIR --output OUTPUT_DIR/ranking_summary.json
```

原始日志保存在本工作树的忽略目录 `logs/peft_unseen_low_snr_gap0_4w_3s4f_20261028/` 和 `logs/peft_unseen_low_snr_600s_4w_3s4f_20261028/`，不会随 Git 提交。

## 结果

| gap | SNR | Frozen 平均 BER | Online 平均 BER | 逐帧改善 / 持平 / 变差 | 同帧 BER 排序一致率 |
|---|---:|---:|---:|---:|---:|
| 0 s | 0 dB | 0.17849 | 0.17849 | 0 / 12 / 0 | 0.333，3 帧、9 对 |
| 0 s | 5 dB | 0.04481 | 0.04481 | 0 / 12 / 0 | 0，1 帧、2 对；另有 7 对 BER 并列 |
| 600 s | 0 dB | 0.38368 | 0.38043 | 3 / 8 / 1 | 0.556，3 帧、9 对 |
| 600 s | 5 dB | 0.35905 | 0.35341 | 1 / 11 / 0 | 0.667，3 帧、9 对 |

零间隔两档每条轨迹首次更新均产生 3 个候选，全部未通过窗口验收，因此 24 帧均未应用 PEFT 更新。600 s 时每档各有一次应用更新，其后参数可能跨帧保留或回滚，故逐帧收益不能只统计执行更新的帧。该强失配样本的绝对 BER 很高，不能解释为在线方法已具备实用性能；仅靠当前统计无法确定造成高 BER 的唯一原因。

两组各 24 条在线行均显式记录 Data 波形和标签未用于在线决策。数据有限，不能推断其他 SNR、seed 或长期稳定性，也不能据此宣称 PEFT 达成目标。

## 固定单候选边界测试

在查看本组结果前，固定已有的 `phase_signal_moderate` 为唯一候选，使用新的 seed 8、9、10；其余配置与上述零间隔测试相同。单候选无需通过 Reward 比较不同候选，Reward 用于验收和跨帧回滚；入口仍会计算历史 Reward 分数字段。此测试验证一个可运行的边界设置，不代表已经解决多候选排序。

| SNR | Frozen 平均 BER | Online 平均 BER | 逐帧改善 / 持平 / 变差 | 更新 / 回滚次数 |
|---:|---:|---:|---:|---:|
| 0 dB | 0.19737 | 0.18924 | 5 / 7 / 0 | 2 / 1 |
| 5 dB | 0.05469 | 0.05469 | 0 / 12 / 0 | 0 / 0 |

0 dB 的五帧收益来自两条轨迹，其中一次更新跨帧保留、另一次在下一帧回滚；不能将五帧视作五次独立成功。两次接受更新的参数变化范数均约为 0.0125。5 dB 的三条轨迹均未通过窗口验收。所有 24 条在线行报告两个 Data 隔离标记均为 false；这只是小样本正面迹象，尚未达到各 SNR 稳定优于 Frozen 的目标。

复现时先复制上述配置到独立路径，将 `online_adaptation_candidates` 改为仅保留原有名称为 `phase_signal_moderate` 的对象；其余字段不变。运行上述命令模板，将 `--seed-start 5` 改为 `--seed-start 8`，使用 `gap=0` 和该独立配置。生成的配置与结果位于未跟踪的 `logs/peft_fixed_candidate_20261028/`。

## 下一步

继续采用预先固定单一候选，或探索使用 Adapt Pilot 内部留出部分选择候选；Reward Pilot 只对选中的一个候选做接受或回滚。固定 checkpoint、Pilot 总长度、信息边界和评估协议后，用独立 seed 检查低接受率的原因及候选影响、接受率与逐帧配对 BER，再决定是否扩大矩阵。不得用 Data 指标调在线阈值；在排序和收益证据不足时不引入 Bandit。

## 2026-10-29：对照已有 PEFT 进度并复核 5 dB 验收

回看已有进度记录后，确认此前较好的 5 dB 数字来自不同实验条件：2026-10-15 的
Reward-logit 守门实验在 seed 72–76、每条 24 帧中报告平均改善 `+0.0705 pp`、
5/5 seed 非负；它使用双窗口、Reward 信号重构检查、logit 改变量上限 `0.1`，并沿用
包含多个候选的配置。固定调度器代码会用 Reward 分数比较通过守门的候选。因此它不能
直接当作“预先固定单一 PEFT 候选、Reward 仅验收/回滚”的复现证据。该 5 dB 结果仍是
有价值的历史筛选结果，但没有证明严格单候选协议在新 seed 上能稳定收益；同一实验的
10 dB 结果也只有 2/5 seed 正收益，0 dB 则被冻结。

为定位近期 5 dB 单候选全部不更新的问题，固定 `phase_signal_moderate` 唯一候选、同一
checkpoint、Level B、gap=0、5 dB、seed 11–15、每条 12 帧，对照不同 Reward 守门：

| 守门设置 | 冻结 BER | 在线 BER | 帧级改善 / 持平 / 变差 | PEFT 更新 / 回滚 |
|---|---:|---:|---:|---:|
| 单窗口，基础守门 | 4.48% | 4.49% | 0 / 55 / 5 | 2 / 1 |
| 双窗口 + Reward 信号重构守门 + logits 改变量 ≤ 0.1 | 4.48% | 4.48% | 0 / 60 / 0 | 0 / 0 |

单窗口只在 seed 14、15 各接受过一次更新，整体平均恶化约 `0.0109 pp`；没有 seed
获得净改善。较严格的双窗口设置在 5 个 seed 上全部拒绝更新，因而安全地退回 Frozen，
却没有提供在线收益。两种设置的 60 帧配对均使用相同冻结 BER；所有在线记录都显式
报告 Data 波形和标签未用于在线决策。

这解释了为何读起来像“退步”：此前 5 dB 的正向结果与当前固定候选测试的**候选选择方式、
守门组合、seed 和帧数都不同**。放宽到单窗口会放进少量有害更新；收紧到双窗口复合守门
则完全不更新。关键瓶颈仍是可迁移的候选收益，不是离线模型或 PEFT 参数路径消失。

复现命令：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
# 单窗口基础守门
D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe compare.py --config logs/peft_fixed_candidate_20261028/config.json --pretrained D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --methods 'Frozen Offline NN' 'Pilot-Driven Online Adaptation' --delays 116 --snrs 5 --seed-start 11 --num-seeds 5 --frames 12 --pilot-total 256 --reward-pilot-total 32 --pilot-layout prefix --acquisition-to-data-gap-seconds 0 --online-freeze-below-snr-db -1 --online-reward-windows 1 --scheduler fixed --cir-update fixed --online-condition-source pilot_cir_phase --device cuda --output-dir logs/peft_fixed_candidate_gap0_5db_w1_seed11_15_12f_20261029
# 双窗口复合守门
D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe compare.py --config logs/peft_fixed_candidate_20261028/config.json --pretrained D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --methods 'Frozen Offline NN' 'Pilot-Driven Online Adaptation' --delays 116 --snrs 5 --seed-start 11 --num-seeds 5 --frames 12 --pilot-total 256 --reward-pilot-total 32 --pilot-layout prefix --acquisition-to-data-gap-seconds 0 --online-freeze-below-snr-db -1 --online-min-reward-improvement 0.0005 --online-reward-windows 2 --online-require-reward-signal-reconstruction --online-reward-signal-tolerance 0 --online-reward-logit-change-tolerance 0.1 --scheduler fixed --cir-update fixed --online-condition-source pilot_cir_phase --device cuda --output-dir logs/peft_fixed_candidate_gap0_5db_w2_signal_logitcap_seed11_15_12f_20261029
```

以上仅为固定 PEFT 策略消融。当前仓库 `AGENTS.md` 已将 RL 调制均衡器列为唯一主
研究路线；任何后续 RL 验证都应单独报告，并确保在线只使用 Pilot reward 和状态，不使用
Data 标签。当前结论仍未达到 PEFT 多 SNR 稳定正收益或项目整体主成功门槛。
