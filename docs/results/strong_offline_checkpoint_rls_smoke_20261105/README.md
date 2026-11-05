# 强离线 checkpoint 在线 RLS 短帧烟测

## 目的与范围

本烟测首次把现有 Pilot-RLS 在线入口接到已核验的强离线模型上，确认模型能载入、更新信息边界正确，并检查在线更新是否真正提交。它只覆盖 5 dB、一个新 seed、8帧，不是在线增益实验，也不支持任何目标成功结论。

离线 checkpoint 为原项目工作区中的 `eme_bce_all_32_20260905_pilot256/model_best.pt`，SHA256=`8cd3912d1c957c9689c51730aff9d27a8f74942186c802a7483e5781af65aec4`。配置为 `configs/eme_long_memory_v2.json`，本 worktree SHA256=`f3952864d26025b9ec5cfc6ea6008080435a7c4e5747881b410eb3d141aebfa5`。信道仍为既有 Level B、116符号记忆、`cfo_phase_tiny`、Pilot=256（Adapt=224、Reward=32），没有更改信道、Pilot或接收窗口。

## 运行与观察

复现命令：

```powershell
& 'D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe' compare.py --config configs/eme_long_memory_v2.json --pretrained D:\Research\RL4EQ\pretrained\eme_bce_all_32_20260905_pilot256\model_best.pt --methods 'Frozen Offline NN' 'Pilot-Driven Online Adaptation' --delays 116 --snrs 5 --num-seeds 1 --seed-start 45000 --frames 8 --pilot-total 256 --reward-pilot-total 32 --pilot-layout prefix --state-split offline_train --impairment-profile cfo_phase_tiny --cir-update fixed --online-algorithm rls --output-dir logs/strong_checkpoint_online_rls_smoke_20261105 --device cuda
```

两臂逐帧接收波形 SHA256 和信道状态完全匹配，说明比较使用同一发射/接收轨迹。Frozen 与 Online 的8帧平均 Data BER 都是 `0.06412761`。在线 RLS 的 Adapt 内部候选有7帧通过，但 Reward Pilot 守门通过0帧，`peft_update_applied` 为0帧，累计参数距离保持0；所以该在线臂实际等于 Frozen。

这定位出强 checkpoint 上的当前首要阻塞：更新器能提出候选，但候选没有通过独立验证 Pilot 门控，在线参数未改变。8帧中6帧候选的 Reward BCE 恶化、1帧持平、1帧改善；有总体 BCE 改善的末帧仍因两个 Reward 子窗口方向不一致而失败。门控还要求 Reward 硬错误数不增加。Reward 只有32个符号，硬 BER 每错1位跳变3.125个百分点，因此小窗口判据很离散。当前 `should_accept_reward_guard` 的调用链优先检查硬 BER，BER相同时才检查窗口BCE改善及参数距离惩罚；配置中的 `online_adaptation_min_reward_improvement=0.0005` 不是这条 RLS 窗口门控的实际接受判据，不能把全拒绝简单归因于这个字段。

不能通过关闭验证门控或读取 Data 真值强行制造增益。每帧均报告 `data_labels_used_online=false` 与 `data_waveform_used_online=false`；Data BER 仅在候选和门控后事后计算。

完整机器记录位于同目录 `frame_metrics.jsonl` 与 `summary.json`。8帧不能估计随帧增长趋势，也不能用于选择超参数。

## 下一步实验

在保持强 checkpoint、信道、Pilot预算和验证集隔离的条件下，使用独立 training/selection 会话筛选 RLS 残差目标与低维参数化，使 Adapt 更新方向更可能改善未参与拟合的 Reward Pilot。先用 Pilot-only 选择更新器与超参数，再固定方案跑新会话；记录候选/守门/提交率、参数距离及分窗 Online−Frozen Data BER。优先比较现有双窗口BCE加硬错误护栏与跨帧累计的留出Pilot统计，不能只降低门槛。若通过帧数仍少，应重新设计离线训练内外循环目标，使合法 Adapt 更新能泛化到后续 Data，而不是放大同一筛查集或削弱守门。
