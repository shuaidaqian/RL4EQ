# 强离线模型单窗 Reward RLS 32 帧开发筛查

## 目的与边界

在先前同 seed、5 dB、8 帧 Pilot-only 烟测中，双 Reward 子窗方向不一致导致所有候选被拒；整段 Reward BCE 改善且硬错误数不增时，1 帧候选可实际提交。此筛查只改变 `online_reward_windows`（2→1），其它配置、RLS、信道和 Pilot 资源保持不变。门控继续要求整段 Reward BCE 达到已有最小改善阈值，且 Reward 硬错误数不增加。

这是一项开发机制筛查，不能作为独立确认，也不能证明在线核心目标成立。3个新会话、每SNR 32帧只覆盖四个8帧窗口。选择单窗的依据来自 Reward Pilot 诊断，不来自 Data；本批 Data 只在所有动作及门控结束后用于事后评分。

## 固定条件

- 来源模型：`eme_bce_all_32_20260905_pilot256/model_best.pt`，SHA256=`8cd3912d1c957c9689c51730aff9d27a8f74942186c802a7483e5781af65aec4`。
- 配置：`configs/eme_long_memory_v2.json`，SHA256=`f3952864d26025b9ec5cfc6ea6008080435a7c4e5747881b410eb3d141aebfa5`。
- Level B、116符号长回波、`cfo_phase_tiny`、SNR 0/5/10/15 dB；Pilot=256（Adapt=224，Reward=32）；每点3个新 seed（45100–45102），每session连续32帧。
- Frozen 与 Online 逐帧接收波形指纹和信道状态匹配。更新仅用Adapt Pilot；门控仅用Reward Pilot；Data标签不用于在线控制。

运行命令：

```powershell
& 'D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe' compare.py --config configs/eme_long_memory_v2.json --pretrained D:\Research\RL4EQ\pretrained\eme_bce_all_32_20260905_pilot256\model_best.pt --methods 'Frozen Offline NN' 'Pilot-Driven Online Adaptation' --delays 116 --snrs 0 5 10 15 --num-seeds 3 --seed-start 45100 --frames 32 --pilot-total 256 --reward-pilot-total 32 --pilot-layout prefix --state-split offline_train --impairment-profile cfo_phase_tiny --cir-update fixed --online-algorithm rls --online-reward-windows 1 --output-dir logs/strong_checkpoint_online_rls_onewindow_32f_dev_20261105 --device cuda
```

## 结果

增益定义为`BER_frozen - BER_online`，正数有利于在线。时间斜率和末窗减首窗在各 session 内先计算，再按3个独立 session 汇总95% t区间；帧不是统计独立单位。

| SNR | 平均收益（95%区间） | 收益斜率/帧（95%区间） | 末窗−首窗（95%区间） | RLS提交次数（3个session） |
|---:|---:|---:|---:|---:|
| 0 dB | -0.00309 [-0.01640, 0.01021] | -3.77e-4 [-1.999e-3, 1.245e-3] | -0.00890 [-0.04718, 0.02939] | 0 / 0 / 1 |
| 5 dB | -0.000258 [-0.00498, 0.00447] | +1.40e-4 [+1.40e-5, +2.67e-4] | +0.00391 [+0.00045, +0.00736] | 4 / 4 / 3 |
| 10 dB | +1.22e-4 [-1.45e-4, 3.90e-4] | -2.03e-6 [-1.08e-5, 6.72e-6] | 0 [-7.00e-4, 7.00e-4] | 6 / 8 / 6 |
| 15 dB | -1.36e-5 [-1.68e-4, 1.41e-4] | -2.03e-6 [-7.09e-6, 3.02e-6] | -5.43e-5 [-2.88e-4, 1.79e-4] | 10 / 6 / 6 |

5 dB 三个 session 的晚窗增益都大于早窗，斜率也都为正，但全程平均收益为负且区间跨零：一个 session 的整体退化抵消了另外两个 session 的改善。0 dB 一个有害更新使单 session BER 退化约0.00928，说明低SNR风险仍未解决。10/15 dB 收益幅度接近误码量化分辨率，没有增长证据。

结论是单窗守门提高了实际参数更新率，并出现值得继续检验的5 dB开发趋势；它没有在全SNR证明正收益，更没有证明收益随传输帧总体增长。不能把5 dB局部结果概括成整体成功，也不应只按结果挑SNR。

完整逐帧原始记录、比较器摘要和会话趋势JSON见本目录。`session_trends.json`记录 checkpoint/config/data 指纹、逐seed BER、窗口收益、斜率、末首差、提交次数与限制。

## 下一步

在独立的 training/selection Pilot 会话上固定双窗与单窗候选门控，再比较安全限幅（如累计参数距离约束）和残差目标尺度，保持硬错误不增护栏。先用Pilot-only证据选择更新机制；随后冻结所有超参，在新会话四SNR、至少128帧评估每8/16/32帧窗口的配对趋势。若0 dB风险仍存在，应先改进离线可适配表示与内外循环目标，而不是放松硬错误护栏。
