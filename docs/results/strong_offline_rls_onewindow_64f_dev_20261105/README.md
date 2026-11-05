# 强离线模型单窗 Reward RLS 64 帧开发复核

## 目的与协议

本轮复核前一轮32帧开发筛查中5 dB出现的正向时间趋势。使用同一强离线 checkpoint 和固定的 one-window Reward BCE + 硬错误不增门控，不再调参；新 seed 为45200–45202，每 SNR 连续64帧。保持 Level B、116符号长回波、`cfo_phase_tiny`、Pilot=256（Adapt=224，Reward=32）不变。Frozen 与 Online 逐帧 RX 指纹及信道状态匹配；Data 真值只在在线动作和Reward门控完成后事后评分。

该结果仍为开发机制筛查，不是正式确认：只有3个独立 session，且每个 session 64帧。收益定义为`BER_frozen - BER_online`；每个 session 按连续8帧算窗口、拟合时间斜率并计算末窗减首窗，再按session汇总95% t区间。

## 运行命令

```powershell
& 'D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe' compare.py --config configs/eme_long_memory_v2.json --pretrained D:\Research\RL4EQ\pretrained\eme_bce_all_32_20260905_pilot256\model_best.pt --methods 'Frozen Offline NN' 'Pilot-Driven Online Adaptation' --delays 116 --snrs 0 5 10 15 --num-seeds 3 --seed-start 45200 --frames 64 --pilot-total 256 --reward-pilot-total 32 --pilot-layout prefix --state-split offline_train --impairment-profile cfo_phase_tiny --cir-update fixed --online-algorithm rls --online-reward-windows 1 --output-dir logs/strong_checkpoint_online_rls_onewindow_64f_dev_20261105 --device cuda
```

## 结果

| SNR | 平均收益（95%区间） | 斜率/帧（95%区间） | 末窗−首窗（95%区间） | 每session RLS提交次数 |
|---:|---:|---:|---:|---:|
| 0 dB | -0.01290 [-0.04431, 0.01851] | -7.12e-4 [-2.26e-3, 8.36e-4] | -0.03277 [-0.12533, 0.05979] | 2 / 2 / 2 |
| 5 dB | -0.00306 [-0.00907, 0.00296] | +1.48e-5 [-1.72e-4, 2.02e-4] | -0.00184 [-0.01328, 0.00959] | 4 / 5 / 4 |
| 10 dB | +2.37e-4 [-3.91e-4, 8.65e-4] | +6.70e-6 [-1.18e-5, 2.52e-5] | +3.26e-4 [-8.87e-4, 1.54e-3] | 5 / 10 / 4 |
| 15 dB | 0 [0, 0] | +4.84e-7 [-1.60e-6, 2.57e-6] | 0 [0, 0] | 12 / 7 / 6 |

上一轮5 dB的正增长趋势没有在新会话复现：平均收益为负、斜率接近零。0 dB三个 session 都退化；10 dB只有极小的正点估计且区间跨零，15 dB更新虽发生但 BER 没有改变。由此，单窗门控能让RLS更新持续发生，却没有形成可重复、随帧增长的检测收益；它不能作为核心研究成果。

本轮结果说明当前障碍已超出门控频率。简单的 Pilot 目标残差更新会在强信道记忆下把 Adapt Pilot 局部拟合泛化成 Data 退化，尤其0/5 dB明显。下一步停止扩增这个 RLS 变体，转向匹配在线更新的离线元训练：内循环复现合法 Adapt 更新及参数持续保留，外循环用隔离 Data 目标训练更新表示/初始化；再以 Frozen/GN/RLS/元初始化更新做同波形比较，并按独立会话报告连续窗口增益。不能把其他 SNR 的小正值平均后宣称成功。

完整逐帧结果、compare摘要和逐session趋势见本目录 `frame_metrics.jsonl`、`compare_summary.json` 与 `session_trends.json`。摘要记录 checkpoint/config/逐帧指纹及数据边界。
