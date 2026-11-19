# 持续均衡候选配对开发比较

**开发诊断，不是独立确认，不证明增长至饱和。**

实验：`p1_backend_gpu_full_receiver_1x4`；1 会话 × 4 帧。

所有候选同波形、同 acquisition，噪声由 acquisition 合法估计。

| 候选 | 接收状态/上下文来源 | 相位估计 | 解码分支 | checkpoint |
|---|---|---|---|---|
| acq_direct | acquisition | unwrap_median | neural | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| acq_fft | acquisition | unwrap_median | neural | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| pilot_direct | recursive_pilot | unwrap_median | neural | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| pilot_fft | recursive_pilot | unwrap_median | neural | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |

| SNR | 候选 | 平均 BER | 早期减晚期 BER | 平均 ms/帧 |
|---|---|---:|---:|---:|
| 0 | acq_direct | 0.255534 | -0.093424 | 1310.37 |
| 0 | acq_fft | 0.255534 | -0.093424 | 576.35 |
| 0 | pilot_direct | 0.282552 | -0.128906 | 1024.37 |
| 0 | pilot_fft | 0.282552 | -0.128906 | 552.37 |
| 5 | acq_direct | 0.056315 | -0.011393 | 962.09 |
| 5 | acq_fft | 0.056315 | -0.011393 | 623.73 |
| 5 | pilot_direct | 0.047526 | -0.009766 | 976.49 |
| 5 | pilot_fft | 0.047526 | -0.009766 | 549.01 |
| 10 | acq_direct | 0.007161 | -0.008464 | 977.96 |
| 10 | acq_fft | 0.007161 | -0.008464 | 543.57 |
| 10 | pilot_direct | 0.006836 | -0.006185 | 975.86 |
| 10 | pilot_fft | 0.006836 | -0.006185 | 553.81 |
| 15 | acq_direct | 0.000977 | -0.001628 | 982.26 |
| 15 | acq_fft | 0.000977 | -0.001628 | 561.51 |
| 15 | pilot_direct | 0.001302 | -0.002604 | 1046.15 |
| 15 | pilot_fft | 0.001302 | -0.002604 | 592.29 |

配对差为前者 BER 减后者 BER；正值表示后者更好。双侧 t 区间未作多重性校正。

| SNR | 比较 | 均值 | 95% 区间 | 正向会话 |
|---|---|---:|---|---:|
| 0 | acq_direct_minus_acq_fft | +0.000000 | 样本不足 | 0/1 |
| 0 | pilot_direct_minus_pilot_fft | +0.000000 | 样本不足 | 0/1 |
| 5 | acq_direct_minus_acq_fft | +0.000000 | 样本不足 | 0/1 |
| 5 | pilot_direct_minus_pilot_fft | +0.000000 | 样本不足 | 0/1 |
| 10 | acq_direct_minus_acq_fft | +0.000000 | 样本不足 | 0/1 |
| 10 | pilot_direct_minus_pilot_fft | +0.000000 | 样本不足 | 0/1 |
| 15 | acq_direct_minus_acq_fft | +0.000000 | 样本不足 | 0/1 |
| 15 | pilot_direct_minus_pilot_fft | +0.000000 | 样本不足 | 0/1 |

物理分支仍保留同一 checkpoint 的已学习同步，不是完全非神经基线。
各窗口、自身改善、优势增长与成本分别报告；短期均值改善不能替代增长与饱和确认。
