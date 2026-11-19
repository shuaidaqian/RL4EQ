# 持续均衡候选配对开发比较

**开发诊断，不是独立确认，不证明增长至饱和。**

实验：`p1_backend_full_receiver_1x8`；1 会话 × 8 帧。

所有候选同波形、同 acquisition，噪声由 acquisition 合法估计。

| 候选 | 接收状态 | 相位估计 | 解码分支 | checkpoint |
|---|---|---|---|---|
| acq_direct | acquisition | unwrap_median | neural | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| acq_fft | acquisition | unwrap_median | neural | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| pilot_direct | recursive_pilot | unwrap_median | neural | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| pilot_fft | recursive_pilot | unwrap_median | neural | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |

| SNR | 候选 | 平均 BER | 早期减晚期 BER | 平均 ms/帧 |
|---|---|---:|---:|---:|
| 0 | acq_direct | 0.138021 | -0.015951 | 578.28 |
| 0 | acq_fft | 0.138021 | -0.015951 | 319.37 |
| 0 | pilot_direct | 0.149577 | +0.025391 | 608.88 |
| 0 | pilot_fft | 0.149577 | +0.025391 | 323.90 |
| 5 | acq_direct | 0.035482 | +0.001953 | 588.07 |
| 5 | acq_fft | 0.035482 | +0.001953 | 320.89 |
| 5 | pilot_direct | 0.034668 | +0.001628 | 604.21 |
| 5 | pilot_fft | 0.034668 | +0.001628 | 352.38 |
| 10 | acq_direct | 0.001790 | -0.000326 | 561.16 |
| 10 | acq_fft | 0.001790 | -0.000326 | 312.80 |
| 10 | pilot_direct | 0.002116 | -0.000651 | 552.55 |
| 10 | pilot_fft | 0.002116 | -0.000651 | 353.11 |
| 15 | acq_direct | 0.000488 | +0.000326 | 500.83 |
| 15 | acq_fft | 0.000488 | +0.000326 | 278.34 |
| 15 | pilot_direct | 0.000488 | +0.000326 | 507.56 |
| 15 | pilot_fft | 0.000488 | +0.000326 | 293.59 |

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
