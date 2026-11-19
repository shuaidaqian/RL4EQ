# 持续均衡候选配对开发比较

**开发诊断，不是独立确认，不证明增长至饱和。**

实验：`p2_circular_phase_four_arms_3x32`；3 会话 × 32 帧。

所有候选同波形、同 acquisition，噪声由 acquisition 合法估计。

| 候选 | 接收状态/上下文来源 | 相位估计 | 解码分支 | checkpoint |
|---|---|---|---|---|
| legacy_recursive_pilot | recursive_pilot | unwrap_median | neural | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| acquisition | acquisition | circular_all | neural | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| current_pilot | current_pilot | circular_all | neural | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| recursive_pilot | recursive_pilot | circular_all | neural | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| recursive_soft | recursive_soft | circular_all | neural | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| pilot_shared_physics | 共享 recursive_pilot 的实际输入 | circular_all | physics_branch | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |

| SNR | 候选 | 平均 BER | 早期减晚期 BER | 平均 ms/帧 |
|---|---|---:|---:|---:|
| 0 | legacy_recursive_pilot | 0.180379 | +0.054579 | 276.06 |
| 0 | acquisition | 0.146010 | -0.004232 | 261.53 |
| 0 | current_pilot | 0.143500 | -0.004232 | 270.35 |
| 0 | recursive_pilot | 0.134196 | +0.007867 | 270.58 |
| 0 | recursive_soft | 0.133681 | +0.003581 | 279.67 |
| 0 | pilot_shared_physics | 0.134101 | +0.007487 | 6.58 |
| 5 | legacy_recursive_pilot | 0.040717 | +0.002224 | 278.87 |
| 5 | acquisition | 0.043498 | +0.000760 | 265.09 |
| 5 | current_pilot | 0.042250 | +0.003689 | 275.42 |
| 5 | recursive_pilot | 0.040012 | +0.003092 | 276.08 |
| 5 | recursive_soft | 0.038968 | +0.002279 | 284.82 |
| 5 | pilot_shared_physics | 0.040025 | +0.003147 | 7.12 |
| 10 | legacy_recursive_pilot | 0.004666 | +0.001085 | 304.32 |
| 10 | acquisition | 0.005276 | +0.000543 | 299.43 |
| 10 | current_pilot | 0.004978 | +0.000705 | 299.18 |
| 10 | recursive_pilot | 0.004734 | +0.000922 | 304.65 |
| 10 | recursive_soft | 0.004083 | +0.001519 | 315.85 |
| 10 | pilot_shared_physics | 0.004734 | +0.000922 | 7.92 |
| 15 | legacy_recursive_pilot | 0.000936 | +0.000651 | 301.32 |
| 15 | acquisition | 0.000990 | +0.000868 | 288.00 |
| 15 | current_pilot | 0.000936 | +0.000760 | 297.90 |
| 15 | recursive_pilot | 0.000949 | +0.000651 | 295.78 |
| 15 | recursive_soft | 0.000936 | +0.000814 | 306.64 |
| 15 | pilot_shared_physics | 0.000949 | +0.000651 | 7.53 |

配对差为前者 BER 减后者 BER；正值表示后者更好。双侧 t 区间未作多重性校正。

| SNR | 比较 | 均值 | 95% 区间 | 正向会话 |
|---|---|---:|---|---:|
| 0 | legacy_recursive_pilot_minus_recursive_pilot | +0.046183 | [-0.046447, +0.138813] | 3/3 |
| 0 | acquisition_minus_recursive_pilot | +0.011814 | [-0.000616, +0.024243] | 3/3 |
| 0 | current_pilot_minus_recursive_pilot | +0.009304 | [+0.000572, +0.018037] | 3/3 |
| 0 | recursive_pilot_minus_recursive_soft | +0.000515 | [-0.007150, +0.008180] | 2/3 |
| 0 | pilot_shared_physics_minus_recursive_pilot | -0.000095 | [-0.000153, -0.000037] | 0/3 |
| 5 | legacy_recursive_pilot_minus_recursive_pilot | +0.000705 | [+0.000284, +0.001126] | 3/3 |
| 5 | acquisition_minus_recursive_pilot | +0.003486 | [-0.001367, +0.008339] | 3/3 |
| 5 | current_pilot_minus_recursive_pilot | +0.002238 | [-0.002402, +0.006878] | 3/3 |
| 5 | recursive_pilot_minus_recursive_soft | +0.001044 | [-0.000885, +0.002974] | 3/3 |
| 5 | pilot_shared_physics_minus_recursive_pilot | +0.000014 | [-0.000045, +0.000072] | 1/3 |
| 10 | legacy_recursive_pilot_minus_recursive_pilot | -0.000068 | [-0.000126, -0.000009] | 0/3 |
| 10 | acquisition_minus_recursive_pilot | +0.000543 | [+0.000288, +0.000797] | 3/3 |
| 10 | current_pilot_minus_recursive_pilot | +0.000244 | [+0.000143, +0.000345] | 3/3 |
| 10 | recursive_pilot_minus_recursive_soft | +0.000651 | [-0.000313, +0.001615] | 3/3 |
| 10 | pilot_shared_physics_minus_recursive_pilot | +0.000000 | [+0.000000, +0.000000] | 0/3 |
| 15 | legacy_recursive_pilot_minus_recursive_pilot | -0.000014 | [-0.000072, +0.000045] | 0/3 |
| 15 | acquisition_minus_recursive_pilot | +0.000041 | [-0.000591, +0.000672] | 1/3 |
| 15 | current_pilot_minus_recursive_pilot | -0.000014 | [-0.000469, +0.000442] | 1/3 |
| 15 | recursive_pilot_minus_recursive_soft | +0.000014 | [-0.000045, +0.000072] | 1/3 |
| 15 | pilot_shared_physics_minus_recursive_pilot | +0.000000 | [+0.000000, +0.000000] | 0/3 |

物理分支仍保留同一 checkpoint 的已学习同步，不是完全非神经基线。
各窗口、自身改善、优势增长与成本分别报告；短期均值改善不能替代增长与饱和确认。
