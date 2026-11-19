# 持续均衡候选配对开发比较

**开发诊断，不是独立确认，不证明增长至饱和。**

实验：`p2_q_physics_vectorized_10x64`；10 会话 × 64 帧。

所有候选同波形、同 acquisition，噪声由 acquisition 合法估计。

| 候选 | 接收状态/上下文来源 | 相位估计 | 解码分支 | checkpoint |
|---|---|---|---|---|
| acquisition | acquisition | circular_all | physics_branch | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| current_pilot | current_pilot | circular_all | physics_branch | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| recursive_q1e4 | recursive_pilot | circular_all | physics_branch | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| recursive_q1e6 | recursive_pilot | circular_all | physics_branch | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| recursive_q0 | recursive_pilot | circular_all | physics_branch | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| soft_q1e6 | recursive_soft | circular_all | physics_branch | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |

| SNR | 候选 | 平均 BER | 早期减晚期 BER | 平均 ms/帧 |
|---|---|---:|---:|---:|
| 0 | acquisition | 0.144242 | +0.002157 | 13.49 |
| 0 | current_pilot | 0.141506 | +0.002677 | 21.26 |
| 0 | recursive_q1e4 | 0.130481 | +0.012386 | 21.70 |
| 0 | recursive_q1e6 | 0.127488 | +0.016968 | 21.53 |
| 0 | recursive_q0 | 0.127417 | +0.016992 | 21.42 |
| 0 | soft_q1e6 | 0.130031 | +0.012760 | 31.50 |
| 5 | acquisition | 0.043545 | +0.001872 | 11.94 |
| 5 | current_pilot | 0.042269 | +0.002653 | 19.01 |
| 5 | recursive_q1e4 | 0.039583 | +0.004606 | 18.82 |
| 5 | recursive_q1e6 | 0.036141 | +0.008960 | 19.08 |
| 5 | recursive_q0 | 0.036174 | +0.008903 | 18.79 |
| 5 | soft_q1e6 | 0.036715 | +0.006038 | 27.93 |
| 10 | acquisition | 0.005396 | -0.000195 | 11.72 |
| 10 | current_pilot | 0.005056 | -0.000155 | 19.34 |
| 10 | recursive_q1e4 | 0.005101 | +0.000342 | 19.03 |
| 10 | recursive_q1e6 | 0.003998 | +0.001530 | 18.93 |
| 10 | recursive_q0 | 0.004014 | +0.001465 | 18.94 |
| 10 | soft_q1e6 | 0.003782 | +0.000977 | 27.89 |
| 15 | acquisition | 0.000789 | -0.000122 | 12.04 |
| 15 | current_pilot | 0.000765 | -0.000098 | 19.34 |
| 15 | recursive_q1e4 | 0.000793 | -0.000090 | 19.04 |
| 15 | recursive_q1e6 | 0.000710 | -0.000041 | 19.26 |
| 15 | recursive_q0 | 0.000726 | -0.000057 | 19.10 |
| 15 | soft_q1e6 | 0.000696 | -0.000041 | 28.88 |

配对差为前者 BER 减后者 BER；正值表示后者更好。双侧 t 区间未作多重性校正。

| SNR | 比较 | 均值 | 95% 区间 | 正向会话 |
|---|---|---:|---|---:|
| 0 | acquisition_minus_recursive_q1e4 | +0.013761 | [+0.012277, +0.015246] | 10/10 |
| 0 | acquisition_minus_recursive_q1e6 | +0.016754 | [+0.015261, +0.018247] | 10/10 |
| 0 | acquisition_minus_recursive_q0 | +0.016825 | [+0.015327, +0.018324] | 10/10 |
| 0 | acquisition_minus_soft_q1e6 | +0.014211 | [+0.011926, +0.016496] | 10/10 |
| 0 | current_pilot_minus_recursive_q1e4 | +0.011025 | [+0.009667, +0.012383] | 10/10 |
| 0 | current_pilot_minus_recursive_q1e6 | +0.014018 | [+0.012774, +0.015262] | 10/10 |
| 0 | current_pilot_minus_recursive_q0 | +0.014089 | [+0.012858, +0.015320] | 10/10 |
| 0 | recursive_q1e4_minus_recursive_q1e6 | +0.002993 | [+0.002179, +0.003806] | 10/10 |
| 0 | recursive_q1e6_minus_recursive_q0 | +0.000071 | [-0.000016, +0.000158] | 8/10 |
| 0 | recursive_q1e6_minus_soft_q1e6 | -0.002543 | [-0.004284, -0.000802] | 1/10 |
| 5 | acquisition_minus_recursive_q1e4 | +0.003961 | [+0.002757, +0.005165] | 9/10 |
| 5 | acquisition_minus_recursive_q1e6 | +0.007404 | [+0.006069, +0.008739] | 10/10 |
| 5 | acquisition_minus_recursive_q0 | +0.007371 | [+0.006118, +0.008624] | 10/10 |
| 5 | acquisition_minus_soft_q1e6 | +0.006830 | [+0.005414, +0.008246] | 10/10 |
| 5 | current_pilot_minus_recursive_q1e4 | +0.002686 | [+0.001776, +0.003595] | 9/10 |
| 5 | current_pilot_minus_recursive_q1e6 | +0.006128 | [+0.005090, +0.007165] | 10/10 |
| 5 | current_pilot_minus_recursive_q0 | +0.006095 | [+0.005138, +0.007053] | 10/10 |
| 5 | recursive_q1e4_minus_recursive_q1e6 | +0.003442 | [+0.002968, +0.003916] | 10/10 |
| 5 | recursive_q1e6_minus_recursive_q0 | -0.000033 | [-0.000153, +0.000088] | 4/10 |
| 5 | recursive_q1e6_minus_soft_q1e6 | -0.000574 | [-0.001196, +0.000049] | 3/10 |
| 10 | acquisition_minus_recursive_q1e4 | +0.000295 | [-0.000044, +0.000634] | 7/10 |
| 10 | acquisition_minus_recursive_q1e6 | +0.001398 | [+0.000899, +0.001897] | 10/10 |
| 10 | acquisition_minus_recursive_q0 | +0.001381 | [+0.000919, +0.001844] | 10/10 |
| 10 | acquisition_minus_soft_q1e6 | +0.001613 | [+0.001099, +0.002128] | 10/10 |
| 10 | current_pilot_minus_recursive_q1e4 | -0.000045 | [-0.000330, +0.000240] | 4/10 |
| 10 | current_pilot_minus_recursive_q1e6 | +0.001058 | [+0.000647, +0.001469] | 10/10 |
| 10 | current_pilot_minus_recursive_q0 | +0.001042 | [+0.000667, +0.001416] | 10/10 |
| 10 | recursive_q1e4_minus_recursive_q1e6 | +0.001103 | [+0.000763, +0.001442] | 10/10 |
| 10 | recursive_q1e6_minus_recursive_q0 | -0.000016 | [-0.000067, +0.000034] | 5/10 |
| 10 | recursive_q1e6_minus_soft_q1e6 | +0.000216 | [+0.000112, +0.000319] | 9/10 |
| 15 | acquisition_minus_recursive_q1e4 | -0.000004 | [-0.000053, +0.000045] | 5/10 |
| 15 | acquisition_minus_recursive_q1e6 | +0.000079 | [+0.000022, +0.000137] | 8/10 |
| 15 | acquisition_minus_recursive_q0 | +0.000063 | [+0.000018, +0.000108] | 8/10 |
| 15 | acquisition_minus_soft_q1e6 | +0.000094 | [+0.000039, +0.000148] | 9/10 |
| 15 | current_pilot_minus_recursive_q1e4 | -0.000028 | [-0.000090, +0.000033] | 3/10 |
| 15 | current_pilot_minus_recursive_q1e6 | +0.000055 | [+0.000009, +0.000101] | 7/10 |
| 15 | current_pilot_minus_recursive_q0 | +0.000039 | [-0.000005, +0.000082] | 6/10 |
| 15 | recursive_q1e4_minus_recursive_q1e6 | +0.000083 | [+0.000027, +0.000139] | 10/10 |
| 15 | recursive_q1e6_minus_recursive_q0 | -0.000016 | [-0.000034, +0.000002] | 2/10 |
| 15 | recursive_q1e6_minus_soft_q1e6 | +0.000014 | [-0.000011, +0.000039] | 6/10 |

物理分支仍保留同一 checkpoint 的已学习同步，不是完全非神经基线。
各窗口、自身改善、优势增长与成本分别报告；短期均值改善不能替代增长与饱和确认。
