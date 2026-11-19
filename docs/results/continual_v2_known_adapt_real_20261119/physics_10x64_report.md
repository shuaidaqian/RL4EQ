# 持续均衡候选配对开发比较

**开发诊断，不是独立确认，不证明增长至饱和。**

实验：`p2_known_adapt_real_physics_10x64`；10 会话 × 64 帧。

所有候选同波形、同 acquisition，噪声由 acquisition 合法估计。

| 候选 | 接收状态/上下文来源 | 相位估计 | 解码分支 | checkpoint |
|---|---|---|---|---|
| acquisition_q1e6 | acquisition | circular_all | physics_branch | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| recursive_pilot_q1e6 | recursive_pilot | circular_all | physics_branch | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| recursive_pilot_known_adapt_real | 共享 recursive_pilot_q1e6 的实际输入 | circular_all | known_adapt_real_physics | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |

| SNR | 候选 | 平均 BER | 早期减晚期 BER | 平均 ms/帧 |
|---|---|---:|---:|---:|
| 0 | acquisition_q1e6 | 0.148142 | -0.004549 | 23.00 |
| 0 | recursive_pilot_q1e6 | 0.130231 | +0.011898 | 36.32 |
| 0 | recursive_pilot_known_adapt_real | 0.121407 | +0.014111 | 9.99 |
| 5 | acquisition_q1e6 | 0.046344 | -0.004386 | 23.23 |
| 5 | recursive_pilot_q1e6 | 0.038393 | +0.004281 | 39.14 |
| 5 | recursive_pilot_known_adapt_real | 0.025553 | +0.003109 | 10.64 |
| 10 | acquisition_q1e6 | 0.006185 | -0.001497 | 22.60 |
| 10 | recursive_pilot_q1e6 | 0.004508 | +0.000505 | 38.51 |
| 10 | recursive_pilot_known_adapt_real | 0.001178 | +0.000269 | 10.36 |
| 15 | acquisition_q1e6 | 0.000867 | -0.000358 | 22.87 |
| 15 | recursive_pilot_q1e6 | 0.000751 | -0.000203 | 38.51 |
| 15 | recursive_pilot_known_adapt_real | 0.000195 | -0.000203 | 10.26 |

配对差为前者 BER 减后者 BER；正值表示后者更好。双侧 t 区间未作多重性校正。

| SNR | 比较 | 均值 | 95% 区间 | 正向会话 |
|---|---|---:|---|---:|
| 0 | acquisition_q1e6_minus_recursive_pilot_q1e6 | +0.017912 | [+0.016305, +0.019519] | 10/10 |
| 0 | acquisition_q1e6_minus_recursive_pilot_known_adapt_real | +0.026735 | [+0.024713, +0.028758] | 10/10 |
| 0 | recursive_pilot_q1e6_minus_recursive_pilot_known_adapt_real | +0.008824 | [+0.007976, +0.009671] | 10/10 |
| 5 | acquisition_q1e6_minus_recursive_pilot_q1e6 | +0.007951 | [+0.006427, +0.009475] | 10/10 |
| 5 | acquisition_q1e6_minus_recursive_pilot_known_adapt_real | +0.020791 | [+0.018514, +0.023067] | 10/10 |
| 5 | recursive_pilot_q1e6_minus_recursive_pilot_known_adapt_real | +0.012840 | [+0.011454, +0.014226] | 10/10 |
| 10 | acquisition_q1e6_minus_recursive_pilot_q1e6 | +0.001676 | [+0.001126, +0.002227] | 10/10 |
| 10 | acquisition_q1e6_minus_recursive_pilot_known_adapt_real | +0.005007 | [+0.003481, +0.006533] | 10/10 |
| 10 | recursive_pilot_q1e6_minus_recursive_pilot_known_adapt_real | +0.003330 | [+0.002226, +0.004435] | 10/10 |
| 15 | acquisition_q1e6_minus_recursive_pilot_q1e6 | +0.000116 | [+0.000039, +0.000193] | 7/10 |
| 15 | acquisition_q1e6_minus_recursive_pilot_known_adapt_real | +0.000671 | [+0.000438, +0.000904] | 10/10 |
| 15 | recursive_pilot_q1e6_minus_recursive_pilot_known_adapt_real | +0.000555 | [+0.000384, +0.000727] | 10/10 |

物理分支仍保留同一 checkpoint 的已学习同步，不是完全非神经基线。
各窗口、自身改善、优势增长与成本分别报告；短期均值改善不能替代增长与饱和确认。
