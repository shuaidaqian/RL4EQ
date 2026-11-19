# 持续均衡候选配对开发比较

**开发诊断，不是独立确认，不证明增长至饱和。**

实验：`p3_known_adapt_budget_validation_10x64`；10 会话 × 64 帧。

所有候选同波形、同 acquisition，噪声由 acquisition 合法估计。

| 候选 | 接收状态/上下文来源 | 相位估计 | 解码分支 | checkpoint |
|---|---|---|---|---|
| acquisition_physics_q1e6 | acquisition | circular_all | physics_branch | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| recursive_pilot_physics_q1e6 | recursive_pilot | circular_all | physics_branch | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| untrained_known_adapt_physics | 共享 recursive_pilot_physics_q1e6 的实际输入 | circular_all | known_adapt_real_physics | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| untrained_known_adapt_neural | 共享 recursive_pilot_physics_q1e6 的实际输入 | circular_all | known_adapt_real_neural | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| train64_r0_recursive_pilot | 共享 recursive_pilot_physics_q1e6 的实际输入 | circular_all | known_adapt_real_neural | `pretrained/continual_v2_p3_known_adapt_20261119/repeat0_recursive_pilot.pt` |
| train64_r1_recursive_pilot | 共享 recursive_pilot_physics_q1e6 的实际输入 | circular_all | known_adapt_real_neural | `pretrained/continual_v2_p3_known_adapt_20261119/repeat1_recursive_pilot.pt` |
| train64_r2_recursive_pilot | 共享 recursive_pilot_physics_q1e6 的实际输入 | circular_all | known_adapt_real_neural | `pretrained/continual_v2_p3_known_adapt_20261119/repeat2_recursive_pilot.pt` |
| train256_r0_recursive_pilot | 共享 recursive_pilot_physics_q1e6 的实际输入 | circular_all | known_adapt_real_neural | `pretrained/continual_v2_p3_known_adapt_20261119_256updates/repeat0_recursive_pilot.pt` |
| train256_r1_recursive_pilot | 共享 recursive_pilot_physics_q1e6 的实际输入 | circular_all | known_adapt_real_neural | `pretrained/continual_v2_p3_known_adapt_20261119_256updates/repeat1_recursive_pilot.pt` |
| train256_r2_recursive_pilot | 共享 recursive_pilot_physics_q1e6 的实际输入 | circular_all | known_adapt_real_neural | `pretrained/continual_v2_p3_known_adapt_20261119_256updates/repeat2_recursive_pilot.pt` |

| SNR | 候选 | 平均 BER | 早期减晚期 BER | 平均 ms/帧 |
|---|---|---:|---:|---:|
| 0 | acquisition_physics_q1e6 | 0.147740 | -0.002344 | 20.82 |
| 0 | recursive_pilot_physics_q1e6 | 0.129716 | +0.015812 | 31.92 |
| 0 | untrained_known_adapt_physics | 0.120422 | +0.016268 | 9.56 |
| 0 | untrained_known_adapt_neural | 0.120420 | +0.015999 | 27.87 |
| 0 | train64_r0_recursive_pilot | 0.120481 | +0.016333 | 27.59 |
| 0 | train64_r1_recursive_pilot | 0.120526 | +0.016284 | 27.69 |
| 0 | train64_r2_recursive_pilot | 0.120398 | +0.016260 | 27.75 |
| 0 | train256_r0_recursive_pilot | 0.120272 | +0.016187 | 27.82 |
| 0 | train256_r1_recursive_pilot | 0.120337 | +0.015934 | 28.21 |
| 0 | train256_r2_recursive_pilot | 0.121533 | +0.016178 | 27.99 |
| 5 | acquisition_physics_q1e6 | 0.044985 | -0.002140 | 19.73 |
| 5 | recursive_pilot_physics_q1e6 | 0.037563 | +0.005160 | 33.20 |
| 5 | untrained_known_adapt_physics | 0.024343 | +0.003337 | 9.91 |
| 5 | untrained_known_adapt_neural | 0.024351 | +0.003328 | 28.79 |
| 5 | train64_r0_recursive_pilot | 0.024394 | +0.003011 | 28.96 |
| 5 | train64_r1_recursive_pilot | 0.024424 | +0.003011 | 28.61 |
| 5 | train64_r2_recursive_pilot | 0.024345 | +0.003231 | 28.42 |
| 5 | train256_r0_recursive_pilot | 0.023901 | +0.003564 | 29.13 |
| 5 | train256_r1_recursive_pilot | 0.023846 | +0.003491 | 28.94 |
| 5 | train256_r2_recursive_pilot | 0.024347 | +0.003190 | 29.08 |
| 10 | acquisition_physics_q1e6 | 0.005322 | -0.001742 | 19.75 |
| 10 | recursive_pilot_physics_q1e6 | 0.004106 | -0.000529 | 32.90 |
| 10 | untrained_known_adapt_physics | 0.000869 | -0.000057 | 9.76 |
| 10 | untrained_known_adapt_neural | 0.000869 | -0.000057 | 30.31 |
| 10 | train64_r0_recursive_pilot | 0.000865 | -0.000041 | 29.97 |
| 10 | train64_r1_recursive_pilot | 0.000867 | -0.000065 | 29.90 |
| 10 | train64_r2_recursive_pilot | 0.000865 | -0.000049 | 29.18 |
| 10 | train256_r0_recursive_pilot | 0.000857 | -0.000024 | 29.27 |
| 10 | train256_r1_recursive_pilot | 0.000859 | -0.000024 | 29.13 |
| 10 | train256_r2_recursive_pilot | 0.000863 | -0.000057 | 28.76 |
| 15 | acquisition_physics_q1e6 | 0.000647 | -0.000301 | 20.68 |
| 15 | recursive_pilot_physics_q1e6 | 0.000557 | -0.000081 | 34.66 |
| 15 | untrained_known_adapt_physics | 0.000114 | -0.000090 | 10.51 |
| 15 | untrained_known_adapt_neural | 0.000114 | -0.000090 | 35.61 |
| 15 | train64_r0_recursive_pilot | 0.000114 | -0.000090 | 33.46 |
| 15 | train64_r1_recursive_pilot | 0.000114 | -0.000090 | 31.04 |
| 15 | train64_r2_recursive_pilot | 0.000114 | -0.000090 | 29.90 |
| 15 | train256_r0_recursive_pilot | 0.000114 | -0.000090 | 29.46 |
| 15 | train256_r1_recursive_pilot | 0.000114 | -0.000090 | 29.37 |
| 15 | train256_r2_recursive_pilot | 0.000114 | -0.000090 | 29.01 |

配对差为前者 BER 减后者 BER；正值表示后者更好。双侧 t 区间未作多重性校正。

| SNR | 比较 | 均值 | 95% 区间 | 正向会话 |
|---|---|---:|---|---:|
| 0 | acquisition_physics_q1e6_minus_recursive_pilot_physics_q1e6 | +0.018024 | [+0.015515, +0.020533] | 10/10 |
| 0 | recursive_pilot_physics_q1e6_minus_untrained_known_adapt_physics | +0.009294 | [+0.008292, +0.010295] | 10/10 |
| 0 | untrained_known_adapt_physics_minus_untrained_known_adapt_neural | +0.000002 | [-0.000036, +0.000041] | 4/10 |
| 0 | recursive_pilot_physics_q1e6_minus_train64_r0_recursive_pilot | +0.009235 | [+0.008235, +0.010234] | 10/10 |
| 0 | untrained_known_adapt_physics_minus_train64_r0_recursive_pilot | -0.000059 | [-0.000194, +0.000076] | 4/10 |
| 0 | untrained_known_adapt_neural_minus_train64_r0_recursive_pilot | -0.000061 | [-0.000183, +0.000061] | 5/10 |
| 0 | recursive_pilot_physics_q1e6_minus_train64_r1_recursive_pilot | +0.009190 | [+0.008195, +0.010185] | 10/10 |
| 0 | untrained_known_adapt_physics_minus_train64_r1_recursive_pilot | -0.000104 | [-0.000275, +0.000067] | 2/10 |
| 0 | untrained_known_adapt_neural_minus_train64_r1_recursive_pilot | -0.000106 | [-0.000278, +0.000067] | 2/10 |
| 0 | recursive_pilot_physics_q1e6_minus_train64_r2_recursive_pilot | +0.009318 | [+0.008336, +0.010300] | 10/10 |
| 0 | untrained_known_adapt_physics_minus_train64_r2_recursive_pilot | +0.000024 | [-0.000096, +0.000145] | 6/10 |
| 0 | untrained_known_adapt_neural_minus_train64_r2_recursive_pilot | +0.000022 | [-0.000085, +0.000130] | 5/10 |
| 0 | recursive_pilot_physics_q1e6_minus_train256_r0_recursive_pilot | +0.009444 | [+0.008421, +0.010468] | 10/10 |
| 0 | untrained_known_adapt_physics_minus_train256_r0_recursive_pilot | +0.000151 | [-0.000262, +0.000563] | 7/10 |
| 0 | untrained_known_adapt_neural_minus_train256_r0_recursive_pilot | +0.000149 | [-0.000266, +0.000563] | 7/10 |
| 0 | recursive_pilot_physics_q1e6_minus_train256_r1_recursive_pilot | +0.009379 | [+0.008361, +0.010397] | 10/10 |
| 0 | untrained_known_adapt_physics_minus_train256_r1_recursive_pilot | +0.000085 | [-0.000325, +0.000496] | 6/10 |
| 0 | untrained_known_adapt_neural_minus_train256_r1_recursive_pilot | +0.000083 | [-0.000327, +0.000494] | 7/10 |
| 0 | recursive_pilot_physics_q1e6_minus_train256_r2_recursive_pilot | +0.008183 | [+0.007000, +0.009365] | 10/10 |
| 0 | untrained_known_adapt_physics_minus_train256_r2_recursive_pilot | -0.001111 | [-0.001571, -0.000651] | 0/10 |
| 0 | untrained_known_adapt_neural_minus_train256_r2_recursive_pilot | -0.001113 | [-0.001580, -0.000645] | 0/10 |
| 0 | train64_r0_recursive_pilot_minus_train256_r0_recursive_pilot | +0.000210 | [-0.000192, +0.000611] | 6/10 |
| 0 | train64_r1_recursive_pilot_minus_train256_r1_recursive_pilot | +0.000189 | [-0.000189, +0.000568] | 6/10 |
| 0 | train64_r2_recursive_pilot_minus_train256_r2_recursive_pilot | -0.001135 | [-0.001530, -0.000740] | 0/10 |
| 5 | acquisition_physics_q1e6_minus_recursive_pilot_physics_q1e6 | +0.007422 | [+0.006255, +0.008589] | 10/10 |
| 5 | recursive_pilot_physics_q1e6_minus_untrained_known_adapt_physics | +0.013220 | [+0.011806, +0.014634] | 10/10 |
| 5 | untrained_known_adapt_physics_minus_untrained_known_adapt_neural | -0.000008 | [-0.000024, +0.000008] | 2/10 |
| 5 | recursive_pilot_physics_q1e6_minus_train64_r0_recursive_pilot | +0.013169 | [+0.011773, +0.014565] | 10/10 |
| 5 | untrained_known_adapt_physics_minus_train64_r0_recursive_pilot | -0.000051 | [-0.000129, +0.000027] | 3/10 |
| 5 | untrained_known_adapt_neural_minus_train64_r0_recursive_pilot | -0.000043 | [-0.000114, +0.000028] | 4/10 |
| 5 | recursive_pilot_physics_q1e6_minus_train64_r1_recursive_pilot | +0.013139 | [+0.011736, +0.014541] | 10/10 |
| 5 | untrained_known_adapt_physics_minus_train64_r1_recursive_pilot | -0.000081 | [-0.000184, +0.000021] | 1/10 |
| 5 | untrained_known_adapt_neural_minus_train64_r1_recursive_pilot | -0.000073 | [-0.000170, +0.000024] | 2/10 |
| 5 | recursive_pilot_physics_q1e6_minus_train64_r2_recursive_pilot | +0.013218 | [+0.011856, +0.014580] | 10/10 |
| 5 | untrained_known_adapt_physics_minus_train64_r2_recursive_pilot | -0.000002 | [-0.000104, +0.000100] | 6/10 |
| 5 | untrained_known_adapt_neural_minus_train64_r2_recursive_pilot | +0.000006 | [-0.000090, +0.000103] | 6/10 |
| 5 | recursive_pilot_physics_q1e6_minus_train256_r0_recursive_pilot | +0.013662 | [+0.012246, +0.015078] | 10/10 |
| 5 | untrained_known_adapt_physics_minus_train256_r0_recursive_pilot | +0.000441 | [+0.000292, +0.000591] | 10/10 |
| 5 | untrained_known_adapt_neural_minus_train256_r0_recursive_pilot | +0.000450 | [+0.000305, +0.000594] | 10/10 |
| 5 | recursive_pilot_physics_q1e6_minus_train256_r1_recursive_pilot | +0.013717 | [+0.012302, +0.015131] | 10/10 |
| 5 | untrained_known_adapt_physics_minus_train256_r1_recursive_pilot | +0.000496 | [+0.000320, +0.000673] | 10/10 |
| 5 | untrained_known_adapt_neural_minus_train256_r1_recursive_pilot | +0.000505 | [+0.000334, +0.000675] | 10/10 |
| 5 | recursive_pilot_physics_q1e6_minus_train256_r2_recursive_pilot | +0.013216 | [+0.011843, +0.014589] | 10/10 |
| 5 | untrained_known_adapt_physics_minus_train256_r2_recursive_pilot | -0.000004 | [-0.000204, +0.000196] | 5/10 |
| 5 | untrained_known_adapt_neural_minus_train256_r2_recursive_pilot | +0.000004 | [-0.000193, +0.000201] | 5/10 |
| 5 | train64_r0_recursive_pilot_minus_train256_r0_recursive_pilot | +0.000492 | [+0.000322, +0.000663] | 10/10 |
| 5 | train64_r1_recursive_pilot_minus_train256_r1_recursive_pilot | +0.000578 | [+0.000376, +0.000780] | 10/10 |
| 5 | train64_r2_recursive_pilot_minus_train256_r2_recursive_pilot | -0.000002 | [-0.000190, +0.000186] | 4/10 |
| 10 | acquisition_physics_q1e6_minus_recursive_pilot_physics_q1e6 | +0.001217 | [+0.000995, +0.001439] | 10/10 |
| 10 | recursive_pilot_physics_q1e6_minus_untrained_known_adapt_physics | +0.003237 | [+0.002364, +0.004110] | 10/10 |
| 10 | untrained_known_adapt_physics_minus_untrained_known_adapt_neural | +0.000000 | 样本不足或方差退化 | 0/10 |
| 10 | recursive_pilot_physics_q1e6_minus_train64_r0_recursive_pilot | +0.003241 | [+0.002367, +0.004115] | 10/10 |
| 10 | untrained_known_adapt_physics_minus_train64_r0_recursive_pilot | +0.000004 | [-0.000002, +0.000010] | 2/10 |
| 10 | untrained_known_adapt_neural_minus_train64_r0_recursive_pilot | +0.000004 | [-0.000002, +0.000010] | 2/10 |
| 10 | recursive_pilot_physics_q1e6_minus_train64_r1_recursive_pilot | +0.003239 | [+0.002366, +0.004112] | 10/10 |
| 10 | untrained_known_adapt_physics_minus_train64_r1_recursive_pilot | +0.000002 | [-0.000006, +0.000010] | 2/10 |
| 10 | untrained_known_adapt_neural_minus_train64_r1_recursive_pilot | +0.000002 | [-0.000006, +0.000010] | 2/10 |
| 10 | recursive_pilot_physics_q1e6_minus_train64_r2_recursive_pilot | +0.003241 | [+0.002372, +0.004110] | 10/10 |
| 10 | untrained_known_adapt_physics_minus_train64_r2_recursive_pilot | +0.000004 | [-0.000005, +0.000013] | 3/10 |
| 10 | untrained_known_adapt_neural_minus_train64_r2_recursive_pilot | +0.000004 | [-0.000005, +0.000013] | 3/10 |
| 10 | recursive_pilot_physics_q1e6_minus_train256_r0_recursive_pilot | +0.003249 | [+0.002369, +0.004129] | 10/10 |
| 10 | untrained_known_adapt_physics_minus_train256_r0_recursive_pilot | +0.000012 | [-0.000010, +0.000034] | 4/10 |
| 10 | untrained_known_adapt_neural_minus_train256_r0_recursive_pilot | +0.000012 | [-0.000010, +0.000034] | 4/10 |
| 10 | recursive_pilot_physics_q1e6_minus_train256_r1_recursive_pilot | +0.003247 | [+0.002366, +0.004128] | 10/10 |
| 10 | untrained_known_adapt_physics_minus_train256_r1_recursive_pilot | +0.000010 | [-0.000010, +0.000030] | 3/10 |
| 10 | untrained_known_adapt_neural_minus_train256_r1_recursive_pilot | +0.000010 | [-0.000010, +0.000030] | 3/10 |
| 10 | recursive_pilot_physics_q1e6_minus_train256_r2_recursive_pilot | +0.003243 | [+0.002374, +0.004112] | 10/10 |
| 10 | untrained_known_adapt_physics_minus_train256_r2_recursive_pilot | +0.000006 | [-0.000011, +0.000023] | 3/10 |
| 10 | untrained_known_adapt_neural_minus_train256_r2_recursive_pilot | +0.000006 | [-0.000011, +0.000023] | 3/10 |
| 10 | train64_r0_recursive_pilot_minus_train256_r0_recursive_pilot | +0.000008 | [-0.000017, +0.000033] | 3/10 |
| 10 | train64_r1_recursive_pilot_minus_train256_r1_recursive_pilot | +0.000008 | [-0.000009, +0.000025] | 3/10 |
| 10 | train64_r2_recursive_pilot_minus_train256_r2_recursive_pilot | +0.000002 | [-0.000014, +0.000018] | 4/10 |
| 15 | acquisition_physics_q1e6_minus_recursive_pilot_physics_q1e6 | +0.000090 | [+0.000037, +0.000142] | 10/10 |
| 15 | recursive_pilot_physics_q1e6_minus_untrained_known_adapt_physics | +0.000444 | [+0.000259, +0.000628] | 10/10 |
| 15 | untrained_known_adapt_physics_minus_untrained_known_adapt_neural | +0.000000 | 样本不足或方差退化 | 0/10 |
| 15 | recursive_pilot_physics_q1e6_minus_train64_r0_recursive_pilot | +0.000444 | [+0.000259, +0.000628] | 10/10 |
| 15 | untrained_known_adapt_physics_minus_train64_r0_recursive_pilot | +0.000000 | 样本不足或方差退化 | 0/10 |
| 15 | untrained_known_adapt_neural_minus_train64_r0_recursive_pilot | +0.000000 | 样本不足或方差退化 | 0/10 |
| 15 | recursive_pilot_physics_q1e6_minus_train64_r1_recursive_pilot | +0.000444 | [+0.000259, +0.000628] | 10/10 |
| 15 | untrained_known_adapt_physics_minus_train64_r1_recursive_pilot | +0.000000 | 样本不足或方差退化 | 0/10 |
| 15 | untrained_known_adapt_neural_minus_train64_r1_recursive_pilot | +0.000000 | 样本不足或方差退化 | 0/10 |
| 15 | recursive_pilot_physics_q1e6_minus_train64_r2_recursive_pilot | +0.000444 | [+0.000259, +0.000628] | 10/10 |
| 15 | untrained_known_adapt_physics_minus_train64_r2_recursive_pilot | +0.000000 | 样本不足或方差退化 | 0/10 |
| 15 | untrained_known_adapt_neural_minus_train64_r2_recursive_pilot | +0.000000 | 样本不足或方差退化 | 0/10 |
| 15 | recursive_pilot_physics_q1e6_minus_train256_r0_recursive_pilot | +0.000444 | [+0.000259, +0.000628] | 10/10 |
| 15 | untrained_known_adapt_physics_minus_train256_r0_recursive_pilot | +0.000000 | 样本不足或方差退化 | 0/10 |
| 15 | untrained_known_adapt_neural_minus_train256_r0_recursive_pilot | +0.000000 | 样本不足或方差退化 | 0/10 |
| 15 | recursive_pilot_physics_q1e6_minus_train256_r1_recursive_pilot | +0.000444 | [+0.000259, +0.000628] | 10/10 |
| 15 | untrained_known_adapt_physics_minus_train256_r1_recursive_pilot | +0.000000 | 样本不足或方差退化 | 0/10 |
| 15 | untrained_known_adapt_neural_minus_train256_r1_recursive_pilot | +0.000000 | 样本不足或方差退化 | 0/10 |
| 15 | recursive_pilot_physics_q1e6_minus_train256_r2_recursive_pilot | +0.000444 | [+0.000259, +0.000628] | 10/10 |
| 15 | untrained_known_adapt_physics_minus_train256_r2_recursive_pilot | +0.000000 | 样本不足或方差退化 | 0/10 |
| 15 | untrained_known_adapt_neural_minus_train256_r2_recursive_pilot | +0.000000 | 样本不足或方差退化 | 0/10 |
| 15 | train64_r0_recursive_pilot_minus_train256_r0_recursive_pilot | +0.000000 | 样本不足或方差退化 | 0/10 |
| 15 | train64_r1_recursive_pilot_minus_train256_r1_recursive_pilot | +0.000000 | 样本不足或方差退化 | 0/10 |
| 15 | train64_r2_recursive_pilot_minus_train256_r2_recursive_pilot | +0.000000 | 样本不足或方差退化 | 0/10 |

物理分支仍保留同一 checkpoint 的已学习同步，不是完全非神经基线。
各窗口、自身改善、优势增长与成本分别报告；短期均值改善不能替代增长与饱和确认。
