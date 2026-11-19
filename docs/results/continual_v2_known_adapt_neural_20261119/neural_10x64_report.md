# 持续均衡候选配对开发比较

**开发诊断，不是独立确认，不证明增长至饱和。**

实验：`p2_known_adapt_real_neural_10x64`；10 会话 × 64 帧。

所有候选同波形、同 acquisition，噪声由 acquisition 合法估计。

| 候选 | 接收状态/上下文来源 | 相位估计 | 解码分支 | checkpoint |
|---|---|---|---|---|
| acquisition_physics_q1e6 | acquisition | circular_all | physics_branch | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| recursive_pilot_physics_q1e6 | recursive_pilot | circular_all | physics_branch | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| recursive_pilot_neural_legacy_input | 共享 recursive_pilot_physics_q1e6 的实际输入 | circular_all | neural | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| recursive_pilot_known_adapt_real_physics | 共享 recursive_pilot_physics_q1e6 的实际输入 | circular_all | known_adapt_real_physics | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| recursive_pilot_known_adapt_real_neural | 共享 recursive_pilot_physics_q1e6 的实际输入 | circular_all | known_adapt_real_neural | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |

| SNR | 候选 | 平均 BER | 早期减晚期 BER | 平均 ms/帧 |
|---|---|---:|---:|---:|
| 0 | acquisition_physics_q1e6 | 0.146326 | -0.001693 | 21.14 |
| 0 | recursive_pilot_physics_q1e6 | 0.128695 | +0.012467 | 32.82 |
| 0 | recursive_pilot_neural_legacy_input | 0.128729 | +0.012638 | 29.85 |
| 0 | recursive_pilot_known_adapt_real_physics | 0.120074 | +0.013062 | 9.70 |
| 0 | recursive_pilot_known_adapt_real_neural | 0.120074 | +0.013053 | 29.11 |
| 5 | acquisition_physics_q1e6 | 0.043339 | -0.002271 | 19.74 |
| 5 | recursive_pilot_physics_q1e6 | 0.036222 | +0.004175 | 32.98 |
| 5 | recursive_pilot_neural_legacy_input | 0.036222 | +0.004207 | 30.45 |
| 5 | recursive_pilot_known_adapt_real_physics | 0.024583 | +0.004508 | 9.93 |
| 5 | recursive_pilot_known_adapt_real_neural | 0.024573 | +0.004525 | 29.43 |
| 10 | acquisition_physics_q1e6 | 0.005011 | -0.001449 | 20.01 |
| 10 | recursive_pilot_physics_q1e6 | 0.003760 | -0.000195 | 33.72 |
| 10 | recursive_pilot_neural_legacy_input | 0.003758 | -0.000195 | 31.11 |
| 10 | recursive_pilot_known_adapt_real_physics | 0.001064 | +0.000122 | 10.04 |
| 10 | recursive_pilot_known_adapt_real_neural | 0.001064 | +0.000122 | 30.89 |
| 15 | acquisition_physics_q1e6 | 0.000736 | -0.000203 | 20.08 |
| 15 | recursive_pilot_physics_q1e6 | 0.000647 | -0.000065 | 33.46 |
| 15 | recursive_pilot_neural_legacy_input | 0.000647 | -0.000065 | 32.01 |
| 15 | recursive_pilot_known_adapt_real_physics | 0.000220 | +0.000138 | 10.00 |
| 15 | recursive_pilot_known_adapt_real_neural | 0.000220 | +0.000138 | 31.68 |

配对差为前者 BER 减后者 BER；正值表示后者更好。双侧 t 区间未作多重性校正。

| SNR | 比较 | 均值 | 95% 区间 | 正向会话 |
|---|---|---:|---|---:|
| 0 | acquisition_physics_q1e6_minus_recursive_pilot_physics_q1e6 | +0.017631 | [+0.015208, +0.020054] | 10/10 |
| 0 | recursive_pilot_physics_q1e6_minus_recursive_pilot_neural_legacy_input | -0.000035 | [-0.000105, +0.000036] | 2/10 |
| 0 | recursive_pilot_physics_q1e6_minus_recursive_pilot_known_adapt_real_physics | +0.008620 | [+0.007758, +0.009482] | 10/10 |
| 0 | recursive_pilot_neural_legacy_input_minus_recursive_pilot_known_adapt_real_neural | +0.008655 | [+0.007826, +0.009483] | 10/10 |
| 0 | recursive_pilot_known_adapt_real_physics_minus_recursive_pilot_known_adapt_real_neural | -0.000000 | [-0.000080, +0.000080] | 4/10 |
| 5 | acquisition_physics_q1e6_minus_recursive_pilot_physics_q1e6 | +0.007117 | [+0.005622, +0.008611] | 10/10 |
| 5 | recursive_pilot_physics_q1e6_minus_recursive_pilot_neural_legacy_input | -0.000000 | [-0.000017, +0.000017] | 3/10 |
| 5 | recursive_pilot_physics_q1e6_minus_recursive_pilot_known_adapt_real_physics | +0.011639 | [+0.010225, +0.013054] | 10/10 |
| 5 | recursive_pilot_neural_legacy_input_minus_recursive_pilot_known_adapt_real_neural | +0.011650 | [+0.010232, +0.013067] | 10/10 |
| 5 | recursive_pilot_known_adapt_real_physics_minus_recursive_pilot_known_adapt_real_neural | +0.000010 | [-0.000007, +0.000027] | 7/10 |
| 10 | acquisition_physics_q1e6_minus_recursive_pilot_physics_q1e6 | +0.001251 | [+0.000759, +0.001743] | 10/10 |
| 10 | recursive_pilot_physics_q1e6_minus_recursive_pilot_neural_legacy_input | +0.000002 | [-0.000003, +0.000007] | 1/10 |
| 10 | recursive_pilot_physics_q1e6_minus_recursive_pilot_known_adapt_real_physics | +0.002696 | [+0.001882, +0.003510] | 10/10 |
| 10 | recursive_pilot_neural_legacy_input_minus_recursive_pilot_known_adapt_real_neural | +0.002694 | [+0.001880, +0.003508] | 10/10 |
| 10 | recursive_pilot_known_adapt_real_physics_minus_recursive_pilot_known_adapt_real_neural | +0.000000 | 样本不足或方差退化 | 0/10 |
| 15 | acquisition_physics_q1e6_minus_recursive_pilot_physics_q1e6 | +0.000090 | [+0.000024, +0.000155] | 8/10 |
| 15 | recursive_pilot_physics_q1e6_minus_recursive_pilot_neural_legacy_input | +0.000000 | 样本不足或方差退化 | 0/10 |
| 15 | recursive_pilot_physics_q1e6_minus_recursive_pilot_known_adapt_real_physics | +0.000427 | [+0.000290, +0.000565] | 10/10 |
| 15 | recursive_pilot_neural_legacy_input_minus_recursive_pilot_known_adapt_real_neural | +0.000427 | [+0.000290, +0.000565] | 10/10 |
| 15 | recursive_pilot_known_adapt_real_physics_minus_recursive_pilot_known_adapt_real_neural | +0.000000 | 样本不足或方差退化 | 0/10 |

物理分支仍保留同一 checkpoint 的已学习同步，不是完全非神经基线。
各窗口、自身改善、优势增长与成本分别报告；短期均值改善不能替代增长与饱和确认。
