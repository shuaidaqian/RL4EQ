# 持续均衡候选配对开发比较

**开发诊断，不是独立确认，不证明增长至饱和。**

实验：`p2_streaming_writer_smoke_1x8`；1 会话 × 8 帧。

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
| 0 | acquisition_physics_q1e6 | 0.147135 | +0.003581 | 133.93 |
| 0 | recursive_pilot_physics_q1e6 | 0.139811 | +0.010742 | 38.62 |
| 0 | recursive_pilot_neural_legacy_input | 0.139648 | +0.013021 | 54.95 |
| 0 | recursive_pilot_known_adapt_real_physics | 0.137044 | +0.014323 | 10.39 |
| 0 | recursive_pilot_known_adapt_real_neural | 0.136882 | +0.014974 | 29.18 |
| 5 | acquisition_physics_q1e6 | 0.047363 | +0.005534 | 18.96 |
| 5 | recursive_pilot_physics_q1e6 | 0.042643 | -0.000977 | 31.79 |
| 5 | recursive_pilot_neural_legacy_input | 0.042806 | -0.001628 | 27.55 |
| 5 | recursive_pilot_known_adapt_real_physics | 0.032389 | +0.006836 | 9.76 |
| 5 | recursive_pilot_known_adapt_real_neural | 0.032389 | +0.006836 | 27.43 |
| 10 | acquisition_physics_q1e6 | 0.004069 | +0.000326 | 19.53 |
| 10 | recursive_pilot_physics_q1e6 | 0.004069 | +0.001302 | 32.62 |
| 10 | recursive_pilot_neural_legacy_input | 0.004069 | +0.001302 | 28.31 |
| 10 | recursive_pilot_known_adapt_real_physics | 0.000814 | +0.001628 | 9.30 |
| 10 | recursive_pilot_known_adapt_real_neural | 0.000814 | +0.001628 | 27.87 |
| 15 | acquisition_physics_q1e6 | 0.001465 | +0.001953 | 19.74 |
| 15 | recursive_pilot_physics_q1e6 | 0.001465 | +0.001953 | 32.38 |
| 15 | recursive_pilot_neural_legacy_input | 0.001465 | +0.001953 | 29.09 |
| 15 | recursive_pilot_known_adapt_real_physics | 0.000326 | +0.000651 | 9.66 |
| 15 | recursive_pilot_known_adapt_real_neural | 0.000326 | +0.000651 | 29.01 |

配对差为前者 BER 减后者 BER；正值表示后者更好。双侧 t 区间未作多重性校正。

| SNR | 比较 | 均值 | 95% 区间 | 正向会话 |
|---|---|---:|---|---:|
| 0 | acquisition_physics_q1e6_minus_recursive_pilot_physics_q1e6 | +0.007324 | 样本不足或方差退化 | 1/1 |
| 0 | recursive_pilot_physics_q1e6_minus_recursive_pilot_neural_legacy_input | +0.000163 | 样本不足或方差退化 | 1/1 |
| 0 | recursive_pilot_physics_q1e6_minus_recursive_pilot_known_adapt_real_physics | +0.002767 | 样本不足或方差退化 | 1/1 |
| 0 | recursive_pilot_neural_legacy_input_minus_recursive_pilot_known_adapt_real_neural | +0.002767 | 样本不足或方差退化 | 1/1 |
| 0 | recursive_pilot_known_adapt_real_physics_minus_recursive_pilot_known_adapt_real_neural | +0.000163 | 样本不足或方差退化 | 1/1 |
| 5 | acquisition_physics_q1e6_minus_recursive_pilot_physics_q1e6 | +0.004720 | 样本不足或方差退化 | 1/1 |
| 5 | recursive_pilot_physics_q1e6_minus_recursive_pilot_neural_legacy_input | -0.000163 | 样本不足或方差退化 | 0/1 |
| 5 | recursive_pilot_physics_q1e6_minus_recursive_pilot_known_adapt_real_physics | +0.010254 | 样本不足或方差退化 | 1/1 |
| 5 | recursive_pilot_neural_legacy_input_minus_recursive_pilot_known_adapt_real_neural | +0.010417 | 样本不足或方差退化 | 1/1 |
| 5 | recursive_pilot_known_adapt_real_physics_minus_recursive_pilot_known_adapt_real_neural | +0.000000 | 样本不足或方差退化 | 0/1 |
| 10 | acquisition_physics_q1e6_minus_recursive_pilot_physics_q1e6 | +0.000000 | 样本不足或方差退化 | 0/1 |
| 10 | recursive_pilot_physics_q1e6_minus_recursive_pilot_neural_legacy_input | +0.000000 | 样本不足或方差退化 | 0/1 |
| 10 | recursive_pilot_physics_q1e6_minus_recursive_pilot_known_adapt_real_physics | +0.003255 | 样本不足或方差退化 | 1/1 |
| 10 | recursive_pilot_neural_legacy_input_minus_recursive_pilot_known_adapt_real_neural | +0.003255 | 样本不足或方差退化 | 1/1 |
| 10 | recursive_pilot_known_adapt_real_physics_minus_recursive_pilot_known_adapt_real_neural | +0.000000 | 样本不足或方差退化 | 0/1 |
| 15 | acquisition_physics_q1e6_minus_recursive_pilot_physics_q1e6 | +0.000000 | 样本不足或方差退化 | 0/1 |
| 15 | recursive_pilot_physics_q1e6_minus_recursive_pilot_neural_legacy_input | +0.000000 | 样本不足或方差退化 | 0/1 |
| 15 | recursive_pilot_physics_q1e6_minus_recursive_pilot_known_adapt_real_physics | +0.001139 | 样本不足或方差退化 | 1/1 |
| 15 | recursive_pilot_neural_legacy_input_minus_recursive_pilot_known_adapt_real_neural | +0.001139 | 样本不足或方差退化 | 1/1 |
| 15 | recursive_pilot_known_adapt_real_physics_minus_recursive_pilot_known_adapt_real_neural | +0.000000 | 样本不足或方差退化 | 0/1 |

物理分支仍保留同一 checkpoint 的已学习同步，不是完全非神经基线。
各窗口、自身改善、优势增长与成本分别报告；短期均值改善不能替代增长与饱和确认。
