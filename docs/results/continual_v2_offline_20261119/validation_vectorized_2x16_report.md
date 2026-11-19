# 持续均衡候选配对开发比较

**开发诊断，不是独立确认，不证明增长至饱和。**

实验：`p3_matched_finetune_validation_vectorized_2x16`；2 会话 × 16 帧。

所有候选同波形、同 acquisition，噪声由 acquisition 合法估计。

| 候选 | 接收状态/上下文来源 | 相位估计 | 解码分支 | checkpoint |
|---|---|---|---|---|
| anchor_current_pilot | current_pilot | circular_all | neural | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| anchor_recursive_pilot | recursive_pilot | circular_all | neural | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| r0_acquisition_current_pilot | current_pilot | circular_all | neural | `pretrained\continual_v2_p3_20261119\repeat0_acquisition.pt` |
| r0_acquisition_recursive_pilot | recursive_pilot | circular_all | neural | `pretrained\continual_v2_p3_20261119\repeat0_acquisition.pt` |
| r0_recursive_pilot_current_pilot | current_pilot | circular_all | neural | `pretrained\continual_v2_p3_20261119\repeat0_recursive_pilot.pt` |
| r0_recursive_pilot_recursive_pilot | recursive_pilot | circular_all | neural | `pretrained\continual_v2_p3_20261119\repeat0_recursive_pilot.pt` |
| r1_acquisition_current_pilot | current_pilot | circular_all | neural | `pretrained\continual_v2_p3_20261119\repeat1_acquisition.pt` |
| r1_acquisition_recursive_pilot | recursive_pilot | circular_all | neural | `pretrained\continual_v2_p3_20261119\repeat1_acquisition.pt` |
| r1_recursive_pilot_current_pilot | current_pilot | circular_all | neural | `pretrained\continual_v2_p3_20261119\repeat1_recursive_pilot.pt` |
| r1_recursive_pilot_recursive_pilot | recursive_pilot | circular_all | neural | `pretrained\continual_v2_p3_20261119\repeat1_recursive_pilot.pt` |
| r2_acquisition_current_pilot | current_pilot | circular_all | neural | `pretrained\continual_v2_p3_20261119\repeat2_acquisition.pt` |
| r2_acquisition_recursive_pilot | recursive_pilot | circular_all | neural | `pretrained\continual_v2_p3_20261119\repeat2_acquisition.pt` |
| r2_recursive_pilot_current_pilot | current_pilot | circular_all | neural | `pretrained\continual_v2_p3_20261119\repeat2_recursive_pilot.pt` |
| r2_recursive_pilot_recursive_pilot | recursive_pilot | circular_all | neural | `pretrained\continual_v2_p3_20261119\repeat2_recursive_pilot.pt` |

| SNR | 候选 | 平均 BER | 早期减晚期 BER | 平均 ms/帧 |
|---|---|---:|---:|---:|
| 0 | anchor_current_pilot | 0.139323 | -0.009603 | 122.75 |
| 0 | anchor_recursive_pilot | 0.132690 | -0.003255 | 70.42 |
| 0 | r0_acquisition_current_pilot | 0.139201 | -0.008626 | 74.76 |
| 0 | r0_acquisition_recursive_pilot | 0.132568 | -0.002767 | 69.90 |
| 0 | r0_recursive_pilot_current_pilot | 0.138997 | -0.009115 | 71.46 |
| 0 | r0_recursive_pilot_recursive_pilot | 0.132609 | -0.002767 | 75.70 |
| 0 | r1_acquisition_current_pilot | 0.139445 | -0.009115 | 73.94 |
| 0 | r1_acquisition_recursive_pilot | 0.132406 | -0.002767 | 74.26 |
| 0 | r1_recursive_pilot_current_pilot | 0.139404 | -0.009277 | 73.28 |
| 0 | r1_recursive_pilot_recursive_pilot | 0.132365 | -0.003092 | 71.92 |
| 0 | r2_acquisition_current_pilot | 0.138997 | -0.009115 | 73.14 |
| 0 | r2_acquisition_recursive_pilot | 0.132772 | -0.001953 | 71.15 |
| 0 | r2_recursive_pilot_current_pilot | 0.138997 | -0.009115 | 70.00 |
| 0 | r2_recursive_pilot_recursive_pilot | 0.132894 | -0.002441 | 75.27 |
| 5 | anchor_current_pilot | 0.043864 | -0.002604 | 68.74 |
| 5 | anchor_recursive_pilot | 0.042033 | -0.004232 | 65.58 |
| 5 | r0_acquisition_current_pilot | 0.043742 | -0.002767 | 69.14 |
| 5 | r0_acquisition_recursive_pilot | 0.042074 | -0.004883 | 64.42 |
| 5 | r0_recursive_pilot_current_pilot | 0.043742 | -0.002767 | 63.13 |
| 5 | r0_recursive_pilot_recursive_pilot | 0.041951 | -0.004395 | 65.88 |
| 5 | r1_acquisition_current_pilot | 0.043823 | -0.002767 | 64.03 |
| 5 | r1_acquisition_recursive_pilot | 0.041789 | -0.004720 | 63.67 |
| 5 | r1_recursive_pilot_current_pilot | 0.043823 | -0.002604 | 63.65 |
| 5 | r1_recursive_pilot_recursive_pilot | 0.041911 | -0.004395 | 64.27 |
| 5 | r2_acquisition_current_pilot | 0.043823 | -0.002930 | 63.27 |
| 5 | r2_acquisition_recursive_pilot | 0.041911 | -0.004883 | 65.53 |
| 5 | r2_recursive_pilot_current_pilot | 0.043823 | -0.002930 | 65.69 |
| 5 | r2_recursive_pilot_recursive_pilot | 0.041951 | -0.004883 | 64.33 |
| 10 | anchor_current_pilot | 0.006510 | +0.000651 | 66.54 |
| 10 | anchor_recursive_pilot | 0.006551 | -0.000488 | 66.31 |
| 10 | r0_acquisition_current_pilot | 0.006510 | +0.000651 | 64.99 |
| 10 | r0_acquisition_recursive_pilot | 0.006592 | -0.000651 | 67.40 |
| 10 | r0_recursive_pilot_current_pilot | 0.006429 | +0.000814 | 64.07 |
| 10 | r0_recursive_pilot_recursive_pilot | 0.006592 | -0.000651 | 63.34 |
| 10 | r1_acquisition_current_pilot | 0.006592 | +0.000651 | 64.77 |
| 10 | r1_acquisition_recursive_pilot | 0.006510 | -0.000163 | 65.64 |
| 10 | r1_recursive_pilot_current_pilot | 0.006510 | +0.000651 | 63.54 |
| 10 | r1_recursive_pilot_recursive_pilot | 0.006470 | -0.000326 | 62.52 |
| 10 | r2_acquisition_current_pilot | 0.006429 | +0.000814 | 63.41 |
| 10 | r2_acquisition_recursive_pilot | 0.006551 | -0.000814 | 65.45 |
| 10 | r2_recursive_pilot_current_pilot | 0.006429 | +0.000814 | 65.93 |
| 10 | r2_recursive_pilot_recursive_pilot | 0.006551 | -0.000814 | 66.43 |
| 15 | anchor_current_pilot | 0.000895 | +0.000163 | 64.91 |
| 15 | anchor_recursive_pilot | 0.001058 | +0.000326 | 63.69 |
| 15 | r0_acquisition_current_pilot | 0.000895 | +0.000163 | 63.79 |
| 15 | r0_acquisition_recursive_pilot | 0.001058 | +0.000326 | 64.50 |
| 15 | r0_recursive_pilot_current_pilot | 0.000895 | +0.000163 | 63.78 |
| 15 | r0_recursive_pilot_recursive_pilot | 0.001058 | +0.000326 | 66.18 |
| 15 | r1_acquisition_current_pilot | 0.000895 | +0.000163 | 64.56 |
| 15 | r1_acquisition_recursive_pilot | 0.001058 | +0.000326 | 62.62 |
| 15 | r1_recursive_pilot_current_pilot | 0.000895 | +0.000163 | 62.69 |
| 15 | r1_recursive_pilot_recursive_pilot | 0.001058 | +0.000326 | 60.97 |
| 15 | r2_acquisition_current_pilot | 0.000895 | +0.000163 | 62.87 |
| 15 | r2_acquisition_recursive_pilot | 0.001058 | +0.000326 | 61.85 |
| 15 | r2_recursive_pilot_current_pilot | 0.000936 | +0.000163 | 62.56 |
| 15 | r2_recursive_pilot_recursive_pilot | 0.001058 | +0.000326 | 71.92 |

配对差为前者 BER 减后者 BER；正值表示后者更好。双侧 t 区间未作多重性校正。

| SNR | 比较 | 均值 | 95% 区间 | 正向会话 |
|---|---|---:|---|---:|
| 0 | anchor_current_pilot_minus_anchor_recursive_pilot | +0.006632 | [+0.003013, +0.010252] | 2/2 |
| 0 | anchor_current_pilot_minus_r0_acquisition_current_pilot | +0.000122 | [-0.000395, +0.000639] | 2/2 |
| 0 | anchor_recursive_pilot_minus_r0_acquisition_recursive_pilot | +0.000122 | [-0.001429, +0.001673] | 1/2 |
| 0 | r0_acquisition_current_pilot_minus_r0_acquisition_recursive_pilot | +0.006632 | [+0.004047, +0.009218] | 2/2 |
| 0 | anchor_current_pilot_minus_r0_recursive_pilot_current_pilot | +0.000326 | [-0.002777, +0.003428] | 2/2 |
| 0 | anchor_recursive_pilot_minus_r0_recursive_pilot_recursive_pilot | +0.000081 | [-0.000953, +0.001115] | 1/2 |
| 0 | r0_recursive_pilot_current_pilot_minus_r0_recursive_pilot_recursive_pilot | +0.006388 | [+0.005871, +0.006905] | 2/2 |
| 0 | anchor_current_pilot_minus_r1_acquisition_current_pilot | -0.000122 | [-0.000639, +0.000395] | 0/2 |
| 0 | anchor_recursive_pilot_minus_r1_acquisition_recursive_pilot | +0.000285 | [-0.001266, +0.001836] | 2/2 |
| 0 | r1_acquisition_current_pilot_minus_r1_acquisition_recursive_pilot | +0.007039 | [+0.004454, +0.009624] | 2/2 |
| 0 | anchor_current_pilot_minus_r1_recursive_pilot_current_pilot | -0.000081 | [-0.001115, +0.000953] | 0/2 |
| 0 | anchor_recursive_pilot_minus_r1_recursive_pilot_recursive_pilot | +0.000326 | [-0.002777, +0.003428] | 2/2 |
| 0 | r1_recursive_pilot_current_pilot_minus_r1_recursive_pilot_recursive_pilot | +0.007039 | [+0.005488, +0.008590] | 2/2 |
| 0 | anchor_current_pilot_minus_r2_acquisition_current_pilot | +0.000326 | [-0.001743, +0.002394] | 2/2 |
| 0 | anchor_recursive_pilot_minus_r2_acquisition_recursive_pilot | -0.000081 | [-0.002149, +0.001987] | 1/2 |
| 0 | r2_acquisition_current_pilot_minus_r2_acquisition_recursive_pilot | +0.006226 | [+0.002606, +0.009845] | 2/2 |
| 0 | anchor_current_pilot_minus_r2_recursive_pilot_current_pilot | +0.000326 | [-0.001743, +0.002394] | 2/2 |
| 0 | anchor_recursive_pilot_minus_r2_recursive_pilot_recursive_pilot | -0.000203 | [-0.002789, +0.002382] | 0/2 |
| 0 | r2_recursive_pilot_current_pilot_minus_r2_recursive_pilot_recursive_pilot | +0.006104 | [+0.003001, +0.009206] | 2/2 |
| 5 | anchor_current_pilot_minus_anchor_recursive_pilot | +0.001831 | [-0.018333, +0.021995] | 2/2 |
| 5 | anchor_current_pilot_minus_r0_acquisition_current_pilot | +0.000122 | [-0.000395, +0.000639] | 2/2 |
| 5 | anchor_recursive_pilot_minus_r0_acquisition_recursive_pilot | -0.000041 | [-0.000558, +0.000476] | 0/2 |
| 5 | r0_acquisition_current_pilot_minus_r0_acquisition_recursive_pilot | +0.001668 | [-0.018495, +0.021832] | 2/2 |
| 5 | anchor_current_pilot_minus_r0_recursive_pilot_current_pilot | +0.000122 | [-0.000395, +0.000639] | 2/2 |
| 5 | anchor_recursive_pilot_minus_r0_recursive_pilot_recursive_pilot | +0.000081 | [-0.001987, +0.002149] | 1/2 |
| 5 | r0_recursive_pilot_current_pilot_minus_r0_recursive_pilot_recursive_pilot | +0.001790 | [-0.020958, +0.024539] | 1/2 |
| 5 | anchor_current_pilot_minus_r1_acquisition_current_pilot | +0.000041 | [-0.000476, +0.000558] | 1/2 |
| 5 | anchor_recursive_pilot_minus_r1_acquisition_recursive_pilot | +0.000244 | [-0.002858, +0.003346] | 1/2 |
| 5 | r1_acquisition_current_pilot_minus_r1_acquisition_recursive_pilot | +0.002035 | [-0.015544, +0.019613] | 2/2 |
| 5 | anchor_current_pilot_minus_r1_recursive_pilot_current_pilot | +0.000041 | [-0.000476, +0.000558] | 1/2 |
| 5 | anchor_recursive_pilot_minus_r1_recursive_pilot_recursive_pilot | +0.000122 | [-0.001429, +0.001673] | 1/2 |
| 5 | r1_recursive_pilot_current_pilot_minus_r1_recursive_pilot_recursive_pilot | +0.001912 | [-0.017217, +0.021042] | 2/2 |
| 5 | anchor_current_pilot_minus_r2_acquisition_current_pilot | +0.000041 | [-0.000476, +0.000558] | 1/2 |
| 5 | anchor_recursive_pilot_minus_r2_acquisition_recursive_pilot | +0.000122 | [-0.000395, +0.000639] | 2/2 |
| 5 | r2_acquisition_current_pilot_minus_r2_acquisition_recursive_pilot | +0.001912 | [-0.018251, +0.022076] | 2/2 |
| 5 | anchor_current_pilot_minus_r2_recursive_pilot_current_pilot | +0.000041 | [-0.000476, +0.000558] | 1/2 |
| 5 | anchor_recursive_pilot_minus_r2_recursive_pilot_recursive_pilot | +0.000081 | 样本不足或方差退化 | 2/2 |
| 5 | r2_recursive_pilot_current_pilot_minus_r2_recursive_pilot_recursive_pilot | +0.001872 | [-0.018809, +0.022552] | 2/2 |
| 10 | anchor_current_pilot_minus_anchor_recursive_pilot | -0.000041 | [-0.005728, +0.005646] | 1/2 |
| 10 | anchor_current_pilot_minus_r0_acquisition_current_pilot | +0.000000 | 样本不足或方差退化 | 0/2 |
| 10 | anchor_recursive_pilot_minus_r0_acquisition_recursive_pilot | -0.000041 | [-0.000558, +0.000476] | 0/2 |
| 10 | r0_acquisition_current_pilot_minus_r0_acquisition_recursive_pilot | -0.000081 | [-0.006286, +0.006123] | 1/2 |
| 10 | anchor_current_pilot_minus_r0_recursive_pilot_current_pilot | +0.000081 | [-0.000953, +0.001115] | 1/2 |
| 10 | anchor_recursive_pilot_minus_r0_recursive_pilot_recursive_pilot | -0.000041 | [-0.000558, +0.000476] | 0/2 |
| 10 | r0_recursive_pilot_current_pilot_minus_r0_recursive_pilot_recursive_pilot | -0.000163 | [-0.005333, +0.005007] | 1/2 |
| 10 | anchor_current_pilot_minus_r1_acquisition_current_pilot | -0.000081 | [-0.001115, +0.000953] | 0/2 |
| 10 | anchor_recursive_pilot_minus_r1_acquisition_recursive_pilot | +0.000041 | [-0.000476, +0.000558] | 1/2 |
| 10 | r1_acquisition_current_pilot_minus_r1_acquisition_recursive_pilot | +0.000081 | [-0.006123, +0.006286] | 1/2 |
| 10 | anchor_current_pilot_minus_r1_recursive_pilot_current_pilot | +0.000000 | 样本不足或方差退化 | 0/2 |
| 10 | anchor_recursive_pilot_minus_r1_recursive_pilot_recursive_pilot | +0.000081 | 样本不足或方差退化 | 2/2 |
| 10 | r1_recursive_pilot_current_pilot_minus_r1_recursive_pilot_recursive_pilot | +0.000041 | [-0.005646, +0.005728] | 1/2 |
| 10 | anchor_current_pilot_minus_r2_acquisition_current_pilot | +0.000081 | [-0.000953, +0.001115] | 1/2 |
| 10 | anchor_recursive_pilot_minus_r2_acquisition_recursive_pilot | +0.000000 | 样本不足或方差退化 | 0/2 |
| 10 | r2_acquisition_current_pilot_minus_r2_acquisition_recursive_pilot | -0.000122 | [-0.004775, +0.004531] | 1/2 |
| 10 | anchor_current_pilot_minus_r2_recursive_pilot_current_pilot | +0.000081 | [-0.000953, +0.001115] | 1/2 |
| 10 | anchor_recursive_pilot_minus_r2_recursive_pilot_recursive_pilot | +0.000000 | 样本不足或方差退化 | 0/2 |
| 10 | r2_recursive_pilot_current_pilot_minus_r2_recursive_pilot_recursive_pilot | -0.000122 | [-0.004775, +0.004531] | 1/2 |
| 15 | anchor_current_pilot_minus_anchor_recursive_pilot | -0.000163 | [-0.001197, +0.000871] | 0/2 |
| 15 | anchor_current_pilot_minus_r0_acquisition_current_pilot | +0.000000 | 样本不足或方差退化 | 0/2 |
| 15 | anchor_recursive_pilot_minus_r0_acquisition_recursive_pilot | +0.000000 | 样本不足或方差退化 | 0/2 |
| 15 | r0_acquisition_current_pilot_minus_r0_acquisition_recursive_pilot | -0.000163 | [-0.001197, +0.000871] | 0/2 |
| 15 | anchor_current_pilot_minus_r0_recursive_pilot_current_pilot | +0.000000 | 样本不足或方差退化 | 0/2 |
| 15 | anchor_recursive_pilot_minus_r0_recursive_pilot_recursive_pilot | +0.000000 | 样本不足或方差退化 | 0/2 |
| 15 | r0_recursive_pilot_current_pilot_minus_r0_recursive_pilot_recursive_pilot | -0.000163 | [-0.001197, +0.000871] | 0/2 |
| 15 | anchor_current_pilot_minus_r1_acquisition_current_pilot | +0.000000 | 样本不足或方差退化 | 0/2 |
| 15 | anchor_recursive_pilot_minus_r1_acquisition_recursive_pilot | +0.000000 | 样本不足或方差退化 | 0/2 |
| 15 | r1_acquisition_current_pilot_minus_r1_acquisition_recursive_pilot | -0.000163 | [-0.001197, +0.000871] | 0/2 |
| 15 | anchor_current_pilot_minus_r1_recursive_pilot_current_pilot | +0.000000 | 样本不足或方差退化 | 0/2 |
| 15 | anchor_recursive_pilot_minus_r1_recursive_pilot_recursive_pilot | +0.000000 | 样本不足或方差退化 | 0/2 |
| 15 | r1_recursive_pilot_current_pilot_minus_r1_recursive_pilot_recursive_pilot | -0.000163 | [-0.001197, +0.000871] | 0/2 |
| 15 | anchor_current_pilot_minus_r2_acquisition_current_pilot | +0.000000 | 样本不足或方差退化 | 0/2 |
| 15 | anchor_recursive_pilot_minus_r2_acquisition_recursive_pilot | +0.000000 | 样本不足或方差退化 | 0/2 |
| 15 | r2_acquisition_current_pilot_minus_r2_acquisition_recursive_pilot | -0.000163 | [-0.001197, +0.000871] | 0/2 |
| 15 | anchor_current_pilot_minus_r2_recursive_pilot_current_pilot | -0.000041 | [-0.000558, +0.000476] | 0/2 |
| 15 | anchor_recursive_pilot_minus_r2_recursive_pilot_recursive_pilot | +0.000000 | 样本不足或方差退化 | 0/2 |
| 15 | r2_recursive_pilot_current_pilot_minus_r2_recursive_pilot_recursive_pilot | -0.000122 | [-0.001673, +0.001429] | 0/2 |

物理分支仍保留同一 checkpoint 的已学习同步，不是完全非神经基线。
各窗口、自身改善、优势增长与成本分别报告；短期均值改善不能替代增长与饱和确认。
