# 持续均衡候选配对开发比较

**开发诊断，不是独立确认，不证明增长至饱和。**

实验：`p2_vectorized_reference_full_trajectory_1x16`；1 会话 × 16 帧。

所有候选同波形、同 acquisition，噪声由 acquisition 合法估计。

| 候选 | 接收状态/上下文来源 | 相位估计 | 解码分支 | checkpoint |
|---|---|---|---|---|
| recursive_pilot_legacy | recursive_pilot | circular_all | neural | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| recursive_pilot_vectorized | recursive_pilot | circular_all | neural | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| recursive_soft_legacy | recursive_soft | circular_all | neural | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| recursive_soft_vectorized | recursive_soft | circular_all | neural | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |

| SNR | 候选 | 平均 BER | 早期减晚期 BER | 平均 ms/帧 |
|---|---|---:|---:|---:|
| 0 | recursive_pilot_legacy | 0.139567 | -0.007487 | 249.35 |
| 0 | recursive_pilot_vectorized | 0.139567 | -0.007487 | 175.92 |
| 0 | recursive_soft_legacy | 0.137695 | -0.006510 | 264.86 |
| 0 | recursive_soft_vectorized | 0.137695 | -0.006510 | 175.06 |
| 5 | recursive_pilot_legacy | 0.040365 | -0.002279 | 237.27 |
| 5 | recursive_pilot_vectorized | 0.040365 | -0.002279 | 163.57 |
| 5 | recursive_soft_legacy | 0.039307 | +0.001953 | 243.49 |
| 5 | recursive_soft_vectorized | 0.039307 | +0.001953 | 169.82 |
| 10 | recursive_pilot_legacy | 0.003092 | +0.000326 | 261.60 |
| 10 | recursive_pilot_vectorized | 0.003092 | +0.000326 | 180.09 |
| 10 | recursive_soft_legacy | 0.003499 | +0.000977 | 268.75 |
| 10 | recursive_soft_vectorized | 0.003499 | +0.000977 | 192.43 |
| 15 | recursive_pilot_legacy | 0.000163 | +0.000326 | 294.66 |
| 15 | recursive_pilot_vectorized | 0.000163 | +0.000326 | 206.64 |
| 15 | recursive_soft_legacy | 0.000163 | +0.000326 | 305.15 |
| 15 | recursive_soft_vectorized | 0.000163 | +0.000326 | 219.07 |

配对差为前者 BER 减后者 BER；正值表示后者更好。双侧 t 区间未作多重性校正。

| SNR | 比较 | 均值 | 95% 区间 | 正向会话 |
|---|---|---:|---|---:|
| 0 | recursive_pilot_legacy_minus_recursive_pilot_vectorized | +0.000000 | 样本不足或方差退化 | 0/1 |
| 0 | recursive_soft_legacy_minus_recursive_soft_vectorized | +0.000000 | 样本不足或方差退化 | 0/1 |
| 5 | recursive_pilot_legacy_minus_recursive_pilot_vectorized | +0.000000 | 样本不足或方差退化 | 0/1 |
| 5 | recursive_soft_legacy_minus_recursive_soft_vectorized | +0.000000 | 样本不足或方差退化 | 0/1 |
| 10 | recursive_pilot_legacy_minus_recursive_pilot_vectorized | +0.000000 | 样本不足或方差退化 | 0/1 |
| 10 | recursive_soft_legacy_minus_recursive_soft_vectorized | +0.000000 | 样本不足或方差退化 | 0/1 |
| 15 | recursive_pilot_legacy_minus_recursive_pilot_vectorized | +0.000000 | 样本不足或方差退化 | 0/1 |
| 15 | recursive_soft_legacy_minus_recursive_soft_vectorized | +0.000000 | 样本不足或方差退化 | 0/1 |

物理分支仍保留同一 checkpoint 的已学习同步，不是完全非神经基线。
各窗口、自身改善、优势增长与成本分别报告；短期均值改善不能替代增长与饱和确认。
