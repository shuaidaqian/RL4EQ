# 持续均衡候选配对开发比较

**开发诊断，不是独立确认，不证明增长至饱和。**

实验：`p3_known_adapt_real_finetune_validation_10x64`；10 会话 × 64 帧。

所有候选同波形、同 acquisition，噪声由 acquisition 合法估计。

| 候选 | 接收状态/上下文来源 | 相位估计 | 解码分支 | checkpoint |
|---|---|---|---|---|
| acquisition_physics_q1e6 | acquisition | circular_all | physics_branch | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| recursive_pilot_physics_q1e6 | recursive_pilot | circular_all | physics_branch | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| untrained_known_adapt_physics | 共享 recursive_pilot_physics_q1e6 的实际输入 | circular_all | known_adapt_real_physics | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| untrained_known_adapt_neural | 共享 recursive_pilot_physics_q1e6 的实际输入 | circular_all | known_adapt_real_neural | `pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` |
| train_r0_acquisition_known_adapt_neural | 共享 recursive_pilot_physics_q1e6 的实际输入 | circular_all | known_adapt_real_neural | `pretrained/continual_v2_p3_known_adapt_20261119/repeat0_acquisition.pt` |
| train_r0_recursive_pilot_known_adapt_neural | 共享 recursive_pilot_physics_q1e6 的实际输入 | circular_all | known_adapt_real_neural | `pretrained/continual_v2_p3_known_adapt_20261119/repeat0_recursive_pilot.pt` |
| train_r1_acquisition_known_adapt_neural | 共享 recursive_pilot_physics_q1e6 的实际输入 | circular_all | known_adapt_real_neural | `pretrained/continual_v2_p3_known_adapt_20261119/repeat1_acquisition.pt` |
| train_r1_recursive_pilot_known_adapt_neural | 共享 recursive_pilot_physics_q1e6 的实际输入 | circular_all | known_adapt_real_neural | `pretrained/continual_v2_p3_known_adapt_20261119/repeat1_recursive_pilot.pt` |
| train_r2_acquisition_known_adapt_neural | 共享 recursive_pilot_physics_q1e6 的实际输入 | circular_all | known_adapt_real_neural | `pretrained/continual_v2_p3_known_adapt_20261119/repeat2_acquisition.pt` |
| train_r2_recursive_pilot_known_adapt_neural | 共享 recursive_pilot_physics_q1e6 的实际输入 | circular_all | known_adapt_real_neural | `pretrained/continual_v2_p3_known_adapt_20261119/repeat2_recursive_pilot.pt` |

| SNR | 候选 | 平均 BER | 早期减晚期 BER | 平均 ms/帧 |
|---|---|---:|---:|---:|
| 0 | acquisition_physics_q1e6 | 0.150016 | +0.000529 | 20.59 |
| 0 | recursive_pilot_physics_q1e6 | 0.131934 | +0.014893 | 32.02 |
| 0 | untrained_known_adapt_physics | 0.123181 | +0.015869 | 9.63 |
| 0 | untrained_known_adapt_neural | 0.123185 | +0.015462 | 27.98 |
| 0 | train_r0_acquisition_known_adapt_neural | 0.123216 | +0.015210 | 27.78 |
| 0 | train_r0_recursive_pilot_known_adapt_neural | 0.123210 | +0.015283 | 27.61 |
| 0 | train_r1_acquisition_known_adapt_neural | 0.123193 | +0.015145 | 27.70 |
| 0 | train_r1_recursive_pilot_known_adapt_neural | 0.123191 | +0.015291 | 27.66 |
| 0 | train_r2_acquisition_known_adapt_neural | 0.123110 | +0.014868 | 28.01 |
| 0 | train_r2_recursive_pilot_known_adapt_neural | 0.123102 | +0.015218 | 28.06 |
| 5 | acquisition_physics_q1e6 | 0.046519 | -0.002767 | 20.68 |
| 5 | recursive_pilot_physics_q1e6 | 0.038829 | +0.004883 | 33.87 |
| 5 | untrained_known_adapt_physics | 0.026453 | +0.003719 | 10.01 |
| 5 | untrained_known_adapt_neural | 0.026444 | +0.003711 | 29.47 |
| 5 | train_r0_acquisition_known_adapt_neural | 0.026428 | +0.003849 | 29.63 |
| 5 | train_r0_recursive_pilot_known_adapt_neural | 0.026408 | +0.003719 | 29.45 |
| 5 | train_r1_acquisition_known_adapt_neural | 0.026430 | +0.003817 | 28.95 |
| 5 | train_r1_recursive_pilot_known_adapt_neural | 0.026438 | +0.003727 | 29.34 |
| 5 | train_r2_acquisition_known_adapt_neural | 0.026388 | +0.003768 | 29.46 |
| 5 | train_r2_recursive_pilot_known_adapt_neural | 0.026355 | +0.003735 | 29.41 |
| 10 | acquisition_physics_q1e6 | 0.006087 | -0.001253 | 20.30 |
| 10 | recursive_pilot_physics_q1e6 | 0.004476 | +0.001025 | 33.40 |
| 10 | untrained_known_adapt_physics | 0.001170 | +0.000521 | 9.99 |
| 10 | untrained_known_adapt_neural | 0.001172 | +0.000521 | 30.88 |
| 10 | train_r0_acquisition_known_adapt_neural | 0.001174 | +0.000496 | 30.76 |
| 10 | train_r0_recursive_pilot_known_adapt_neural | 0.001164 | +0.000513 | 30.10 |
| 10 | train_r1_acquisition_known_adapt_neural | 0.001178 | +0.000496 | 29.53 |
| 10 | train_r1_recursive_pilot_known_adapt_neural | 0.001176 | +0.000496 | 29.15 |
| 10 | train_r2_acquisition_known_adapt_neural | 0.001168 | +0.000513 | 29.15 |
| 10 | train_r2_recursive_pilot_known_adapt_neural | 0.001172 | +0.000513 | 28.57 |
| 15 | acquisition_physics_q1e6 | 0.000842 | +0.000155 | 20.20 |
| 15 | recursive_pilot_physics_q1e6 | 0.000708 | +0.000252 | 33.55 |
| 15 | untrained_known_adapt_physics | 0.000203 | +0.000146 | 9.85 |
| 15 | untrained_known_adapt_neural | 0.000203 | +0.000146 | 33.53 |
| 15 | train_r0_acquisition_known_adapt_neural | 0.000203 | +0.000106 | 32.99 |
| 15 | train_r0_recursive_pilot_known_adapt_neural | 0.000203 | +0.000106 | 30.78 |
| 15 | train_r1_acquisition_known_adapt_neural | 0.000205 | +0.000098 | 29.10 |
| 15 | train_r1_recursive_pilot_known_adapt_neural | 0.000205 | +0.000098 | 28.56 |
| 15 | train_r2_acquisition_known_adapt_neural | 0.000205 | +0.000098 | 28.49 |
| 15 | train_r2_recursive_pilot_known_adapt_neural | 0.000205 | +0.000098 | 28.12 |

配对差为前者 BER 减后者 BER；正值表示后者更好。双侧 t 区间未作多重性校正。

| SNR | 比较 | 均值 | 95% 区间 | 正向会话 |
|---|---|---:|---|---:|
| 0 | acquisition_physics_q1e6_minus_recursive_pilot_physics_q1e6 | +0.018083 | [+0.015950, +0.020215] | 10/10 |
| 0 | recursive_pilot_physics_q1e6_minus_untrained_known_adapt_physics | +0.008752 | [+0.007578, +0.009927] | 10/10 |
| 0 | untrained_known_adapt_physics_minus_untrained_known_adapt_neural | -0.000004 | [-0.000053, +0.000045] | 4/10 |
| 0 | recursive_pilot_physics_q1e6_minus_train_r0_acquisition_known_adapt_neural | +0.008718 | [+0.007617, +0.009818] | 10/10 |
| 0 | untrained_known_adapt_physics_minus_train_r0_acquisition_known_adapt_neural | -0.000035 | [-0.000156, +0.000087] | 4/10 |
| 0 | untrained_known_adapt_neural_minus_train_r0_acquisition_known_adapt_neural | -0.000031 | [-0.000179, +0.000118] | 4/10 |
| 0 | recursive_pilot_physics_q1e6_minus_train_r0_recursive_pilot_known_adapt_neural | +0.008724 | [+0.007643, +0.009805] | 10/10 |
| 0 | untrained_known_adapt_physics_minus_train_r0_recursive_pilot_known_adapt_neural | -0.000028 | [-0.000169, +0.000112] | 5/10 |
| 0 | untrained_known_adapt_neural_minus_train_r0_recursive_pilot_known_adapt_neural | -0.000024 | [-0.000197, +0.000148] | 5/10 |
| 0 | train_r0_acquisition_known_adapt_neural_minus_train_r0_recursive_pilot_known_adapt_neural | +0.000006 | [-0.000046, +0.000058] | 4/10 |
| 0 | recursive_pilot_physics_q1e6_minus_train_r1_acquisition_known_adapt_neural | +0.008740 | [+0.007640, +0.009841] | 10/10 |
| 0 | untrained_known_adapt_physics_minus_train_r1_acquisition_known_adapt_neural | -0.000012 | [-0.000126, +0.000101] | 5/10 |
| 0 | untrained_known_adapt_neural_minus_train_r1_acquisition_known_adapt_neural | -0.000008 | [-0.000146, +0.000130] | 4/10 |
| 0 | recursive_pilot_physics_q1e6_minus_train_r1_recursive_pilot_known_adapt_neural | +0.008742 | [+0.007643, +0.009841] | 10/10 |
| 0 | untrained_known_adapt_physics_minus_train_r1_recursive_pilot_known_adapt_neural | -0.000010 | [-0.000131, +0.000111] | 4/10 |
| 0 | untrained_known_adapt_neural_minus_train_r1_recursive_pilot_known_adapt_neural | -0.000006 | [-0.000156, +0.000144] | 5/10 |
| 0 | train_r1_acquisition_known_adapt_neural_minus_train_r1_recursive_pilot_known_adapt_neural | +0.000002 | [-0.000029, +0.000033] | 5/10 |
| 0 | recursive_pilot_physics_q1e6_minus_train_r2_acquisition_known_adapt_neural | +0.008824 | [+0.007713, +0.009934] | 10/10 |
| 0 | untrained_known_adapt_physics_minus_train_r2_acquisition_known_adapt_neural | +0.000071 | [-0.000058, +0.000200] | 5/10 |
| 0 | untrained_known_adapt_neural_minus_train_r2_acquisition_known_adapt_neural | +0.000075 | [-0.000090, +0.000241] | 6/10 |
| 0 | recursive_pilot_physics_q1e6_minus_train_r2_recursive_pilot_known_adapt_neural | +0.008832 | [+0.007739, +0.009925] | 10/10 |
| 0 | untrained_known_adapt_physics_minus_train_r2_recursive_pilot_known_adapt_neural | +0.000079 | [-0.000091, +0.000250] | 6/10 |
| 0 | untrained_known_adapt_neural_minus_train_r2_recursive_pilot_known_adapt_neural | +0.000083 | [-0.000114, +0.000281] | 7/10 |
| 0 | train_r2_acquisition_known_adapt_neural_minus_train_r2_recursive_pilot_known_adapt_neural | +0.000008 | [-0.000081, +0.000097] | 4/10 |
| 5 | acquisition_physics_q1e6_minus_recursive_pilot_physics_q1e6 | +0.007690 | [+0.006309, +0.009072] | 10/10 |
| 5 | recursive_pilot_physics_q1e6_minus_untrained_known_adapt_physics | +0.012376 | [+0.011343, +0.013409] | 10/10 |
| 5 | untrained_known_adapt_physics_minus_untrained_known_adapt_neural | +0.000008 | [-0.000008, +0.000024] | 4/10 |
| 5 | recursive_pilot_physics_q1e6_minus_train_r0_acquisition_known_adapt_neural | +0.012400 | [+0.011349, +0.013451] | 10/10 |
| 5 | untrained_known_adapt_physics_minus_train_r0_acquisition_known_adapt_neural | +0.000024 | [-0.000050, +0.000099] | 6/10 |
| 5 | untrained_known_adapt_neural_minus_train_r0_acquisition_known_adapt_neural | +0.000016 | [-0.000049, +0.000082] | 5/10 |
| 5 | recursive_pilot_physics_q1e6_minus_train_r0_recursive_pilot_known_adapt_neural | +0.012421 | [+0.011349, +0.013493] | 10/10 |
| 5 | untrained_known_adapt_physics_minus_train_r0_recursive_pilot_known_adapt_neural | +0.000045 | [-0.000042, +0.000131] | 7/10 |
| 5 | untrained_known_adapt_neural_minus_train_r0_recursive_pilot_known_adapt_neural | +0.000037 | [-0.000041, +0.000114] | 7/10 |
| 5 | train_r0_acquisition_known_adapt_neural_minus_train_r0_recursive_pilot_known_adapt_neural | +0.000020 | [-0.000039, +0.000079] | 5/10 |
| 5 | recursive_pilot_physics_q1e6_minus_train_r1_acquisition_known_adapt_neural | +0.012398 | [+0.011326, +0.013471] | 10/10 |
| 5 | untrained_known_adapt_physics_minus_train_r1_acquisition_known_adapt_neural | +0.000022 | [-0.000071, +0.000116] | 4/10 |
| 5 | untrained_known_adapt_neural_minus_train_r1_acquisition_known_adapt_neural | +0.000014 | [-0.000070, +0.000099] | 4/10 |
| 5 | recursive_pilot_physics_q1e6_minus_train_r1_recursive_pilot_known_adapt_neural | +0.012390 | [+0.011320, +0.013460] | 10/10 |
| 5 | untrained_known_adapt_physics_minus_train_r1_recursive_pilot_known_adapt_neural | +0.000014 | [-0.000079, +0.000107] | 3/10 |
| 5 | untrained_known_adapt_neural_minus_train_r1_recursive_pilot_known_adapt_neural | +0.000006 | [-0.000076, +0.000088] | 4/10 |
| 5 | train_r1_acquisition_known_adapt_neural_minus_train_r1_recursive_pilot_known_adapt_neural | -0.000008 | [-0.000033, +0.000017] | 3/10 |
| 5 | recursive_pilot_physics_q1e6_minus_train_r2_acquisition_known_adapt_neural | +0.012441 | [+0.011379, +0.013503] | 10/10 |
| 5 | untrained_known_adapt_physics_minus_train_r2_acquisition_known_adapt_neural | +0.000065 | [-0.000015, +0.000145] | 8/10 |
| 5 | untrained_known_adapt_neural_minus_train_r2_acquisition_known_adapt_neural | +0.000057 | [-0.000021, +0.000135] | 7/10 |
| 5 | recursive_pilot_physics_q1e6_minus_train_r2_recursive_pilot_known_adapt_neural | +0.012474 | [+0.011407, +0.013540] | 10/10 |
| 5 | untrained_known_adapt_physics_minus_train_r2_recursive_pilot_known_adapt_neural | +0.000098 | [+0.000015, +0.000181] | 8/10 |
| 5 | untrained_known_adapt_neural_minus_train_r2_recursive_pilot_known_adapt_neural | +0.000090 | [+0.000007, +0.000172] | 8/10 |
| 5 | train_r2_acquisition_known_adapt_neural_minus_train_r2_recursive_pilot_known_adapt_neural | +0.000033 | [+0.000015, +0.000050] | 9/10 |
| 10 | acquisition_physics_q1e6_minus_recursive_pilot_physics_q1e6 | +0.001611 | [+0.001010, +0.002213] | 10/10 |
| 10 | recursive_pilot_physics_q1e6_minus_untrained_known_adapt_physics | +0.003306 | [+0.002387, +0.004225] | 10/10 |
| 10 | untrained_known_adapt_physics_minus_untrained_known_adapt_neural | -0.000002 | [-0.000007, +0.000003] | 0/10 |
| 10 | recursive_pilot_physics_q1e6_minus_train_r0_acquisition_known_adapt_neural | +0.003302 | [+0.002387, +0.004217] | 10/10 |
| 10 | untrained_known_adapt_physics_minus_train_r0_acquisition_known_adapt_neural | -0.000004 | [-0.000016, +0.000007] | 2/10 |
| 10 | untrained_known_adapt_neural_minus_train_r0_acquisition_known_adapt_neural | -0.000002 | [-0.000013, +0.000009] | 2/10 |
| 10 | recursive_pilot_physics_q1e6_minus_train_r0_recursive_pilot_known_adapt_neural | +0.003312 | [+0.002389, +0.004235] | 10/10 |
| 10 | untrained_known_adapt_physics_minus_train_r0_recursive_pilot_known_adapt_neural | +0.000006 | [-0.000008, +0.000020] | 2/10 |
| 10 | untrained_known_adapt_neural_minus_train_r0_recursive_pilot_known_adapt_neural | +0.000008 | [-0.000004, +0.000020] | 2/10 |
| 10 | train_r0_acquisition_known_adapt_neural_minus_train_r0_recursive_pilot_known_adapt_neural | +0.000010 | [-0.000006, +0.000026] | 4/10 |
| 10 | recursive_pilot_physics_q1e6_minus_train_r1_acquisition_known_adapt_neural | +0.003298 | [+0.002387, +0.004209] | 10/10 |
| 10 | untrained_known_adapt_physics_minus_train_r1_acquisition_known_adapt_neural | -0.000008 | [-0.000024, +0.000008] | 2/10 |
| 10 | untrained_known_adapt_neural_minus_train_r1_acquisition_known_adapt_neural | -0.000006 | [-0.000022, +0.000009] | 2/10 |
| 10 | recursive_pilot_physics_q1e6_minus_train_r1_recursive_pilot_known_adapt_neural | +0.003300 | [+0.002388, +0.004212] | 10/10 |
| 10 | untrained_known_adapt_physics_minus_train_r1_recursive_pilot_known_adapt_neural | -0.000006 | [-0.000024, +0.000012] | 2/10 |
| 10 | untrained_known_adapt_neural_minus_train_r1_recursive_pilot_known_adapt_neural | -0.000004 | [-0.000022, +0.000014] | 2/10 |
| 10 | train_r1_acquisition_known_adapt_neural_minus_train_r1_recursive_pilot_known_adapt_neural | +0.000002 | [-0.000003, +0.000007] | 1/10 |
| 10 | recursive_pilot_physics_q1e6_minus_train_r2_acquisition_known_adapt_neural | +0.003308 | [+0.002388, +0.004229] | 10/10 |
| 10 | untrained_known_adapt_physics_minus_train_r2_acquisition_known_adapt_neural | +0.000002 | [-0.000009, +0.000013] | 3/10 |
| 10 | untrained_known_adapt_neural_minus_train_r2_acquisition_known_adapt_neural | +0.000004 | [-0.000005, +0.000013] | 3/10 |
| 10 | recursive_pilot_physics_q1e6_minus_train_r2_recursive_pilot_known_adapt_neural | +0.003304 | [+0.002384, +0.004224] | 10/10 |
| 10 | untrained_known_adapt_physics_minus_train_r2_recursive_pilot_known_adapt_neural | -0.000002 | [-0.000013, +0.000009] | 2/10 |
| 10 | untrained_known_adapt_neural_minus_train_r2_recursive_pilot_known_adapt_neural | +0.000000 | [-0.000010, +0.000010] | 2/10 |
| 10 | train_r2_acquisition_known_adapt_neural_minus_train_r2_recursive_pilot_known_adapt_neural | -0.000004 | [-0.000010, +0.000002] | 0/10 |
| 15 | acquisition_physics_q1e6_minus_recursive_pilot_physics_q1e6 | +0.000134 | [+0.000015, +0.000253] | 9/10 |
| 15 | recursive_pilot_physics_q1e6_minus_untrained_known_adapt_physics | +0.000505 | [+0.000370, +0.000639] | 10/10 |
| 15 | untrained_known_adapt_physics_minus_untrained_known_adapt_neural | +0.000000 | 样本不足或方差退化 | 0/10 |
| 15 | recursive_pilot_physics_q1e6_minus_train_r0_acquisition_known_adapt_neural | +0.000505 | [+0.000368, +0.000641] | 10/10 |
| 15 | untrained_known_adapt_physics_minus_train_r0_acquisition_known_adapt_neural | +0.000000 | [-0.000007, +0.000007] | 1/10 |
| 15 | untrained_known_adapt_neural_minus_train_r0_acquisition_known_adapt_neural | +0.000000 | [-0.000007, +0.000007] | 1/10 |
| 15 | recursive_pilot_physics_q1e6_minus_train_r0_recursive_pilot_known_adapt_neural | +0.000505 | [+0.000366, +0.000643] | 10/10 |
| 15 | untrained_known_adapt_physics_minus_train_r0_recursive_pilot_known_adapt_neural | +0.000000 | [-0.000007, +0.000007] | 1/10 |
| 15 | untrained_known_adapt_neural_minus_train_r0_recursive_pilot_known_adapt_neural | +0.000000 | [-0.000007, +0.000007] | 1/10 |
| 15 | train_r0_acquisition_known_adapt_neural_minus_train_r0_recursive_pilot_known_adapt_neural | +0.000000 | [-0.000007, +0.000007] | 1/10 |
| 15 | recursive_pilot_physics_q1e6_minus_train_r1_acquisition_known_adapt_neural | +0.000503 | [+0.000367, +0.000639] | 10/10 |
| 15 | untrained_known_adapt_physics_minus_train_r1_acquisition_known_adapt_neural | -0.000002 | [-0.000010, +0.000006] | 1/10 |
| 15 | untrained_known_adapt_neural_minus_train_r1_acquisition_known_adapt_neural | -0.000002 | [-0.000010, +0.000006] | 1/10 |
| 15 | recursive_pilot_physics_q1e6_minus_train_r1_recursive_pilot_known_adapt_neural | +0.000503 | [+0.000367, +0.000639] | 10/10 |
| 15 | untrained_known_adapt_physics_minus_train_r1_recursive_pilot_known_adapt_neural | -0.000002 | [-0.000010, +0.000006] | 1/10 |
| 15 | untrained_known_adapt_neural_minus_train_r1_recursive_pilot_known_adapt_neural | -0.000002 | [-0.000010, +0.000006] | 1/10 |
| 15 | train_r1_acquisition_known_adapt_neural_minus_train_r1_recursive_pilot_known_adapt_neural | +0.000000 | 样本不足或方差退化 | 0/10 |
| 15 | recursive_pilot_physics_q1e6_minus_train_r2_acquisition_known_adapt_neural | +0.000503 | [+0.000365, +0.000640] | 10/10 |
| 15 | untrained_known_adapt_physics_minus_train_r2_acquisition_known_adapt_neural | -0.000002 | [-0.000010, +0.000006] | 1/10 |
| 15 | untrained_known_adapt_neural_minus_train_r2_acquisition_known_adapt_neural | -0.000002 | [-0.000010, +0.000006] | 1/10 |
| 15 | recursive_pilot_physics_q1e6_minus_train_r2_recursive_pilot_known_adapt_neural | +0.000503 | [+0.000365, +0.000640] | 10/10 |
| 15 | untrained_known_adapt_physics_minus_train_r2_recursive_pilot_known_adapt_neural | -0.000002 | [-0.000010, +0.000006] | 1/10 |
| 15 | untrained_known_adapt_neural_minus_train_r2_recursive_pilot_known_adapt_neural | -0.000002 | [-0.000010, +0.000006] | 1/10 |
| 15 | train_r2_acquisition_known_adapt_neural_minus_train_r2_recursive_pilot_known_adapt_neural | +0.000000 | 样本不足或方差退化 | 0/10 |

物理分支仍保留同一 checkpoint 的已学习同步，不是完全非神经基线。
各窗口、自身改善、优势增长与成本分别报告；短期均值改善不能替代增长与饱和确认。
