# 持续均衡 v2：四臂开发筛查

状态：**开发诊断，不是独立确认，不证明增长至饱和。**

实验：`p1_four_arms_estimated_noise_3x32`；3 个会话 × 32 帧；每臂每帧一次神经前向。

噪声来自 acquisition 残差估计；所有臂同波形、同初始化、同 checkpoint。

| SNR | acquisition | 当前 Pilot | 递归 Pilot | 递归 soft | 估噪/真值 |
|---|---:|---:|---:|---:|---:|
| 0 | 0.232042 | 0.181546 | 0.172499 | 0.169800 | 1.005 |
| 5 | 0.048164 | 0.045573 | 0.043783 | 0.041802 | 1.005 |
| 10 | 0.004869 | 0.004829 | 0.004801 | 0.004055 | 1.005 |
| 15 | 0.000583 | 0.000583 | 0.000705 | 0.000570 | 1.006 |

## 会话级配对差

正值表示后一方法 BER 更低。区间为未作多重性校正的双侧 t 区间，仅作开发诊断。

| SNR | 对比（前减后） | 均值 | 95% 区间 | 正向会话 |
|---|---|---:|---|---:|
| 0 | acquisition_minus_recursive_pilot | +0.059543 | [+0.040257, +0.078829] | 3/3 |
| 0 | current_pilot_minus_recursive_pilot | +0.009047 | [-0.045586, +0.063680] | 2/3 |
| 0 | recursive_pilot_minus_recursive_soft | +0.002699 | [-0.019287, +0.024685] | 2/3 |
| 5 | acquisition_minus_recursive_pilot | +0.004381 | [-0.004012, +0.012774] | 3/3 |
| 5 | current_pilot_minus_recursive_pilot | +0.001790 | [-0.002317, +0.005897] | 2/3 |
| 5 | recursive_pilot_minus_recursive_soft | +0.001980 | [-0.000143, +0.004104] | 3/3 |
| 10 | acquisition_minus_recursive_pilot | +0.000068 | [-0.001422, +0.001558] | 2/3 |
| 10 | current_pilot_minus_recursive_pilot | +0.000027 | [-0.000784, +0.000838] | 1/3 |
| 10 | recursive_pilot_minus_recursive_soft | +0.000746 | [+0.000363, +0.001129] | 3/3 |
| 15 | acquisition_minus_recursive_pilot | -0.000122 | [-0.000526, +0.000282] | 1/3 |
| 15 | current_pilot_minus_recursive_pilot | -0.000122 | [-0.000487, +0.000242] | 1/3 |
| 15 | recursive_pilot_minus_recursive_soft | +0.000136 | [-0.000075, +0.000346] | 3/3 |

## 边界

- 当前是既有 checkpoint 的接收机制筛查，尚未进行闭环重训。
- 噪声残差可能包含同步/信道模型误差，估计比值只作事后诊断。
- 首帧已经使用当前 Pilot，首帧优势不能作为跨帧增长证据。
- 自身 BER 与相对优势分别检查；短时开发曲线不能证明饱和。
- 各窗口、逐会话差值与耗时见同目录统计 JSON；无选择性删 seed。
