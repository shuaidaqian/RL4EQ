# 更新后目标与普通 Data 重加权的梯度分解

这是训练范围内的探索性机制诊断，不是新的留出验证，也不参与在线动作或模型
更新。准备脚本时相位四臂测试尚在运行；本诊断不根据其 BER 调参。

固定源为相位 A/B 的共同初始化：`pretrained/head_transfer_post256_20261101/model_best.pt`，
SHA256 `b8281a8132759865b1b9c303aed60067eaae72a6917f1a0ace6457c4f5b4b7e5`。
使用 offline_train 新 seed=38000–38002、每条 2 帧、四主 SNR；共 24 个帧上下文。
使用相位实验的原 Adapt-only 内循环，不改变步长、信道或网络。

对相同上下文计算基础参数上的三个精确梯度：

```text
g_all    = grad BCE_all(theta)
g_before = grad BCE_data(theta)
g_post   = grad BCE_data(Update_Adapt(theta))
g_B      = base_weight*g_all + post_weight*g_post
g_rew    = base_weight*g_all + post_weight*g_before
r        = g_B - g_rew
```

报告 `norm(g_post-g_before)/norm(g_before)`、`norm(r)/norm(g_rew)` 及对应余弦。
分别汇总全部基础参数和 phase 参数组，先按 seed 平均，再计算区间。
接近零的残差说明：在这些样本、参数化和更新预算下，新增目标的瞬时梯度接近
普通 Data 重加权。它不证明长期 AdamW 轨迹相同，也不证明元训练一般无效。
中位数并列点仍使用当前实现选定的导数，不把局部梯度扩展为全局结论。

该诊断使用离线 Data 标签，没有在线控制器，不保存新 checkpoint，不作任何
Data 标签驱动的在线候选。真实二阶图和恒等内循环退化检查通过后运行。

```powershell
$env:RL4EQ_LINEAR_OPERATOR_BACKEND = 'fft'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& D:/Research/RL4EQ/.venv-gpu/Scripts/python.exe scripts/diagnose_transfer_gradient.py --config configs/eme_long_memory_v2_phase_transfer.json --pretrained pretrained/head_transfer_post256_20261101/model_best.pt --seed-start 38000 --num-seeds 3 --frames 2 --device cuda --output logs/phase_transfer_gradient_source_3s2f_20261101.json
```
