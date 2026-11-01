# 接收波形置零与 Pilot 梯度失配审计

本阶段不修改网络、正式 RL 动作或在线信息边界，仅用训练范围进行事后梯度分析。
主契约禁止使用 Data 标签和发送符号作为在线监督；它同时规定整帧非因果接收。
现有实现额外将非 Adapt 接收波形在更新时置零，将 Data 接收波形在验收时置零。
本轮检验这个额外限制是否改变 Pilot 梯度及其与 Data 梯度的对齐。

完整接收波形、未知发送符号和 Data 标签是不同对象，不能混为一谈。是否把
完整接收波形用于后续实际适配的边界选择已向用户提问，未答复前保持现有控制器。

## 看新诊断输出前固定

- 模型固定为上一阶段相位 B 的 256 步终点：
  `pretrained/phase_transfer_post256_20261101/model_best.pt`；不根据本轮挑 checkpoint。
- 不更新任何参数，不生成 Data 梯度驱动候选，不选择动作或保存新模型。
- offline_train，新 seed=39000–39002，每条 2 帧，0/5/10/15 dB，共 24 个上下文。
- acquisition CIR/CFO、当前 Adapt 条件和旧 tail 不变；head、phase 两个已有组分别统计。
- 对相同 Pilot 标签比较四个梯度：Adapt-only 波形/Adapt BCE、整帧波形/Adapt BCE、
  Adapt+Reward 波形/Reward BCE、整帧波形/Reward BCE。
- 所有 Pilot 梯度完成后，才计算整帧输入下的 Data 梯度作为离线参照。
- 检查 Pilot 视图前后梯度余弦、相对差范数，以及对 Data 梯度的对齐变化。
  先按 seed 平均，再计算探索性配对 t 区间；3 seed 不支持正式主结论。
- Data 标签和未知发送符号污染不应改变任何 Pilot 梯度；改 Data 接收波形允许
  改变整帧输入的 Pilot 梯度，但不能改变置零输入的 Pilot 梯度。

即便整帧输入的对齐更好，也不推出 BER 改善；必须经独立候选、Reward 验收、
同 checkpoint Frozen 与普通监督配对验证后，才能讨论累计和正式 RL。

```powershell
$env:RL4EQ_LINEAR_OPERATOR_BACKEND = 'fft'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& D:/Research/RL4EQ/.venv-gpu/Scripts/python.exe scripts/diagnose_receiver_visibility.py --config configs/eme_long_memory_v2_phase_transfer.json --pretrained pretrained/phase_transfer_post256_20261101/model_best.pt --seed-start 39000 --num-seeds 3 --frames 2 --device cuda --output logs/receiver_visibility_phase256_3s2f_20261101.json
```

## 首轮输出后的数值审计增补

首次 24 帧输出出现高 SNR 的参考梯度接近零，相对范数比被放大、余弦不可解释。
因此在同一批样本上补记各视图 Pilot BER/BCE 及 Data 梯度范数；梯度范数不超过
1e-8 时将方向和相应分母的比值记为 null，汇总显式记录可判读的帧数和 seed 数。
这个阈值仅用于数值解释，不进入优化、动作或验收。增补属于事后审计，重算不作
独立样本，也不据此宣称输入边界变更已经有效。重算输出使用新路径：
`logs/receiver_visibility_phase256_3s2f_metrics_20261101.json`，保留首次原始结果。
