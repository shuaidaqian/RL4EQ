# 长记忆算子加速与前缀边界诊断（2026-11-01）

## 目的

支持后续离线适配训练与独立 seed 验证，降低长记忆算子反复逐 tap 调用的成本。
新增 FFT 后端与原有线性卷积及伴随在数学上等价，保留原 `direct` 默认实现。
它不增加循环前缀，不把线性信道改成循环信道，也不改变接收机可见信息。

## 等价关系

前向输入是长度 N+L 的 `[tail, x]`。使用长度至少 N+L 的 FFT 做卷积，
截取 `[L, L+N)`。不足完整线性卷积长度的循环混叠只影响被丢弃的前 L 个位置，
因此保留区间等于原逐 tap 实现。伴随使用零填充后的频域相关并截取前 N 个位置。

测试覆盖非零 tail、L>N、小尺寸、1024/116 主尺寸、实数/复数、非 batch 输入、
伴随内积恒等式，以及对信号/CIR/tail 的一阶梯度和 CIR 的二阶梯度。

## 验证

- FFT 后端完整 pytest：`653 passed`，240.32 秒，测试进程使用 CPU。
- 定向的算子、接收机与 Pilot 迁移检查：40 项通过。
- GTX 1650、1024 帧长、116 记忆、同一历史 checkpoint、独立 seed=41997、
  10 dB 的推理微基准：逐 tap 单帧 501.35 ms，FFT 单帧 35.18 ms，约 14.25 倍。
- 算子前向加伴随：28.45 ms 降到 0.465 ms。
- 同一帧最大/平均 logit 差异约 7.25e-5 / 1.50e-5；Data 硬判决差异为 0。

耗时来自预热后的少量重复，属于本机微基准；单帧硬判决一致不能证明任意轨迹
在舍入误差下完全一致。后续同一次实验的所有分支须使用相同后端，记录后端，
不能把不同后端下接近阈值的变化误认为训练收益。首轮四臂结果仍来自 direct。

## 使用

```powershell
$env:RL4EQ_LINEAR_OPERATOR_BACKEND = 'fft'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& D:/Research/RL4EQ/.venv-gpu/Scripts/python.exe scripts/benchmark_linear_operator.py --config configs/eme_long_memory_v2_pilot_transfer.json --pretrained D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --device cuda --output logs/linear_operator_fft_gpu_benchmark_20261101.json
```

未设置环境变量时继续使用 direct；显式传入算子 `backend` 优先于环境变量。
新增训练和四臂评估记录实际后端。实际微调仍由原 Pilot 信息边界约束。

## 前缀边界

`scripts/diagnose_pilot_boundary.py` 固定完整已知 Pilot 卷积输入，仅改变信号
重构评分窗。跳过前 L 个位置后，不再依赖历史 tail；该路径显式使用零 tail
计算评分区间，避免 FFT 对无关未知尾部数值的舍入传播。
它是诊断选项，默认评分窗仍为全部 Adapt，与正式 PPO 动作保持一致。
3 seed 的结果没有通过稳定迁移门槛，详见[首轮报告](pilot_transfer_stage_20261101.md)。

机器可读微基准：[结果 JSON](results/linear_operator_fft_gpu_benchmark_20261101.json)。
