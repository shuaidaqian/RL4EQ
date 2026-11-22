# GPU 接收吞吐基准（2026-11-22）

## 状态

- 开发资源测量，**不是**效果或确认证据；只用计时专用 seed 240012，不做 BER/NMSE 评分。
- 设备：NVIDIA GeForce GTX 1650（4 GB）；线性算子后端 `fft`（与正式先导一致）。
- 配置、checkpoint、manifest 与 soft Data 方差先导相同；原始输出见 [`bench_fft.json`](bench_fft.json)。
- 脚本：[`scripts/bench_receiver_gpu.py`](../../../scripts/bench_receiver_gpu.py)。

## 结果（SNR 0 dB，32 帧真实接收路径）

| 项目 | 中位数 |
|---|---|
| 单臂完整 `step`（相位 + 后验 + 前向 + 诊断） | pilot_only 62 ms / soft 75 ms |
| 整帧两臂完整 `step` | 137 ms（均值 176 ms） |
| 单臂解码器前向（batch=1） | 27–28 ms |
| 两臂合并一次前向（batch=2） | 29–31 ms |
| batch=2 + CUDA Graph 回放 | **16.5 ms** |
| batch=1 与 batch=2 logits 最大差 | 1.19e-7 |
| `torch.compile(mode="reduce-overhead")` | 不可用（`TritonMissing`，Windows 无 Triton） |

## 解读

- 解码器前向只占整帧约 34%；其余是相位特征、Pilot/soft 后验和逐帧诊断的 CPU 派发与小算子开销。
- 两臂合并成 batch=2 基本不增加单次前向成本（27 → 29–31 ms），可把两臂前向从约 55 ms 降到约 30 ms。
- CUDA Graph 把 batch=2 前向压到 16.5 ms；若只图化解码器，整帧预计从约 137 ms 降到约 100–120 ms（约 1.2–1.4×）。
- 要显著拉高 GPU 利用率，需要把相位/后验/诊断也纳入图化（整步 CUDA Graph）；这需要改动接收机核心代码、重新预登记并使用新 seed。
- 本基准不能与已完成先导的逐帧延迟直接相加或替换；实现变更后必须重新验证数值一致性与协议。

## 复现

```powershell
.venv-gpu\Scripts\python.exe scripts/bench_receiver_gpu.py `
  --manifest docs/protocols/continual_v2_soft_data_variance_pilot_10x1024_20261122.json `
  --seed 240012 --snr 0 --frames 32 --reps 20 --mode both
```
