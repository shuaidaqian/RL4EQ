# 接收加速的等价性与速度检查（2026-11-23）

## 状态

- 只读开发检查：验证两臂合并前向（batch=2）与 CUDA Graph 在同一波形、同一 checkpoint 下与逐臂基线等价；不做 BER 确认、不产生效果结论。
- 使用计时/等价专用 seed 240013（不进入后续筛查或确认）。
- 设备 GTX 1650，`fft` 线性算子后端；配置与 soft Data 方差先导相同。
- 脚本：[`scripts/check_accel_equivalence.py`](../../../scripts/check_accel_equivalence.py)；原始输出：[`equivalence.json`](equivalence.json)。

## 实现改动

1. `training/continual_accel.py`：`BatchedForward` 把同帧两条臂的前向合并为 batch=2；`GraphBatchedForward` 在固定形状下捕获一次并回放。
2. `training/continual_receiver.py`：`step` 拆为 `prepare` + `finish`，批量路径在前向之外各执行一次；`_design` 由逐 tap Python 循环改为单次 gather（数值相同，消除每帧数百次小启动）。
3. `scripts/compare_continual_variants.py`：新增 `--accel {none,batch,graph}`；默认 `none`，只支持两条 neural 臂、schema 2 流式、共享 checkpoint，禁止长诊断/共享上下文。
4. `training/pilot_transfer.py`：抽出 `context_model_inputs` 供批量路径复用，`context_logits` 行为不变。

## 等价性（0 dB，32 帧，seed 240013）

| 模式 | 帧均值 / 中位数 | 相对基线速度（均值 / 中位数） | logits 最大差 | 诊断最大差 | BER 错误差异 |
|---|---:|---:|---:|---:|---:|
| 基线（逐臂） | 119.5 / 86.3 ms | — | — | — | — |
| batch=2 | 59.6 / 59.1 ms | 2.01× / 1.46× | 3.3e-6 | 8.3e-7 | 0（32 帧全部一致） |
| batch=2 + CUDA Graph | 45.8 / 45.5 ms | 2.61× / 1.90× | 3.3e-6 | 8.3e-7 | 0（32 帧全部一致） |

- CUDA Graph 首帧含捕获约 232 ms，已排除在稳态统计外。
- 两臂的逐步错误数与基线逐帧完全一致；递归状态轨迹未因批量化改变。
- `_design` 向量化单独把 prepare+finish 从约 81 ms 降到约 30 ms/帧。

## 端到端烟雾

- `--accel graph` 与 `--accel none` 各完成 1 seed × 0 dB × 128 帧流式烟雾（seed 240018/240019），状态 `completed`，行数与臂结构完整。
- 含模型加载的墙钟：graph 19 s，none 25 s；graph 每臂记录的共享前向中位数约 17.9 ms，none 约 46.7/53.0 ms。
- 烟雾结果只验证执行与记录路径，不用于 BER 比较或效果判断。

## 限制与使用条件

- 只验证了 0 dB、一个 seed、32 帧与 128 帧烟雾；其他 SNR、种子和更长会话需在新实验中重新审计。
- CUDA Graph 要求固定形状与固定非张量输入；当前配置无 `enable_channel_state_branch`，因此 `frame_index` 不进入模型。若未来启用该分支，graph 模式会被拒绝。
- 批量前向的浮点归约顺序与逐臂略有差异（logits 差 ~1e-6）；对已完成的旧实验不做混用，新实验需在协议中记录 `acceleration` 字段。
- 每臂 `elapsed_ms` 在加速模式下是共享前向的用时；`model_calls`/`neural_forwards` 仍按每臂逻辑前向记为 1，实际 GPU 调用次数减半。成本解释以帧级墙钟与实现记录为准。

## 对 P5 成本的影响

- 未优化实现约 0.163 s/帧，四 SNR 方案约 12.8 GPU 日；
- `_design` 向量化后基线约 0.09–0.12 s/帧；
- batch=2 + CUDA Graph 稳态约 46 ms/帧，加运行器评分/落盘开销后估计约 60–80 ms/帧，四 SNR 方案约 **5–6 GPU 日**。

## 复现

```powershell
.venv-gpu\Scripts\python.exe scripts/check_accel_equivalence.py `
  --manifest docs/protocols/continual_v2_soft_data_variance_pilot_10x1024_20261122.json `
  --seed 240013 --snr 0 --frames 32 --mode both `
  --output docs/results/continual_v2_accel_equivalence_20261122/equivalence.json
```
