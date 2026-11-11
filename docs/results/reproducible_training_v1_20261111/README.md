# 离线训练可复现性工程（2026-11-11）

状态：**训练复现工程，不是新的方法或确认。** 目标：把此前“复现链不完整”的离线训练
补到可重放，并证明重训 checkpoint 与原 checkpoint 的评估指标在容差内一致。

## 背景

`docs/offline_equalizer_training_and_baseline_20261105.md` 记录：
原始训练命令、源码 revision、Python/PyTorch/CUDA 环境、完整优化器超参数与显式随机种子
没有随 checkpoint 保存，因此当前 checkout 不能称为严格复现。

本次从当前代码、锁定环境与种子重训 32 步，并记录完整指纹。

## 锁定条件

| 项 | 值 |
|---|---|
| git commit | `600cca769e146917cc9a78a0da29f022ff7e1b22`（训练代码干净） |
| Python | 3.12.7 |
| PyTorch | 2.11.0+cu128 |
| CUDA / cuDNN | 12.8 / 9.19 |
| GPU | NVIDIA GeForce GTX 1650 |
| 配置 | `configs/eme_long_memory_v2.json`，SHA256 `4f104bc0a8fad33c89ec794694bc77bbc3f9c64aee94279a9575f75daab44263` |
| 种子 | `curriculum_seed=80000`（config 未显式设置，使用默认）；采样 `seed = 80000 + sample_index` |
| 训练 | 32 步，batch=1，accumulation=1，AMP=false，`bce_all` |
| 原 checkpoint | `eme_bce_all_32_20260905_pilot256/model_best.pt`，SHA256 `8cd3912d...` |

## 训练命令

```powershell
$env:PYTHONUTF8=1
python pretrain.py `
  --config configs/eme_long_memory_v2.json `
  --stage all --steps 32 --batch-size 1 --accumulation-steps 1 `
  --save-dir docs/results/reproducible_training_v1_20261111/retrained
```

重训 checkpoint SHA256：`4d5147f3caf25a189c014cbe19b10a5e50747a8500dfa737c226ce5be25550b3`。

## 结果

### 训练 loss 曲线（32 步，逐 step 相对差）

| 指标 | 值 |
|---|---:|
| 第 1 步相对差 | **0（逐位相同）** |
| 最终步相对差 | `2.80e-5` |
| 最大相对差 | `5.02e-2`（第 24 步，此时 loss 已接近 0，相对差被放大） |

第 1 步 loss 与原 checkpoint 完全相同（`0.5020070672035217`），说明数据序列、模型初始化
与优化器路径一致；后续差异来自 GPU 浮点非确定性累积。

### 验证指标

| 指标 | 原 checkpoint | 重训 checkpoint | 相对差 |
|---|---:|---:|---:|
| `best_model_metric`（offline_nn mean BER） | 0.0657552 | 0.0651042 | 1.0% |

### 独立评估（新 seed 170021，16 帧，同配置）

| SNR | 原 Frozen | 重训 Frozen | 原 Bayes | 重训 Bayes | 原 Bayes NMSE | 重训 Bayes NMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 dB | 0.20964 | 0.20947 | 0.18986 | 0.18905 | 0.07122 | 0.07120 |
| 5 dB | 0.05314 | 0.05322 | 0.04980 | 0.04989 | 0.02325 | 0.02325 |
| 10 dB | 0.008952 | 0.008952 | 0.008057 | 0.008057 | 0.0092 | 0.0092 |
| 15 dB | 0.000732 | 0.000732 | 0.000651 | 0.000651 | 0.0054 | 0.0054 |

**结论：**

- 10/15 dB 的 Frozen/Bayes BER 与 NMSE **逐位相同**；
- 5 dB 的 BER 差 `8.1e-5`，NMSE 差 `2.4e-6`；
- 0 dB 的 BER 差 `8.1e-4`（Bayes）/`1.6e-4`（Frozen），NMSE 差 `2.1e-5`；
- 全部在预定容差内（BER 绝对差 ≤ `1e-3`，NMSE 相对差 ≤ 1%）。

## 结论与边界

- **训练链已可重放（指标级）**：同一 commit、环境、config、种子和命令下，
  第 1 步 loss 逐位相同，最终验证指标相对差 1%，独立评估在容差内一致。
- **不是逐位复现**：GPU 浮点非确定性使后续 step 的 loss 与权重逐位不同；
  这是预期行为，不影响指标级复现。
- 原 checkpoint 的 history 没有采样 seed/输入哈希，**无法逐条证明同一数据序列**；
  本次以“第 1 步 loss 逐位相同 + 最终指标一致”作为证据。当前代码的 history 已记录
  `seed`、`offline_sample_input_sha256` 等字段，今后的训练可做逐条比对。
- 本次不改变任何方法、判据或确认结果；只是补齐训练复现链。

## 文件

| 文件 | 内容 |
|---|---|
| `environment.json` | commit、环境、config、命令、种子、指纹 |
| `summary.json` | loss 曲线差异、验证指标、独立评估对比 |
| `train.log` | 训练日志 |
| `retrained/` | 重训 checkpoint、metrics、config |
| `eval_original.json` / `eval_retrained.json` | 独立评估逐帧结果 |
| `retrained.sha256` | 重训 checkpoint 与 metrics 的 SHA256 |
