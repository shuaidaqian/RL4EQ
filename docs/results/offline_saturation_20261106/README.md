# 离线网络饱和与可修复空间开发诊断（2026-11-06）

状态：**开发诊断，未进入正式确认；oracle 分支不可部署。**
本目录所有 Data 标签只用于事后评分与隔离 oracle，未回流任何可部署控制。

## 结论摘要

- **训练预算/宽度维度已接近饱和**：从同一 checkpoint 继续训练 32/64 步，或在相同
  32 步预算下把 `iterations` 4→6、`d_model` 64→96，Frozen Data BER 与原始 checkpoint
  基本重合；最大点估计改善仅约 `-1.8e-4`（5 dB，64 步续训），对应 12 帧约 1–2 个比特，
  其余点多数为 0 或跨零。
- **表示/读出仍有可修复空间，但主要是逐帧的**：
  - 同帧 oracle head（Data 标签、每次 K 步 Adam、在相同帧上评分）在 5/10 dB 显著：
    K=64 时 5 dB `BER 0.05283→0.05049`（`-2.34e-3`，95% CI `[-3.50e-3,-1.19e-3]`），
    10 dB `0.00700→0.00622`（`-7.81e-4`，CI `[-1.38e-3,-1.85e-4]`）；
    但 0 dB 为 `-6.19e-3` 且 CI 跨零。
  - 会话内留出帧 oracle（偶数帧优化、奇数帧评分）时这些增益基本消失：
    5 dB K=64 `-1.43e-3`（CI 跨零），10 dB `-3.26e-4`（CI 跨零），0 dB 反而
    `+1.99e-2`（CI 跨零）。说明单帧/单会话静态 head 修正不能稳定迁移到同会话其他帧。
- **CIR 估计误差是 5/10 dB 的主要剩余误差来源**：把条件 CIR 换成该帧真实 CIR 后，
  Frozen BER 在 5 dB 从 `0.05283` 降到 `0.04199`（`-1.08e-2`，CI `[-1.57e-2,-5.94e-3]`），
  10 dB 从 `0.00700` 降到 `0.00465`（`-2.34e-3`，CI `[-3.23e-3,-1.46e-3]`）；
  0 dB 点估计 `-4.60e-2`（5/5 seed 同向），但 seed 91003 的 BER 0.311 使区间跨零。
- **建议下一步**：优先做合法的 CIR/物理状态跟踪改善（Adapt Pilot 条件、DD-CIR 更新、
  条件分布增广），而不是继续增加离线训练步数或模型宽度；若做在线 PEFT，应针对
  逐帧条件化读出，而不是持久会话级静态 head。

## 目的与实验设计

判断当前强离线 checkpoint（32 步、`configs/eme_long_memory_v2.json`、116 符号长回波）
在当前损失/结构/数据预算下是否饱和，并量化可修复空间：

1. **Oracle head 上界**（不可部署）：用 Data 真标签对 `model.head` 做 K=1/4/16/64 步
   Adam（lr=1e-2）全 Data BCE 优化。通过缓存冻结 head 输入与固定加性项，
   精确复现 `logits = additive + neural_residual_scale * head(hidden)`；
   优化只改 65 个 head 参数，不改变物理状态/tail。
2. **Oracle CIR 上界**（不可部署）：用 `frame.true_cir` 重建条件并重新递推 soft-tail，
   与估计 CIR 的 Frozen 轨迹比较。
3. **更长训练**：从同一 checkpoint `--resume` 继续训练 32/64 步。
4. **更大模型**：`iterations=6` 或 `d_model=96`，同 32 步预算从头训练。

## 固定条件

- 信道配置：`configs/eme_long_memory_v2.json`，Level B，116 符号记忆，1024 帧，
  prefix Pilot=256（Adapt=224 / Reward=32），SNR 0/5/10/15 dB，`state_split=offline_train`。
  未修改信道、Pilot、SNR、数据划分或现有源码。
- 评估 seed：`91001–91005`（新开发 seed，不参与训练）；实验 3/4 的配对评估使用
  `91001–91003`、每 seed/SNR 12 帧。
- Oracle 诊断：5 seed × 4 SNR × 8 帧；实验 3/4：3 seed × 4 SNR × 12 帧。
- Frozen soft-tail：上一帧最后 `max_delay` 个 logits 的 `tanh(logit/2)`，`alpha=1.0`，
  与 `training/curriculum.py` 的 `online_soft` 一致。
  现有 `scripts/prototype_head_rank1_adapt.py` 的 `next_pilot_tail` 在 prefix 布局下
  不更新帧尾，本次不沿用。
- 线性算子后端：训练沿用默认 `direct`；评估使用 `RL4EQ_LINEAR_OPERATOR_BACKEND=fft`。
  已核对 seed 91001、2 帧（0/5/10/15 dB）Frozen 逐帧判决与 `direct` 完全一致。
- head oracle 的精确性：与逐帧全模型反传版本在 1 seed/2 帧/4 SNR 上对比，
  K=1 完全相同；K=4 仅在 0 dB 差 1 bit（浮点结合差异）。后续均用特征缓存版。

## 源 checkpoint 与配置指纹

| 文件 | SHA256 |
| --- | --- |
| `D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt` | `8cd3912d1c957c9689c51730aff9d27a8f74942186c802a7483e5781af65aec4` |
| `configs/eme_long_memory_v2.json` | `f3952864d26025b9ec5cfc6ea6008080435a7c4e5747881b410eb3d141aebfa5` |
| `configs/eme_long_memory_v2_iter6_satdev_v1.json` | `ba5c43b14fc530ba3b881aa891f810c783900aa5837a6b251aafa5c4607f316b` |
| `configs/eme_long_memory_v2_dmodel96_satdev_v1.json` | `28e139e2a41a848241e0180cb03ae8591958f849900861456ed43e67a33594f0` |

源 checkpoint 元数据：`resume_loaded=False`（32 步从头训练），189,350 参数，
`iterations=4`、`d_model=64`、`physics_warm_start_iterations=8`；
离线验证 `mean_ber_data≈0.0658`。

生成的 checkpoint：

| 实验 | model_best.pt SHA256 |
| --- | --- |
| continued32 | `cbbc1678263c1fdf91d3b053ccec5e04edac414ed72444f105d62d56c71b6847` |
| continued64 | `356080abe9df5cada7cc2c4dbec3386ecf7ece4d17144ddf04883fa1ea26cf2e` |
| iter6_scratch32 | `a1070079bc51d357638c978114520b22cf3e88f89c728aff17af41eb1bef01f1` |
| dmodel96_scratch32 | `2182c887a75613c3c9e54e54017be4ede23889b02ee17d5c637a6eacc08ceb1f` |

## 精确命令

以下 `<PY>` = `D:/Research/RL4EQ/.venv-gpu/Scripts/python.exe`，工作目录为项目根。

### 1) 原始 checkpoint：Frozen + Oracle head + Oracle CIR（同帧）

```bash
RL4EQ_LINEAR_OPERATOR_BACKEND=fft <PY> -I -X utf8 scripts/diagnose_offline_saturation.py \
  --seeds 91001 91002 91003 91004 91005 --frames 8 --mode all \
  --head-k 1 4 16 64 --head-lr 1e-2 --state-split offline_train \
  --output-dir docs/results/offline_saturation_20261106/original_checkpoint --device cuda
```

耗时 124.6 s，CUDA 峰值 35.6 MB。

### 2) 原始 checkpoint：会话内留出帧 head oracle

```bash
RL4EQ_LINEAR_OPERATOR_BACKEND=fft <PY> -I -X utf8 scripts/diagnose_offline_saturation.py \
  --seeds 91001 91002 91003 91004 91005 --frames 8 --mode all --head-split heldout \
  --head-k 1 4 16 64 --head-lr 1e-2 --state-split offline_train \
  --output-dir docs/results/offline_saturation_20261106/oracle_head_heldout --device cuda
```

耗时 114.4 s，CUDA 峰值 35.6 MB。

### 3) 实验 3/4 的配对参考 Frozen

```bash
RL4EQ_LINEAR_OPERATOR_BACKEND=fft <PY> -I -X utf8 scripts/diagnose_offline_saturation.py \
  --seeds 91001 91002 91003 --frames 12 --mode frozen --state-split offline_train \
  --output-dir docs/results/offline_saturation_20261106/reference_frozen12 --device cuda
```

耗时 51.3 s，CUDA 峰值 18.6 MB。

### 4) 继续训练 32/64 步

```bash
<PY> -X utf8 pretrain.py --config configs/eme_long_memory_v2.json --stage all \
  --steps 32 --batch-size 1 \
  --resume D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt \
  --save-dir docs/results/offline_saturation_20261106/continued32 --device cuda

<PY> -X utf8 pretrain.py --config configs/eme_long_memory_v2.json --stage all \
  --steps 64 --batch-size 1 \
  --resume D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt \
  --save-dir docs/results/offline_saturation_20261106/continued64 --device cuda
```

`pretrain.py` 不能加 `-I`：`-I` 会把工作目录移出 `sys.path`，触发
`ModuleNotFoundError: No module named 'training'`。因此训练命令使用 `-X utf8`。
续训只加载模型权重，Adam 状态重置；训练样本计数器从 0 开始，即复用相同
seed 80000–80031 的 32/64 个样本（同配置、同分布）。
耗时约 1.5–3 分钟/次（由输出目录时间戳估计，未单独插桩）；无 OOM。

### 5) 更大模型（同 32 步预算从头训练）

```bash
<PY> -X utf8 pretrain.py \
  --config docs/results/offline_saturation_20261106/configs/eme_long_memory_v2_iter6_satdev_v1.json \
  --stage all --steps 32 --batch-size 1 \
  --save-dir docs/results/offline_saturation_20261106/iter6_scratch32 --device cuda

<PY> -X utf8 pretrain.py \
  --config docs/results/offline_saturation_20261106/configs/eme_long_memory_v2_dmodel96_satdev_v1.json \
  --stage all --steps 32 --batch-size 1 \
  --save-dir docs/results/offline_saturation_20261106/dmodel96_scratch32 --device cuda
```

无法从源 checkpoint `--resume` 到更大结构：`iterations` 4→6 会改变 `alpha/damping`
形状，`d_model` 64→96 会改变全部层形状，`PilotTransferTrainer/CurriculumTrainer`
的 strict-load 会拒绝。源 checkpoint 本身也是 32 步从头训练（`resume_loaded=False`），
因此这里采用相同 32 步预算从头训练作为公平比较。
耗时约 2 分钟/次（1.8–2.4 分钟）；无 OOM。

### 6) 四个变体的 Frozen 配对评估

```bash
for name in continued32 continued64 iter6_scratch32 dmodel96_scratch32; do
  cfg=configs/eme_long_memory_v2.json
  case $name in iter6_*) cfg=docs/results/offline_saturation_20261106/configs/eme_long_memory_v2_iter6_satdev_v1.json;; dmodel96_*) cfg=docs/results/offline_saturation_20261106/configs/eme_long_memory_v2_dmodel96_satdev_v1.json;; esac
  RL4EQ_LINEAR_OPERATOR_BACKEND=fft <PY> -I -X utf8 scripts/diagnose_offline_saturation.py \
    --config "$cfg" --checkpoint docs/results/offline_saturation_20261106/$name/model_best.pt \
    --seeds 91001 91002 91003 --frames 12 --mode frozen --state-split offline_train \
    --output-dir docs/results/offline_saturation_20261106/eval_$name \
    --paired-reference docs/results/offline_saturation_20261106/reference_frozen12/summary.json \
    --device cuda
done
```

每次 51–59 s，CUDA 峰值 18.6–20.6 MB。配对差定义为
`BER_current - BER_reference`，负值表示改善。

## 结果

### 实验 1a：同帧 oracle head（5 seed × 8 帧，6144 bit/seed/SNR）

Frozen Data BER（seed 级 95% t 区间）：

| SNR | Frozen BER | 95% CI | 逐 seed |
| --- | --- | --- | --- |
| 0 dB | 0.20472 | [0.12983, 0.27961] | 0.1847 / 0.1655 / 0.3110 / 0.1912 / 0.1711 |
| 5 dB | 0.05283 | [0.04908, 0.05658] | 0.0479 / 0.0544 / 0.0542 / 0.0522 / 0.0555 |
| 10 dB | 0.00700 | [0.00411, 0.00989] | 0.0036 / 0.0096 / 0.0085 / 0.0073 / 0.0060 |
| 15 dB | 0.00065 | [0.00020, 0.00110] | 0.0002 / 0.0011 / 0.0007 / 0.0008 / 0.0005 |

Oracle head 在**相同帧**上优化并评分（`Δ = BER_after - BER_before`）：

| SNR | K=1 Δ | K=4 Δ | K=16 Δ | K=64 Δ [95% CI] | K=64 修复/伤害错误 |
| --- | --- | --- | --- | --- | --- |
| 0 dB | -3.3e-5 | -1.95e-4 | -1.40e-3 | -6.19e-3 [-1.30e-2, +5.8e-4] | 883 / 693 |
| 5 dB | +3.3e-5 | +1.63e-4 | -8.14e-4 [-1.10e-3,-5.28e-4] | -2.34e-3 [-3.50e-3,-1.19e-3] | 139 / 67 |
| 10 dB | 0 | -6.5e-5 | -1.63e-4 [-3.65e-4,+3.9e-5] | -7.81e-4 [-1.38e-3,-1.85e-4] | 38 / 14 |
| 15 dB | 0 | +6.5e-5 | 0 | -3.3e-5 [-2.54e-4,+1.89e-4] | 4 / 3 |

逐 seed K=64 Δ：0 dB 为 `+2.3e-3/-5.5e-3/-12.4e-3/-9.1e-3/-6.2e-3`（5 个中 4 个改善）；
5 dB 全部为负；10 dB 全部为负。

### 实验 1b：会话内留出帧 oracle head（偶数帧优化、奇数帧评分）

| SNR | 留出帧 before | K=64 Δ [95% CI] | 修复/伤害错误 |
| --- | --- | --- | --- |
| 0 dB | 0.1765 | +1.99e-2 [-4.62e-2,+8.60e-2] | 485 / 791 |
| 5 dB | 0.05618 | -1.43e-3 [-3.69e-3,+8.29e-4] | 67 / 45 |
| 10 dB | 0.00736 | -3.26e-4 [-7.30e-4,+7.9e-5] | 19 / 14 |
| 15 dB | 0.00065 | +1.95e-4 [-2.6e-5,+4.17e-4] | 1 / 4 |

同帧增益在留出帧上消失（5/10 dB 区间跨零；0 dB 反而退化），说明 head-only
可修复空间主要是逐帧的，不是一个稳定的会话级 head 修正。

### 实验 2：Oracle CIR（真实 CIR 条件 + 重新递推 tail）

| SNR | 估计 CIR BER | 真实 CIR BER | Δ=true-est [95% CI] | 逐 seed Δ |
| --- | --- | --- | --- | --- |
| 0 dB | 0.20472 | 0.15869 | -4.60e-2 [-1.07e-1,+1.54e-2] | -5.97e-2 / -2.21e-2 / -1.27e-1 / -7.8e-3 / -1.37e-2 |
| 5 dB | 0.05283 | 0.04199 | -1.08e-2 [-1.57e-2,-5.94e-3] | 全部为负（-5.5e-3 … -1.42e-2） |
| 10 dB | 0.00700 | 0.00465 | -2.34e-3 [-3.23e-3,-1.46e-3] | 全部为负（-1.3e-3 … -3.1e-3） |
| 15 dB | 0.00065 | 0.00059 | -6.5e-5 [-2.96e-4,+1.65e-4] | 近似 0 |

0 dB 的点估计很大且 5/5 seed 同向，但 seed 91003 的信道实现使 BER 达 0.311，
区间被拉宽跨零。

### 实验 3：继续训练 32/64 步

参考（原始 checkpoint，3 seed × 12 帧，9216 bit/seed/SNR）：
0 dB `0.20779`，5 dB `0.05067`，10 dB `0.00713`，15 dB `0.00054`。

| 变体 | 0 dB Δ [CI] | 5 dB Δ [CI] | 10 dB Δ [CI] | 15 dB Δ [CI] |
| --- | --- | --- | --- | --- |
| continued32 | -2.17e-4 [-9.3e-4,+5.0e-4] | +3.6e-5 [-3.8e-4,+4.5e-4] | -3.6e-5 [-1.9e-4,+1.2e-4] | 0 [0,0] |
| continued64 | -5.79e-4 [-2.53e-3,+1.37e-3] | **-1.81e-4 [-3.36e-4,-2.5e-5]** | -3.6e-5 [-1.9e-4,+1.2e-4] | 0 [0,0] |

续训没有降低训练损失平台：0 dB BCE 仍约 0.44，5 dB 约 0.21，10 dB 约 0.01，
15 dB 约 0。continued64 的 10 dB 训练损失甚至升到约 0.058，但留出 BER 未变。
`offline_nn_validation` 均值：continued32 0.0651，continued64 0.06576
（原始 0.06576；差异来自单 seed/单帧验证，不作为主证据）。

### 实验 4：更大模型（同 32 步预算从头训练）

| 变体 | 参数量 | 0 dB Δ [CI] | 5 dB Δ [CI] | 10 dB Δ | 15 dB Δ |
| --- | --- | --- | --- | --- | --- |
| iterations=6 | 189,354 | -1.81e-4 [-4.92e-4,+1.30e-4] | -3.6e-5 [-1.9e-4,+1.2e-4] | 0 | 0 |
| d_model=96 | 379,238 | -3.6e-5 [-1.9e-4,+1.2e-4] | -1.45e-4 [-3.0e-4,+1.1e-5] | 0 | 0 |

同 32 步预算下增加深度或宽度没有带来可确认的 Frozen BER 改善。

## 失败与未运行记录

- **`pretrain.py -I` 失败**：`-I` 移除工作目录导致
  `ModuleNotFoundError: No module named 'training'`。训练改用 `-X utf8`；诊断脚本
  自身会插入项目根，因此可继续用 `-I`。
- **大模型无法 `--resume`**：strict-load 拒绝 `alpha/damping`（iterations）与
  全部层形状（d_model）不匹配；改为同 32 步从头训练，源 checkpoint 也是
  `resume_loaded=False` 的 32 步从头训练。
- **两次中间实现被停止**：逐帧全模型反传的 head oracle 太慢（首个 session >4 分钟），
  以及批量全模型反传版本仍偏慢；均未产生科学结果，输出目录已删除。最终改用
  精确的 head 特征缓存，结果与逐帧版本一致（仅 K=4 在 0 dB 差 1 bit）。
- **未运行**：真实 CIR 下的 head oracle 组合、跨会话/跨 drift 的留出 oracle、
  以及正式确认集。均不在本次开发诊断范围。
- 实验 3/4 的训练耗时未单独插桩，仅按输出时间戳估计；显存峰值使用评估脚本的
  `torch.cuda.max_memory_allocated()`，训练峰值未记录但全程未 OOM。

## 结果文件

- `original_checkpoint/summary.json`：Frozen + 同帧 head oracle + CIR oracle
  （SHA256 `00244c83b6613f006aca7c7be8a3131971790c37587e65d43c2efac34dbbb1cc`）。
- `oracle_head_heldout/summary.json`：留出帧 head oracle
  （`810448d4c1472da1f4b8e3860e2e93330155784f8f35440fea1f0282af80feb0`）。
- `reference_frozen12/summary.json`：实验 3/4 参考
  （`33c68e618c31cd4ddad56c13a726abe2b3d0700de4a455794268ad334f525454`）。
- `eval_*/summary.json`：四个变体的 Frozen 评估与配对差
  （`9cf08e59…` / `7be7b7f6…` / `68d21b3a…` / `a323955d…`）。
- `continued32/`、`continued64/`、`iter6_scratch32/`、`dmodel96_scratch32/`：
  训练产出（`model_best.pt`、`model_final.pt`、`pretrain_metrics.json`、
  `model_config.json`）。
- `configs/`：本次新增的两个开发配置（`iter6`、`dmodel96`）。
- 脚本：`scripts/diagnose_offline_saturation.py`（新增只读诊断入口）。
