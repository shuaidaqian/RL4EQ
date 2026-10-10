# Pilot 作用位置扫描与信号重构在线微调阶段结果

日期：2026-10-10

## 本阶段目的

本阶段保持离线 checkpoint 不变，只用 Adapt Pilot 更新均衡器内部参数，检查
“更新确实改变 Data 输出”之后，哪一种参数位置和 Pilot-only 目标最有希望把改善迁移到
Data 段。Reward Pilot 仍只做接受、拒绝和回滚，Data 标签只用于事后统计。

固定对象如下：

- checkpoint：`pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`
- 信道：Level B、`eme_long_memory_v2`、`delay=116`
- Pilot：prefix，总长度 256，Adapt 224，Reward 32
- 诊断状态：`acquisition_to_data_gap_seconds=120` 或 `600`
- 所有结果都是 Frozen/Online 配对 replay，不进入 Level B 主平均

## 代码修正

本阶段先修正了三个会影响实验解释的实现问题：

1. `replay_pilot_candidate_ranking.py` 现在统一登记 `phase_trend`、输入/输出 affine/FIR、
   `conditioner_film`、Adapter、LoRA、`pilot_encoder` 和 `head` 等作用位置，并记录
   Data logits 平均变化、方向翻转比例、soft output 变化和参数增量。
2. 诊断前向现在显式传入当前帧 Adapt Pilot。此前 `pilot_conditioned` 模型在 replay 中
   实际收到的是空 Pilot context，内部 `pilot_encoder` 候选因此没有有效梯度。
3. `PilotDrivenOnlineAdapter` 增加可审计的 `sgd/adamw` 选择。候选 replay 使用 AdamW，
   所以真实在线流程可以显式用 `--online-optimizer adamw` 对齐；默认仍是 SGD，避免
   改变既有配置的含义。

新增的 `pilot_signal_reconstruction` 目标只校正 `phase_trend` Adapter：先用该 Adapter
校正接收 IQ，再用当前 acquisition CIR、Adapt Pilot 符号和历史 soft tail 重构 Pilot 接收
信号。目标函数只读取 Adapt Pilot 的 `tx/rx/mask`，不访问 Reward/Data 标签。

## Pilot-only 作用位置结果

在 gap=120 s、3 seeds、5/10 dB、4 帧的更新幅度扫描中，普通内部
`conditioner_film`、`adapter`、`attention_lora`、`ffn_lora` 和 `head` 都只能带来微小
连续输出变化，硬 BER 没有稳定正收益。`channel_residual + pilot_reconstruction` 在
较大更新下 Data BCE 全部下降，但 6 个 seed-SNR 组合只有 4 个硬 BER 改善，不能作为
稳定主路线。

`phase_trend + pilot_signal_reconstruction` 的 3 seeds × 2 SNR × 2 帧 replay 结果为：

| 指标 | 平均结果 |
|---|---:|
| Adapt BCE 改善 | `6.39e-5` |
| Pilot 信号重构改善 | `7.75e-4` |
| Data BCE 改善 | `3.78e-3` |
| Data BER 改善 | `6.51e-4` |
| Data logits 平均绝对变化 | `0.273` |
| Data BER 为正的 seed-SNR 组合 | `2/6` |
| Pilot-only Data BCE 与 Data BER 的 Spearman | `0.617`（小样本，不能替代正式统计） |

这说明直接重构 Pilot 信号比单纯判决 BCE 更接近 Data 连续目标，但硬 BER 方向仍不够
稳定，因此没有进入 5 seeds × 60 frames 正式矩阵，也没有引入 Contextual Bandit。

## 真实逐帧配对结果

使用 `phase_trend + pilot_signal_reconstruction`、AdamW、5 步、学习率 `5e-3`，并由
Reward Pilot 做窗口验收：

### gap=120 s，3 seeds × 2 SNR × 4 帧

| SNR | Frozen BER | Online BER | 配对收益 |
|---:|---:|---:|---:|
| 5 dB | `35.8290%` | `35.0043%` | `+0.8247 pp` |
| 10 dB | `31.7600%` | `28.7109%` | `+3.0490 pp` |

收益集中在状态老化较强的轨迹，5 dB 有一个 seed 基本不变；不能据此宣称跨配置稳定。

### gap=0，3 seeds × 4 SNR × 4 帧

| SNR | Frozen BER | Online BER | 配对收益 |
|---:|---:|---:|---:|
| 0 dB | `21.5929%` | `21.5929%` | `0.0000 pp` |
| 5 dB | `6.0872%` | `6.0330%` | `+0.0543 pp` |
| 10 dB | `0.9549%` | `1.0308%` | `-0.0760 pp` |
| 15 dB | `0.2496%` | `0.2821%` | `-0.0326 pp` |

主 gap=0 场景仍没有稳定超过 Frozen；10/15 dB 还有轻微退化。当前 checkpoint 在无明显
状态老化时已经接近其可达性能，Adapt Pilot 的自监督重构不能保证额外收益。

### gap=600 s，5 seeds × 60 帧压力矩阵

使用同一 `phase_trend + pilot_signal_reconstruction`、AdamW、5 步、漂移门限 `0.7`，
并由 Reward Pilot 逐帧验收和回滚：

| SNR | Frozen BER | Online BER | 平均配对收益 | 正收益 seed |
|---:|---:|---:|---:|---:|
| 5 dB | `29.0881%` | `28.4683%` | `+0.6198 pp` | `4/5` |
| 10 dB | `26.6133%` | `25.8537%` | `+0.7595 pp` | `4/5` |

5 dB 的 seed 配对收益（pp）为 `+0.577/+0.675/+0.762/-0.855/+1.940`，10 dB 为
`-0.347/+1.022/+1.699/+0.135/+1.289`。前后 30 帧均值显示，多数 seed 的收益在后半段
保持或扩大，但单帧仍有较大正负摆动；总计 600 帧中 158 帧真正保留 PEFT 更新、116 帧
触发回滚。该结果证明路线在强 acquisition 老化下有可重复的平均收益，但仍未满足
“每个配置和至少 5/5 seed 稳定超过 Frozen”的主目标，也不应作为 gap=0 主平均结果。

## 漂移门控

`compare.py` 新增 `--online-drift-gate-threshold`。它用 Adapt Pilot 估计 CIR 相对
acquisition 条件的距离，仅决定是否尝试 PEFT 更新，不把估计 CIR 写回模型条件，因此
不会把“CIR 状态恢复”冒充成参数微调。

阈值 `0.7` 的 smoke 结果：gap=0 时 5/10 dB 的距离约 `0.15–0.30`，更新被跳过；gap=600
时距离约 `0.79–1.07`，允许尝试更新。该门控已通过 smoke，但尚未通过正式多 seed 统计，
不能作为稳定收益结论。

## 当前结论与下一步

当前最有证据的方向是：

```text
Adapt Pilot 信号重构
-> phase_trend 参数微调（AdamW/受限步长）
-> Pilot-only 漂移门控
-> Reward Pilot 多帧验收、回滚
```

它已经在强状态老化诊断中产生明显的配对 BER 收益，但还没有满足“主 gap=0、多个 SNR 和
seed 稳定超过 Frozen Offline”的目标。因此本阶段只提交可复现的目标、作用位置和门控
接口，不把诊断收益写成主论文成功结果。5 seeds × 60 frames 的 gap=600 压力矩阵已经
完成但仍有 1/5 seed 负收益。下一步要把 Reward Pilot 的验收从单纯 logits loss/硬 BER
扩展为同一 Reward Pilot 上的信号重构一致性与跨帧最坏退化约束，先解决这个负 seed，
再单独设计能观测 residual CFO/慢相位的主配置实验；在此之前不把该方法提升为主路线，
也不接入 Contextual Bandit。

## 复现实验命令

```powershell
.\.venv-gpu\Scripts\python.exe scripts/replay_pilot_candidate_ranking.py `
  --config configs/eme_long_memory_v2.json `
  --pretrained pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt `
  --output logs/pilot_phase_signal_reconstruction_3s2snr2f_20261010.json `
  --snrs 5 10 --seeds 0 1 2 --frames 2 --delay 116 `
  --pilot-total 256 --gap-seconds 120 --lr 5e-3 --steps 5 `
  --candidates phase_trend_signal_reconstruction --device cuda
```

```powershell
.\.venv-gpu\Scripts\python.exe compare.py `
  --config configs/eme_long_memory_v2.json --method-group main `
  --pretrained pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt `
  --delays 116 --snrs 5 10 --num-seeds 3 --frames 4 `
  --pilot-total 256 --pilot-layout prefix `
  --acquisition-to-data-gap-seconds 600 `
  --online-groups phase_trend --online-algorithm sgd `
  --online-objective pilot_signal_reconstruction --online-optimizer adamw `
  --online-learning-rate 5e-3 --online-steps 5 `
  --online-drift-gate-threshold 0.7 `
  --scheduler fixed --update-interval 1 `
  --online-min-reward-improvement 0 --online-reward-windows 2 `
  --resume --output-dir logs/phase_signal_gap600_driftgate
```
