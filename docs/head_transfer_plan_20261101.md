# 实际神经 head 的离线更新后泛化实验

## 首轮留下的问题

相位趋势动作首轮四臂结果没有显示新增外层目标优于普通监督。下一轮检验
真实神经参数的 Adapt Pilot 更新，沿用现成 `peft_head_fast` 动作，不增加动作维度。
保持 Level B、prefix、0/5/10/15 dB、原信道状态范围与信息边界。

## 预先固定的首轮设置

- 原 checkpoint 不变，训练 A/B 均从它开始；A 是整帧 BCE，B 是整帧 BCE
  加上内循环更新后的 Data BCE，权重均为 1。
- 仅更新现有 head：AdamW、lr=5e-4、1 步、梯度范数限制 1、增量上限 0.05。
  内循环使用 Adapt-only 整帧视图及已知 Pilot BCE，不能看到 Reward/Data 波形或标签。
- 两分支各 256 个外层优化步骤，batch=1、外层 lr=1e-4，训练 seed=34000–34255。
  SNR 按主网格轮换，使用 `offline_train`，每个样本包含一帧历史与一帧训练目标。
- 所有新实验使用 FFT 算子；两组相同。CUDA 外层反传需要高阶导数，两组统一
  使用 math attention；高效 attention backward 当前不支持该二阶导数。
- 第一个配对机制筛查预先选定 `heldout_edge`、seed=43001–43005、每条轨迹 8 帧。
  每帧重置候选，先验证单次更新迁移；跨帧累计和 PPO 调度随后另行验证。
- 不使用测试集选 checkpoint，均使用固定预算终点。若结果不足，不扩大正式主表。

## 实现验证

内循环已经扩展为可配置的既有 PEFT 组与 BCE/信号重构目标。新增检查证明
head 和 adapter_lora_head 的 BCE AdamW 候选与正式 PPO 更新数值相符；
改变 Reward/Data 波形和标签不改变更新；head 更新后外层梯度通过有限差分核验。
真实 1024/116 CUDA 两步 smoke 已通过。

## 复现

```powershell
$taskPython = 'D:/Research/RL4EQ/.venv-gpu/Scripts/python.exe'
$taskSource = 'D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt'
$env:RL4EQ_LINEAR_OPERATOR_BACKEND = 'fft'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& $taskPython pretrain.py --config configs/eme_long_memory_v2_head_transfer.json --stage pilot_transfer --transfer-objective baseline --resume $taskSource --steps 256 --batch-size 1 --device cuda --save-dir pretrained/head_transfer_baseline256_20261101
& $taskPython pretrain.py --config configs/eme_long_memory_v2_head_transfer.json --stage pilot_transfer --transfer-objective post_adapt --resume $taskSource --steps 256 --batch-size 1 --device cuda --save-dir pretrained/head_transfer_post256_20261101
& $taskPython compare.py --config configs/eme_long_memory_v2_head_transfer.json --pretrained pretrained/head_transfer_baseline256_20261101/model_best.pt --transfer-checkpoint pretrained/head_transfer_post256_20261101/model_best.pt --snrs 0 5 10 15 --seed-start 43001 --num-seeds 5 --frames 8 --state-split heldout_edge --device cuda --output-dir logs/head_transfer_four_arm256_5s8f_20261101
```
