# 离线适应信道、在线微调均衡：Pilot 更新后泛化实验

## 目标与验收

用户已确认：Level B、前缀 Pilot、0/5/10/15 dB 逐配置超过校准后的传统
非神经、非 RL baseline；在线相对同 checkpoint Frozen 的 BER 增益需有
独立 seed 的可信区间支持，且后期平均优势比早期更明显。暂不规定固定降幅。
允许持续使用本机 GPU，按照机制实验、独立确认、正式主比较的门槛推进。

## 首阶段假设

当前整帧 BCE checkpoint 具备较强的更新前均衡能力，但没有直接优化
“只在 Adapt Pilot 更新后，对后续 Data 更好”。首阶段检验离线训练目标
能否改善这一迁移性质；暂不把缺乏迁移归因于单一根因。

同一历史 checkpoint 建立两个分支：A 继续整帧 BCE；B 使用整帧 BCE 加上
Adapt 更新后 Data BCE。两组使用相同训练 seed、SNR 调度、样本量和优化器，
记录实际耗时；额外的计算开销明确报告，不能声称完全相同 FLOPs。

## 实现边界

- 使用真实 `eme_long_memory_v2`，训练仅使用 `offline_train`，测试使用独立
  seed 的 `heldout_edge`。不扩大扰动范围以追求结果。
- 内循环复现现有窗口 PPO 的 `peft_phase_signal_moderate`：Adapt-only 波形、
  信号重构、AdamW、梯度范数限制、5 步、lr=0.0025、参数增量上限 0.05。
- 外层训练在离线仿真中使用 Data 标签。Reward Pilot 不作为新增外层目标；
  在线阶段继续只用于动作后的 reward 和留出验收。
- 基础 checkpoint 不包含临时 phase_trend 参数，保持现有 strict-load 格式。
  临时 Adapter 初始为零，每个离线训练样本重新从相同起点模拟更新。
- 历史 soft-tail 只由 Adapt-only 接收视图递推，不读 Data 波形或标签。
- 与正式 PPO 一致，只有 Pilot 覆盖的帧尾位置可写回尾状态；当前 prefix
  配置不覆盖帧尾，因此保留原有尾状态，不能用注意力外推的未知尾部替换。
- 首阶段先检验当前帧更新后 Data 的迁移；跨帧参数累计与 PPO 调度要随后单独
  验证，不能由首阶段结果推断长期收益。
- 首阶段使用固定训练预算终点，不使用测试 BER 选择 checkpoint。

## 必须通过的检查

1. 可微 AdamW 候选与正式 PPO 更新在数值上相符，包括参数投影。
2. 改动 Reward/Data 标签与接收波形不改变内循环更新。
3. 更新算子不原地修改基础模型；外层导数通过有限差分核验。
4. 导出的模型可由现有接收机严格加载。
5. 小规模真实信道 GPU 训练、原入口 smoke 和完整 pytest。

## 后续门槛

配对比较 A0/A1 与 B0/B1：0 为冻结，1 为同预算固定更新组件消融。
检查各分支相对自身 Frozen 的增益，同时比较 B1 与 A1 的绝对 BER，避免
因 B0 变差造成虚假的“更大适配增益”。小规模筛查只用于判断机制，独立确认
和较长轨迹才用于收益与后期优势结论。PPO 保持最终主方法的调度器。

若更新后泛化仍不成立，回到 Pilot 可观测性、物理状态恢复和 PEFT 表达能力
分解误差，不能反复修改测试配置或仅凭正均值宣布目标完成。

## 首轮复现入口

本机 GPU Python 为 `D:/Research/RL4EQ/.venv-gpu/Scripts/python.exe`；GTX 1650
显存 4 GB。历史 checkpoint SHA256 为
`8cd3912d1c957c9689c51730aff9d27a8f74942186c802a7483e5781af65aec4`。
新路径暂不启用 AMP，避免可微 AdamW 内循环的精度差异。

```powershell
$taskPython = 'D:/Research/RL4EQ/.venv-gpu/Scripts/python.exe'
$taskSource = 'D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt'
& $taskPython pretrain.py --config configs/eme_long_memory_v2_pilot_transfer.json --stage pilot_transfer --transfer-objective baseline --resume $taskSource --steps 64 --batch-size 1 --device cuda --save-dir pretrained/pilot_transfer_baseline64_20261101
& $taskPython pretrain.py --config configs/eme_long_memory_v2_pilot_transfer.json --stage pilot_transfer --transfer-objective post_adapt --resume $taskSource --steps 64 --batch-size 1 --device cuda --save-dir pretrained/pilot_transfer_post64_20261101
& $taskPython compare.py --config configs/eme_long_memory_v2_pilot_transfer.json --pretrained pretrained/pilot_transfer_baseline64_20261101/model_best.pt --transfer-checkpoint pretrained/pilot_transfer_post64_20261101/model_best.pt --snrs 0 5 10 15 --seed-start 41001 --num-seeds 5 --frames 8 --state-split heldout_edge --device cuda --output-dir logs/pilot_transfer_four_arm64_5s8f_20261101
```

四臂机制实验默认每帧从零 Adapter 模拟同一动作，隔离“单次更新能否迁移”；
加 `--transfer-persistent` 才跨帧累计提交的 PEFT。机制实验不包含 PPO 窗口回滚，
不冒充正式主结果。两组原始接收波形逐帧 SHA256 必须匹配。
统计先在每条 seed 轨迹内平均，再计算 seed 级配对 t 区间；单 seed 不报告推断区间。

## 实现阶段验证记录

- 完整 pytest：`644 passed`，390.72 秒；为与训练分离，测试进程隐藏 CUDA，
  使用共享 GPU 虚拟环境中的 CPU 后端执行。
- 新增定向检查 11 项通过，覆盖正式 PPO 数值一致性、隐藏信息隔离、
  有限差分、零初始化 checkpoint、尾状态边界、四臂配对与 seed 级统计。
- 原 `pretrain.py --stage all --steps 2`、`online_train.py --frames 8` 和
  `compare.py --method-group main --frames 1` 回归 smoke 均通过；本次回归
  smoke 使用 CPU，以免占用正在进行的 GPU 机制实验。
- 新训练入口已完成真实 Level B GPU 训练。A/B 均为 64 步、每步一个样本，
  训练 seed 为 31000–31063，SNR 按 0/5/10/15 dB 轮换。
- A 耗时 164.39 秒，B 耗时 209.08 秒；两组未并发使用 GPU，但同时有 CPU
  测试进程，耗时仅作资源记录，不作为严格性能基准。
- 四臂入口同 checkpoint 自对照 smoke 通过：两分支输出与增量逐项一致。

这些记录证明实现可运行与信息边界成立，不构成在线 BER 收益证据。
未见信道轨迹的配对机制结果单独整理后再更新研究结论。
