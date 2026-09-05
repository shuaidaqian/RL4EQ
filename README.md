# RL4EQ：Level B 极端长回波下的 RL-Modulated Neural Block Equalizer

> 当前主配置复核（2026-09-04）：`eme_long_memory_v2` Level B 使用 `0 s`
> acquisition 空档、`coherence_time_seconds=1200` 和 `physics_warm_start_iterations=8`。
> 旧的 Frozen/Online 数值来自已废弃的条件边界，不能作为当前主结果。当前主实验要求重新
> 在同一 Level B 轨迹上比较参数冻结但使用当前帧 Pilot 条件的 Frozen 与 Pilot 驱动恢复/微调的
> Online，并单独报告 Online 相对 Frozen 的配对增益；acquisition 老化只作为独立压力测试。

本项目研究 EME 启发但不等同于完整物理 EME 的极端稀疏长时延扩展信道。当前主线使用独立的 `eme_long_memory_v2` 长记忆 profile；`eme_measurement_v1` 保留作历史兼容和对照，不把二者混合平均。主线是：

```text
Level A/B/C 可控信道族
-> 全程 Pilot 条件课程监督预训练
-> 整帧缓冲、非因果神经块均衡器
-> Adapt Pilot 驱动受限在线元适配
-> Reward Pilot 安全门控与跨帧 soft-tail 递推
-> 可选 RL 选择离散更新强度和更新率
-> 与传统非神经、非 RL baseline 公平比较
```

研究目标不是超过所有可能的强模型驱动序列检测器，也不是只找到一个传统方法很弱的工作区。当前第一性目标是在 Level B 主论文场景中，让同一离线 checkpoint 初始化的 `Pilot-Driven Online Adaptation` 相对 `Frozen Offline NN` 产生可重复的额外收益，并同时明显超过传统非神经、非 RL 均衡器。`BER_data < 0.01` 保留为辅助系统指标，不再作为第一成功门槛。

## 研究契约

- 主论文场景是 Level B：`eme_long_memory_v2` 使用 11.6 ms 雷达深度约束、10 ksymbol/s 离散化和 116 个符号最大记忆，主 SNR 为 `[0, 5, 10, 15]` dB。
- Pilot 只放在帧前缀；`two_block` 和 `multi_block` 不再作为主论文 Pilot 布局。
- Level A 用于课程学习和可达性校准；Level C 只作为压力测试，不混入 Level B 主平均。
- 接收机是整帧缓冲、非因果块神经均衡器；“在线”指信道运行期间按帧持续适配，不是逐符号即时输出。
- 当前主比较只保留传统均衡器、`Frozen Offline NN` 和 `Pilot-Driven Online Adaptation`；早期 PPO/调制策略仅作为代码兼容或专项消融，不作为第二研究点的主对照。
- 传统 baseline 不使用神经网络，不使用 RL，只使用 acquisition/Adapt Pilot、接收信号和传统自适应规则；在 CFO/慢相位扰动实验中必须包含基于 Pilot 的合理补偿，不能人为打残 baseline。
- Reward Pilot 只用于动作后的 reward 与留出评估；Data 标签只用于离线监督和仿真 `BER_data` 评估。
- 在线 observation、reward、动作选择和调制更新不使用数据标签上界。
- 当前 RL 路线采用离散安全动作和窗口级 reward；逐帧连续 modulation 不再作为主实施路线。
- 当前在线均衡主线是 Adapt Pilot 驱动的两时间尺度约束元适配；RL 只调度安全动作，不直接生成高维参数增量，PPO 仅作调度器消融。
- 为隔离在线微调本身的增量，可用 `compare.py --online-condition-source acquisition` 固定 acquisition 条件；该消融仍只用 Adapt Pilot 更新 PEFT、只用 Reward Pilot 验收/回滚。默认主线为 `pilot_cir_phase`。
- 所有神经方法共享同一 `tail_update_alpha` 跨帧 soft-tail 递推；Frozen 与 Online 的差异不能再来自不一致的尾状态更新。
- Data Oracle 不恢复。
- clean Level B 作为 sanity check，用来证明传统均衡器在干净线性 BPSK 下确实很强；主攻场景逐级加入 residual CFO 与慢相位扰动。
- 非线性、信道编码和高阶调制只作为后续按需扩展，不进入当前主实验。

## 方法分组

正式主比较包含：

```text
traditional:
  LMMSE-FIR
  LMS
  NLMS
  RLS Linear
  DFE-RLS
  SC-FDE-MMSE
  CFO-Corrected LMMSE-FIR
  CFO-Corrected DFE-RLS
  CFO+DD-Phase LMMSE-FIR
  CFO+DD-Phase DFE-RLS

proposed:
  Frozen Offline NN
  Pilot-Driven Online Adaptation
```

`Frozen Offline NN` 与在线方法加载完全相同的离线 checkpoint，并共享同一条信道轨迹、帧结构和跨帧 soft-tail；Frozen 只冻结网络参数和 CIR 更新，但使用当前帧已知 Pilot 生成模型所需的 phase/CFO 条件。Online 在此基础上使用 Adapt Pilot 恢复物理状态并更新受限 PEFT 参数，再由 Reward Pilot 验收或回滚。因此论文中真正要解释的是“离线基础能力”与“在线状态恢复及参数微调增益”的差别，而不是把不同网络或不同信道拿来比较。

诊断参考单独报告，不进入主成功门槛：

```text
diagnostic:
  Perfect-CSI Block
  Fixed CG-BPSK-DD Block Detector
```

## 当前代码入口

```powershell
.\.venv-gpu\Scripts\python.exe pretrain.py --config configs/continual_ppo.json --stage all --steps 2 --batch-size 1 --amp --save-dir pretrained/final_smoke
.\.venv-gpu\Scripts\python.exe pretrain.py --config configs/eme_long_memory_v2.json --stage online_meta --steps 2 --batch-size 1 --amp --save-dir pretrained/eme_online_meta_smoke
.\.venv-gpu\Scripts\python.exe online_train.py --config configs/continual_ppo.json --pretrained pretrained/final_smoke/model_best.pt --frames 8 --num-seeds 1 --window-size 4 --update-interval 4 --delays 20 --snrs 10 --pilot-total 64 --pilot-layout prefix --amp --output-dir logs/final_online_smoke
.\.venv-gpu\Scripts\python.exe scripts/diagnose_action_delay.py logs/final_online_smoke --output logs/final_online_smoke/action_delay.json
.\.venv-gpu\Scripts\python.exe compare.py --config configs/continual_ppo.json --method-group main --pretrained pretrained/final_smoke/model_best.pt --delays 20 --snrs 10 --num-seeds 1 --frames 1 --pilot-total 64 --pilot-layout prefix --resume --output-dir logs/final_compare_smoke
.\.venv-gpu\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

GPU 环境安装：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_gpu_env.ps1 -Recreate
.\.venv-gpu\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

PyTorch wheel 自带 CUDA runtime；脚本不修改系统 CUDA Toolkit。

## 目录结构

```text
agent/
  cir_estimator.py              Adapt/acquisition 驱动 CIR 条件
  modulation.py                 有界低维调制状态
  discrete_safe_policy.py       离散安全动作 PPO 策略
  rl_modulator.py               observation 编码器；旧连续 policy 仅保留兼容
  unfolded_equalizer.py         整帧非因果神经块均衡器
baseline/
  traditional_equalizers.py     传统非神经、非 RL baseline
  block_equalizers.py           强模型驱动诊断检测器
env/
  channel_profiles.py           Level A/B/C 信道族
  eme_channel_profiles.py       文献包络约束的 EME profile 采样器
  extreme_delay_channel.py      长回波、漂移、跨帧 ISI
  impairments.py                residual CFO 与慢相位扰动
  frame_structure.py            前缀 Pilot 帧结构；其他布局仅保留消融/诊断兼容
  comm_env.py                   acquisition + 普通帧 episode 环境
training/
  curriculum.py                 真实信道 Pilot 条件课程预训练
  windowed_discrete_ppo.py      离散安全动作 + 窗口级 reward 在线 runner
  rl_modulated_online.py        旧连续调制 runner，仅保留历史诊断兼容
  continual_ppo.py              旧兼容 smoke runner，不作为新主路线
evaluation/
  metrics.py                    BER、goodput、Spearman、逐配置汇总
  bootstrap.py                  seed + 连续帧块 bootstrap
```

## 门槛式自动推进

执行顺序固定为：

```text
1. 真实传统 baseline 可运行且标签隔离
2. 离线预训练 checkpoint strict-load
3. Windowed Discrete PPO 在线 smoke
4. append-safe compare smoke
5. 小样本门槛矩阵
6. 只有前置门槛通过，才进入 pilot sweep
7. 只有 pilot sweep 通过，才进入正式主矩阵
```

任何前置门槛失败时，必须停止并分析原因，不能用总体平均、强诊断方法或 Data 标签上界掩盖失败。

## 当前实现边界

当前代码已经补齐真实传统 baseline、神经均衡器最小链路、离散安全动作窗口级 PPO、真实信道预训练数据流、strict-load checkpoint、append-safe compare smoke。当前执行门槛调整为：先用传统难度校准脚本确认 clean、CFO、慢相位、CFO+慢相位四级复杂度下传统 baseline 的真实难度；再训练 Proposed，使其在选定 Level B impaired 主场景逐配置明显超过带合理补偿的传统 baseline。未通过前不能启动正式主矩阵或撰写“神经/RL 显著优于传统”的结论。
