# RL4EQ：Level B 极端长回波下的 RL-Modulated Neural Block Equalizer

> 2026-11-02：[匹配前缀短窗训练与梯度解释](docs/prefix_transfer_stage_20261102.md)已完成。
> A/B 各 256 步、5 个独立 seed × 8 帧 × 四 SNR 的在线改善区间均包含零；
> 更高接受率没有证明在线泛化。用户已明确持续跟踪场景，正在按
> [128 帧结果](docs/persistent_prefix_stage_20261102.md)也未建立平均或后期优势，
> 下一步在训练分布检验 Pilot 聚合的方向稳定性；主目标未完成。

> 2026-11-01：新增真实 Level B 的 Pilot 更新后 Data 泛化训练与四臂配对机制实验。
> 用户已授权重新训练适配友好的 checkpoint；原 checkpoint 保留为对照。
> 训练/验证边界与分阶段验收见[实验计划](docs/pilot_transfer_plan_20261101.md)。
> 该阶段只检验更新机制，不能替代正式 PPO、传统 baseline 或长期收益比较。
> 首轮结果见[四臂机制实验](docs/pilot_transfer_stage_20261101.md)：64 步下两分支在线
> BER 基本重合，新增外层目标尚未证明额外适配收益。
> 后续[神经 head 四臂结果](docs/head_transfer_stage_20261101.md)也未通过：256 步后
> 10/15 dB 在线动作没有改变 Data 判决，0/5 dB 增益区间跨零。下一步先核验
> 神经调制对主要均衡误差的作用能力，并完成传统 baseline 的记忆与边界校准。
> [传统线性校准与独立核验](docs/linear_baseline_calibration_stage_20261101.md)已完成
> 首轮预设网格：同步和块边界会显著增强部分传统参考；主方法组暂未改变。
> 当前第一性原理拆解、未验证假设和后续依赖见[路线复核](docs/first_principles_20261101.md)。
> [统一校准比较](docs/calibrated_compare_stage_20261101.md)已完成：显式冻结协议接入
> `compare.py`，800 条同轨迹记录中 RL BER 与 Frozen 一致；对较强 FDE 的神经
> 优势仅在 15 dB 有配对区间支持，整体目标仍未达成。
> [现有相位组作用能力诊断](docs/existing_phase_capacity_stage_20261101.md)显示：
> 相位组比 head 更能改变 Data 输出，但三臂均未证明稳定 BER 收益；下一步在
> 现有结构内验证其更新后的离线泛化，不新增正式 RL 动作。
> [现有神经相位组四臂训练](docs/phase_transfer_stage_20261101.md)现已完成：A/B 各
> 256 步后，B 的四 SNR 在线改善区间均包含零，也未可靠超过 A Online。普通
> 监督 A 在 0 dB 有探索性正向区间，不能据此宣布主目标完成。停止扩大同配置。
> [相位作用几何](docs/phase_modulation_geometry_20261101.md)和离线梯度诊断区分了
> 有限输出族、目标梯度差异与实际 BER 收益，下一步先确认可观测误差与调制位置。
> [接收波形置零审计](docs/receiver_visibility_stage_20261101.md)发现：适配/验收与
> 整帧推理的输入模型存在差异，10/15 dB 的置零 Reward 错误在整帧输入下消失。
> 这仍是离线诊断，未证明在线收益，尚未修改实际控制器的信息边界。
> [DFE 先验与遗忘因子校准](docs/dfe_rls_calibration_stage_20261101.md)已完成 36 项
> 网格、冻结和独立核验；训练选优没有获得四 SNR 可靠留出收益，保留原传统参考。
> [接收输入四组合组件实验](docs/receiver_input_candidate_stage_20261101.md)已完成
> 48 个上下文、384 条配对记录；仅改变更新/验收的波形范围没有获得可靠 BER 增量。
> 原默认控制器保持不变；中高 SNR 的 Reward 饱和仍须与候选作用能力分别处理。
> [DFE 已知 Pilot 反馈核查](docs/dfe_known_feedback_stage_20261101.md)在固定参数的
> 48 个配对帧上没有改变 Data BER；保留可选实现，不替换默认基线或扩大矩阵。
> 用户现已允许在线读取整帧接收波形并严格隔离未知发送符号和 Data 标签。
> [匹配整帧输入训练计划](docs/full_rx_transfer_plan_20261101.md)使用现有神经 phase
> 组比较普通监督与更新后目标；新配置显式声明输入协议，历史配置保留原语义。
> [匹配整帧输入四臂结果](docs/full_rx_transfer_stage_20261101.md)已完成 A/B 各 256 步：
> 0 dB 两分支均有在线正向区间，但 B Online 比 A 更差；5/10 dB 收益仍不确定，
> 15 dB 没有接受更新。整体目标未达成，不扩大同配置矩阵。
> 用户已指示按建议继续，包括必要的神经调制结构调整。
> [冻结错误定位](docs/error_geometry_stage_20261101.md)发现 15 dB Data 错误集中于
> 帧尾，原神经残差未改变这些判决；[神经符号残差计划](docs/symbol_transfer_plan_20261101.md)
> 据此增加观测覆盖感知的小型分支，离线训练 1153 参数、在线只微调 17 参数 head。
> [神经符号残差四臂结果](docs/symbol_transfer_stage_20261101.md)：5/10 dB 的 Frozen
> 相对原模型有可靠配对改善，但在线额外收益未成立；下一步验证 Pilot 任务与帧尾覆盖的匹配。
> [前缀短窗与更新幅度诊断](docs/prefix_task_stage_20261102.md)已完成：短窗验收
> 可以改变接受率，安全半径内较大更新可以改变判决，但均没有可靠在线增益。
> 下一步直接验证匹配边界任务的离线训练，不继续单独扩大步幅或 PPO 预算。

> 当前主配置复核（2026-09-04）：`eme_long_memory_v2` Level B 使用 `0 s`
> acquisition 空档、`coherence_time_seconds=1200` 和 `physics_warm_start_iterations=8`。
> 当前主结果以 `AGENTS.md` 的离散安全动作 PPO 路线为准。`Frozen Offline NN` 是同轨迹
> 参考，用来测量在线动作带来的增量；Pilot-Driven 单候选 PEFT 结果保留为历史专项消融。
> acquisition 老化测试和 gap=0 主场景分开报告。

本项目研究 EME 启发但不等同于完整物理 EME 的极端稀疏长时延扩展信道。当前主线使用独立的 `eme_long_memory_v2` 长记忆 profile；`eme_measurement_v1` 保留作历史兼容和对照，不把二者混合平均。主线是：

```text
Level A/B/C 可控信道族
-> 全程 Pilot 条件课程监督预训练
-> 整帧缓冲、非因果神经块均衡器
-> 离散安全 PPO 按 Pilot 状态选择有界均衡器动作
-> Adapt Pilot 更新选定的 Adapter/LoRA/head 参数
-> Reward Pilot 验收、拒绝和回滚；窗口级 reward 更新策略
-> 与传统非神经、非 RL baseline 公平比较
```

研究目标不是超过所有可能的强模型驱动序列检测器，也不是只找到一个传统方法很弱的工作区。当前第一性目标是在 Level B 主论文场景中，让离散安全动作 PPO 调制的神经均衡器逐配置明显超过传统非神经、非 RL 均衡器，并相对同 checkpoint 的 `Frozen Offline NN` 展现可重复的在线增益。`BER_data < 0.01` 保留为辅助系统指标，不再作为第一成功门槛。

## 研究契约

- 主论文场景是 Level B：`eme_long_memory_v2` 使用 11.6 ms 雷达深度约束、10 ksymbol/s 离散化和 116 个符号最大记忆，主 SNR 为 `[0, 5, 10, 15]` dB。
- Pilot 只放在帧前缀；`two_block` 和 `multi_block` 不再作为主论文 Pilot 布局。
- Level A 用于课程学习和可达性校准；Level C 只作为压力测试，不混入 Level B 主平均。
- 接收机是整帧缓冲、非因果块神经均衡器；“在线”指信道运行期间按帧持续适配，不是逐符号即时输出。
- 当前 proposed 主路线是离散安全动作 PPO 调制的神经块均衡器；主对照包含传统非神经、非 RL 均衡器和同轨迹 `Frozen Offline NN` 参考。旧 `Pilot-Driven Online Adaptation` 仅保留为专项消融，不混入主方法组。
- 2026-10-29 的 RL-PEFT 执行链路审计和 seed 复核见[研究记录](docs/rl_peft_execution_audit_20261029.md)：修正 Reward 回滚时序、前缀 Pilot soft-tail 可观测范围、策略状态初始化和 PPO 动作编号；当前 SNR 复核尚未证明稳定 BER 收益。
- 2026-10-29 的 Pilot-only PEFT 排序复核见[阶段结果](docs/pilot_peft_stage_20261029.md)：修复 Data 波形隔离漏洞，并完成累计目标、Reward 评分和当前 PEFT 候选的留 seed 排序诊断；排序门槛尚未通过，因此未启动正式大矩阵。
- 2026-10-30 的 Pilot 状态 embedding 复核见[阶段结果](docs/rl_peft_pilot_state_embedding_update16_20261030.md)：Adapt-only 状态特征已接入 PPO，但在 5 seed × 60 帧中没有改变动作或 BER；在线收益仍小且 seed 区间包含零。
- 2026-10-30 的 PPO 动作概率和学习率复核见[阶段结果](docs/rl_peft_policy_probability_lr_20261030.md)：高学习率在 5 dB 的新 seed 上改变了部分 PEFT 动作，10 seed 全部略有收益，但它相对默认学习率的增量仍不确定。
- 2026-10-30 的四 SNR 学习率筛查见[阶段结果](docs/rl_peft_multisnr_lr1e3_20261030.md)：`1e-3` 在四个 SNR 的平均 BER 都略好于 Frozen，但各 SNR 配对区间均跨零。
- 2026-10-30 的 Level B 主方法比较见[阶段结果](docs/rl_peft_main_comparison_l1e3_20261030.md)，identity-only 因果对照见[后续结果](docs/rl_peft_identity_control_20261030.md)：在线收益确实来自 PEFT 参数更新，但四个 SNR 上都很小且不确定；下一步在更强状态失配下复核。
- 2026-10-30 的更强 CFO/相位失配复核见[阶段结果](docs/rl_peft_strong_gap_identity_control_20261030.md)：Adapt Pilot 驱动的 `phase_trend` PEFT 更新确实改变输出，但在线 BER 增益在四个 SNR 上仍不显著；下一步改测 Pilot-only 候选排序与更新强度，不重训离线 checkpoint，也不让 Data 参与在线更新。
- 2026-10-30 的 Pilot-only PEFT 总复核见[阶段记录](docs/rl_peft_pilot_multiscale_ranking_strong_gap0_20261030.md)、[Reward Pilot 长度对照](docs/rl_peft_reward_pilot_length_replay_20261030.md)和[PEFT 更新半径筛查](docs/rl_peft_update_radius_replay_20261030.md)：候选排序 Spearman 最高为 0.415；Reward 从 32 增至 64 未消除有害更新；半径 0.01/0.025/0.05 在 5 seed × 4 SNR 下都未证明稳定收益。在线更新目标、参数范围、验收样本量和步幅的诊断仍未达到“稳定超过 Frozen”。下一步需要决定是否允许离线训练一份专门支持 Pilot 驱动 PEFT 的新 checkpoint，当前 checkpoint 保留为 Frozen 基线。
- Pilot-Driven 单候选 PEFT 的历史回放见 [统一 Pilot replay 协议复核](docs/pilot_replay_protocol_audit_20261014.md) 与 [Reward logits 守门复核](docs/online_logit_guard_replay_20261015.md)。它曾在部分 SNR/seed 上获得小幅正增益，但更新目标和候选选择协议与当前 PPO 主方法不同，不能并入当前主结果，也没有证明所有主 SNR 稳定提升。
- [10 dB 长窗口复核](docs/formal_signal_guard_logitcap01_gap0_10db_seed77_81_60f.md) 及其[严格门控对照](docs/formal_signal_guard_logitcap01_signalmin0005_gap0_10db_seed77_81_60f.md)表明，更多帧或更严格的 Reward 门控单独都不能解决 Pilot 到 Data 的迁移问题；均为 Pilot-Driven 历史路线的专项结果。
- 传统 baseline 不使用神经网络，不使用 RL，只使用 acquisition/Adapt Pilot、接收信号和传统自适应规则；在 CFO/慢相位扰动实验中必须包含基于 Pilot 的合理补偿，不能人为打残 baseline。
- Reward Pilot 只用于动作后的 reward 与留出评估；Data 标签只用于离线监督和仿真 `BER_data` 评估。
- 在线 observation、reward、动作选择和调制更新不使用数据标签上界。
- 当前 RL 路线采用离散安全动作和窗口级 reward；逐帧连续 modulation 不再作为主实施路线。
- 当前在线均衡主线是 Adapt Pilot 驱动的受限 PEFT 更新和 Reward Pilot 验收；由于 Pilot-only reward 尚未在多个 SNR/seed 上稳定排序 Data 收益，Contextual Bandit 暂不进入主比较。Bandit 不直接生成高维参数增量，只有排序门槛通过后才考虑接入。
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
.\.venv-gpu\Scripts\python.exe online_train.py --config configs/continual_ppo.json --pretrained pretrained/final_smoke/model_best.pt --frames 8 --num-seeds 1 --window-size 4 --update-interval 4 --delays 20 --snrs 10 --pilot-total 64 --pilot-layout prefix --amp --output-dir logs/final_online_smoke
.\.venv-gpu\Scripts\python.exe scripts/diagnose_action_delay.py logs/final_online_smoke --output logs/final_online_smoke/action_delay.json
.\.venv-gpu\Scripts\python.exe compare.py --config configs/continual_ppo.json --method-group main --pretrained pretrained/final_smoke/model_best.pt --delays 20 --snrs 10 --num-seeds 1 --frames 1 --pilot-total 64 --pilot-layout prefix --resume --output-dir logs/final_compare_smoke
# compare.py 未显式指定调度器时沿用配置中的 `online_scheduler`；当前主配置使用固定候选选择器。
# 只有 Pilot-only 排序通过稳定性门槛后，才显式使用：--scheduler bandit
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
