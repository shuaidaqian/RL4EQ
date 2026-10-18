# AGENTS.md

我是中文语境开发者，所有对话和代码注释都用中文。

本仓库当前唯一主路线是 **Level B 极端稀疏长记忆 ISI 信道上的冻结神经均衡器与 Pilot-only 物理状态后验跟踪**。离线阶段预训练基础均衡能力；在线阶段只读取帧前缀 Pilot，跟踪稀疏 CIR、residual CFO、慢相位和跨帧 soft-tail belief。在线不更新神经主干参数，PEFT、Bandit、PPO 只能作为通过代理排序门槛后的历史或专项消融。

必须遵守以下契约：

- 主论文场景为 Level B `eme_long_memory_v2`；Level C 只能做压力测试，不混入主平均。
- 主论文 SNR 固定为 `0/5/10/15 dB`，主时延为物理配置确定的 `116` 个符号。
- Pilot 只放在帧前缀；two-block/multi-block 仅保留兼容和诊断用途。
- 接收机是整帧缓冲、非因果块均衡器；在线状态按帧提交，跨帧维护 soft-tail 均值、方差和置信度。
- Adapt Pilot 是物理状态更新的唯一输入。Reward Pilot 只可用于已知 Pilot 的留出物理一致性检查和事后排序验证，不得用其 BCE 直接驱动未经验证的参数更新。
- Data 波形可以进入冻结均衡器做检测；Data 标签只用于离线监督和事后硬 BER，不能进入在线状态更新、风险计算、动作选择或提交/回滚。
- 任何状态候选都必须原子化提交 CIR、phase/CFO、tail belief；低置信度、OOD 或代理排序未通过时必须 abstain 并保持 Frozen。
- 未来风险代理与 Data BER 的正相关只能作为独立信道轨迹上的实证验收门槛，不能声称对任意时变信道有理论保证。
- 正式第一目标是 proposed 在 Level B 主配置逐配置超过带合理 Pilot 补偿的传统非神经、非 RL baseline，并报告随在线帧数的变化；`BER_data < 0.01` 只是辅助指标。
- `Perfect-CSI`、oracle CIR/tail 和固定强块检测器只作诊断参考，不纳入主成功门槛。
- 不恢复旧 A2C、逐符号 PPO、Data Oracle、多载波、MIMO、RIS、非线性、编码和高阶调制主路线。
- 旧 PEFT/PPO/Bandit 代码和实验结果可以保留用于复现失败证据，但必须标注为历史路线，不能成为默认入口。

唯一权威配置为 `configs/level_b_state_tracking.json`。默认入口：

```powershell
python pretrain.py --config configs/level_b_state_tracking.json --stage all --steps 2 --batch-size 1 --amp --save-dir pretrained/state_smoke
python online_train.py --config configs/level_b_state_tracking.json --pretrained pretrained/state_smoke/model_best.pt --method state --frames 8 --num-seeds 1 --delays 116 --snrs 10 --pilot-total 128 --pilot-layout prefix --output-dir logs/state_smoke
python compare.py --config configs/level_b_state_tracking.json --method-group main --pretrained pretrained/state_smoke/model_best.pt --delays 116 --snrs 10 --num-seeds 1 --frames 1 --pilot-total 128 --pilot-layout prefix --output-dir logs/compare_smoke
python -m pytest -q -p no:cacheprovider
```

`online_train.py --method pilot/ppo` 只用于历史诊断；神经方法缺少 checkpoint 时必须直接失败，禁止静默随机初始化。

兼容说明：`RL-Modulated Neural Block Equalizer` 是已退出主路线的历史实现；接收机仍是整帧缓冲、非因果，正式目标是 proposed 明显超过传统，且在 CFO、相位扰动下传统方法必须有合理补偿。在线流程不使用数据标签上界。
