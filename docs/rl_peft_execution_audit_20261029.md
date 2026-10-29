# RL-PEFT 执行链路审计与阶段结果（2026-10-29）

## 研究问题

固定离线 checkpoint，用前缀 Adapt Pilot 更新均衡器内部 PEFT 参数，Reward Pilot 决定保留或回滚；Data 波形和标签只在控制流程结束后评估。本阶段沿当前仓库要求的离散安全动作 PPO 路线检查：参数更新是否真正执行，Reward 拒绝后 Data 是否使用恢复后的参数，冻结身份动作是否与 Frozen 接收状态一致，以及 acquisition 老化时参数动作是否带来可测的 Data 收益。

本文件补充而不覆盖 [PEFT 历史进度对照](peft_within_frame_audit_20261028.md)。10 月 15 日的多候选 Reward 排序结果与此处的单动作 RL 选择协议不是同一方法，不能拼成一条实验曲线。

## 执行链路审计发现

1. PEFT 更新原来先应用到当前帧，再等窗口结束才做总体验收。Reward Pilot 已经能在当前帧验收，因此代码现改为逐帧验收；拒绝后立刻恢复参数，再继续更新接收状态和事后 Data 评估。回归测试强制拒绝候选，确认两次 Data 评估前向都看到恢复后的参数。
2. 前缀 Pilot 之外的尾部位置输入被置零。旧代码却把这些位置的模型输出当成观测值更新 soft-tail。现在仅当 Adapt 或 Reward Pilot 覆盖该位置时才更新；未观测位置保留旧 tail。
3. RL Data 评估旧代码使用更新前 soft-tail，Frozen 使用当前帧 Pilot 更新后的 soft-tail，造成 identity/no-PEFT 对照不公平。现在两次事后 Data 前向都使用 Pilot 更新后的同一个 tail。修复后，同一组 5 seed × 24 帧中，104/104 个无 PEFT 更新帧的 RL BER 与 Frozen 一致。
4. PPO 安全初始化旧代码把编码器和 GRU 一并置零，使策略状态无法表达 Pilot 观测。现在保留特征网络随机权重，只把动作输出层置零并加安全先验；新增测试验证不同 Pilot 状态可学出不同动作概率。
5. 旧代码把 `rollback_identity` 的采样动作先改名为 `identity`，造成 PPO 更新时使用的动作编号与采样编号不一致。现在只在执行阶段映射为 identity，训练记录保留原始采样动作；新增回归测试验证动作索引和 log-prob 一致。

相关验证覆盖在 `tests/test_windowed_discrete_ppo.py`：Adapt/Data 波形隔离、Reward 拒绝后参数回滚、前缀 Pilot tail 可观测性、Frozen/identity 状态一致性、状态条件策略更新和 PPO 动作索引一致性。

## 固定条件

- Level B `eme_long_memory_v2`，delay `116`，prefix Pilot 总长 `256`（Adapt `224`、Reward `32`），SNR `5 dB`。
- checkpoint：`pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`。
- checkpoint SHA256：`8CD3912D1C957C9689C51730AFF9D27A8F74942186C802A7483E5781AF65AEC4`；离线权重没有修改。
- RL 使用独立随机初始化的安全 PPO 策略，不加载训练完成的 policy checkpoint；窗口长 `8`，每 `8` 帧更新策略，CIR 固定、条件来源 `pilot_cir_phase`。
- 所有在线候选学习只使用 Adapt Pilot；Reward Pilot 用于当前帧验收及窗口回滚；Data 标签只用于最终 BER 与连续指标统计。
- `gap=0` 是 acquisition 与在线信道无老化的 sanity check；`gap=600 s` 是 acquisition 老化诊断，单独报告，不与主结果混合。

可复现的独立安全动作配置：

- [相位趋势 PEFT](../configs/rl_peft_phase_signal_diagnostic.json)
- [Adapter/LoRA 保守步长](../configs/rl_peft_adapter_lora_diagnostic.json)
- [Adapter/LoRA 轻量步长](../configs/rl_peft_adapter_lora_light_diagnostic.json)

三者均保留 identity 并只启用一个 PEFT 候选，避免把不同参数对象混入同一次比较。配置设置 `diagnostic_only=true`，不属于正式主矩阵。

## 结果

修正后的最终复核使用预先声明的不同 seed，并记录 BER、PEFT 接受/拒绝、Data BCE、logit 变化、判决方向翻转与 soft-output 变化。原始 GPU 日志在本机被 `.gitignore` 排除，版本库保留汇总表和配置；运行中途或修正前的日志只作执行诊断，不作为最终估计。

| gap | 单动作 | seed × 帧 | Frozen BER | 在线 BER | 平均改善 | 正收益 seed | PEFT 尝试 / 接受 / 拒绝 | 判断 |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0 s | 相位趋势信号重构 | 5 × 24 | 5.0499% | 5.0532% | -0.0033 pp | 1/5 | 32 / 19 / 13 | 策略修正后复核；未证明增益 |
| 600 s | 相位趋势信号重构 | 5 × 24（策略修正后） | 27.7311% | 27.7203% | +0.0109 pp | 2/5 | 32 / 27 / 5 | 2 seed 未更新；总体收益很小 |
| 600 s | 相位趋势信号重构 | 5 × 60（旧策略） | 31.9961% | 31.9740% | +0.0221 pp | 3/5 | 32 / 26 / 6 | 仅作策略修正前诊断 |
| 600 s | Adapter/LoRA 保守，BCE | 5 × 60（旧策略） | 31.9961% | 31.9961% | 0 pp | 0/5 | 32 / 0 / 32 | Reward 拒绝所有更新 |
| 600 s | Adapter/LoRA 轻量，BCE | 5 × 24（旧策略） | 32.6769% | 32.6769% | 0 pp | 0/5 | 16 / 0 / 16 | Reward 拒绝所有更新 |

`pp` 表示 BER 百分点。两组策略修正后的 5 dB 回放里，gap=0 配对收益依 seed 为 `-0.0163, 0, +0.0054, -0.0054, 0 pp`；gap=600 s 为 `0, 0, +0.0271, -0.0054, +0.0326 pp`。gap=600 s 的 32 次 PEFT 尝试有 27 次通过当前帧 Reward；接受帧事后 Data BCE 平均改善 `0.00172`，logit 平均改变量 `0.00871`，方向翻转率 `0.092%`，soft-output 平均变化 `0.00075`。这些改善仍没有转成跨 seed 稳定的硬 BER 收益。

策略修正前的相位趋势 5 × 60 试验按 20 帧区间的平均改善依次为 `+0.004`、`-0.003`、`+0.065 pp`；当时两个 seed 从未抽到 PEFT 动作。这组旧结果只用于解释之前的探索缺陷，不作为当前策略收益估计。

Adapter/LoRA 保守步长的 32 次尝试中，Reward Pilot BCE 平均只改善约 `1.6e-6`，候选参数变化惩罚大于该改善，故全部拒绝。这说明“简单扩大可训练参数组”并未自动改善 Reward 到 Data 的迁移。

**表中注明“旧策略”的三组实验仍采用旧全零状态初始化和错误的 rollback 动作记账，因此只能判断安全门控和 PEFT 更新对象，不能作为状态条件 PPO 有效性的结论。**策略状态初始化和动作编号修复后的两组结果也未达到稳定提升门槛。

## 当前判断与下一步

- 执行实现已证明参数更新、Reward 拒绝回滚和 Data 事后评估可以按要求隔离；离线 checkpoint 没有变化。
- 真实配对测试已证明未发生 PEFT 更新时，RL identity 与 Frozen 完全一致；先前的身份帧差异来自 tail 传递不一致，修复前的 RL BER 不应用于 PEFT 收益判断。
- 当前还没有证明在线微调稳定超过 Frozen，更没有完成 Level B 四个主 SNR 的完整矩阵。
- 下一步针对 gap=600 时 Adapt/Reward 改善偶尔无法稳定迁移到 Data BER 的问题，做 Pilot-only replay：一次只变一个可训练参数组或更新目标，测 Adapt/Reward 指标对事后 Data 连续指标的排序。稳定性门槛通过前不扩大 SNR 矩阵，也不引入 bandit。

修正后复现命令模板：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe compare.py `
  --config configs/rl_peft_phase_signal_diagnostic.json `
  --pretrained D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt `
  --methods 'Frozen Offline NN' 'RL-Modulated Neural Block Equalizer' `
  --delays 116 --snrs 5 --seed-start 52 --num-seeds 5 --frames 24 `
  --pilot-total 256 --reward-pilot-total 32 --pilot-layout prefix `
  --acquisition-to-data-gap-seconds 0 --update-interval 8 --cir-update fixed `
  --online-condition-source pilot_cir_phase --device cuda `
  --output-dir logs/rl_peft_phase_signal_gap0_5db_5s24f_seed52_56
```

将命令中的 gap、seed 起点和输出目录固定记录后，再运行 `gap=600` 的独立诊断；两种 gap 不能混合汇总。
