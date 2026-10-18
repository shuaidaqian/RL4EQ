# RL4EQ：长记忆 ISI 信道的 Pilot-only 状态跟踪接收机

项目目标是在 Level B 极端稀疏长回波信道上，离线预训练基础神经块均衡器，在线仅使用帧前缀 Pilot 适配时变状态：多径 tap 慢漂移、residual CFO、慢相位噪声、acquisition-to-data 老化和跨帧 soft-tail 误差。Data 段只做事后硬 BER 评估。

当前主线是：

```text
Level B 长记忆信道轨迹
-> 在线 soft-tail 条件下离线 BCE 预训练
-> 冻结离线均衡器主干
-> Adapt Pilot 更新 CIR/CFO/phase 后验
-> 计算 Pilot-only 物理风险与不确定度
-> 低置信度时 abstain，安全时原子提交状态
-> 冻结均衡器整帧检测并维护下一帧 tail belief
-> 用 Data 硬 BER 做事后评估
```

当前主方法不是在线 PEFT 或 PPO。历史 PEFT、Bandit、窗口 PPO 仍可用于失败复现和消融，但它们不能绕过 Pilot 代理排序验证直接改变模型参数。Reward Pilot BCE 已经出现与 Data BER 负相关，因而不再是默认控制目标。

## 研究契约

- 主配置：`eme_long_memory_v2`、Level B、物理最大记忆 116、SNR `0/5/10/15 dB`、prefix Pilot。
- 在线状态：稀疏 CIR 均值/方差、residual CFO、phase0、innovation variance、soft-tail 均值/方差/置信度。
- 信息边界：Adapt Pilot 可更新状态；Reward Pilot 只用于已知 Pilot 物理一致性留出；Data 标签只用于离线监督和最终 BER。
- 主比较：传统 LMMSE/DFE/RLS/SC-FDE 与 `Frozen Offline NN`、`Pilot State Tracking Frozen NN`。Perfect-CSI 和 oracle 只作诊断。
- 代理验收：在独立状态轨迹上按 SNR、delay、漂移和 gap 分桶检查 Spearman/Kendall、top-1 误选率、置信区间和 abstain 率；不通过就保持 Frozen。

## 入口

```powershell
python pretrain.py --config configs/level_b_state_tracking.json --stage all --steps 2 --batch-size 1 --amp --save-dir pretrained/state_smoke
python online_train.py --config configs/level_b_state_tracking.json --pretrained pretrained/state_smoke/model_best.pt --method state --frames 8 --num-seeds 1 --delays 116 --snrs 10 --pilot-total 128 --pilot-layout prefix --output-dir logs/state_smoke
python compare.py --config configs/level_b_state_tracking.json --method-group main --pretrained pretrained/state_smoke/model_best.pt --delays 116 --snrs 10 --num-seeds 1 --frames 1 --pilot-total 128 --pilot-layout prefix --output-dir logs/compare_smoke
python -m pytest -q -p no:cacheprovider
```

正式在线运行必须提供存在且 strict-load 成功的 `model_best.pt`。入口会在输出中记录 `pretrained_loaded`、`online_parameter_update`、`data_labels_used_online`、`online_waveform_visibility` 和 Pilot 风险字段。

## 目录

- `agent/state_tracking.py`：物理状态后验、tail belief 和原子提交。
- `agent/pilot_risk.py`：Pilot-only 重构、创新白化、不确定度和 abstain 风险代理。
- `agent/cir_estimator.py`：稀疏 CIR、phase/CFO 的 Pilot 估计器。
- `agent/unfolded_equalizer.py`：离线预训练的整帧非因果块均衡器。
- `training/state_tracking_online.py`：正式冻结主干在线入口。
- `training/online_adaptation.py`、`training/windowed_discrete_ppo.py`：历史 PEFT/PPO 诊断入口。
- `docs/state_tracking_receiver_design.md`：唯一权威接收机设计。
- `docs/state_tracking_validation_protocol.md`：分层基线与代理验收协议。
- `docs/legacy_online_routes.md`：旧路线和失败证据索引。

旧文档中的实验数字保持追溯性，但均以“历史诊断/已被当前主路线取代”解释，不可直接作为当前主方法定义。

历史兼容说明：`RL-Modulated Neural Block Equalizer` 曾作为 proposed；接收机仍是整帧缓冲、非因果，主研究要求在 CFO、相位扰动下以合理补偿为前提明显超过传统，并且不使用数据标签上界。
