# 离线神经均衡器训练过程与传统基线结果

## 结论

已存在的 Level B 多 seed 主矩阵支持一个明确结论：**在相同长回波信道和公平 Pilot 条件下，离线训练得到的神经均衡器比本矩阵中的强传统均衡器低很多 BER。** 0/5/10/15 dB 的 Frozen Offline NN BER 为 22.19% / 5.96% / 1.06% / 0.216%；与 CFO+DD-Phase DFE-RLS 相比，BER 点估计分别降低 49.6% / 67.4% / 84.7% / 95.1%。三个独立 seed 中每个 SNR 的 Frozen 均低于该传统基线。

这证明离线模型提供了强均衡基础。它**不等于**在线神经参数微调已经带来同样的额外收益；在线结果须单独与同源 Frozen 网络比较。

## 训练过程与信息边界

当前有完整指标与权重指纹的离线模型是 `eme_bce_all_32_20260905_pilot256`。配置为 `configs/eme_long_memory_v2.json`，损失为全帧等权二元交叉熵：

```text
BCE_all = mean(BCE(logits_all, bits_all))
```

训练样本生成接收帧和完整离线监督标签。模型的已知符号输入只在 Adapt Pilot 位置填入 Pilot，其余位置置零；整帧 Data 标签仅用于离线反向传播和 checkpoint 验证，不作为推理输入。checkpoint 的选择指标是独立验证切片的 Data BER，而不是 Pilot BCE。

配置与权重元数据显示：BPSK 单载波、1024 符号帧、最大信道时延 116 符号；4层迭代展开、`d_model=64`、4个注意力头、Adapter/LoRA rank 8、Pilot-conditioned 网络、8次物理 warm start。训练日志记录 32 个 `estimated_cir_level_b` 更新步，条件 CIR 来源为 acquisition。Pilot=256，前缀布局，Adapt=224、Reward=32。

原始命令没有随 checkpoint 保存。按配置和训练指标可以重构为：

```powershell
python pretrain.py --config configs/eme_long_memory_v2.json --stage all --steps 32 --batch-size 1 --save-dir pretrained/eme_bce_all_32_20260905_pilot256
```

这是**重构的复现命令**；元数据不能确认当时是否显式指定 GPU、AMP 和随机种子。内部 checkpoint 验证每个 SNR 只有1帧（共4帧，平均 Data BER 6.58%），因此强基线结论主要由下述独立主矩阵支持，而不是由这4帧验证集单独支撑。

当前复现链仍不完整：checkpoint 没保存原始源码 revision、Python/PyTorch/CUDA 环境、完整优化器超参数及显式随机种子；在这些信息恢复前，上述命令不能称为严格复现。不要用当前 checkout 的默认值推定历史运行值。

## 主矩阵与 BER

结果来自 `logs/eme_pilot256_bandit_pilotsparse_main_60f_3s_final/`，同一 Level B 长回波 profile、116符号最大时延、1200秒相干时间、零 acquisition 空档、`cfo_phase_tiny`；0/5/10/15 dB，每点3个独立 seed、每个 seed 连续60帧。12种方法共8640条逐帧记录。传统对照使用合法 Pilot 驱动的 CFO/相位补偿 LMMSE-FIR 与 DFE-RLS，均为非神经、非RL接收机。

| SNR | Frozen Offline NN | CFO+DD-Phase DFE-RLS | CFO+DD-Phase LMMSE-FIR | NN相对DFE的BER降低 |
|---:|---:|---:|---:|---:|
| 0 dB | 22.1947% | 44.0524% | 44.0054% | 49.6% |
| 5 dB | 5.9599% | 18.2986% | 22.4306% | 67.4% |
| 10 dB | 1.0598% | 6.9090% | 7.2229% | 84.7% |
| 15 dB | 0.2156% | 4.3743% | 5.0456% | 95.1% |

该离线表不把同一矩阵的 Online 接收机总 BER 并入神经 PEFT 增益；Online 分支还包含物理状态跟踪、Bandit 调度等流程，而该矩阵中没有有效接受的 PEFT 更新。在线研究要证明的是在同源 Frozen 基础上额外的、可归因的神经参数更新收益。

这里的 `Frozen Offline NN` 冻结神经权重，但仍按公平协议使用当前帧 Pilot 形成推理条件；传统接收机也使用合法 Pilot 和其校准的 CFO/相位补偿。因此这里的准确表述是“离线训练的神经均衡器在公平 Pilot 条件下优于本矩阵中的传统均衡器”，不应说成完全不使用当前 Pilot 的接收机。3个 seed 的方向一致，但统计单位仍是独立信道会话；每个 SNR 仅3个 seed，不能把180帧当作180个独立样本来夸大显著性。

逐 seed 主矩阵、每臂错误数及来源文件指纹见机器快照。现有展示的四个 SNR 聚合 BER 已足以说明该开发矩阵中的方向一致性；每 SNR 只有3个独立会话，结论不替代更大样本的正式统计检验。

## 资产、版本与复现限制

权威 checkpoint 位于原项目工作区 `D:\Research\RL4EQ\pretrained\eme_bce_all_32_20260905_pilot256\model_best.pt`，SHA256 为 `8cd3912d1c957c9689c51730aff9d27a8f74942186c802a7483e5781af65aec4`。训练指标、模型配置和独立主矩阵的文件指纹及逐 seed BER 已汇总到[机器可读结果快照](results/offline_equalizer_baseline_snapshot_20261105.json)。主矩阵原始摘要哈希为 `3148679b040806af8ebf605b542b7970639d510dadcba505e2afd4fa58203913`，逐帧日志哈希为 `872e8ad0e6a496b9cf63c4e0069f6a4d00a2365ebcb08917353de4d2b581aa82`。

checkpoint 与原始大日志目前不在这个 Codex worktree 的 `pretrained/` 和 `logs/` 目录中，但仍在原项目路径可读。当前 worktree 无法仅凭 Git checkout 独立重跑该矩阵；要在此 worktree 复现，需要恢复 checkpoint 和原始评估资产，并核对对应的训练源码版本。历史的 Pilot=128、30秒 acquisition 空档、旧 Frozen 条件边界结果不并入这张主表。跨帧 34 个 memory scale 参数的 A/B 试验也不是传统均衡器对照，不作为离线模型胜过 baseline 的证据。

## 与在线研究的接口

这张表固定了研究的起点：离线均衡能力已经明显优于传统 baseline。后续在线试验必须从同一个离线 checkpoint 出发，Frozen 和 Online 匹配当前 Pilot 推理条件、物理状态跟踪、接收窗口、soft-tail递推、Pilot预算和信道波形。只有神经参数被在线更新的分支相对 Frozen 产生额外收益，才能归因于用户的核心研究点。

后续持续适配还新增时间趋势假设：**随着会话推进，Online 相对 Frozen 的配对 BER 收益应总体增加。** 用独立会话的固定长度窗口计算 `gain = BER_frozen - BER_online`，比较早期与晚期窗口并估计时间斜率；不要求噪声下每帧严格单调，但须完整报告四个 SNR、每个独立会话曲线、累积曲线、回滚和计算代价。统计单位是会话/seed，而不是帧或 bit。在线 Data 真值仍只能用于事后评分。
