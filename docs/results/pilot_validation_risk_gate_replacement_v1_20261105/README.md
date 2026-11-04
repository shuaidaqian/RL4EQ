# 第十三轮：Pilot-only 风险分数替代验证硬门控的反事实

## 固定比较

本轮使用 v12 训练出的同一个线性风险模型、岭系数0.1、固定决策阈值0和同一8维v11神经均衡器，分别回放：原验证Pilot硬门控、在硬门控后附加风险过滤、以及由风险分数替代验证Pilot BCE/硬BER门控。替代策略仍保留逐步/累计投影及离线初始值回退；不接收Data或真实信道真值。三种策略各自维护在线系数历史。信道、Pilot预算、同步/物理状态跟踪和接收窗口完全不变；无SNR专属门槛。

清单SHA256：`d6e66049437a0c884218b8a668c98d10e43cdfd87f782981a88e0c2fafe2a90a`。风险回归器没有重新拟合；仍只用131个训练候选，并在66个选择候选上报告6个误收、8个误拒。评估角色继续是已用于开发的会话，不是独立确认。

## 四SNR反事实结果

有害接受定义为事后候选 Data BCE 未改善；该 Data 量只用于离线评分。

| SNR | 原硬门控：接受/有害 | 附加过滤：接受/有害 | 风险分数替代：接受/有害 | BER增益：硬门控 / 过滤 / 替代 |
|---:|---:|---:|---:|---:|
| 0 dB | 45 / 3 | 41 / 3 | 28 / 4 | `+1.36e-5 / +1.36e-5 / 0` |
| 5 dB | 43 / 6 | 38 / 5 | 19 / 4 | `-1.36e-5 / -1.36e-5 / 0` |
| 10 dB | 3 / 3 | 2 / 2 | 36 / 16 | `0 / 0 / 0` |
| 15 dB | 0 / 0 | 0 / 0 | 16 / 6 | `0 / 0 / 0` |

替代门控在0/5 dB分别避免原方法的小幅改善/退化，在线BER都回到无变化；10/15 dB风险分数放行了更多动作，其中不少动作事后Data BCE有害，BER仍没有改变。附加过滤没有改变四点BER，0 dB还拒绝了有益候选。以上动作风险比例是同一开发评估会话上的离线统计，不构成独立性能推断。

## 结论与边界

简单将同一个风险回归器作为附加门控或替代门控，都没有形成在线BER增益。替代门控在低SNR抑制了更新次数，在高SNR却放行许多没有硬误码收益、部分Data BCE退化的动作；因此当前线性分数不适合作为部署策略。风险模型的选择角色相关约0.465，不能支撑更强的判决主张。

这轮反事实不证明所有Pilot-only风险估计都不可行，但当前提案已被否定，不应继续靠在同一评估集调阈值、按SNR开关或扩大同类seed来找收益。**核心目标仍未完成，不能进入R3。**下一步先暂停风险门控分支，汇总现有发现并冻结应用主要区间、退化界限和统计方案；之后再决定研究更丰富但仍只读Pilot的风险特征，或把资源转回离线表示/更新幅度。独立确认只能用新的会话和预注册判据。

## 复现

```powershell
& 'D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe' -m pytest -q -p no:cacheprovider tests/test_detection_subspace.py
& 'D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe' scripts/train_pilot_risk_gate.py --manifest configs/audits/pilot_validation_risk_gate_replacement_v1.json --base-training-dir logs/detection_subspace_training_v11_smooth_ber_outer_v1_20261104 --output-dir logs/pilot_validation_risk_gate_replacement_v1_20261104 --device cuda
```

归档包括固定清单、复用的风险模型与逐动作反事实记录。特征缓存未提交。

## 软件验证

- v13定向测试：31项通过。
- v13代码改动后的全量测试：`python -m pytest -q -p no:cacheprovider`，932项通过，耗时370.78秒。
- `git diff --check` 在提交前执行；本结果只验证代码一致性，不改变上述研究结论。
